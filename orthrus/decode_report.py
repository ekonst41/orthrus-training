"""Analysis of decoding-cycle experiments (orthrus.decode_exp): headline across models, checks,
TPF with confidence intervals, anatomy of rejected blocks, offline search over tail policies.

    python -m orthrus.decode_report --config configs/decode/tail-v1.yaml [--runs a,b]
        [--grid full|small|none] [--workers 4]

Reads experiments/<run>/ (downloaded from the bucket into <storage.local_dir>/experiments/<run>/;
works while the job is still running), prints a Markdown report and writes it, with summary.json
and policy_grid.csv, to <storage.local_dir>/experiments/<run>/analysis/.

Every TPF here is replayed from the draft tables (decoding.OracleBackend) on every decoded prompt:
exact for greedy decoding, which the run checks against real decodes (sim_match, draft_mismatch,
match_base, match_ar: section "Checks"). Definitions (per suite over all its prompts; "avg" = mean
over suites, like the paper's Table 1):
  tpf           new tokens / forward passes (prefill, drafts, verifications, tail passes)
  acceptance    accepted drafts per verification of a block (the bonus token excluded)
  tail yield    tokens a tail pass keeps (accepted + the AR token): a third pass raises TPF only
                if its yield exceeds the TPF
  verify tokens input tokens per AR pass (its compute: tree hypotheses enlarge it)
Confidence intervals: 95% percentile bootstrap over prompts (2000 resamples, seed 0), suites
resampled separately; differences to base are paired (the same resampled prompts).
Policy search: deployable policies are ranked by their dev average (even ranks) and reported on
the test prompts (odd ranks). "oracle>=k" policies peek at the outcome before a tail pass (a
perfect gate): an upper bound for any gating rule, never selected.
"""

import argparse
import csv
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from orthrus.config import Config, parse_args
from orthrus.decoding import OracleBackend, Strategy, decode, parse_strategies
from orthrus.storage import Bucket

log = logging.getLogger("orthrus.decode_report")

PAPER_TPF = {  # arXiv 2605.12825, Table 1, T = 0 (benchmarks we also run)
    "1.7B": {"gsm8k": 4.20, "math500": 4.71, "humaneval": 2.75, "mbpp": 2.76},
    "4B": {"gsm8k": 4.91, "math500": 6.02, "humaneval": 3.57, "mbpp": 3.74},
    "8B": {"gsm8k": 5.30, "math500": 6.35, "humaneval": 3.94, "mbpp": 3.95},
}
SHIFT_SETS = ((0,), (1,), (-1,), (1, 0), (0, -1), (1, 0, -1), (1, 0, -1, -2), (1, 0, -1, -2, -3))
TREE = (1, 0, -1, -2)
CEILINGS = [
    Strategy(f"perfect gate >={k}", "tail", TREE, oracle_min_tokens=k) for k in range(1, 11)
]
ANATOMY_SHIFTS = (1, 0, -1, -2, -3)
BOOTSTRAP, SEED = 2000, 0


# --- Loading and replay ----------------------------------------------------------------------


def load_traces(path: Path, max_new_tokens: int | None = None) -> list[dict]:
    data = np.load(path)
    offsets = data["offsets"]
    traces = []
    for i, rank in enumerate(data["rank"]):
        rows = slice(offsets[i], offsets[i + 1])
        drafts = data["drafts"][rows]
        traces.append(
            {
                "suite": str(data["suite"][i]),
                "rank": int(rank),
                "split": "dev" if rank % 2 == 0 else "test",
                "prompt_len": int(data["prompt_len"][i]),
                "reference": data["reference"][rows].tolist(),
                "block_size": int(data["block_size"]),
                "max_new_tokens": int(max_new_tokens or data["max_new_tokens"]),
                "drafts": [row[row >= 0].tolist() for row in drafts],
                "confidence": data["confidence"][rows].tolist(),
            }
        )
    return traces


def replay(trace: dict, strategy: Strategy, eos: set[int], backend_cls=OracleBackend):
    backend = backend_cls(
        trace["prompt_len"],
        trace["reference"],
        trace["drafts"],
        trace["confidence"],
        trace["block_size"],
    )
    gen = decode(backend, strategy, trace["prompt_len"], trace["max_new_tokens"], eos)
    if gen.tokens != trace["reference"]:
        raise RuntimeError(f"replay of {strategy.name} did not reproduce the reference")
    return gen, backend


def _outcome(gen) -> tuple:
    """(tokens, passes, cycles, accepted, tail passes, tail tokens kept, AR input, AR passes)."""
    ar_kinds = ("verify", "tail")
    return (
        len(gen.tokens),
        gen.forward_passes,
        len(gen.accepted),
        sum(gen.accepted),
        len(gen.tails),
        sum(t[2] for t in gen.tails) + len(gen.tails),
        sum(gen.input_tokens.get(k, 0) for k in ar_kinds),
        sum(gen.passes.get(k, 0) for k in ar_kinds),
    )


def _replay_job(job) -> list[tuple]:
    trace, policies, eos = job
    return [_outcome(replay(trace, policy, eos)[0]) for policy in policies]


def replay_all(traces: list[dict], policies: list[Strategy], eos: set[int], workers: int):
    """outcomes[prompt][policy] (see _outcome)."""
    jobs = [(trace, policies, eos) for trace in traces]
    if workers > 1 and len(jobs) > 8:
        with ProcessPoolExecutor(workers) as pool:
            return list(pool.map(_replay_job, jobs, chunksize=4))
    return [_replay_job(job) for job in jobs]


# --- Statistics ------------------------------------------------------------------------------


def bootstrap_tpf(tokens: np.ndarray, passes: np.ndarray, base: int = 0) -> list[dict]:
    """tokens, passes: [strategy, prompt] of one suite. Per strategy: TPF, 95% CI, the paired
    difference to row `base` with its CI, and the shares of prompts with fewer / more passes."""
    rng = np.random.default_rng(SEED)
    n = tokens.shape[1]
    index = rng.integers(0, n, size=(BOOTSTRAP, n))
    sampled = tokens[:, index].sum(-1) / passes[:, index].sum(-1)  # [strategy, resample]
    point = tokens.sum(-1) / passes.sum(-1)
    delta = sampled - sampled[base]
    return [
        {
            "tpf": float(point[i]),
            "ci": [float(x) for x in np.percentile(sampled[i], [2.5, 97.5])],
            "delta": float(point[i] - point[base]),
            "delta_ci": [float(x) for x in np.percentile(delta[i], [2.5, 97.5])],
            "fewer_passes": float(np.mean(passes[i] < passes[base])),
            "more_passes": float(np.mean(passes[i] > passes[base])),
        }
        for i in range(tokens.shape[0])
    ]


def bootstrap_avg(per_suite: list[tuple[np.ndarray, np.ndarray]], base: int = 0) -> list[dict]:
    """Mean over suites of the per-suite TPF, each suite resampled separately; relative change to
    row `base` in percent with its paired CI."""
    rng = np.random.default_rng(SEED)
    sampled, point = 0.0, 0.0
    for tokens, passes in per_suite:
        n = tokens.shape[1]
        index = rng.integers(0, n, size=(BOOTSTRAP, n))
        sampled = sampled + tokens[:, index].sum(-1) / passes[:, index].sum(-1)
        point = point + tokens.sum(-1) / passes.sum(-1)
    sampled, point = sampled / len(per_suite), point / len(per_suite)
    relative = 100 * (sampled / sampled[base] - 1)
    return [
        {
            "tpf": float(point[i]),
            "ci": [float(x) for x in np.percentile(sampled[i], [2.5, 97.5])],
            "change_pct": float(100 * (point[i] / point[base] - 1)),
            "change_ci": [float(x) for x in np.percentile(relative[i], [2.5, 97.5])],
        }
        for i in range(len(point))
    ]


def _arrays(outcomes: list[list[tuple]], traces: list[dict], suite: str, policies: list[int]):
    rows = [i for i, t in enumerate(traces) if t["suite"] == suite]
    tokens = np.array([[outcomes[i][p][0] for i in rows] for p in policies], float)
    passes = np.array([[outcomes[i][p][1] for i in rows] for p in policies], float)
    return tokens, passes


def _cycle_stats(outcomes: list[list[tuple]], p: int) -> dict:
    total = np.array([o[p] for o in outcomes], float).sum(0)
    tokens, passes, cycles, accepted, tails, kept, ar_input, ar_passes = total
    return {
        "tpf": tokens / passes,
        "acceptance": accepted / cycles if cycles else None,
        "tokens_per_cycle": (tokens - len(outcomes)) / cycles if cycles else None,
        "tail_rate": tails / cycles if cycles else None,
        "tail_yield": kept / tails if tails else None,
        "verify_tokens": ar_input / ar_passes if ar_passes else None,
    }


def _mean(values) -> float | None:
    values = [float(v) for v in values if v is not None]
    return sum(values) / len(values) if values else None


# --- Anatomy of rejected blocks ----------------------------------------------------------------


class _RecordingOracle(OracleBackend):
    def __init__(self, *args):
        super().__init__(*args)
        self.blocks: list[tuple[int, list[int]]] = []

    def draft(self, start, anchor, block_len, confidence):
        drafts = super().draft(start, anchor, block_len, confidence)
        self.blocks.append((start, drafts.tokens))
        return drafts


def _prefix_match(tokens: list[int], reference: list[int], begin: int) -> int:
    n = 0
    while n < len(tokens) and begin + n < len(reference) and tokens[n] == reference[begin + n]:
        n += 1
    return n


def anatomy(traces: list[dict], eos: set[int]) -> dict:
    """What the rejected tails of the paper's cycle hold: for every base cycle that rejected a
    draft (except a generation's last cycle), the tokens each shift hypothesis would keep."""
    rows, full, cycles = [], 0, 0
    for trace in traces:
        gen, backend = replay(trace, Strategy("base"), eos, _RecordingOracle)
        p, ref = trace["prompt_len"], trace["reference"]
        max_length = p + trace["max_new_tokens"]
        blocks = list(zip(backend.blocks, gen.accepted, strict=True))
        for c, ((start, drafts), accepted) in enumerate(blocks):
            cycles += 1
            if accepted == len(drafts):
                full += 1
                continue
            bonus_at = start + accepted + 1
            room = max_length - 1 - bonus_at
            if c == len(blocks) - 1 or room <= 0:
                continue
            kept = {}
            for shift in ANATOMY_SHIFTS:
                begin = accepted + 1 - shift
                tail = drafts[begin : begin + room]
                kept[shift] = _prefix_match(tail, ref, bonus_at - p + 1) if tail else None
            rows.append({"tail": len(drafts) - accepted - 1, "kept": kept})
    best = [max((v for v in r["kept"].values() if v is not None), default=0) for r in rows]
    by_shift = {}
    for shift in ANATOMY_SHIFTS:
        values = [r["kept"][shift] for r in rows if r["kept"][shift] is not None]
        by_shift[shift] = {
            "any_kept": _mean(v > 0 for v in values),
            "mean_kept": _mean(values),
        }
    buckets = {"0": 0, "1-2": 0, "3-5": 0, "6-10": 0, "11+": 0}
    for b in best:
        key = "0" if b == 0 else "1-2" if b <= 2 else "3-5" if b <= 5 else "6-10"
        buckets[key if b <= 10 else "11+"] += 1
    return {
        "cycles": cycles,
        "fully_accepted": full / cycles if cycles else None,
        "rejections_with_tail": len(rows),
        "mean_tail_length": _mean(r["tail"] for r in rows),
        "by_shift": by_shift,
        "tree_pass_yield": 1 + _mean(best) if rows else None,
        "best_kept_distribution": {k: v / len(rows) for k, v in buckets.items()} if rows else {},
    }


# --- Report helpers --------------------------------------------------------------------------


def _fmt(value, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(c if isinstance(c, str) else _fmt(c) for c in row) + " |")
    return "\n".join(lines)


def _tpf_ci(entry: dict) -> str:
    return f"{entry['tpf']:.2f} [{entry['ci'][0]:.2f}, {entry['ci'][1]:.2f}]"


def _delta_ci(entry: dict) -> str:
    low, high = entry["delta_ci"]
    return f"{entry['delta']:+.3f} [{low:+.3f}, {high:+.3f}]"


def _pct_ci(entry: dict) -> str:
    low, high = entry["change_ci"]
    return f"{entry['change_pct']:+.1f}% [{low:+.1f}, {high:+.1f}]"


def fetch_run(cfg: Config, run: str) -> Path:
    root = Path(cfg.storage.local_dir) / "experiments" / run
    bucket = Bucket(cfg.storage.bucket)
    if bucket.enabled:
        bucket.download(f"experiments/{run}", root)
    if not (root / "run.json").exists():
        raise FileNotFoundError(f"no experiment {run} locally or in the bucket")
    return root


def _load_model_dir(directory: Path) -> tuple[dict, list[dict], list[dict]]:
    """model.json, real-decode records and traces of the complete rounds of one model."""
    info = json.loads((directory / "model.json").read_text(encoding="utf-8"))
    records, traces = [], []
    for path in sorted(directory.glob("round-*.jsonl")):  # the .jsonl marks a complete round
        records += [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
        traces += load_traces(path.with_suffix(".npz"))
    return info, records, traces


# --- Analysis --------------------------------------------------------------------------------


def analyze_model(info, records, traces, strategies, suites, grid: str, workers: int) -> dict:
    eos = set(info["eos_token_ids"])
    suites = [s for s in suites if any(t["suite"] == s for t in traces)]
    policies = [*strategies, *CEILINGS]
    outcomes = replay_all(traces, policies, eos, workers)
    names = [s.name for s in strategies]
    result = {"prompts": len(traces), "suites": suites, "per_suite": {}, "strategies": names}
    per_suite_arrays = []
    for suite in suites:
        tokens, passes = _arrays(outcomes, traces, suite, list(range(len(strategies))))
        per_suite_arrays.append((tokens, passes))
        result["per_suite"][suite] = dict(zip(names, bootstrap_tpf(tokens, passes), strict=True))
    result["avg"] = dict(zip(names, bootstrap_avg(per_suite_arrays), strict=True))
    result["cycles"] = {name: _cycle_stats(outcomes, i) for i, name in enumerate(names)}
    # perfect-gate ceiling: the best threshold (chosen on all prompts: an upper bound)
    ceiling = []
    for c, policy in enumerate(CEILINGS):
        index = [0, len(strategies) + c]
        arrays = [_arrays(outcomes, traces, s, index) for s in suites]
        ceiling.append((bootstrap_avg(arrays)[1], policy.name))
    best_ceiling, ceiling_name = max(ceiling, key=lambda x: x[0]["tpf"])
    result["ceiling"] = {"policy": ceiling_name, **best_ceiling}
    result["anatomy"] = {s: anatomy([t for t in traces if t["suite"] == s], eos) for s in suites}
    result["checks"] = _checks(records)
    result["timing"] = _timing(records)
    if grid != "none":
        result["grid"] = search(policy_grid(grid), traces, suites, eos, workers)
    return result


def _checks(records: list[dict]) -> dict:
    out = {}
    for name in dict.fromkeys(r["strategy"] for r in records):
        rows = [r for r in records if r["strategy"] == name]
        checked = sum(r.get("draft_checked", 0) for r in rows)
        out[name] = {
            "decodes": len(rows),
            "match_base": _mean(r["match_base"] for r in rows),
            "match_ar": _mean(r.get("match_ar") for r in rows),
            "sim_match": _mean(r.get("sim_match") for r in rows),
            "draft_mismatch": (
                sum(r.get("draft_mismatch", 0) for r in rows) / checked if checked else None
            ),
        }
    return out


def _timing(records: list[dict]) -> dict:
    base = [r for r in records if r["strategy"] == "base"]
    if not base:
        return {}
    passes = sum(r["passes"] for r in base)
    return {
        "base_tokens_per_second": sum(r["new_tokens"] for r in base)
        / sum(r["seconds"] for r in base),
        "base_ms_per_pass": 1000 * sum(r["seconds"] for r in base) / passes,
        "mean_new_tokens": sum(r["new_tokens"] for r in base) / len(base),
        "truncated": _mean(not r["eos"] for r in base),
    }


def policy_grid(kind: str) -> list[Strategy]:
    small = kind == "small"
    shift_sets = ((0,), TREE) if small else SHIFT_SETS
    policies = [Strategy("base")]
    for shifts in shift_sets:
        tag = ",".join(f"{s:+d}" for s in shifts)
        policies.append(Strategy(f"reuse [{tag}]", kind="reuse", shifts=shifts))
        for min_tail in (1, 8) if small else (1, 4, 8, 16):
            for min_expected in (0, 3) if small else (0, 2, 3, 4, 5, 6):
                for passes in (1,) if small else (1, 2):
                    policies.append(
                        Strategy(
                            f"tail [{tag}] min_tail={min_tail} min_expected={min_expected} "
                            f"passes={passes}",
                            kind="tail",
                            shifts=shifts,
                            min_tail=min_tail,
                            min_expected=min_expected,
                            max_tail_passes=passes,
                        )
                    )
    return policies


def search(policies: list[Strategy], traces: list[dict], suites: list[str], eos, workers: int):
    """Replay every deployable policy; rank on dev, report on test (paired bootstrap vs base)."""
    outcomes = replay_all(traces, policies, eos, workers)
    table = []
    for p, policy in enumerate(policies):
        entry = {"policy": policy.name, **policy.to_dict()}
        for split in ("dev", "test"):
            tpfs = []
            for suite in suites:
                rows = [o[p] for o, t in zip(outcomes, traces, strict=True)
                        if t["suite"] == suite and t["split"] == split]  # fmt: skip
                tpf = sum(x[0] for x in rows) / sum(x[1] for x in rows) if rows else None
                entry[f"{split}/{suite}"] = tpf
                tpfs.append(tpf)
            entry[f"{split}/avg"] = _mean(tpfs)
        entry["tail_yield"] = _cycle_stats(outcomes, p)["tail_yield"]
        table.append(entry)
    best = max(table[1:], key=lambda e: e["dev/avg"] or 0)
    p_best = table.index(best)
    test = [i for i, t in enumerate(traces) if t["split"] == "test"]
    arrays = []
    for suite in suites:
        rows = [i for i in test if traces[i]["suite"] == suite]
        if rows:
            tokens = np.array([[outcomes[i][p][0] for i in rows] for p in (0, p_best)], float)
            passes = np.array([[outcomes[i][p][1] for i in rows] for p in (0, p_best)], float)
            arrays.append((tokens, passes))
    best_test = bootstrap_avg(arrays)[1] if arrays else None
    return {"table": table, "best": best["policy"], "best_test": best_test}


def analyze(cfg: Config, run: str, grid: str, workers: int) -> tuple[str, dict]:
    root = fetch_run(cfg, run)
    manifest = json.loads((root / "run.json").read_text(encoding="utf-8"))
    run_cfg = manifest["config"]["decode"]
    strategies = parse_strategies(manifest["config"]["decode"]["strategies"])
    names = [s.name for s in strategies]
    tags = [t for t in manifest.get("models", {}) if (root / t / "model.json").exists()]
    results, infos = {}, {}
    for tag in tags:
        info, records, traces = _load_model_dir(root / tag)
        if traces:
            infos[tag] = info
            results[tag] = analyze_model(
                info, records, traces, strategies, run_cfg["suites"], grid, workers
            )
    out = [
        f"# Decoding cycles: {run}",
        "",
        f"Official Orthrus checkpoints, {run_cfg['dtype']}, greedy, batch 1, at most "
        f"{run_cfg['max_new_tokens']} new tokens; prompt set `{run_cfg['prompt_set']}` "
        f"(sha256 {manifest['prompt_set_sha256'][:12]}); code `{manifest['git_commit'][:12]}`; "
        f"GPU {manifest.get('gpu')}; jobs "
        + ", ".join(str(s.get("datasphere_job_id")) for s in manifest["segments"])
        + ".",
        "",
        "## Headline: average TPF over suites, change vs the paper's cycle (95% CI)",
        "",
    ]
    rows = []
    for tag, r in results.items():
        avg = r["avg"]
        tail_yield = r["cycles"].get("tail-tree", {}).get("tail_yield")
        rows.append(
            [
                tag,
                f"{r['prompts']} ({r['prompts'] // max(len(r['suites']), 1)}/suite)",
                _tpf_ci(avg[names[0]]),
                *[_pct_ci(avg[name]) for name in names[1:]],
                _pct_ci(r["ceiling"]),
                f"{_fmt(tail_yield)} vs {avg[names[0]]['tpf']:.2f}",
            ]
        )
    out.append(
        _table(
            [
                "model",
                "prompts",
                f"{names[0]} TPF",
                *names[1:],
                "perfect gate (ceiling)",
                "tail-pass yield vs TPF",
            ],
            rows,
        )  # fmt: skip
    )
    out += [
        "",
        "Strategies: " + "; ".join(_describe(s) for s in strategies) + ". The perfect gate "
        "runs a tail pass (all shifts) only when it would keep at least k tokens, best k: no "
        "gating rule can do better. A third pass helps only if it keeps more tokens than an "
        "average pass of the cycle, i.e. its yield must exceed the TPF.",
    ]
    out += ["", "## Checks (real decodes vs replay and plain AR)", ""]
    rows = []
    for tag, r in results.items():
        for name, c in r["checks"].items():
            rows.append([tag, name, c["decodes"], c["match_base"], c["match_ar"], c["sim_match"],
                         c["draft_mismatch"]])  # fmt: skip
    out.append(
        _table(["model", "decode", "n", "=base", "=AR", "replay exact", "draft mism."], rows)
    )
    out.append(
        "\n`=base`: output equals the base output; `=AR`: equals plain greedy AR; `replay exact`: "
        "the replay from the draft table gives the same passes, acceptances and tails; `draft "
        "mism.`: share of visited anchors whose real drafts differ from the table."
    )
    for tag, r in results.items():
        out += _model_section(tag, infos[tag], r, names, run_cfg["dtype"])
    text = "\n".join(out) + "\n"
    analysis_dir = root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "report.md").write_text(text, encoding="utf-8")
    summary = {t: {k: v for k, v in r.items() if k != "grid"} for t, r in results.items()}
    (analysis_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    grids = {t: r["grid"]["table"] for t, r in results.items() if "grid" in r}
    if grids:
        write_grid_csv(analysis_dir / "policy_grid.csv", grids)
    return text, results


def _describe(s: Strategy) -> str:
    if s.kind == "base":
        return f"`{s.name}` draft -> verify (paper)"
    shifts = ",".join(f"{x:+d}" for x in s.shifts)
    if s.kind == "reuse":
        return f"`{s.name}` next verification also tests the rejected tail, shifts {shifts}"
    gate = f", if expected yield >= {s.min_expected}" if s.min_expected else ""
    return f"`{s.name}` + third pass on the rejected tail, shifts {shifts}{gate}"


def _model_section(tag: str, info: dict, r: dict, names: list[str], dtype: str) -> list[str]:
    out = ["", f"## {tag}: `{info['base']}@{(info.get('revision') or '')[:10]}`", ""]
    suites = r["suites"]
    out.append("TPF per suite [95% CI]:")
    out.append("")
    rows = [[n, *[_tpf_ci(r["per_suite"][s][n]) for s in suites], _tpf_ci(r["avg"][n])]
            for n in names]  # fmt: skip
    out.append(_table(["strategy", *suites, "avg"], rows))
    out += ["", "Difference to base [95% CI] (share of prompts with fewer / more passes):", ""]
    rows = []
    for n in names[1:]:
        cells = [
            f"{_delta_ci(r['per_suite'][s][n])} ({r['per_suite'][s][n]['fewer_passes']:.0%} / "
            f"{r['per_suite'][s][n]['more_passes']:.0%})"
            for s in suites
        ]
        rows.append([n, *cells, _pct_ci(r["avg"][n])])
    out.append(_table(["strategy", *suites, "avg"], rows))
    out += ["", "Cycle statistics (all prompts):", ""]
    rows = [[n, c["tpf"], c["acceptance"], c["tokens_per_cycle"], c["tail_rate"], c["tail_yield"],
             c["verify_tokens"]] for n, c in r["cycles"].items()]  # fmt: skip
    out.append(_table(["strategy", "tpf", "acceptance", "tokens/cycle", "tail passes/cycle",
                       "tail yield", "verify tokens"], rows))  # fmt: skip
    size = next((k for k in PAPER_TPF if f"-{k}" in info["base"]), None)
    if size:
        rows = [[s, PAPER_TPF[size][s], r["per_suite"][s][names[0]]["tpf"]]
                for s in suites if s in PAPER_TPF[size]]  # fmt: skip
        out += ["", "Base vs the paper (Table 1, T=0; paper length limit not stated):", ""]
        out.append(_table(["suite", "paper TPF", "our base TPF"], rows))
    t = r["timing"]
    if t:
        out.append(
            f"\nBase decodes: {t['mean_new_tokens']:.0f} new tokens on average, "
            f"{_fmt(t['truncated'])} cut by the length limit; {t['base_ms_per_pass']:.1f} ms per "
            f"pass, {t['base_tokens_per_second']:.1f} tok/s ({dtype}, HF eager, batch 1)."
        )
    out += [
        "",
        "Anatomy of the rejected blocks of the base cycle (tokens each tail hypothesis would "
        "keep after the bonus token; shift s continues with drafts[a+1-s:]):",
        "",
    ]
    rows = []
    for s, a in r["anatomy"].items():
        rows.append([
            s, a["cycles"], a["fully_accepted"], a["mean_tail_length"],
            *[f"{_fmt(a['by_shift'][x]['any_kept'])} / {_fmt(a['by_shift'][x]['mean_kept'])}"
              for x in ANATOMY_SHIFTS],
            a["tree_pass_yield"], r["per_suite"][s][names[0]]["tpf"],
        ])  # fmt: skip
    out.append(_table(["suite", "cycles", "full accept", "tail len",
                       *[f"s={x:+d}: P(kept>0) / mean" for x in ANATOMY_SHIFTS],
                       "tree yield", "base TPF"], rows))  # fmt: skip
    if "grid" in r:
        g = r["grid"]
        out += [
            "",
            f"Policy search ({len(g['table'])} policies): best on dev `{g['best']}`, on test "
            f"{_pct_ci(g['best_test']) if g['best_test'] else '- (no test prompts yet)'} vs base.",
            "",
        ]
        top = sorted(g["table"][1:], key=lambda e: -(e["dev/avg"] or 0))[:8]
        rows = [[e["policy"], e["dev/avg"], e["test/avg"], e["tail_yield"]]
                for e in [g["table"][0], *top]]  # fmt: skip
        out.append(_table(["policy", "dev avg TPF", "test avg TPF", "tail yield"], rows))
    return out


def write_grid_csv(path: Path, grids: dict[str, list[dict]]) -> None:
    rows = [{"model": tag, **entry} for tag, table in grids.items() for entry in table]
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()}
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--runs", help="comma-separated run names (default: the config's)")
    parser.add_argument("--grid", default="small", choices=("full", "small", "none"))
    parser.add_argument("--workers", type=int, default=4)
    known, rest = parser.parse_known_args(argv)
    cfg = parse_args(__doc__, rest)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for run in known.runs.split(",") if known.runs else [cfg.run_name]:
        text, _ = analyze(cfg, run, known.grid, known.workers)
        print(text)


if __name__ == "__main__":
    main()
