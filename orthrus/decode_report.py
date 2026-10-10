"""Analysis of decoding-cycle experiments (orthrus.decode_exp): checks, TPF with confidence
intervals, anatomy of rejected blocks, offline search over tail policies.

    python -m orthrus.decode_report --config configs/decode/qwen3-1.7b.yaml [--runs a,b]
        [--grid full|small|none] [--workers 4]

Reads experiments/<run>/ (downloaded from the bucket into <storage.local_dir>/experiments/<run>/),
prints a Markdown report and writes it, with summary.json and policy_grid.csv, to
<storage.local_dir>/experiments/<run>/analysis/.

Definitions (per suite over all its prompts; "avg" = mean over suites, like the paper's Table 1):
  tpf               new tokens / forward passes (prefill, drafts, verifications, tail passes)
  acceptance        accepted drafts per verification of a block (the bonus token excluded)
  tokens/cycle      new tokens after the first / cycles (a cycle includes its tail passes)
  tail rate         tail passes per cycle; tail yield: tokens kept per tail pass, including the
                    AR token. A tail pass raises TPF only if its yield exceeds the TPF.
  verify tokens     input tokens per AR pass (its compute; tree hypotheses enlarge it)
  tok/s, speedup    wall clock; speedup over plain AR on the prompts also decoded with AR
Confidence intervals: 95% percentile bootstrap over prompts (2000 resamples, seed 0); differences
to base are paired (the same resampled prompts for both strategies).
Policy search: each grid policy is replayed from the draft tables (decoding.OracleBackend, exact
for greedy decoding: checked by sim_match in the run); deployable policies are ranked by their dev
average and reported on the test prompts. "oracle>=k" policies peek at the outcome before a tail
pass (a perfect gate): they bound what any gating rule can reach and are never selected.
"""

import argparse
import csv
import json
import logging
import statistics
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
}
SHIFT_SETS = ((0,), (1,), (-1,), (1, 0), (0, -1), (1, 0, -1), (1, 0, -1, -2), (1, 0, -1, -2, -3))
ANATOMY_SHIFTS = (1, 0, -1, -2, -3)
BOOTSTRAP, SEED = 2000, 0


# --- Summaries of real runs -----------------------------------------------------------------


def _mean(values) -> float | None:
    values = [float(v) for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _metrics(rows: list[dict]) -> dict:
    tokens = sum(r["new_tokens"] for r in rows)
    passes = sum(r["passes"] for r in rows)
    cycles = sum(len(r["accepted"]) for r in rows)
    tails = [t for r in rows for t in r["tails"]]
    ar_passes = sum(r["passes_by_kind"].get(k, 0) for r in rows for k in ("verify", "tail", "ar"))
    ar_inputs = sum(r["input_tokens"].get(k, 0) for r in rows for k in ("verify", "tail", "ar"))
    checked = sum(r.get("draft_checked", 0) for r in rows)
    return {
        "prompts": len(rows),
        "tokens": tokens,
        "passes": passes,
        "tpf": tokens / max(passes, 1),
        "acceptance": sum(sum(r["accepted"]) for r in rows) / cycles if cycles else None,
        "tokens_per_cycle": (tokens - len(rows)) / cycles if cycles else None,
        "tail_rate": len(tails) / cycles if cycles else None,
        "tail_yield": (sum(t[2] for t in tails) + len(tails)) / len(tails) if tails else None,
        "verify_tokens": ar_inputs / ar_passes if ar_passes else None,
        "tokens_per_second": tokens / max(sum(r["seconds"] for r in rows), 1e-9),
        "mean_new_tokens": tokens / max(len(rows), 1),
        "truncated": _mean(not r["eos"] for r in rows),
        "match_base": _mean(r["match_base"] for r in rows),
        "match_ar": _mean(r.get("match_ar") for r in rows),
        "sim_match": _mean(r.get("sim_match") for r in rows),
        "draft_mismatch": sum(r.get("draft_mismatch", 0) for r in rows) / checked
        if checked
        else None,
    }


def summarize(records: list[dict]) -> dict:
    """{strategy: {suite: metrics, ..., "avg": paper-style means over suites}}."""
    suites = list(dict.fromkeys(r["suite"] for r in records))
    strategies = list(dict.fromkeys(r["strategy"] for r in records))
    ar_seconds = {(r["suite"], r["rank"]): r["seconds"] for r in records if r["strategy"] == "ar"}
    summary = {}
    for strategy in strategies:
        rows = [r for r in records if r["strategy"] == strategy]
        per_suite = {s: _metrics([r for r in rows if r["suite"] == s]) for s in suites}
        per_suite = {s: m for s, m in per_suite.items() if m["prompts"]}
        for suite, metrics in per_suite.items():
            paired = [r for r in rows if r["suite"] == suite and (suite, r["rank"]) in ar_seconds]
            if paired and strategy != "ar":
                metrics["speedup"] = sum(ar_seconds[(suite, r["rank"])] for r in paired) / sum(
                    r["seconds"] for r in paired
                )
        avg = {
            key: _mean(m.get(key) for m in per_suite.values())
            for key in ("tpf", "acceptance", "tokens_per_cycle", "speedup", "tokens_per_second")
        }
        summary[strategy] = {**per_suite, "avg": avg, "all": _metrics(rows)}
    return summary


def summary_table(summary: dict) -> str:
    """Plain-text TPF table with the run's validity checks (for job logs)."""
    suites = [s for s in next(iter(summary.values())) if s not in ("avg", "all")]
    lines = [
        f"{'strategy':<22}"
        + "".join(f"{s:>10}" for s in suites)
        + f"{'avg':>8}{'base=':>7}{'ar=':>6}{'sim=':>6}"
    ]
    for strategy, values in summary.items():
        if strategy == "ar":
            continue
        cells = "".join(
            f"{values[s]['tpf']:>10.3f}" if s in values else f"{'-':>10}" for s in suites
        )
        checks = [values["all"][k] for k in ("match_base", "match_ar", "sim_match")]
        base, ar, sim = ("-" if v is None else f"{v:.2f}" for v in checks)
        lines.append(f"{strategy:<22}{cells}{values['avg']['tpf']:>8.3f}{base:>7}{ar:>6}{sim:>6}")
    return "\n".join(lines)


# --- Bootstrap -------------------------------------------------------------------------------


def _paired_arrays(records: list[dict], strategies: list[str], suite: str):
    """tokens, passes [strategy, prompt] for the prompts every strategy decoded."""
    by_key = {(r["strategy"], r["rank"]): r for r in records if r["suite"] == suite}
    ranks = sorted({rank for (s, rank) in by_key if s == strategies[0]})
    ranks = [k for k in ranks if all((s, k) in by_key for s in strategies)]
    tokens = np.array([[by_key[(s, k)]["new_tokens"] for k in ranks] for s in strategies], float)
    passes = np.array([[by_key[(s, k)]["passes"] for k in ranks] for s in strategies], float)
    return tokens, passes


def bootstrap_tpf(tokens: np.ndarray, passes: np.ndarray, base: int = 0) -> dict:
    """Per strategy row: TPF, its 95% CI, and the paired difference to row `base` with its CI
    and the share of prompts where the strategy needs fewer passes."""
    rng = np.random.default_rng(SEED)
    n = tokens.shape[1]
    index = rng.integers(0, n, size=(BOOTSTRAP, n))
    sampled = tokens[:, index].sum(-1) / passes[:, index].sum(-1)  # [strategy, resample]
    point = tokens.sum(-1) / passes.sum(-1)
    delta = sampled - sampled[base]
    out = []
    for i in range(tokens.shape[0]):
        out.append(
            {
                "tpf": float(point[i]),
                "ci": [float(x) for x in np.percentile(sampled[i], [2.5, 97.5])],
                "delta": float(point[i] - point[base]),
                "delta_ci": [float(x) for x in np.percentile(delta[i], [2.5, 97.5])],
                "fewer_passes": float(np.mean(passes[i] < passes[base])),
                "more_passes": float(np.mean(passes[i] > passes[base])),
            }
        )
    return out


# --- Traces and replay -----------------------------------------------------------------------


def load_traces(path: Path) -> list[dict]:
    data = np.load(path)
    offsets = data["offsets"]
    traces = []
    for i, rank in enumerate(data["rank"]):
        rows = slice(offsets[i], offsets[i + 1])
        trace = {
            "rank": int(rank),
            "prompt_len": int(data["prompt_len"][i]),
            "reference": data["reference"][rows].tolist(),
            "block_size": int(data["block_size"]),
            "max_new_tokens": int(data["max_new_tokens"]),
        }
        if "drafts" in data:
            drafts = data["drafts"][rows]
            trace["drafts"] = [row[row >= 0].tolist() for row in drafts]
            trace["confidence"] = data["confidence"][rows].tolist()
        traces.append(trace)
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


def _replay_job(job) -> list[tuple[int, int, int, int]]:
    trace, policies, eos = job
    out = []
    for policy in policies:
        gen, _ = replay(trace, policy, eos)
        out.append(
            (len(gen.tokens), gen.forward_passes, len(gen.tails), sum(t[2] for t in gen.tails))
        )
    return out


def policy_grid(kind: str) -> list[Strategy]:
    if kind == "none":
        return []
    small = kind == "small"
    shift_sets = ((0,), (1, 0, -1, -2)) if small else SHIFT_SETS
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
        for gate in (3, 5) if small else (2, 3, 4, 5, 6, 8):
            policies.append(
                Strategy(
                    f"tail [{tag}] oracle>={gate}",
                    kind="tail",
                    shifts=shifts,
                    oracle_min_tokens=gate,
                )
            )
    return policies


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
    rows = []
    full = cycles = 0
    for trace in traces:
        gen, backend = replay(trace, Strategy("base"), eos, _RecordingOracle)
        p, ref = trace["prompt_len"], trace["reference"]
        max_length = p + trace["max_new_tokens"]
        for c, ((start, drafts), accepted) in enumerate(
            zip(backend.blocks, gen.accepted, strict=True)
        ):
            cycles += 1
            if accepted == len(drafts):
                full += 1
                continue
            bonus_at = start + accepted + 1
            room = max_length - 1 - bonus_at
            if c == len(gen.accepted) - 1 or room <= 0:
                continue
            kept = {}
            for shift in ANATOMY_SHIFTS:
                begin = accepted + 1 - shift
                tail = drafts[begin : begin + room]
                kept[shift] = _prefix_match(tail, ref, bonus_at - p + 1) if tail else None
            rows.append({"accepted": accepted, "tail": len(drafts) - accepted - 1, "kept": kept})
    best = [max((v for v in r["kept"].values() if v is not None), default=0) for r in rows]
    by_shift = {}
    for shift in ANATOMY_SHIFTS:
        values = [r["kept"][shift] for r in rows if r["kept"][shift] is not None]
        winners = sum(
            1
            for r, b in zip(rows, best, strict=True)
            if b > 0
            and r["kept"][shift] == b
            and all(v is None or v < b for s, v in r["kept"].items() if s != shift)
        )
        by_shift[shift] = {
            "hypotheses": len(values),
            "any_kept": _mean(v > 0 for v in values),
            "mean_kept": _mean(values),
            "unique_best": winners / len(rows) if rows else None,
        }
    buckets = {"0": 0, "1-2": 0, "3-5": 0, "6-10": 0, "11+": 0}
    for b in best:
        key = (
            "0"
            if b == 0
            else "1-2"
            if b <= 2
            else "3-5"
            if b <= 5
            else "6-10"
            if b <= 10
            else "11+"
        )
        buckets[key] += 1
    return {
        "cycles": cycles,
        "fully_accepted": full / cycles if cycles else None,
        "rejections_with_tail": len(rows),
        "mean_tail_length": _mean(r["tail"] for r in rows),
        "by_shift": by_shift,
        "best_kept_mean": _mean(best),
        "tree_pass_yield": 1 + _mean(best) if rows else None,  # tokens per tail pass, all shifts
        "best_kept_distribution": {k: v / len(rows) for k, v in buckets.items()} if rows else {},
    }


# --- Report ----------------------------------------------------------------------------------


def _fmt(value, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += [
        "| " + " | ".join(_fmt(c) if not isinstance(c, str) else c for c in row) + " |"
        for row in rows
    ]
    return "\n".join(lines)


def _interval(entry: dict, key: str = "tpf", ci: str = "ci") -> str:
    low, high = entry[ci]
    if key == "delta":
        return f"{entry[key]:+.3f} [{low:+.3f}, {high:+.3f}]"
    return f"{entry[key]:.3f} [{low:.3f}, {high:.3f}]"


def fetch_run(cfg: Config, run: str) -> Path:
    root = Path(cfg.storage.local_dir) / "experiments" / run
    bucket = Bucket(cfg.storage.bucket)
    if bucket.enabled:
        bucket.download(f"experiments/{run}", root)
    if not (root / "run.json").exists():
        raise FileNotFoundError(f"no experiment {run} locally or in the bucket")
    return root


def analyze(cfg: Config, run: str, grid: str, workers: int) -> tuple[str, dict]:
    root = fetch_run(cfg, run)
    manifest = json.loads((root / "run.json").read_text(encoding="utf-8"))
    run_cfg = manifest["config"]
    eos = set(manifest["eos_token_ids"])
    suites = [
        s for s in run_cfg["decode"]["suites"] if (root / "suites" / s / "records.jsonl").exists()
    ]
    records = [
        json.loads(line)
        for s in suites
        for line in (root / "suites" / s / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    strategies = [s["name"] for s in manifest["strategies"]]
    summary = summarize(records)
    model = run_cfg["model"]
    out = [
        f"# Decoding cycles: {run}",
        "",
        f"Model `{model['base']}@{(model.get('revision') or '')[:10]}`, "
        f"{run_cfg['decode']['dtype']}, greedy, batch 1, max "
        f"{run_cfg['decode']['max_new_tokens']} new tokens; prompt set "
        f"`{run_cfg['decode']['prompt_set']}` (sha256 {manifest['prompt_set_sha256'][:12]}), "
        f"{run_cfg['decode']['prompts_per_suite']} prompts per suite; code "
        f"`{manifest['git_commit'][:12]}`; GPU {manifest.get('gpu')}; jobs "
        + ", ".join(str(s.get("datasphere_job_id")) for s in manifest["segments"])
        + ".",
        f"Suites complete: {', '.join(suites)}.",
        "",
        "## Checks",
        "",
        "All strategies must reproduce the base output (`=base`), plain AR on the AR subset "
        "(`=AR`), and their replay from the draft table (`replay`); `draft mism.` is the share of "
        "visited anchors whose real drafts differ from the table.",
        "",
    ]
    rows = []
    for name in [*strategies, "ar"]:
        if name not in summary:
            continue
        m = summary[name]["all"]
        rows.append(
            [
                name,
                m["prompts"],
                m["match_base"],
                m["match_ar"],
                m["sim_match"],
                m["draft_mismatch"],
                m["truncated"],
                m["mean_new_tokens"],
            ]
        )
    out.append(
        _table(
            [
                "strategy",
                "prompts",
                "=base",
                "=AR",
                "replay",
                "draft mism.",
                "truncated",
                "new tokens",
            ],
            rows,
        )
    )

    # TPF with bootstrap, per suite
    boot = {}
    for suite in suites:
        tokens, passes = _paired_arrays(records, strategies, suite)
        boot[suite] = dict(zip(strategies, bootstrap_tpf(tokens, passes), strict=True))
    out += ["", "## TPF (95% CI) and paired difference to base", ""]
    headers = ["strategy", *suites, "avg"]
    rows = []
    for name in strategies:
        cells = [_interval(boot[s][name]) for s in suites]
        rows.append([name, *cells, summary[name]["avg"]["tpf"]])
    out.append(_table(headers, rows))
    out.append("")
    rows = []
    for name in strategies[1:]:
        cells = [
            f"{_interval(boot[s][name], 'delta', 'delta_ci')} "
            f"({boot[s][name]['fewer_passes']:.0%} / {boot[s][name]['more_passes']:.0%})"
            for s in suites
        ]
        avg_delta = summary[name]["avg"]["tpf"] - summary[strategies[0]]["avg"]["tpf"]
        rows.append([name, *cells, f"{avg_delta:+.3f}"])
    out.append("Delta TPF vs base [95% CI] (share of prompts with fewer / more passes):")
    out.append("")
    out.append(_table(headers, rows))

    out += ["", "## Cycle statistics (all suites)", ""]
    rows = []
    for name in [*strategies, "ar"]:
        if name not in summary:
            continue
        m, avg = summary[name]["all"], summary[name]["avg"]
        rows.append(
            [
                name,
                m["tpf"],
                m["acceptance"],
                m["tokens_per_cycle"],
                m["tail_rate"],
                m["tail_yield"],
                m["verify_tokens"],
                m["tokens_per_second"],
                avg["speedup"],
            ]
        )
    out.append(
        _table(
            [
                "strategy",
                "tpf",
                "acceptance",
                "tokens/cycle",
                "tail rate",
                "tail yield",
                "verify tokens",
                "tok/s",
                "speedup",
            ],
            rows,
        )
    )
    if run_cfg["decode"]["dtype"] == "float32":
        out.append(
            "\nWall-clock numbers are fp32 (no TF32): compute-bound and not representative "
            "of bf16 deployment; TPF does not depend on it."
        )

    size = next((k for k in PAPER_TPF if f"-{k}" in model["base"]), None)
    if size:
        out += ["", f"## Base vs the paper (Table 1, Orthrus-Qwen3-{size}, T=0)", ""]
        rows = [
            [s, PAPER_TPF[size][s], summary[strategies[0]][s]["tpf"]]
            for s in suites
            if s in PAPER_TPF[size]
        ]
        out.append(_table(["suite", "paper TPF", "our base TPF"], rows))

    traces = {}
    for suite in suites:
        path = root / "suites" / suite / "traces.npz"
        if path.exists():
            traces[suite] = load_traces(path)
    analysis = {"summary": summary, "bootstrap": boot}
    if traces and all("drafts" in t[0] for t in traces.values() if t):
        out += ["", "## Anatomy of rejected blocks (base cycle)", ""]
        out.append(
            "For every base cycle that rejected a draft, the tokens each tail hypothesis would "
            "keep after the bonus token (shift s continues with drafts[a+1-s:]). `unique best`: "
            "the hypothesis alone keeps the most. `tree yield` = 1 + best kept: tokens per tail "
            "pass that verifies all shifts at once; compare with the base TPF."
        )
        out.append("")
        rows = []
        analysis["anatomy"] = {}
        for suite, suite_traces in traces.items():
            a = anatomy(suite_traces, eos)
            analysis["anatomy"][suite] = a
            shift_cells = [
                f"{_fmt(a['by_shift'][s]['any_kept'])} / {_fmt(a['by_shift'][s]['mean_kept'])}"
                f" / {_fmt(a['by_shift'][s]['unique_best'])}"
                for s in ANATOMY_SHIFTS
            ]
            rows.append(
                [
                    suite,
                    a["cycles"],
                    a["fully_accepted"],
                    a["mean_tail_length"],
                    *shift_cells,
                    a["tree_pass_yield"],
                    summary[strategies[0]][suite]["tpf"],
                ]
            )
        out.append(
            _table(
                [
                    "suite",
                    "cycles",
                    "full accept",
                    "tail len",
                    *[f"s={s:+d}: P(kept>0) / mean / unique best" for s in ANATOMY_SHIFTS],
                    "tree yield",
                    "base TPF",
                ],
                rows,
            )
        )
        dist_rows = [
            [s, *[_fmt(v) for v in analysis["anatomy"][s]["best_kept_distribution"].values()]]
            for s in traces
        ]
        out += ["", "Distribution of the best kept tail length:", ""]
        out.append(_table(["suite", "0", "1-2", "3-5", "6-10", "11+"], dist_rows))

        policies = policy_grid(grid)
        if policies:
            out += ["", "## Policy search (replayed from the draft tables)", ""]
            grid_rows, analysis["grid"] = search(policies, traces, eos, workers)
            out += grid_rows
            write_grid_csv(root / "analysis" / "policy_grid.csv", analysis["grid"])
    text = "\n".join(out) + "\n"
    analysis_dir = root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "report.md").write_text(text, encoding="utf-8")
    (analysis_dir / "summary.json").write_text(
        json.dumps({k: v for k, v in analysis.items() if k != "grid"}, indent=2), encoding="utf-8"
    )
    return text, analysis


def search(policies: list[Strategy], traces: dict, eos: set[int], workers: int):
    """Replay every policy on every prompt; rank deployable policies on dev, report on test."""
    jobs, keys = [], []
    for suite, suite_traces in traces.items():
        for trace in suite_traces:
            jobs.append((trace, policies, eos))
            keys.append((suite, "dev" if trace["rank"] % 2 == 0 else "test"))
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            results = list(pool.map(_replay_job, jobs, chunksize=4))
    else:
        results = [_replay_job(job) for job in jobs]
    suites = list(traces)
    table = []
    for p, policy in enumerate(policies):
        entry = {"policy": policy.name, **policy.to_dict(), "ceiling": policy.oracle_min_tokens > 0}
        for split in ("dev", "test"):
            tpfs = []
            for suite in suites:
                rows = [r[p] for r, k in zip(results, keys, strict=True) if k == (suite, split)]
                tpf = sum(x[0] for x in rows) / sum(x[1] for x in rows)
                entry[f"{split}/{suite}"] = tpf
                tpfs.append(tpf)
            entry[f"{split}/avg"] = statistics.mean(tpfs)
        tails = [r[p] for r in results]
        entry["tail_passes_per_prompt"] = statistics.mean(x[2] for x in tails)
        entry["tail_yield"] = (
            (sum(x[3] for x in tails) + sum(x[2] for x in tails)) / sum(x[2] for x in tails)
            if sum(x[2] for x in tails)
            else None
        )
        table.append(entry)
    base = table[0]
    deployable = sorted((e for e in table if not e["ceiling"]), key=lambda e: -e["dev/avg"])
    best = deployable[0]
    lines = [
        f"{len(policies)} policies x {len(jobs)} prompts. Best deployable policy on dev: "
        f"`{best['policy']}`: test avg TPF {best['test/avg']:.3f} vs base {base['test/avg']:.3f} "
        f"({best['test/avg'] - base['test/avg']:+.3f}).",
        "",
    ]
    p_best = next(i for i, e in enumerate(table) if e is best)
    for suite in suites:  # paired bootstrap of best vs base on the test prompts
        index = [i for i, k in enumerate(keys) if k == (suite, "test")]
        tokens = np.array([[results[i][p][0] for i in index] for p in (0, p_best)], float)
        passes = np.array([[results[i][p][1] for i in index] for p in (0, p_best)], float)
        stats = bootstrap_tpf(tokens, passes)[1]
        lines.append(f"- {suite} test: {_interval(stats, 'delta', 'delta_ci')} TPF vs base")
    lines.append("")
    top = deployable[:12]
    ceilings = sorted((e for e in table if e["ceiling"]), key=lambda e: -e["dev/avg"])[:5]
    headers = [
        "policy",
        "dev avg",
        "test avg",
        *[f"test {s}" for s in suites],
        "tails/prompt",
        "tail yield",
    ]
    rows = [
        [
            e["policy"],
            e["dev/avg"],
            e["test/avg"],
            *[e[f"test/{s}"] for s in suites],
            e["tail_passes_per_prompt"],
            e["tail_yield"],
        ]
        for e in [base, *top]
    ]
    lines.append("Top deployable policies (ranked on dev):")
    lines.append("")
    lines.append(_table(headers, rows))
    lines += ["", "Perfect-gate ceilings (not deployable):", ""]
    rows = [
        [
            e["policy"],
            e["dev/avg"],
            e["test/avg"],
            *[e[f"test/{s}"] for s in suites],
            e["tail_passes_per_prompt"],
            e["tail_yield"],
        ]
        for e in ceilings
    ]
    lines.append(_table(headers, rows))
    return lines, table


def write_grid_csv(path: Path, table: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(table[0]))
        writer.writeheader()
        for entry in table:
            writer.writerow(
                {k: (json.dumps(v) if isinstance(v, list) else v) for k, v in entry.items()}
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--runs", help="comma-separated run names (default: the config's)")
    parser.add_argument("--grid", default="full", choices=("full", "small", "none"))
    parser.add_argument("--workers", type=int, default=4)
    known, rest = parser.parse_known_args(argv)
    cfg = parse_args(__doc__, rest)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parse_strategies(cfg.decode.strategies)  # validate the config even if unused here
    for run in known.runs.split(",") if known.runs else [cfg.run_name]:
        text, _ = analyze(cfg, run, known.grid, known.workers)
        print(text)


if __name__ == "__main__":
    main()
