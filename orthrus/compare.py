"""Runs side by side: the position-weighting ablation (train.loss_gamma) against the baseline.

    python -m orthrus.compare --config configs/qwen3-0.6b.yaml --runs RUN[,RUN...]

Reads what training and evaluation keep in the bucket (or the local run directories without a
bucket), like orthrus.report, and prints three tables:
  1. progress and the latest held-out evaluation during training: teacher agreement at block
     offsets 1/4/8/16, acceptance proxy, unweighted KL and TPF of real diffusion decoding;
  2. TPF of real decoding at every evaluation step, runs side by side;
  3. the final evaluation (orthrus.evaluate, fp32, 512 new tokens): TPF per suite, AR parity and
     the change of the mean TPF relative to the first run.
"""

import argparse
from pathlib import Path

from orthrus.config import parse_args
from orthrus.report import fetch, read_json, read_jsonl, table

SUITES = ("heldout", "gsm8k", "math500", "humaneval", "mbpp")
FINAL_EVAL = "eval-final-512-float32.json"
OFFSETS = (1, 4, 8, 16)


def load_run(cfg, run: str, out: Path) -> dict:
    root = fetch(cfg, run, out)
    manifest = read_json(root / "run.json")
    return {
        "name": run,
        "records": read_jsonl(root / "train_metrics.jsonl"),
        "total_steps": manifest.get("total_steps"),
        # None: the run has not started; 0 or missing (older runs): equal weights
        "gamma": (manifest["config"]["train"].get("loss_gamma") or 0) if manifest else None,
        "final": read_json(root / FINAL_EVAL),
    }


def progress_table(runs: list[dict]) -> str:
    rows = []
    for run in runs:
        trained = [r for r in run["records"] if "train/loss" in r]
        evals = [r for r in run["records"] if "eval/kl" in r]
        last = evals[-1] if evals else {}
        step = trained[-1]["step"] if trained else 0
        rows.append(
            [
                run["name"],
                "0 (equal)" if run["gamma"] == 0 else run["gamma"],
                f"{step}/{run['total_steps'] or '?'}",
                last.get("step"),
                *(last.get(f"eval/agree_k{k}") for k in OFFSETS),
                last.get("eval/acceptance_proxy"),
                last.get("eval/kl"),
                last.get("generate/tpf"),
            ]
        )
    header = ["run", "gamma", "trained", "eval@", *(f"agree_k{k}" for k in OFFSETS)]
    return table([*header, "accept_proxy", "kl", "gen_tpf"], rows)


def tpf_by_step_table(runs: list[dict]) -> str:
    tpf = {
        run["name"]: {r["step"]: r["generate/tpf"] for r in run["records"] if "generate/tpf" in r}
        for run in runs
    }
    steps = sorted({step for values in tpf.values() for step in values})
    rows = [[step, *(tpf[run["name"]].get(step) for run in runs)] for step in steps]
    return table(["step", *(run["name"] for run in runs)], rows)


def final_table(runs: list[dict]) -> str:
    def mean_tpf(final: dict) -> float | None:
        values = [final[s]["tpf"] for s in SUITES if s in final]
        return sum(values) / len(values) if values else None

    reference = mean_tpf(runs[0]["final"])
    rows = []
    for run in runs:
        final = run["final"]
        mean = mean_tpf(final)
        change = None
        if mean is not None and reference:
            change = f"{100 * (mean / reference - 1):+.1f}%"
        parity = [final[s]["ar_match"] for s in SUITES if s in final]
        rows.append(
            [
                run["name"],
                *(final.get(s, {}).get("tpf") for s in SUITES),
                mean,
                change,
                min(parity) if parity else None,
            ]
        )
    header = ["run", *SUITES, "mean", f"vs {runs[0]['name']}", "ar_match"]
    return table(header, rows)


def table_or_none(text: str) -> str:
    """The table, or a note under its header when it has no rows yet."""
    return text if text.count("\n") > 1 else text + "\n(no data yet)"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--runs", required=True, help="run names, comma-separated")
    known, rest = parser.parse_known_args(argv)
    cfg = parse_args(__doc__, rest)
    out = Path(cfg.storage.local_dir) / "report"
    runs = [load_run(cfg, name, out) for name in known.runs.split(",")]
    print("1. training progress and the latest held-out evaluation (higher agree/tpf is better)")
    print(table_or_none(progress_table(runs)))
    print("\n2. TPF of real diffusion decoding during training (held-out prompts, 256 tokens)")
    print(table_or_none(tpf_by_step_table(runs)))
    print(f"\n3. final evaluation ({FINAL_EVAL}): TPF per suite; ar_match must be 1")
    print(table_or_none(final_table(runs)))


if __name__ == "__main__":
    main()
