"""Metrics of runs in the terminal and in a local Trackio dashboard (no HF Space needed).

    python -m orthrus.report --config configs/qwen3-0.6b.yaml [--run NAME[,NAME...]] [--show]

Reads what train / evaluate / datagen keep in the bucket (or in the local run directory without a
bucket): runs/<run>/run.json, train_metrics.jsonl, eval-*.json, and the dataset's manifest.json and
datagen_metrics.jsonl. Prints job segments (hours, cost), the current training state, the training
curve, evaluations during training, final evaluations and data generation. --show loads the
metric histories into a local Trackio project and opens its dashboard in the browser.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

from orthrus.config import parse_args
from orthrus.storage import Bucket

RUB_PER_GPU_HOUR = 543  # DataSphere g2.1 (1x A100 80GB)
CURVE = ("train/loss", "train/top1_agreement", "train/agree_k1", "train/agree_k8",
         "train/acceptance_proxy", "perf/seconds_per_step")  # fmt: skip
EVAL = ("eval/kl", "eval/top1_agreement", "eval/acceptance_proxy", "generate/tpf",
        "generate/acceptance_length", "generate/math/tpf", "generate/code/tpf",
        "generate/chat/tpf")  # fmt: skip
SUITE = ("prompts", "tpf", "acceptance_length", "tokens_per_cycle", "ar_match", "speedup",
         "tokens_per_second", "mean_new_tokens")  # fmt: skip


def fetch(cfg, run: str, out: Path) -> Path:
    """Copy a run's records (no checkpoints or weights) to out/<run>; return that directory."""
    target = out / run
    shutil.rmtree(target, ignore_errors=True)
    bucket = Bucket(cfg.storage.bucket)
    data_files = ("manifest.json", "datagen_metrics.jsonl")
    if bucket.enabled:
        prefix = f"runs/{run}/"
        for path in bucket.list(f"runs/{run}"):
            rel = path[len(prefix) :]
            if not rel.startswith(("checkpoints/", "final/")):
                bucket.download_file(path, target / rel)
        for name in data_files:
            bucket.download_file(f"data/{cfg.data.dataset}/{name}", target / "data" / name)
    else:
        local = Path(cfg.storage.local_dir)
        source = local / "runs" / run
        if source.exists():
            shutil.copytree(
                source, target, ignore=shutil.ignore_patterns("checkpoints", "final", "*.snapshot")
            )
        for name in data_files:
            path = local / "data" / cfg.data.dataset / name
            if path.exists():
                (target / "data").mkdir(parents=True, exist_ok=True)
                shutil.copy(path, target / "data" / name)
    return target


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def table(header: list[str], rows: list[list]) -> str:
    cells = [header] + [[fmt(v) for v in row] for row in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(header))]
    lines = ["  ".join(c.rjust(w) for c, w in zip(row, widths, strict=True)) for row in cells]
    return "\n".join([lines[0], "  ".join("-" * w for w in widths), *lines[1:]])


def short(name: str) -> str:
    return name.split("/", 1)[-1]


def report(run: str, root: Path) -> str:
    manifest = read_json(root / "run.json")
    records = read_jsonl(root / "train_metrics.jsonl")
    train = [r for r in records if "train/loss" in r]
    evals = [r for r in records if "eval/kl" in r or "generate/tpf" in r]
    parts = [f"=== run {run}"]
    if manifest:
        config = manifest.get("config", {})
        model = config.get("model", {})
        parts.append(
            f"model {model.get('base')}@{(model.get('revision') or 'main')[:10]} | data "
            f"{manifest.get('data')} | {manifest.get('train_rows')} rows | "
            f"{manifest.get('total_steps')} steps"
        )
        segments = manifest.get("segments", [])
        rows = [
            [
                i + 1,
                s.get("datasphere_job_id") or s.get("host"),
                (s.get("git_commit") or "")[:12],
                f"{s.get('resumed_at_step')}->{s.get('ended_at_step', '...')}",
                s.get("hours"),
                s.get("stop_reason", "running?"),
            ]
            for i, s in enumerate(segments)
        ]
        hours = sum(s.get("hours") or 0 for s in segments)
        if rows:
            parts += [
                table(["#", "job", "commit", "steps", "hours", "end"], rows),
                f"wall-clock {hours:.2f} h (~{hours * RUB_PER_GPU_HOUR:.0f} RUB on g2.1)",
            ]
    if train:
        last, total = train[-1], manifest.get("total_steps")
        done = (
            f"{last['step']}/{total} ({100 * last['step'] / total:.1f}%)" if total else last["step"]
        )
        parts.append(
            f"\nstep {done} | epoch {fmt(last.get('train/epoch'))} | loss "
            f"{fmt(last.get('train/loss'))} | top-1 {fmt(last.get('train/top1_agreement'))} | lr "
            f"{fmt(last.get('train/lr'))} | {fmt(last.get('perf/seconds_per_step'))} s/step | ETA "
            f"{fmt(last.get('perf/eta_hours'))} h | peak {fmt(last.get('perf/max_memory_gb'))} GB"
            f" | GPU {fmt(last.get('perf/gpu_util'))}%"
        )
        picks = sorted({train[round(i * (len(train) - 1) / 9)]["step"] for i in range(10)})
        rows = [[r["step"], *(r.get(k) for k in CURVE)] for r in train if r["step"] in picks]
        parts += ["\ntraining curve", table(["step", *map(short, CURVE)], rows)]
    if evals:
        rows = [[r["step"], *(r.get(k) for k in EVAL)] for r in evals]
        parts += [
            "\nevaluations during training (generate/*: diffusion decoding of held-out prompts)",
            table(["step", *(k.replace("generate/", "gen/") for k in EVAL)], rows),
        ]
    for path in sorted(root.glob("eval-*.json")):
        results = read_json(path)
        names = path.stem.split("-")[1:]  # eval-<model>-<tokens>-<dtype> (older: no <model>)
        meta = dict(zip(("model", "max_new_tokens", "dtype")[-len(names) :], names, strict=False))
        meta.update(results.pop("_meta", {}))
        rows = [[suite, *(m.get(k) for k in SUITE)] for suite, m in results.items()]
        parts += [
            f"\n{path.name}: model {meta.get('model', '?')}, {meta.get('dtype', '?')}, "
            f"{meta.get('max_new_tokens', '?')} new tokens",
            table(["suite", *SUITE], rows),
        ]
    data_manifest = read_json(root / "data" / "manifest.json")
    shards = read_jsonl(root / "data" / "datagen_metrics.jsonl")
    if data_manifest or shards:
        speed = [s["datagen/output_tokens_per_second"] for s in shards]
        lengths = [s["datagen/mean_response_tokens"] for s in shards]
        parts.append(
            f"\ndata: {len(shards)}/{data_manifest.get('shards', '?')} shards | "
            f"{data_manifest.get('train_prompts', '?')} prompts | "
            f"{sum(speed) / max(len(speed), 1):.0f} tok/s | "
            f"{sum(lengths) / max(len(lengths), 1):.0f} tokens per response | model "
            f"{data_manifest.get('model')}@{(data_manifest.get('model_revision') or '')[:10]}"
        )
    return "\n".join(parts)


def show(runs: dict[str, Path], project: str) -> None:
    """Load metric histories into a fresh local Trackio project and open its dashboard."""
    import trackio
    from trackio.sqlite_storage import SQLiteStorage

    db = SQLiteStorage.get_project_db_path(project)
    for suffix in ("", "-wal", "-shm"):
        Path(f"{db}{suffix}").unlink(missing_ok=True)
    for run, root in runs.items():
        for kind, path in (
            ("train", root / "train_metrics.jsonl"),
            ("datagen", root / "data" / "datagen_metrics.jsonl"),
        ):
            records = read_jsonl(path)
            numeric = [
                {k: float(v) for k, v in r.items() if k not in ("step", "time")
                 and isinstance(v, (int, float)) and not isinstance(v, bool)}
                for r in records
            ]  # fmt: skip
            keep = [i for i, m in enumerate(numeric) if m]
            if not keep:
                continue
            SQLiteStorage.bulk_log(
                project=project,
                run=f"{run}" if kind == "train" else f"{run}-datagen",
                metrics_list=[numeric[i] for i in keep],
                steps=[int(records[i].get("step", 0)) for i in keep],
                timestamps=[
                    time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(records[i].get("time", 0)))
                    for i in keep
                ],
            )
    trackio.show(project=project)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run", help="run name(s), comma-separated (default: the config's)")
    parser.add_argument("--show", action="store_true", help="open a local Trackio dashboard")
    known, rest = parser.parse_known_args(argv)
    cfg = parse_args(__doc__, rest)
    out = Path(cfg.storage.local_dir) / "report"
    runs = {}
    for run in (known.run or cfg.run_name).split(","):
        runs[run] = fetch(cfg, run, out)
        print(report(run, runs[run]), end="\n\n", flush=True)
    if known.show:
        show(runs, project=f"orthrus-{'-'.join(runs)}"[:80])


if __name__ == "__main__":
    main()
