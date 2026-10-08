"""Experiment records: metrics JSONL, run manifest, DataSphere progress, GPU stats, stall watchdog.

Every metric is appended to <run_dir>/<kind>_metrics.jsonl (kind: train, eval, datagen), which is
uploaded to the bucket next to the checkpoints; `python -m orthrus.report` shows these files (and
opens them in a local Trackio dashboard). Trackio inside the job is optional and off by default:
without an HF PRO Space its database stays in the job container. Tracking failures are logged and
ignored: they must never stop training.
"""

import faulthandler
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import threading
import time
from importlib import metadata
from pathlib import Path

import torch

from orthrus.config import Config

log = logging.getLogger(__name__)
PACKAGES = ("torch", "triton", "transformers", "huggingface_hub", "vllm", "trackio")


class Tracker:
    def __init__(self, cfg: Config, run_dir: Path, kind: str):
        self.metrics = (run_dir / f"{kind}_metrics.jsonl").open("a", encoding="utf-8")
        self.progress_file = os.environ.get("JOB_PROGRESS_FILENAME")  # set inside DataSphere jobs
        self.run = None
        if not cfg.tracking.enabled:
            return
        try:
            import trackio

            self.run = trackio.init(
                project=cfg.tracking.project,
                name=cfg.run_name if kind == "train" else f"{cfg.run_name}-{kind}",
                config=json.loads(json.dumps(cfg.to_dict())),
                space_id=cfg.tracking.space_id or None,
                bucket_id=cfg.storage.bucket or None,
                private=True,
                resume="allow",
                auto_log_gpu=torch.cuda.is_available(),
            )
        except Exception as error:
            log.warning("Trackio disabled (%s); metrics still go to %s", error, self.metrics.name)

    def log(self, metrics: dict, step: int) -> None:
        record = {"step": step, "time": round(time.time(), 1), **metrics}
        self.metrics.write(json.dumps(record) + "\n")
        self.metrics.flush()
        if self.run is not None:
            try:
                self.run.log(metrics, step=step)
            except Exception as error:
                log.warning("Trackio log failed: %s", error)

    def progress(self, fraction: float, message: str) -> None:
        """Progress bar of the DataSphere job page (no-op outside DataSphere). Rewritten each time:
        DataSphere kept showing the first line of an appended file."""
        if self.progress_file:
            entry = {"progress": round(100 * min(max(fraction, 0.0), 1.0)), "message": message}
            with open(self.progress_file, "w", encoding="utf-8") as file:
                file.write(json.dumps(entry) + "\n")

    def close(self) -> None:
        if self.run is not None:
            try:
                self.run.finish()
            except Exception as error:
                log.warning("Trackio finish failed: %s", error)
        self.metrics.close()


def restore_history(bucket, remote: str, local: Path, max_step: int | None = None) -> None:
    """Continue an append-only JSONL log of an earlier job: download it when the job starts with an
    empty run directory (else the next upload would replace the history), and drop records after
    max_step (they are recomputed after resuming from that step)."""
    if not local.exists():
        bucket.download_file(remote, local)
    if max_step is None or not local.exists():
        return
    lines = local.read_text(encoding="utf-8").splitlines()
    kept = [line for line in lines if line and json.loads(line).get("step", 0) <= max_step]
    if len(kept) != len(lines):
        local.write_text("".join(line + "\n" for line in kept), encoding="utf-8")


def gpu_stats() -> dict:
    """Current GPU utilization (%), power (W) and reserved memory (GB); {} without CUDA/NVML."""
    if not torch.cuda.is_available():
        return {}
    stats = {"perf/memory_reserved_gb": torch.cuda.memory_reserved() / 2**30}
    try:
        stats["perf/gpu_util"] = torch.cuda.utilization()
        stats["perf/gpu_power_w"] = torch.cuda.power_draw() / 1000
    except Exception:  # NVML not available
        pass
    return stats


class Watchdog:
    """Abort the process when no progress is reported for `minutes`: a hung job (seen once in
    vLLM teardown) otherwise idles on a paid GPU until the job's timeout. Thread stacks are dumped
    to stderr first; exit code 3."""

    def __init__(self, minutes: float, what: str):
        self.limit, self.what, self.last = minutes * 60, what, time.monotonic()
        if minutes > 0:
            threading.Thread(target=self._watch, daemon=True).start()

    def beat(self) -> None:
        self.last = time.monotonic()

    def _watch(self) -> None:
        while True:
            time.sleep(30)
            if time.monotonic() - self.last > self.limit:
                log.error("no %s for %.0f min: aborting", self.what, self.limit / 60)
                faulthandler.dump_traceback(all_threads=True)
                logging.shutdown()
                os._exit(3)


def _git_revision() -> str:
    """Commit of the code: $ORTHRUS_GIT_COMMIT (set by scripts/ds.py for DataSphere) or git."""
    if os.environ.get("ORTHRUS_GIT_COMMIT"):
        return os.environ["ORTHRUS_GIT_COMMIT"]
    try:
        root = Path(__file__).resolve().parents[1]
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
        return commit + ("-dirty" if dirty.strip() else "")
    except Exception:
        return "unknown"


def _versions() -> dict:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    versions["cuda"] = torch.version.cuda
    return versions


def write_manifest(path: Path, cfg: Config, lock: str = "train", **extra) -> dict:
    """Everything needed to reproduce a run: config, code, environment, hardware, data.

    A run that continues in a later job keeps one entry per job in `segments` (code version, job,
    steps covered), so a run assembled from several jobs stays traceable."""
    # In DataSphere jobs requirements/ arrives as a job input in the working directory.
    candidates = [
        Path(f"requirements/{lock}.txt"),
        Path(__file__).resolve().parents[1] / f"requirements/{lock}.txt",
    ]
    lock_file = next((p for p in candidates if p.exists()), candidates[0])
    previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    segment = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": _git_revision(),
        "datasphere_job_id": os.environ.get("JOB_ID"),
        "host": platform.node(),
        **({"resumed_at_step": extra["resumed_at_step"]} if "resumed_at_step" in extra else {}),
    }
    manifest = {
        "run_name": cfg.run_name,
        "config": cfg.to_dict(),
        "git_commit": segment["git_commit"],
        "lock_file": f"requirements/{lock}.txt",
        "lock_sha256": hashlib.sha256(lock_file.read_bytes()).hexdigest()
        if lock_file.exists()
        else None,
        "python": sys.version.split()[0],
        **_versions(),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "host": platform.node(),
        "datasphere_job_id": segment["datasphere_job_id"],
        "started_at": segment["started_at"],
        **extra,
        "segments": [*previous.get("segments", []), segment],
    }
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest


def end_segment(path: Path, **result) -> None:
    """Record how the current job ended (steps reached, reason, hours) in the manifest."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    segment = manifest["segments"][-1]
    started = time.mktime(time.strptime(segment["started_at"][:19], "%Y-%m-%dT%H:%M:%S"))
    segment.update(result, ended_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    segment["hours"] = round((time.time() - started) / 3600, 3)
    manifest["total_hours"] = round(sum(s.get("hours", 0) for s in manifest["segments"]), 3)
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
