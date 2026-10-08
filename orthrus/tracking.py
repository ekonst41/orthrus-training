"""Experiment tracking: Trackio dashboard + a local JSONL copy + DataSphere progress + run manifest.

Trackio (https://huggingface.co/docs/trackio) keeps its database in the storage bucket under
trackio/ and, when a Space is configured, serves a private dashboard there. Every metric is also
appended to <run_dir>/metrics.jsonl, which is uploaded with the run, so results never depend on the
dashboard. Tracking failures are logged and ignored: they must never stop training.
"""

import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

from orthrus.config import Config

log = logging.getLogger(__name__)


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
        """Progress bar of the DataSphere job page (no-op outside DataSphere)."""
        if self.progress_file:
            entry = {"progress": round(100 * min(max(fraction, 0.0), 1.0)), "message": message}
            with open(self.progress_file, "a", encoding="utf-8") as file:
                file.write(json.dumps(entry) + "\n")

    def close(self) -> None:
        if self.run is not None:
            try:
                self.run.finish()
            except Exception as error:
                log.warning("Trackio finish failed: %s", error)
        self.metrics.close()


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


def write_manifest(path: Path, cfg: Config, **extra) -> dict:
    """Everything needed to reproduce a run: config, code, environment, hardware, data."""
    import transformers

    # In DataSphere jobs requirements/ arrives as a job input in the working directory.
    candidates = [
        Path("requirements/train.txt"),
        Path(__file__).resolve().parents[1] / "requirements/train.txt",
    ]
    lock = next((p for p in candidates if p.exists()), candidates[0])
    manifest = {
        "run_name": cfg.run_name,
        "config": cfg.to_dict(),
        "git_commit": _git_revision(),
        "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "host": platform.node(),
        "datasphere_job_id": os.environ.get("JOB_ID"),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **extra,
    }
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest
