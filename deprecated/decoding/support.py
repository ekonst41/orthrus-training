"""Local bookkeeping for deprecated experiments."""
import json
import time
from pathlib import Path
from src.test.benchmark import load_eval_prompts as read_prompts


def load_eval_prompts(bucket, dataset, local_root):
    return read_prompts(bucket, dataset, Path(local_root) / "data")


class Tracker:
    def __init__(self, cfg, run_dir, kind):
        self.path = run_dir / f"{kind}_metrics.jsonl"
    def log(self, metrics, step):
        with self.path.open("a") as f:
            f.write(json.dumps({"step": step, **metrics}) + "\n")
    def progress(self, fraction, message):
        pass
    def close(self):
        pass


class Watchdog:
    # No process-killing watchdog in the archived local runner.
    def __init__(self, *args):
        pass
    def beat(self):
        pass


def write_manifest(path, cfg, **extra):
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(config=cfg.to_dict(), started_at=time.time(), **extra)
    path.write_text(json.dumps(old, indent=2))


def end_segment(path, **extra):
    old = json.loads(path.read_text())
    old.update(ended_at=time.time(), **extra)
    path.write_text(json.dumps(old, indent=2))
