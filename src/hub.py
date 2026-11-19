"""HF Storage Bucket transfers. Remote paths retain the old data/... and runs/... layout."""

import argparse
from pathlib import Path

from huggingface_hub import batch_bucket_files, download_bucket_files, list_bucket_tree
from huggingface_hub.errors import EntryNotFoundError

from src.config import parse_args


class Bucket:
    def __init__(self, bucket_id: str, upload: bool = True):
        self.bucket_id = bucket_id
        self.can_upload = bool(bucket_id) and upload

    @property
    def enabled(self):
        return bool(self.bucket_id)

    def list(self, prefix: str) -> list[str]:
        if not self.enabled:
            return []
        prefix = prefix.strip("/")
        try:
            entries = list_bucket_tree(self.bucket_id, prefix=prefix, recursive=True)
            return sorted(
                e.path
                for e in entries
                if getattr(e, "type", "file") == "file"
                and (e.path == prefix or e.path.startswith(prefix + "/"))
            )
        except EntryNotFoundError:
            return []  # a missing prefix is expected; authentication/network errors propagate

    def download_file(self, remote: str, local: Path) -> bool:
        remote, local = remote.strip("/"), Path(local)
        if remote not in self.list(remote):
            return False
        local.parent.mkdir(parents=True, exist_ok=True)
        tmp = local.with_name(local.name + ".download")
        download_bucket_files(self.bucket_id, [(remote, tmp)], raise_on_missing_files=True)
        tmp.replace(local)
        return True

    def download(self, remote: str, local: Path) -> bool:
        remote, local = remote.strip("/"), Path(local)
        paths = self.list(remote)
        if not paths:
            return False
        for path in paths:
            self.download_file(path, local / Path(path).relative_to(remote))
        return True

    def upload(self, local: Path, remote: str) -> None:
        if not self.can_upload:
            return
        local, remote = Path(local), remote.strip("/")
        if not local.exists():
            raise FileNotFoundError(local)
        if local.is_dir():
            pairs = [
                (str(p), f"{remote}/{p.relative_to(local).as_posix()}")
                for p in sorted(local.rglob("*"))
                if p.is_file() and "__pycache__" not in p.parts and not p.name.startswith(".")
            ]
        else:
            pairs = [(str(local), remote)]
        if pairs:
            batch_bucket_files(self.bucket_id, add=pairs)


def upload_checkpoint(bucket: Bucket, path: Path, remote: str):
    # Completion marker goes last: an interrupted upload cannot become the latest checkpoint.
    for name in ("trainable.safetensors", "optimizer.pt", "trainer_state.json"):
        bucket.upload(path / name, f"{remote}/{path.name}/{name}")


def upload_run(cfg, bucket):
    root, remote = cfg.run_path, f"runs/{cfg.run_name}"
    if not root.exists():
        raise FileNotFoundError(root)
    for path in sorted((root / "checkpoints").glob("step-*")):
        if (path / "trainer_state.json").exists():
            upload_checkpoint(bucket, path, f"{remote}/checkpoints")
    for name in ("final", "config.yaml", "run.json", "train_metrics.jsonl", "logs"):
        if (root / name).exists():
            bucket.upload(root / name, f"{remote}/{name}")
    for path in sorted(root.glob("eval-*.json")):
        bucket.upload(path, f"{remote}/{path.name}")


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--kind", choices=("data", "run", "all"), default="all")
    known, rest = parser.parse_known_args(argv)
    cfg = parse_args(__doc__, rest)
    bucket = Bucket(cfg.hf.bucket, cfg.hf.upload)
    if not bucket.can_upload:
        raise ValueError("set hf.bucket / ORTHRUS_BUCKET and hf.upload=true")
    if known.kind in ("data", "all"):
        root = cfg.dataset_path
        if not (root / "manifest.json").exists():
            raise FileNotFoundError(f"no dataset manifest in {root}")
        for name in ("prompts.parquet", "eval_prompts.jsonl", "raw", "manifest.json"):
            bucket.upload(root / name, f"data/{cfg.data.dataset}/{name}")
    if known.kind in ("run", "all"):
        upload_run(cfg, bucket)


if __name__ == "__main__":
    main()
