"""Storage probe: HF token rights and HF Storage Bucket throughput from a DataSphere job.

Usage: python scripts/probe_storage.py [report.json]   (needs HF_TOKEN in the environment)
Creates (or reuses) the private bucket <user>/orthrus-training, uploads and downloads a 1 GiB
checkpoint-like folder under probe/, checks integrity and Xet deduplication, then deletes probe/.
"""

import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

from huggingface_hub import HfApi, bucket_info, create_bucket, sync_bucket
from huggingface_hub.errors import RepositoryNotFoundError

BUCKET = "orthrus-training"
MIB = 1 << 20
GIB_IN_MB = (1 << 30) / 1e6


def write_random(path: Path, mib: int) -> None:
    with path.open("wb") as file:
        for _ in range(mib):
            file.write(os.urandom(MIB))  # incompressible, like trained weights


def digest(folder: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir())}


def timed(fn) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def main() -> None:
    report_path = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/probe_storage.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report: dict = {}
    try:
        whoami = HfApi().whoami()
        token = whoami.get("auth", {}).get("accessToken", {})
        report["user"] = whoami["name"]
        report["token_role"] = token.get("role")
        report["token_fine_grained"] = bool(token.get("fineGrained"))
        bucket_id = f"{whoami['name']}/{BUCKET}"
        try:  # a fine-grained token may write to an existing bucket without being allowed to create
            info = bucket_info(bucket_id)
        except RepositoryNotFoundError:
            create_bucket(bucket_id, private=True)
            info = bucket_info(bucket_id)
        report["bucket"] = bucket_id
        report["bucket_private"] = info.private
        remote = f"hf://buckets/{bucket_id}/probe"

        work = Path(tempfile.mkdtemp(prefix="storage_probe_"))
        v1, v2, back = work / "v1", work / "v2", work / "back"
        for folder in (v1, v2, back):
            folder.mkdir()
        write_random(v1 / "frozen.bin", 512)  # stays identical between saves
        write_random(v1 / "trainable.bin", 512)  # changes on every save
        (v1 / "state.json").write_text(json.dumps({"step": 1}))
        shutil.copy2(v1 / "frozen.bin", v2 / "frozen.bin")
        write_random(v2 / "trainable.bin", 512)
        (v2 / "state.json").write_text(json.dumps({"step": 2}))

        up1 = timed(lambda: sync_bucket(str(v1), f"{remote}/ckpt"))
        up2 = timed(lambda: sync_bucket(str(v2), f"{remote}/ckpt", delete=True))
        down = timed(lambda: sync_bucket(f"{remote}/ckpt", str(back)))
        report.update(
            upload_1gib_s=round(up1, 1),
            upload_mb_per_s=round(GIB_IN_MB / up1, 1),
            reupload_half_changed_s=round(up2, 1),
            download_1gib_s=round(down, 1),
            download_mb_per_s=round(GIB_IN_MB / down, 1),
            integrity_ok=digest(back) == digest(v2),
        )
        empty = work / "empty"
        empty.mkdir()
        report["cleanup_s"] = round(timed(lambda: sync_bucket(str(empty), remote, delete=True)), 1)
        shutil.rmtree(work, ignore_errors=True)
    except Exception as error:  # report what failed instead of losing the whole run
        report["error"] = f"{type(error).__name__}: {error}"[:500]
    report["hf_token_present"] = bool(os.environ.get("HF_TOKEN"))
    print(json.dumps(report, indent=2), flush=True)
    report_path.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
