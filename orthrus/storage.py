"""Persistent storage in a Hugging Face Storage Bucket (mutable, no git history, Xet deduplication).

Layout inside the bucket:
    data/<dataset>/              raw/shard-*.parquet, prompts.parquet (cached prompt selection),
                                 eval_prompts.jsonl, manifest.json, datagen_metrics.jsonl
    runs/<run_name>/             checkpoints/step-*/, final/ (export), train_metrics.jsonl,
                                 run.json (one segment per job), logs/, eval-*.json
    bench/                       benchmark results (orthrus.bench, orthrus.genbench)
    runs/<run>/STOP, data/<dataset>/STOP   stop requests (scripts/ds.py stop): the job saves,
                                 exits and removes the file
    trackio/...                  Trackio database, only with tracking.enabled (managed by Trackio)

With an empty bucket id every method is a no-op, so the code also runs fully locally.
"""

import logging
import queue
import threading
from collections.abc import Callable
from pathlib import Path

from huggingface_hub import (
    batch_bucket_files,
    bucket_info,
    download_bucket_files,
    list_bucket_tree,
    sync_bucket,
)

log = logging.getLogger(__name__)


class Bucket:
    def __init__(self, bucket_id: str):
        self.bucket_id = bucket_id
        self._tasks: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._errors: list[BaseException] = []
        if bucket_id and not bucket_info(bucket_id).private:
            log.warning("bucket %s is PUBLIC: make it private in its settings", bucket_id)

    @property
    def enabled(self) -> bool:
        return bool(self.bucket_id)

    def url(self, path: str) -> str:
        return f"hf://buckets/{self.bucket_id}/{path.strip('/')}"

    def list(self, prefix: str) -> list[str]:
        """Paths of files under prefix (empty if the prefix does not exist)."""
        if not self.enabled:
            return []
        try:
            entries = list_bucket_tree(self.bucket_id, prefix=prefix.strip("/"), recursive=True)
            return sorted(e.path for e in entries if getattr(e, "type", "file") == "file")
        except Exception:  # a missing prefix is not an error
            return []

    def exists(self, remote: str) -> bool:
        return remote.strip("/") in self.list(remote)

    def download(self, remote: str, local: Path) -> bool:
        """Mirror a remote prefix into a local directory; False if nothing is there."""
        if not self.list(remote):
            return False
        local.mkdir(parents=True, exist_ok=True)
        sync_bucket(self.url(remote), str(local), quiet=True)
        return True

    def download_file(self, remote: str, local: Path) -> bool:
        """Download one file; False if it does not exist."""
        if remote.strip("/") not in self.list(remote):
            return False
        local.parent.mkdir(parents=True, exist_ok=True)
        download_bucket_files(self.bucket_id, files=[(remote.strip("/"), str(local))])
        return True

    def upload(self, local: Path, remote: str) -> None:
        """Copy a local directory (or file) under the remote prefix, retrying network errors."""
        if not self.enabled:
            return
        for attempt in range(3):
            try:
                if Path(local).is_dir():
                    sync_bucket(str(local), self.url(remote), quiet=True)
                else:
                    batch_bucket_files(self.bucket_id, add=[(str(local), remote.strip("/"))])
                return
            except Exception as error:
                if attempt == 2:
                    raise
                log.warning("upload of %s failed (%s), retrying", remote, error)

    def delete(self, prefix: str) -> None:
        """Remove every file under the remote prefix."""
        paths = self.list(prefix)
        if paths:
            batch_bucket_files(self.bucket_id, delete=paths)

    def submit(self, task: Callable[[], None]) -> None:
        """Run task in the background thread; tasks run one by one in submission order."""
        if self._worker is None:
            self._worker = threading.Thread(target=self._run, daemon=True)
            self._worker.start()
        self._tasks.put(task)

    def wait(self) -> None:
        """Block until all submitted tasks finished; re-raise the first failure."""
        if self._worker is not None:
            self._tasks.join()
        if self._errors:
            raise RuntimeError("background storage task failed") from self._errors[0]

    def _run(self) -> None:
        while True:
            task = self._tasks.get()
            try:
                task()
            except Exception as error:
                log.error("background storage task failed: %s", error)
                self._errors.append(error)
            finally:
                self._tasks.task_done()
