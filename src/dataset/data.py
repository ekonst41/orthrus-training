"""Token-exact packing and deterministic, disjoint batches for one or several GPUs."""

import hashlib
import json
import logging
import os
from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow.parquet as pq
import torch

log = logging.getLogger(__name__)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(cfg, bucket):
    path = cfg.dataset_path / "manifest.json"
    if not path.exists():
        bucket.download_file(f"data/{cfg.data.dataset}/manifest.json", path)
    return json.loads(path.read_text()) if path.exists() else {}


def fetch_raw(cfg, bucket):
    raw = cfg.dataset_path / "raw"
    remote = f"data/{cfg.data.dataset}/raw"
    for name in bucket.list(remote):
        target = raw / Path(name).name
        if not target.exists():
            bucket.download_file(name, target)
    if not list(raw.glob("*.parquet")):
        raise FileNotFoundError(f"no raw shards in {raw}; run scripts/prepare.sh")
    return raw


def pack(raw_dir: Path, seq_len: int, seed: int, out_dir: Path | None = None):
    """Place each conversation at its shuffled offset, optionally writing directly to .npy."""
    files = sorted(raw_dir.glob("*.parquet"))

    def token_batches():
        columns = ["prompt_ids", "response_ids"]
        for file in files:
            with pq.ParquetFile(file) as parquet:
                for batch in parquet.iter_batches(batch_size=1024, columns=columns):
                    yield batch.column("prompt_ids"), batch.column("response_ids")

    lengths = [
        np.diff(p.offsets.to_numpy()) + np.diff(r.offsets.to_numpy()) for p, r in token_batches()
    ]
    lengths = np.concatenate(lengths) if lengths else np.empty(0, dtype=np.int64)
    order = np.random.default_rng(seed).permutation(len(lengths))
    shuffled = lengths[order]
    starts = np.empty(len(order), dtype=np.int64)
    starts[order] = np.cumsum(shuffled, dtype=np.int64) - shuffled
    shape = (int(lengths.sum(dtype=np.int64)) // seq_len, seq_len)

    def allocate(name, dtype):
        if out_dir is None:
            return np.empty(shape, dtype=dtype)
        return np.lib.format.open_memmap(out_dir / name, mode="w+", dtype=dtype, shape=shape)

    ids = allocate("input_ids.npy", np.int32)
    mask = allocate("assistant_mask.npy", np.uint8)
    flat_ids, flat_mask = ids.reshape(-1), mask.reshape(-1)
    row = 0
    for prompts, responses in token_batches():
        dest = starts[row : row + len(prompts)]
        for column, is_response in ((prompts, False), (responses, True)):
            offsets, values = column.offsets.to_numpy(), column.values.to_numpy()
            for start, lo, hi in zip(dest, offsets[:-1], offsets[1:], strict=True):
                target = flat_ids[start : start + (hi - lo)]
                target[:] = values[lo:hi][: target.size]  # drop the incomplete tail
                flat_mask[start : start + target.size] = is_response
            dest = dest + np.diff(offsets)
        row += len(prompts)
    log.info("packed %d conversations into %d rows", len(order), shape[0])
    return ids, mask


def load_packed(raw_dir: Path, cache_root: Path, seq_len: int, seed: int):
    files = sorted(raw_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(raw_dir)
    # Contents, not file sizes, identify the data used by a resumable run.
    key = json.dumps([[p.name, file_sha256(p)] for p in files] + [seq_len, seed])
    cache = cache_root / f"packed-{hashlib.sha256(key.encode()).hexdigest()[:20]}"
    if not (cache / "assistant_mask.npy").exists():
        cache.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=cache) as tmp:
            pack(raw_dir, seq_len, seed, Path(tmp))
            for name in ("input_ids.npy", "assistant_mask.npy"):
                os.replace(Path(tmp) / name, cache / name)
    return (
        np.load(cache / "input_ids.npy", mmap_mode="r"),
        np.load(cache / "assistant_mask.npy", mmap_mode="r"),
        cache.name,
    )


def batches(
    ids: np.ndarray,
    mask: np.ndarray,
    batch_size: int,
    seed: int,
    epoch: int,
    *,
    rank: int = 0,
    world_size: int = 1,
    global_batch_size: int | None = None,
    start_step: int = 0,
) -> Iterator[tuple[int, torch.Tensor, torch.Tensor]]:
    """Yield this rank's complete micro-batches, dropping the incomplete global batch.

    start_step is the optimizer step within this epoch, not a micro-batch offset.
    No padding or duplicated rows: every rank has the same number of backwards.
    """
    global_batch_size = global_batch_size or batch_size * world_size
    if world_size < 1 or not 0 <= rank < world_size or batch_size < 1:
        raise ValueError("invalid rank, world_size or batch_size")
    if global_batch_size < 1 or global_batch_size % (batch_size * world_size):
        raise ValueError("global_batch_size must divide into equal rank micro-batches")
    if len(ids) != len(mask):
        raise ValueError("ids and mask must have the same number of rows")
    steps = len(ids) // global_batch_size
    if not 0 <= start_step <= steps:
        raise ValueError("start_step is outside the epoch")
    accum = global_batch_size // (batch_size * world_size)
    order = np.random.default_rng([seed, epoch]).permutation(len(ids))
    for local_index in range(start_step * accum, steps * accum):
        index = local_index * world_size + rank
        rows = np.sort(order[index * batch_size : (index + 1) * batch_size])
        yield (
            index,
            torch.from_numpy(ids[rows].astype(np.int64)),
            torch.from_numpy(mask[rows].astype(np.bool_)),
        )
