"""Training data: generated conversations packed into fixed-length rows with an assistant mask.

Raw shards (written by orthrus.datagen) hold token ids, so the training sequence is token-exact
with what the model saw at generation time: chat-template prompt (thinking disabled) followed by
the model's own response. Packing follows the official code: shuffle conversations, concatenate
them into one stream and cut it into rows of seq_len tokens (conversations may span two rows).
"""

import hashlib
import json
import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from orthrus.storage import Bucket

log = logging.getLogger(__name__)


def fetch_raw(bucket: Bucket, dataset: str, local_root: Path) -> Path:
    """Make data/<dataset>/raw available locally (download from the bucket if needed)."""
    raw_dir = local_root / "data" / dataset / "raw"
    if bucket.enabled:
        bucket.download(f"data/{dataset}/raw", raw_dir)
    if not sorted(raw_dir.glob("*.parquet")):
        raise FileNotFoundError(f"no raw shards in {raw_dir}: run orthrus.datagen first")
    return raw_dir


def pack(raw_dir: Path, seq_len: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Pack all raw shards into (input_ids int32 [rows, seq_len], assistant_mask uint8)."""
    files = sorted(raw_dir.glob("*.parquet"))
    columns = ["prompt_ids", "response_ids"]
    table = pa.concat_tables([pq.read_table(f, columns=columns) for f in files])
    prompts = table.column("prompt_ids").combine_chunks()
    responses = table.column("response_ids").combine_chunks()
    p_off, p_val = prompts.offsets.to_numpy(), prompts.values.to_numpy()
    r_off, r_val = responses.offsets.to_numpy(), responses.values.to_numpy()

    order = np.random.default_rng(seed).permutation(len(prompts))
    total = int((np.diff(p_off) + np.diff(r_off)).sum())
    ids = np.empty(total, dtype=np.int32)
    mask = np.zeros(total, dtype=np.uint8)
    pos = 0
    for i in order:
        prompt = p_val[p_off[i] : p_off[i + 1]]
        response = r_val[r_off[i] : r_off[i + 1]]
        ids[pos : pos + len(prompt)] = prompt
        pos += len(prompt)
        ids[pos : pos + len(response)] = response
        mask[pos : pos + len(response)] = 1
        pos += len(response)
    rows = total // seq_len
    log.info("packed %d conversations (%d tokens) into %d rows", len(order), total, rows)
    used = rows * seq_len  # the incomplete tail is dropped
    return ids[:used].reshape(rows, seq_len), mask[:used].reshape(rows, seq_len)


def load_packed(raw_dir: Path, cache_root: Path, seq_len: int, seed: int):
    """Packed rows, cached on disk (memory-mapped) under a fingerprint of the inputs."""
    files = sorted(raw_dir.glob("*.parquet"))
    key = json.dumps([[f.name, f.stat().st_size] for f in files] + [seq_len, seed])
    cache = cache_root / f"packed-{hashlib.sha256(key.encode()).hexdigest()[:16]}"
    if not (cache / "assistant_mask.npy").exists():
        ids, mask = pack(raw_dir, seq_len, seed)
        cache.mkdir(parents=True, exist_ok=True)
        np.save(cache / "input_ids.npy", ids)
        np.save(cache / "assistant_mask.npy", mask)  # written last: marks a complete cache
    return (
        np.load(cache / "input_ids.npy", mmap_mode="r"),
        np.load(cache / "assistant_mask.npy", mmap_mode="r"),
        cache.name,
    )


def batches(
    ids: np.ndarray, mask: np.ndarray, batch_size: int, seed: int, epoch: int, start: int = 0
) -> Iterator[tuple[int, torch.Tensor, torch.Tensor]]:
    """Deterministic shuffled micro-batches of one epoch, resumable at micro-batch `start`."""
    order = np.random.default_rng([seed, epoch]).permutation(len(ids))
    for index in range(start, len(order) // batch_size):
        # Sorted row ids read the memory-mapped arrays sequentially; order inside a batch is moot.
        rows = np.sort(order[index * batch_size : (index + 1) * batch_size])
        yield (
            index,
            torch.from_numpy(ids[rows].astype(np.int64)),
            torch.from_numpy(mask[rows].astype(np.bool_)),
        )
