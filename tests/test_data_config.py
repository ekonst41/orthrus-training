import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from orthrus import data
from orthrus.config import load_config


def write_shard(path, prompts, responses):
    path.parent.mkdir(parents=True, exist_ok=True)
    int_lists = pa.list_(pa.int32())
    table = pa.table(
        {
            "prompt_ids": pa.array(prompts, type=int_lists),
            "response_ids": pa.array(responses, type=int_lists),
        }
    )
    pq.write_table(table, path)


def test_pack_marks_exactly_the_response_tokens(tmp_path):
    raw = tmp_path / "raw"
    prompts = [[1, 2, 3], [4, 5], [6]]
    responses = [[10, 11], [12, 13, 14, 15], [16, 17, 18]]
    write_shard(raw / "shard-00000.parquet", prompts, responses)
    ids, mask = data.pack(raw, seq_len=4, seed=0)
    assert ids.shape == (3, 4) and mask.shape == ids.shape  # 15 tokens -> 3 full rows
    assert set(ids[mask.astype(bool)].tolist()) <= {10, 11, 12, 13, 14, 15, 16, 17, 18}
    assert set(ids[~mask.astype(bool)].tolist()) <= {1, 2, 3, 4, 5, 6}
    again, _ = data.pack(raw, seq_len=4, seed=0)
    assert np.array_equal(ids, again)  # deterministic for a seed


def test_batches_are_deterministic_and_resumable():
    ids = np.arange(40, dtype=np.int32).reshape(20, 2)
    mask = np.ones_like(ids, dtype=np.uint8)
    full = [b.tolist() for _, b, _ in data.batches(ids, mask, 4, seed=3, epoch=1)]
    resumed = [b.tolist() for _, b, _ in data.batches(ids, mask, 4, seed=3, epoch=1, start=2)]
    assert resumed == full[2:]
    other_epoch = [b.tolist() for _, b, _ in data.batches(ids, mask, 4, seed=3, epoch=2)]
    assert other_epoch != full
    assert sorted(sum((sum(b, []) for b in full), [])) == list(range(40))


def test_config_overrides_and_types(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text("run_name: a\ntrain:\n  micro_batch_size: 8\n", encoding="utf-8")
    cfg = load_config(
        path,
        [
            "train.learning_rate=1e-4",
            "train.adam_betas=[0.8, 0.9]",
            "datagen.domains=[math]",
            "eval.check_ar_parity=true",
        ],
    )
    assert cfg.run_name == "a" and cfg.train.micro_batch_size == 8
    assert cfg.train.learning_rate == pytest.approx(1e-4)
    assert cfg.train.adam_betas == (0.8, 0.9) and cfg.datagen.domains == ("math",)
    assert cfg.eval.check_ar_parity is True
    with pytest.raises(ValueError):
        load_config(path, ["train.no_such_key=1"])
    with pytest.raises(ValueError):
        load_config(path, ["train.global_batch_size=10"])  # not a multiple of 8


def test_storage_and_space_fall_back_to_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("ORTHRUS_BUCKET", "someone/bucket")
    monkeypatch.setenv("ORTHRUS_TRACKIO_SPACE", "someone/space")
    cfg = load_config(None)
    assert cfg.storage.bucket == "someone/bucket" and cfg.tracking.space_id == "someone/space"
    assert load_config(None, ["storage.bucket="]).storage.bucket == ""  # explicit: local only
