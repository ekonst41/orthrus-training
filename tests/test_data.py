import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from src.config import load_config
from src.dataset import data, prepare
from src.hub import Bucket
from src.test.trace import trace_orthrus


def write_shard(path, response):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "prompt_ids": pa.array([[1, 2], [3]], pa.list_(pa.int32())),
                "response_ids": pa.array([response, [10, 11, 12]], pa.list_(pa.int32())),
            }
        ),
        path,
    )


def test_packing_masks_and_content_fingerprint(tmp_path):
    raw = tmp_path / "raw"
    write_shard(raw / "shard-00000.parquet", [7, 8, 9])
    ids, mask = data.pack(raw, 4, 42)
    assert set(ids[mask.astype(bool)]) <= {7, 8, 9, 10, 11, 12}
    assert set(ids[~mask.astype(bool)]) <= {1, 2, 3}
    _, _, first = data.load_packed(raw, tmp_path / "packed", 4, 42)
    write_shard(raw / "shard-00000.parquet", [6, 8, 9])
    again, _, second = data.load_packed(raw, tmp_path / "packed", 4, 42)
    assert first != second
    assert not np.array_equal(ids, again)


def test_prompt_selection_is_cached(tmp_path, monkeypatch):
    calls = []
    train = [{"uuid": "t", "domain": "math", "prompt_ids": [1, 2]}]
    heldout = [{"uuid": "v", "domain": "chat", "prompt_ids": [3, 4]}]

    def sample(config, tokenizer):
        calls.append(config.samples_per_domain)
        return train, heldout, "revision"

    monkeypatch.setattr(prepare, "sample_prompts", sample)
    cfg = load_config(overrides=["hf.bucket="])
    bucket = Bucket("")
    assert prepare.selected_prompts(cfg, None, bucket, tmp_path) == (train, heldout, "revision")
    assert prepare.selected_prompts(cfg, None, bucket, tmp_path) == (train, heldout, "revision")
    assert len(calls) == 1
    cfg.datagen.samples_per_domain += 1
    prepare.selected_prompts(cfg, None, bucket, tmp_path)
    assert len(calls) == 2


def test_trace_preserves_output_and_block_alignment(tiny_model):
    model = tiny_model.eval()
    prompt = torch.tensor([[5, 10, 15]])
    expected, stats = model.diffusion_generate(prompt, 24)
    before = len(model._forward_hooks)
    ids, blocks, traced = trace_orthrus(model, prompt, 24)
    assert ids == expected[0].tolist()
    assert traced.accepted == stats.accepted
    assert len(model._forward_hooks) == before
    for block in blocks:
        assert ids[block["start"]] == block["anchor"]
        assert (
            ids[block["start"] + 1 : block["start"] + 1 + block["accepted"]]
            == block["draft"][: block["accepted"]]
        )
        assert len(block["top_ids"]) == len(block["draft"])
