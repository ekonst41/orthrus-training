"""End-to-end training on CPU with a tiny Qwen3 checkpoint, local storage and no tracker."""

import json
import shutil

import numpy as np
import pytest
import torch
import yaml
from safetensors.torch import load_file
from test_data_config import write_shard
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from orthrus import train
from orthrus.checkpoint import load_model
from orthrus.config import ModelConfig
from orthrus.storage import Bucket


def make_base_checkpoint(path):
    """A plain (non-Orthrus) Qwen3 checkpoint with a word-level tokenizer."""
    config = Qwen3Config(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=1024,
        tie_word_embeddings=True,
        eos_token_id=[510, 509],
    )
    torch.manual_seed(0)
    Qwen3ForCausalLM(config).save_pretrained(path)
    tokenizer = Tokenizer(models.WordLevel({f"t{i}": i for i in range(512)}, unk_token="t0"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, eos_token="t510", unk_token="t0"
    ).save_pretrained(path)


def write_dataset(root, name):
    rng = np.random.default_rng(0)
    prompts = [rng.integers(1, 500, rng.integers(3, 12)).tolist() for _ in range(60)]
    responses = [rng.integers(1, 500, rng.integers(5, 30)).tolist() for _ in range(60)]
    write_shard(root / "data" / name / "raw" / "shard-00000.parquet", prompts, responses)
    lines = [json.dumps({"domain": "test", "prompt_ids": p}) + "\n" for p in prompts[:2]]
    (root / "data" / name / "eval_prompts.jsonl").write_text("".join(lines), encoding="utf-8")


def tiny_config(tmp_path):
    make_base_checkpoint(tmp_path / "base")
    write_dataset(tmp_path, "tiny")
    return {
        "run_name": "tiny",
        "model": {"base": str(tmp_path / "base"), "block_size": 4, "mask_token_id": 511},
        "data": {"dataset": "tiny", "seq_len": 32, "eval_rows": 4, "seed": 0},
        "train": {
            "epochs": 1,
            "global_batch_size": 4,
            "micro_batch_size": 2,
            "max_steps": 4,
            "num_anchor_blocks": 4,
            "save_every": 2,
            "eval_every": 2,
            "log_every": 1,
            "kl_chunk_size": 16,
        },
        "eval": {"prompts": 2, "max_new_tokens": 8, "check_ar_parity": True},
        "storage": {"bucket": "", "local_dir": str(tmp_path)},
        "tracking": {"enabled": False},
    }


def test_training_resumes_bit_exactly(tmp_path):
    config = tiny_config(tmp_path)
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    run_dir = tmp_path / "runs" / "tiny"

    train.main(["--config", str(config_path)])
    checkpoints = run_dir / "checkpoints"
    assert sorted(p.name for p in checkpoints.iterdir()) == ["step-0000002", "step-0000004"]
    exported = json.loads((run_dir / "final" / "config.json").read_text())
    assert exported["auto_map"]["AutoModelForCausalLM"] == "modeling_orthrus.OrthrusLM"
    assert exported["block_size"] == 4 and exported["mask_token_id"] == 511
    assert (run_dir / "final" / "modeling_orthrus.py").exists()
    metrics = [
        json.loads(line) for line in (run_dir / "train_metrics.jsonl").read_text().splitlines()
    ]
    for key in (
        "train/loss",
        "eval/kl",
        "eval/acceptance_proxy",
        "generate/tpf",
        "generate/ar_match",
    ):
        assert any(key in record for record in metrics), key
    assert all(r["generate/ar_match"] == 1.0 for r in metrics if "generate/ar_match" in r)
    uninterrupted = load_file(checkpoints / "step-0000004" / "trainable.safetensors")

    # Interrupt after step 2, then rerun: the result must equal the uninterrupted run.
    shutil.rmtree(checkpoints / "step-0000004")
    shutil.rmtree(run_dir / "final")
    train.main(["--config", str(config_path)])
    resumed = load_file(checkpoints / "step-0000004" / "trainable.safetensors")
    assert uninterrupted.keys() == resumed.keys()
    for name, tensor in uninterrupted.items():
        assert torch.equal(tensor, resumed[name]), name

    # One manifest segment per job; the metrics history continues without duplicate steps.
    manifest = json.loads((run_dir / "run.json").read_text())
    assert [s["resumed_at_step"] for s in manifest["segments"]] == [0, 2]
    assert [s["ended_at_step"] for s in manifest["segments"]] == [4, 4]
    records = [
        json.loads(line) for line in (run_dir / "train_metrics.jsonl").read_text().splitlines()
    ]
    train_steps = [r["step"] for r in records if "train/loss" in r]
    assert train_steps == [1, 2, 3, 4]
    assert len(list((run_dir / "logs").glob("train-*.log"))) >= 1

    # A changed recipe (here the learning rate) cannot continue the same run.
    shutil.rmtree(checkpoints / "step-0000004")
    config["train"]["learning_rate"] = 1e-3
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(RuntimeError, match="train.learning_rate"):
        train.main(["--config", str(config_path)])

    # The exported model reloads with its trained diffusion weights (no warm start from AR).
    model, _, _ = load_model(
        ModelConfig(base=str(run_dir / "final"), block_size=4), torch.device("cpu"), torch.float32
    )
    q_ar = model.model.layers[0].self_attn.q_proj.weight
    q_diff = model.model.layers[0].self_attn.q_proj_diff.weight
    assert not torch.equal(q_ar, q_diff)


def test_stop_request_saves_and_exits(tmp_path, monkeypatch):
    """runs/<run>/STOP in the bucket (scripts/ds.py stop): checkpoint, exit, remove the request."""
    config = tiny_config(tmp_path)
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    deleted = []
    monkeypatch.setattr(Bucket, "exists", lambda self, path: path == "runs/tiny/STOP")
    monkeypatch.setattr(Bucket, "delete", lambda self, prefix: deleted.append(prefix))
    train.main(["--config", str(config_path)])
    run_dir = tmp_path / "runs" / "tiny"
    assert sorted(p.name for p in (run_dir / "checkpoints").iterdir()) == ["step-0000001"]
    assert not (run_dir / "final").exists()
    segment = json.loads((run_dir / "run.json").read_text())["segments"][-1]
    assert segment["stop_reason"] == "stop requested" and segment["ended_at_step"] == 1
    assert deleted == ["runs/tiny/STOP"]
