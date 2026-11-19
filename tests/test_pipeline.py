"""CPU end-to-end checks, with a tiny local Qwen3 and no network/HF credentials."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import yaml
from safetensors.torch import load_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from src.config import load_config
from src.dataset import data
from src.hub import Bucket, upload_checkpoint
from src.test import benchmark
from src.train import train
from src.train.checkpoint import load_model


def setup_experiment(root):
    base = root / "base"
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
    Qwen3ForCausalLM(config).save_pretrained(base)
    tokenizer = Tokenizer(models.WordLevel({f"t{i}": i for i in range(512)}, unk_token="t0"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, eos_token="t510", unk_token="t0"
    ).save_pretrained(base)
    raw = root / "data/tiny/raw"
    raw.mkdir(parents=True)
    rng = np.random.default_rng(0)
    prompts = [rng.integers(1, 500, 5).tolist() for _ in range(60)]
    responses = [rng.integers(1, 500, 20).tolist() for _ in range(60)]
    pq.write_table(
        pa.table(
            {
                "prompt_ids": pa.array(prompts, pa.list_(pa.int32())),
                "response_ids": pa.array(responses, pa.list_(pa.int32())),
            }
        ),
        raw / "shard-00000.parquet",
    )
    (raw.parent / "eval_prompts.jsonl").write_text(
        json.dumps({"domain": "math", "prompt_ids": prompts[0]}) + "\n"
    )
    values = {
        "run_name": "tiny",
        "data_dir": str(root / "data"),
        "output_dir": str(root / "outputs"),
        "model": {"base": str(base), "block_size": 4, "mask_token_id": 511},
        "data": {"dataset": "tiny", "seq_len": 32, "eval_rows": 3, "seed": 0},
        "train": {
            "epochs": 1,
            "global_batch_size": 4,
            "micro_batch_size": 1,
            "max_steps": 4,
            "num_anchor_blocks": 4,
            "save_every": 2,
            "eval_every": 2,
            "log_every": 1,
            "kl_chunk_size": 16,
            "compile": False,
        },
        "eval": {"prompts": 1, "max_new_tokens": 8, "suites": ["heldout"]},
        "hf": {"bucket": ""},
    }
    path = root / "tiny.yaml"
    path.write_text(yaml.safe_dump(values))
    return path


def test_training_resume_export_and_benchmark(tmp_path):
    config_path = setup_experiment(tmp_path)
    train.main(["--config", str(config_path)])
    run = tmp_path / "outputs/tiny"
    full = load_file(run / "checkpoints/step-0000004/trainable.safetensors")
    shutil.rmtree(run / "checkpoints/step-0000004")
    shutil.rmtree(run / "final")
    train.main(["--config", str(config_path)])
    resumed = load_file(run / "checkpoints/step-0000004/trainable.safetensors")
    assert full.keys() == resumed.keys()
    for name in full:
        assert torch.equal(full[name], resumed[name]), name
    cfg = load_config(config_path)
    model, _, _ = load_model(cfg.model, torch.device("cpu"), torch.float32)
    # Export is genuinely independent of src/: it reloads with the HF remote-code interface.
    exported = AutoModelForCausalLM.from_pretrained(
        run / "final", trust_remote_code=True, dtype=torch.float32
    )
    q = exported.model.layers[0].self_attn.q_proj_diff.weight
    assert torch.equal(q, full["model.layers.0.self_attn.q_proj_diff.weight"].bfloat16().float())
    assert not torch.equal(q, model.model.layers[0].self_attn.q_proj_diff.weight)
    records = [json.loads(s) for s in (run / "train_metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in records if "train/loss" in r] == [1, 2, 3, 4]
    assert any("eval/kl" in r for r in records)
    assert any("generate/tpf" in r for r in records)
    benchmark.main(["--config", str(config_path)])
    result = json.loads((run / "eval-final-8-float32.json").read_text())
    assert result["heldout"]["ar_match"] == 1.0
    with pytest.raises(ValueError, match="cannot resume"):
        train.main(["--config", str(config_path), "train.learning_rate=1e-3"])


@pytest.mark.parametrize("loss_type", ["kl", "ce"])
def test_dflash_training_export_and_greedy_parity(tmp_path, loss_type):
    config_path = setup_experiment(tmp_path)
    args = [
        "--config", str(config_path), "model.kind=dflash", "model.draft_layers=1",
        f"train.loss_type={loss_type}", "train.max_steps=1", "train.activation_checkpointing=true",
    ]
    train.main(args)
    run = tmp_path / "outputs/tiny"
    records = [json.loads(s) for s in (run / "train_metrics.jsonl").read_text().splitlines()]
    assert any(f"eval/{loss_type}" in record for record in records)
    assert any("generate/tpf" in record for record in records)
    saved = load_file(run / "checkpoints/step-0000001/trainable.safetensors")
    assert "context_proj.weight" in saved
    assert all(not key.startswith("target.") for key in saved)
    benchmark.main(args)
    result = json.loads((run / "eval-final-8-float32.json").read_text())
    assert result["heldout"]["ar_match"] == 1.0


def test_stop_saves_before_exit(tmp_path):
    config_path = setup_experiment(tmp_path)
    run = tmp_path / "outputs/tiny"
    run.mkdir(parents=True)
    (run / "STOP").touch()
    train.main(["--config", str(config_path)])
    assert (run / "checkpoints/step-0000001/trainer_state.json").exists()
    assert not (run / "final").exists()
    assert not (run / "STOP").exists()


def test_distributed_training_matches_serial_and_resumes(tmp_path):
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch.distributed as d; from datetime import timedelta; "
            f"d.init_process_group('gloo', init_method='file://{tmp_path}/gloo-probe', "
            "rank=0, world_size=1, timeout=timedelta(seconds=3)); d.destroy_process_group()",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if "Operation not permitted" in probe.stderr:
        pytest.skip("this environment forbids Gloo TCP sockets; run this test on the server")
    assert probe.returncode == 0, probe.stdout + probe.stderr
    config_path = setup_experiment(tmp_path)
    # Exercise real torchrun, Gloo collectives and optimizer updates on two CPU processes.
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        "--module",
        "src.train.train",
        "--config",
        str(config_path),
        "run_name=parallel",
    ]
    result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    run = tmp_path / "outputs/parallel"
    full = load_file(run / "checkpoints/step-0000004/trainable.safetensors")
    shutil.rmtree(run / "checkpoints/step-0000004")
    result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    resumed = load_file(run / "checkpoints/step-0000004/trainable.safetensors")
    for name in full:
        assert torch.equal(full[name], resumed[name]), name
    train.main(["--config", str(config_path)])
    serial = load_file(tmp_path / "outputs/tiny/checkpoints/step-0000004/trainable.safetensors")
    for name in full:
        assert torch.allclose(full[name], serial[name], atol=3e-6, rtol=1e-4), name


def test_rank_batches_are_disjoint_and_resumable():
    ids = np.arange(48).reshape(24, 2)
    mask = np.ones_like(ids)
    batches = [
        list(data.batches(ids, mask, 2, 42, 0, rank=rank, world_size=3, global_batch_size=12))
        for rank in range(3)
    ]
    all_rows = [tuple(row) for part in batches for _, rows, _ in part for row in rows.tolist()]
    assert len(all_rows) == len(set(all_rows)) == 24
    for rank in range(3):
        tail = list(
            data.batches(
                ids, mask, 2, 42, 0, rank=rank, world_size=3, global_batch_size=12, start_step=1
            )
        )
        assert [x[0] for x in tail] == [x[0] for x in batches[rank][2:]]
        assert all(torch.equal(a[1], b[1]) for a, b in zip(tail, batches[rank][2:], strict=True))


def test_checkpoint_marker_is_uploaded_last(tmp_path, monkeypatch):
    bucket = Bucket("test/bucket")
    calls = []
    monkeypatch.setattr(bucket, "upload", lambda path, remote: calls.append(Path(path).name))
    upload_checkpoint(bucket, tmp_path / "step-0000001", "runs/test/checkpoints")
    assert calls == ["trainable.safetensors", "optimizer.pt", "trainer_state.json"]


def test_config_validates_keys_and_local_mode(monkeypatch):
    monkeypatch.setenv("ORTHRUS_BUCKET", "test/bucket")
    assert load_config().hf.bucket == "test/bucket"
    assert load_config(overrides=["hf.bucket="]).hf.bucket == ""
    assert load_config(overrides=["train.learning_rate=1e-4"]).train.learning_rate == 1e-4
    with pytest.raises(ValueError, match="unknown"):
        load_config(overrides=["train.typo=1"])
    with pytest.raises(ValueError, match="divisible"):
        load_config(overrides=["train.global_batch_size=3"])
