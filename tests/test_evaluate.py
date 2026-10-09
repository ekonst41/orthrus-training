import json

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from orthrus import generate, report
from orthrus.config import load_config
from orthrus.evaluate import balanced_prompts, generation_metrics

CHAT_TEMPLATE = (
    "{% for m in messages %}t1 {{ m['content'] }} t2 {% endfor %}"
    "{% if add_generation_prompt %}t3{% endif %}"
)


def test_balanced_prompts_alternate_domains():
    prompts = [{"domain": d, "prompt_ids": [i]} for d in ("math", "code", "chat") for i in range(3)]
    picked = balanced_prompts(prompts, 4)
    assert [p["domain"] for p in picked] == ["math", "code", "chat", "math"]
    assert len(balanced_prompts(prompts, 0)) == 9


def test_generation_metrics_per_domain_and_exact_parity(tiny_model):
    torch.manual_seed(0)
    prompts = [torch.randint(0, 500, (6,)).tolist() for _ in range(4)]
    eos = set(tiny_model.config.eos_token_id)
    metrics = generation_metrics(
        tiny_model, prompts, 12, eos, compare_ar=True, domains=["math", "code", "math", "code"]
    )
    assert metrics["ar_match"] == 1.0  # fp32 on CPU: lossless
    assert {"math/tpf", "code/tpf", "math/acceptance_length"} <= metrics.keys()
    assert 0 < metrics["tpf"] <= tiny_model.config.block_size


def tiny_tokenizer():
    vocab = {f"t{i}": i for i in range(512)}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="t0"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="t0")
    fast.chat_template = CHAT_TEMPLATE
    return fast


def test_generate_cli_reports_decoding_stats(tiny_model, monkeypatch, capsys):
    monkeypatch.setattr(
        generate, "load_for_eval", lambda cfg, device, dtype: (tiny_model, tiny_tokenizer(), "x")
    )
    generate.main(
        ["--config", "configs/smoke.yaml", "--prompt", "t5 t6", "--max-new-tokens", "6",
         "--compare-ar", "storage.bucket="]
    )  # fmt: skip
    out = capsys.readouterr().out
    assert "TPF" in out and "identical answer" in out


def test_report_summarizes_local_run(tmp_path):
    run_dir = tmp_path / "runs" / "demo"
    run_dir.mkdir(parents=True)
    segment = {"resumed_at_step": 0, "ended_at_step": 2, "hours": 0.5, "stop_reason": "complete"}
    manifest = {"config": {"model": {"base": "m"}}, "total_steps": 2, "segments": [segment]}
    (run_dir / "run.json").write_text(json.dumps(manifest))
    records = [
        {"step": 1, "train/loss": 3.0, "perf/seconds_per_step": 2.0},
        {"step": 2, "train/loss": 2.0, "perf/seconds_per_step": 2.0},
        {"step": 2, "eval/kl": 1.5, "generate/tpf": 1.2, "generate/math/tpf": 1.3},
    ]
    (run_dir / "train_metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    results = {"_meta": {"model": "final", "dtype": "float32"}, "heldout": {"tpf": 1.4}}
    (run_dir / "eval-final-512-float32.json").write_text(json.dumps(results))
    cfg = load_config(overrides=["storage.bucket=", f"storage.local_dir={tmp_path}"])
    root = report.fetch(cfg, "demo", tmp_path / "report")
    text = report.report("demo", root)
    assert "step 2/2 (100.0%)" in text and "~272 RUB" in text
    assert "eval-final-512-float32.json" in text and "1.4" in text
