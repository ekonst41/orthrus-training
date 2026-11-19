"""Generation speed and AR parity on benchmark prompts; this does not score task accuracy."""

import dataclasses
import itertools
import json
import logging
import time
from pathlib import Path

import torch
from transformers.cache_utils import DynamicCache

from src.config import parse_args
from src.hub import Bucket

log = logging.getLogger(__name__)


def load_eval_prompts(bucket, dataset, local_root):
    """local_root is the parent of dataset directories (normally data/)."""
    path = Path(local_root) / dataset / "eval_prompts.jsonl"
    if not path.exists():
        bucket.download_file(f"data/{dataset}/eval_prompts.jsonl", path)
    return (
        [json.loads(line) for line in path.read_text().splitlines() if line]
        if path.exists()
        else []
    )


def balanced_prompts(prompts: list[dict], count: int) -> list[dict]:
    """Up to `count` prompts taken round-robin across domains (count <= 0: all). The held-out file
    is grouped by domain, so a plain prefix would contain one domain only."""
    groups: dict[str, list[dict]] = {}
    for prompt in prompts:
        groups.setdefault(prompt.get("domain", ""), []).append(prompt)
    mixed = [p for row in itertools.zip_longest(*groups.values()) for p in row if p is not None]
    return mixed[:count] if count > 0 else mixed


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def ar_generate(model, prompt, max_new_tokens, eos):
    """Greedy AR reference; only the last vocabulary projection is needed."""
    cache = DynamicCache(config=model.config)
    tokens = []
    current = prompt
    for _ in range(max_new_tokens):
        logits = model(
            input_ids=current, past_key_values=cache, use_cache=True, logits_to_keep=1
        ).logits
        current = logits[:, -1].argmax(-1, keepdim=True)
        tokens.append(int(current))
        if tokens[-1] in eos:
            break
    return tokens


def _summary(records: list[dict], compare_ar: bool) -> dict:
    total = {k: sum(r[k] for r in records) for k in ("tokens", "passes", "cycles", "accepted")}
    seconds = sum(r["seconds"] for r in records)
    cycles = max(total["cycles"], 1)
    metrics = {
        "acceptance_length": total["accepted"] / cycles,
        "tpf": total["tokens"] / max(total["passes"], 1),
        "tokens_per_cycle": max(total["tokens"] - len(records), 0) / cycles,
        "tokens_per_second": total["tokens"] / max(seconds, 1e-9),
        "mean_new_tokens": total["tokens"] / max(len(records), 1),
    }
    if compare_ar:
        metrics["ar_match"] = sum(r["match"] for r in records) / max(len(records), 1)
        metrics["speedup"] = sum(r["ar_seconds"] for r in records) / max(seconds, 1e-9)
    return metrics


@torch.inference_mode()
def generation_metrics(
    model,
    prompts: list[list[int]],
    max_new_tokens: int,
    eos: set[int],
    compare_ar: bool = False,
    domains: list[str] | None = None,
) -> dict:
    """Decode each prompt with Orthrus (and optionally plain AR) and aggregate the metrics, overall
    and per domain ("<domain>/<metric>") when domains are given."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    if prompts:  # warm-up (kernel selection, allocator) outside the timed decodes
        warm = torch.tensor([prompts[0]], device=device)
        model.diffusion_generate(warm, 8, eos_token_id=list(eos))
        if compare_ar:
            ar_generate(model, warm, 8, eos)
    records = []
    for prompt_ids in prompts:
        prompt = torch.tensor([prompt_ids], device=device)
        _sync(device)
        start = time.perf_counter()
        output, stats = model.diffusion_generate(prompt, max_new_tokens, eos_token_id=list(eos))
        _sync(device)
        record = {
            "seconds": time.perf_counter() - start,
            "tokens": stats.new_tokens,
            "passes": stats.forward_passes,
            "cycles": stats.cycles,
            "accepted": sum(stats.accepted),
        }
        if compare_ar:
            start = time.perf_counter()
            reference = ar_generate(model, prompt, max_new_tokens, eos)
            _sync(device)
            generated = output[0, len(prompt_ids) :].tolist()
            record.update(
                ar_seconds=time.perf_counter() - start,
                match=generated == reference,
            )
        records.append(record)
    model.train(was_training)
    metrics = _summary(records, compare_ar)
    if domains is not None and len(set(domains)) > 1:
        for domain in sorted(set(domains)):
            subset = [r for r, d in zip(records, domains, strict=True) if d == domain]
            for key, value in _summary(subset, compare_ar).items():
                if key in ("tpf", "acceptance_length", "ar_match", "speedup"):
                    metrics[f"{domain}/{key}"] = value
    return metrics


def eos_token_ids(model, tokenizer) -> set[int]:
    """All end-of-turn ids (Qwen3 generation config: <|im_end|> and <|endoftext|>)."""
    ids = model.generation_config.eos_token_id or tokenizer.eos_token_id
    return set(ids if isinstance(ids, (list, tuple)) else [ids])


def chat_prompt(tokenizer, user: str) -> list[int]:
    """Same format as training data and the official README: empty system turn, no thinking."""
    messages = [{"role": "system", "content": ""}, {"role": "user", "content": user}]
    encoded = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=False, tokenize=True, return_dict=True
    )
    return list(encoded["input_ids"])


BENCHMARKS = {
    "gsm8k": ("openai/gsm8k", "main", "question", "740312add88f781978c0658806c59bc2815b9866"),
    "math500": (
        "HuggingFaceH4/MATH-500",
        None,
        "problem",
        "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
    ),
    "humaneval": (
        "openai/openai_humaneval",
        None,
        "prompt",
        "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
    ),
    "mbpp": (
        "google-research-datasets/mbpp",
        "sanitized",
        "prompt",
        "4bb6404fdc6cacfda99d4ac4205087b89d32030c",
    ),
}


def load_for_eval(cfg, device, dtype):
    from src.train.checkpoint import (
        DFLASH_CONFIG,
        WEIGHTS,
        CheckpointManager,
        load_checkpoint,
        load_model,
    )

    bucket = Bucket(cfg.hf.bucket, cfg.hf.upload)
    if cfg.eval.checkpoint:
        manager = CheckpointManager(cfg.run_path, bucket, f"runs/{cfg.run_name}/checkpoints", 0)
        path = (
            manager.latest()
            if cfg.eval.checkpoint == "latest"
            else manager.get(cfg.eval.checkpoint)
        )
        if path is None:
            raise FileNotFoundError("no training checkpoint")
        model, tokenizer, _ = load_model(cfg.model, device, dtype)
        load_checkpoint(path, model)
        return model, tokenizer, path.name
    final = cfg.run_path / "final"
    if cfg.model.kind == "dflash":
        if not all((final / name).exists() for name in (DFLASH_CONFIG, WEIGHTS)):
            if not bucket.download(f"runs/{cfg.run_name}/final", final):
                raise FileNotFoundError(final)
        spec = dataclasses.replace(cfg.model, base=str(final), revision=None)
        model, tokenizer, _ = load_model(spec, device, dtype)
        return model, tokenizer, "final"
    index = final / "model.safetensors.index.json"
    weights = (
        set(json.loads(index.read_text())["weight_map"].values())
        if index.exists()
        else {"model.safetensors"}
    )
    if not all(
        (final / name).exists() for name in ("config.json", "tokenizer_config.json", *weights)
    ):
        if not bucket.download(f"runs/{cfg.run_name}/final", final):
            raise FileNotFoundError(final)
    spec = dataclasses.replace(cfg.model, base=str(final), revision=None)
    model, tokenizer, _ = load_model(spec, device, dtype)
    return model, tokenizer, "final"


def main(argv=None):
    from datasets import load_dataset

    cfg = parse_args(__doc__, argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, cfg.eval.dtype) if device.type == "cuda" else torch.float32
    # Strict fp32 parity is checked separately from bf16 speed measurements.
    torch.backends.cuda.matmul.allow_tf32 = False
    model, tokenizer, label = load_for_eval(cfg, device, dtype)
    bucket = Bucket(cfg.hf.bucket, cfg.hf.upload)
    results = {
        "_meta": {
            "model": label,
            "dtype": str(dtype),
            "max_new_tokens": cfg.eval.max_new_tokens,
            "prompts_per_suite": cfg.eval.prompts,
            "task_accuracy_scored": False,
        }
    }
    for suite in cfg.eval.suites:
        if suite == "heldout":
            rows = balanced_prompts(
                load_eval_prompts(bucket, cfg.data.dataset, cfg.data_dir), cfg.eval.prompts
            )
            prompts, domains = [r["prompt_ids"] for r in rows], [r["domain"] for r in rows]
        else:
            repo, subset, field, revision = BENCHMARKS[suite]
            rows = load_dataset(repo, subset, split="test", revision=revision)
            rows = (
                rows.select(range(min(cfg.eval.prompts, len(rows)))) if cfg.eval.prompts else rows
            )
            prompts, domains = [chat_prompt(tokenizer, r[field]) for r in rows], None
        if not prompts:
            log.warning("%s: no prompts, skipped", suite)
            continue
        metrics = generation_metrics(
            model,
            prompts,
            cfg.eval.max_new_tokens,
            eos_token_ids(model, tokenizer),
            compare_ar=True,
            domains=domains,
        )
        results[suite] = {"prompts": len(prompts), **metrics}
        log.info("%s: %s", suite, json.dumps(results[suite]))
    out = cfg.run_path / f"eval-{label}-{cfg.eval.max_new_tokens}-{cfg.eval.dtype}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    bucket.upload(out, f"runs/{cfg.run_name}/{out.name}")


if __name__ == "__main__":
    main()
