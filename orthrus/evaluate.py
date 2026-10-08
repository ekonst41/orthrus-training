"""Speed evaluation (paper, Sec. 4.2): forward passes per token, acceptance length, AR parity.

    python -m orthrus.evaluate --config configs/qwen3-0.6b.yaml [eval.dtype=float32]
        [eval.checkpoint=step-0002000|latest] [eval.prompts=32] [eval.max_new_tokens=512]

Evaluates the exported model (runs/<run>/final) or, with eval.checkpoint, a training checkpoint, on
the held-out prompts (balanced across domains) and benchmark prompts, eval.prompts per suite.
Decoding is greedy (temperature 0), batch size 1, thinking disabled (official README).
Metrics (paper definitions; Table 1 and Fig. 4 agree only with these):
  acceptance_length  accepted draft tokens per diffusion + verification cycle (bonus excluded)
  tpf                generated tokens / forward passes, forward passes = 1 + 2 * cycles
  tokens_per_cycle   generated tokens per verification pass (what Zarya's script calls TPF)
  ar_match           fraction of prompts whose output equals plain AR greedy decoding (1.0 in
                     fp32; in bf16 near-tied AR logits make it lower, see ar_divergence_logit_gap)
  speedup            wall-clock time of AR decoding / time of Orthrus decoding, same prompts
Per-domain values: <domain>/<metric>. Results: runs/<run>/eval-<model>-<N>-<dtype>.json
"""

import itertools
import json
import logging
import time
from pathlib import Path

import torch
from transformers.cache_utils import DynamicCache

from orthrus.config import parse_args
from orthrus.storage import Bucket

log = logging.getLogger("orthrus.evaluate")


def load_eval_prompts(bucket: Bucket, dataset: str, local_root: Path) -> list[dict]:
    """Held-out prompts written by datagen (never trained on): [{"domain", "prompt_ids"}, ...]."""
    path = local_root / "data" / dataset / "eval_prompts.jsonl"
    if bucket.enabled and not path.exists():
        bucket.download_file(f"data/{dataset}/eval_prompts.jsonl", path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


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
def ar_generate(
    model, prompt: torch.Tensor, max_new_tokens: int, eos: set[int]
) -> tuple[list[int], list[float]]:
    """Plain greedy decoding with the frozen AR view (reference for parity and speed).
    Also returns the top-1 minus top-2 logit gap per token, to tell rounding ties from bugs."""
    cache = DynamicCache(config=model.config)
    logits = model(input_ids=prompt, past_key_values=cache, use_cache=True).logits
    tokens: list[int] = []
    gaps = []
    while True:
        token = logits[:, -1].argmax(-1, keepdim=True)  # argmax tie-breaking, as in decoding
        top2 = logits[0, -1].topk(2).values
        gaps.append(top2[0] - top2[1])
        tokens.append(int(token))
        if tokens[-1] in eos or len(tokens) == max_new_tokens:
            return tokens, torch.stack(gaps).float().tolist()
        logits = model(input_ids=token, past_key_values=cache, use_cache=True).logits


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
        diverged = [r for r in records if not r["match"]]
        if diverged:  # where outputs first differ, and how close AR's top-2 logits were there
            metrics["ar_first_divergence"] = sum(r["divergence"] for r in diverged) / len(diverged)
            gaps = sorted(r["gap"] for r in diverged)
            metrics["ar_divergence_logit_gap"] = gaps[len(gaps) // 2]
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
            reference, reference_gaps = ar_generate(model, prompt, max_new_tokens, eos)
            _sync(device)
            generated = output[0, len(prompt_ids) :].tolist()
            first = next(
                (i for i, (a, b) in enumerate(zip(generated, reference, strict=False)) if a != b),
                min(len(generated), len(reference)),
            )
            record.update(
                ar_seconds=time.perf_counter() - start,
                match=generated == reference,
                divergence=first,
                gap=reference_gaps[min(first, len(reference_gaps) - 1)],
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


def _load(name: str, split: str, field: str, config: str | None = None) -> list[str]:
    from datasets import load_dataset

    return [row[field] for row in load_dataset(name, config, split=split)]


# Benchmarks of the paper's Table 1 that are plain prompt -> completion tasks (zero-shot).
BENCHMARKS = {
    "gsm8k": lambda: _load("openai/gsm8k", "test", "question", "main"),
    "math500": lambda: _load("HuggingFaceH4/MATH-500", "test", "problem"),
    "humaneval": lambda: _load("openai/openai_humaneval", "test", "prompt"),
    "mbpp": lambda: _load("google-research-datasets/mbpp", "test", "prompt", "sanitized"),
}


def load_for_eval(cfg, device: torch.device, dtype: torch.dtype):
    """(model, tokenizer, label): the exported model, or the base model with the diffusion weights
    of a training checkpoint when cfg.eval.checkpoint is set ("step-0001000" or "latest")."""
    from orthrus.checkpoint import CheckpointManager, load_checkpoint, load_model

    run_dir = Path(cfg.storage.local_dir) / "runs" / cfg.run_name
    bucket = Bucket(cfg.storage.bucket)
    if cfg.eval.checkpoint:
        manager = CheckpointManager(run_dir, bucket, f"runs/{cfg.run_name}/checkpoints", keep=0)
        name = cfg.eval.checkpoint
        path = manager.latest() if name == "latest" else manager.get(name)
        if path is None:
            raise FileNotFoundError(f"run {cfg.run_name} has no checkpoint yet")
        model, tokenizer, _ = load_model(cfg.model, device, dtype)
        load_checkpoint(path, model)
        return model, tokenizer, path.name
    final_dir = run_dir / "final"
    if not (final_dir / "config.json").exists():
        if not bucket.download(f"runs/{cfg.run_name}/final", final_dir):
            raise FileNotFoundError(f"no exported model for run {cfg.run_name}")
    model_cfg = cfg.model
    model_cfg.base, model_cfg.revision = str(final_dir), None
    model, tokenizer, _ = load_model(model_cfg, device, dtype)
    return model, tokenizer, "final"


def main(argv: list[str] | None = None) -> None:
    from orthrus.tracking import Tracker, Watchdog, _git_revision

    cfg = parse_args(__doc__, argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    watchdog = Watchdog(cfg.train.stall_minutes, "evaluation progress")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_root = Path(cfg.storage.local_dir)
    run_dir = local_root / "runs" / cfg.run_name
    bucket = Bucket(cfg.storage.bucket)
    dtype = getattr(torch, cfg.eval.dtype) if device.type == "cuda" else torch.float32
    model, tokenizer, label = load_for_eval(cfg, device, dtype)
    eos = eos_token_ids(model, tokenizer)
    watchdog.beat()

    heldout = balanced_prompts(load_eval_prompts(bucket, cfg.data.dataset, local_root), 0)
    suites = {"heldout": ([p["prompt_ids"] for p in heldout], [p["domain"] for p in heldout])}
    for name, load in BENCHMARKS.items():
        suites[name] = ([chat_prompt(tokenizer, q) for q in load()], None)
    results = {
        "_meta": {
            "model": label,
            "dtype": cfg.eval.dtype,
            "max_new_tokens": cfg.eval.max_new_tokens,
            "prompts_per_suite": cfg.eval.prompts,
            "git_commit": _git_revision(),
        }
    }
    tracker = Tracker(cfg, run_dir, "eval")
    for step, (name, (prompts, domains)) in enumerate(suites.items()):
        if cfg.eval.prompts > 0:
            prompts = prompts[: cfg.eval.prompts]
            domains = domains[: cfg.eval.prompts] if domains else None
        if not prompts:
            continue
        metrics = generation_metrics(
            model, prompts, cfg.eval.max_new_tokens, eos, compare_ar=True, domains=domains
        )
        results[name] = {"prompts": len(prompts), **metrics}
        log.info("%s: %s", name, json.dumps(results[name]))
        tracker.log({f"{name}/{k}": v for k, v in metrics.items()}, step=step)
        watchdog.beat()
    out = run_dir / f"eval-{label}-{cfg.eval.max_new_tokens}-{cfg.eval.dtype}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    bucket.upload(out, f"runs/{cfg.run_name}/{out.name}")
    tracker.close()


if __name__ == "__main__":
    main()
