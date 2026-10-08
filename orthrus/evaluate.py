"""Speed evaluation (paper, Sec. 4.2): forward passes per token, acceptance length, AR parity.

    python -m orthrus.evaluate --config configs/qwen3-0.6b.yaml [eval.max_new_tokens=1024]

Decoding is greedy (temperature 0), batch size 1, thinking disabled, as in the paper.
Metrics (paper definitions; Table 1 and Fig. 4 agree only with these):
  acceptance_length  accepted draft tokens per diffusion + verification cycle (bonus excluded)
  tpf                generated tokens / forward passes, forward passes = 1 + 2 * cycles
  tokens_per_cycle   generated tokens per verification pass (what Zarya's script calls TPF)
  ar_match           fraction of prompts whose output equals plain AR greedy decoding
  speedup            wall-clock time of AR decoding / time of Orthrus decoding, same prompts
"""

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


@torch.inference_mode()
def generation_metrics(
    model, prompts: list[list[int]], max_new_tokens: int, eos: set[int], compare_ar: bool = False
) -> dict:
    """Decode each prompt with Orthrus (and optionally plain AR) and aggregate the metrics."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    totals = {"tokens": 0, "passes": 0, "cycles": 0, "accepted": 0, "seconds": 0.0}
    ar_seconds, matches, divergences, gaps = 0.0, 0, [], []
    for prompt_ids in prompts:
        prompt = torch.tensor([prompt_ids], device=device)
        _sync(device)
        start = time.perf_counter()
        output, stats = model.diffusion_generate(prompt, max_new_tokens, eos_token_id=list(eos))
        _sync(device)
        totals["seconds"] += time.perf_counter() - start
        totals["tokens"] += stats.new_tokens
        totals["passes"] += stats.forward_passes
        totals["cycles"] += stats.cycles
        totals["accepted"] += sum(stats.accepted)
        if compare_ar:
            start = time.perf_counter()
            reference, reference_gaps = ar_generate(model, prompt, max_new_tokens, eos)
            _sync(device)
            ar_seconds += time.perf_counter() - start
            generated = output[0, len(prompt_ids) :].tolist()
            if generated == reference:
                matches += 1
            else:
                first = next(
                    (
                        i
                        for i, (a, b) in enumerate(zip(generated, reference, strict=False))
                        if a != b
                    ),
                    min(len(generated), len(reference)),
                )
                divergences.append(first)
                gaps.append(reference_gaps[min(first, len(reference_gaps) - 1)])
    model.train(was_training)
    cycles = max(totals["cycles"], 1)
    metrics = {
        "acceptance_length": totals["accepted"] / cycles,
        "tpf": totals["tokens"] / max(totals["passes"], 1),
        "tokens_per_cycle": max(totals["tokens"] - len(prompts), 0) / cycles,
        "tokens_per_second": totals["tokens"] / max(totals["seconds"], 1e-9),
        "mean_new_tokens": totals["tokens"] / max(len(prompts), 1),
    }
    if compare_ar:
        metrics["ar_match"] = matches / max(len(prompts), 1)
        metrics["speedup"] = ar_seconds / max(totals["seconds"], 1e-9)
        if divergences:  # where outputs first differ, and how close AR's top-2 logits were there
            metrics["ar_first_divergence"] = sum(divergences) / len(divergences)
            metrics["ar_divergence_logit_gap"] = sorted(gaps)[len(gaps) // 2]
    return metrics


def eos_token_ids(model, tokenizer) -> set[int]:
    """All end-of-turn ids (Qwen3 generation config: <|im_end|> and <|endoftext|>)."""
    ids = model.generation_config.eos_token_id or tokenizer.eos_token_id
    return set(ids if isinstance(ids, (list, tuple)) else [ids])


def _chat_prompt(tokenizer, user: str) -> list[int]:
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


def main(argv: list[str] | None = None) -> None:
    from orthrus.checkpoint import load_model
    from orthrus.tracking import Tracker

    cfg = parse_args(__doc__, argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_root = Path(cfg.storage.local_dir)
    run_dir = local_root / "runs" / cfg.run_name
    bucket = Bucket(cfg.storage.bucket)
    final_dir = run_dir / "final"
    if not (final_dir / "config.json").exists():
        if not bucket.download(f"runs/{cfg.run_name}/final", final_dir):
            raise FileNotFoundError(f"no exported model for run {cfg.run_name}")
    model_cfg = cfg.model
    model_cfg.base = str(final_dir)
    dtype = getattr(torch, cfg.eval.dtype) if device.type == "cuda" else torch.float32
    model, tokenizer, _ = load_model(model_cfg, device, dtype)
    eos = eos_token_ids(model, tokenizer)

    results, tracker = {}, Tracker(cfg, run_dir, "eval")
    suites = {
        "heldout": [
            p["prompt_ids"] for p in load_eval_prompts(bucket, cfg.data.dataset, local_root)
        ]
    }
    suites.update(
        {name: [_chat_prompt(tokenizer, q) for q in load()] for name, load in BENCHMARKS.items()}
    )
    for step, (name, prompts) in enumerate(suites.items()):
        prompts = prompts[: cfg.eval.prompts] if cfg.eval.prompts > 0 else prompts
        if not prompts:
            continue
        metrics = generation_metrics(model, prompts, cfg.eval.max_new_tokens, eos, compare_ar=True)
        results[name] = {"prompts": len(prompts), **metrics}
        log.info("%s: %s", name, json.dumps(results[name]))
        tracker.log({f"{name}/{k}": v for k, v in metrics.items()}, step=step)
    out = run_dir / f"eval-{cfg.eval.max_new_tokens}-{cfg.eval.dtype}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    bucket.upload(out, f"runs/{cfg.run_name}/{out.name}")
    tracker.close()


if __name__ == "__main__":
    main()
