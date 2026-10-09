"""Chat with a trained Orthrus model and see how its parallel decoding performs.

    python -m orthrus.generate --config configs/qwen3-0.6b.yaml --prompt "Solve 2x + 3 = 11"
    python -m orthrus.generate --config configs/qwen3-0.6b.yaml [--model DIR] [--compare-ar]
        [--max-new-tokens 512] [eval.checkpoint=latest] [eval.dtype=float32]

Without --prompt, prompts are read from stdin (one per line; an empty line quits). The model is the
run's export (downloaded from the bucket if needed), an exported directory given with --model (for
example the official chiennv/Orthrus-Qwen3-1.7B), or a training checkpoint (eval.checkpoint). The
answer is streamed, followed by decoding statistics: accepted draft tokens per cycle, tokens per
forward pass (TPF, paper definition) and tokens per second; --compare-ar also decodes with the
plain AR view and reports its time and whether the two answers are identical.
"""

import argparse
import dataclasses
import sys
import time

import torch
from transformers import TextStreamer

from orthrus.config import parse_args
from orthrus.evaluate import ar_generate, chat_prompt, eos_token_ids, load_for_eval


def _timed(device, fn):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return result, time.perf_counter() - start


def main(argv: list[str] | None = None) -> None:
    from orthrus.checkpoint import load_model

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--prompt")
    parser.add_argument("--model", help="exported Orthrus directory or Hub id")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--compare-ar", action="store_true")
    known, rest = parser.parse_known_args(argv)
    cfg = parse_args(__doc__, rest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, cfg.eval.dtype) if device.type == "cuda" else torch.float32
    if known.model:
        model_cfg = dataclasses.replace(cfg.model, base=known.model, revision=None)
        model, tokenizer, _ = load_model(model_cfg, device, dtype)
    else:
        model, tokenizer, _ = load_for_eval(cfg, device, dtype)
    model.eval()
    eos = eos_token_ids(model, tokenizer)

    prompts = [known.prompt] if known.prompt else iter(lambda: input("\n> ").strip(), "")
    for text in prompts:
        prompt = torch.tensor([chat_prompt(tokenizer, text)], device=device)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        (output, stats), seconds = _timed(
            device,
            lambda p=prompt, s=streamer: model.diffusion_generate(
                p, known.max_new_tokens, eos_token_id=list(eos), streamer=s
            ),
        )
        cycles = max(stats.cycles, 1)
        print(
            f"\n[{stats.new_tokens} tokens in {seconds:.2f} s, {stats.new_tokens / seconds:.1f} "
            f"tok/s | {stats.cycles} cycles, {sum(stats.accepted) / cycles:.2f} accepted drafts "
            f"per cycle | TPF {stats.new_tokens / stats.forward_passes:.2f}]",
            flush=True,
        )
        if known.compare_ar:
            (reference, _), ar_seconds = _timed(
                device, lambda p=prompt: ar_generate(model, p, known.max_new_tokens, eos)
            )
            same = output[0, prompt.shape[1] :].tolist() == reference
            print(
                f"[AR: {len(reference)} tokens in {ar_seconds:.2f} s, speedup "
                f"{ar_seconds / seconds:.2f}x, {'identical' if same else 'different'} answer]",
                flush=True,
            )
        sys.stdout.flush()


if __name__ == "__main__":
    main()
