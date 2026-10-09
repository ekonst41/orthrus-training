"""Data-generation benchmark on one GPU: throughput and output agreement of vLLM settings.

    python -m orthrus.genbench --config configs/qwen3-0.6b.yaml [--prompts-per-domain 700]
        [--variants baseline,ngram_gpu_k4]

The prompts are the first N training prompts per domain of the real selection (what the full run
answers first), with the full run's limits. Each variant runs in its own process (clean GPU memory,
isolated crashes):
  baseline      datagen settings before the speed work (detokenized outputs)
  no_detok      token ids only (no detokenization)
  ngram_gpu_k4  n-gram speculative decoding with the GPU proposer, 4 draft tokens
  ngram_k4      the same with the CPU proposer; ngram_gpu_k8 with 8 draft tokens
  suffix        suffix decoding (Arctic Inference, built on demand)
  seqs256       no_detok with 256 instead of 512 concurrent sequences
Reported per variant: output tokens/s, response lengths, preemptions, speculative acceptance and
agreement with the baseline outputs (speculative greedy decoding gives the same tokens up to bf16
rounding of near-ties). One `BENCH {json}` line each; the list goes to bench/ in the bucket.
"""

import argparse
import dataclasses
import json
import logging
import os
import subprocess
import sys
import time
from itertools import zip_longest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from orthrus.config import parse_args
from orthrus.datagen import (
    build_engine,
    end_tokens,
    exit_now,
    responses,
    sample_prompts,
    sampling_params,
)
from orthrus.storage import Bucket

log = logging.getLogger("orthrus.genbench")
NGRAM = {"prompt_lookup_min": 2, "prompt_lookup_max": 5}
SUFFIX_BUILD = [  # build-only dependencies; --no-deps keeps vLLM's protobuf and grpcio
    ["-m", "pip", "install", "-q", "--no-deps", "cmake", "nanobind==2.9.2", "grpcio-tools==1.84.0"],
    ["-m", "pip", "install", "-q", "--no-deps", "--no-build-isolation", "arctic-inference==0.3.0"],
]


def speculative(method: str, tokens: int, **extra) -> dict:
    config = {"method": method, "num_speculative_tokens": tokens, **extra}
    return {"engine": {"speculative_config": config}}


VARIANTS = {  # run in this order: the most informative first
    "baseline": {"detokenize": True},
    "no_detok": {},
    "ngram_gpu_k4": speculative("ngram_gpu", 4, **NGRAM),
    "ngram_k4": speculative("ngram", 4, **NGRAM),
    "ngram_gpu_k8": speculative("ngram_gpu", 8, **NGRAM),
    "suffix": {**speculative("suffix", 24), "build": SUFFIX_BUILD},  # 24 = max tree depth
    "seqs256": {"engine": {"max_num_seqs": 256}},
}
METRICS = (
    "vllm:num_preemptions",
    "vllm:generation_tokens",
    "vllm:prompt_tokens",
    "vllm:spec_decode_num_drafts",
    "vllm:spec_decode_num_draft_tokens",
    "vllm:spec_decode_num_accepted_tokens",
    "vllm:spec_decode_num_accepted_tokens_per_pos",
)


def report(results: list, **result) -> None:
    results.append(result)
    print("BENCH " + json.dumps(result), flush=True)


def engine_metrics(llm) -> dict:
    """Totals of the vLLM counters above (summed over label sets)."""
    totals: dict = {}
    for metric in llm.get_metrics():
        if metric.name not in METRICS:
            continue
        if hasattr(metric, "values"):  # per-position vector
            old = totals.get(metric.name, [])
            totals[metric.name] = [a + b for a, b in zip_longest(old, metric.values, fillvalue=0)]
        elif hasattr(metric, "value"):
            totals[metric.name] = totals.get(metric.name, 0) + metric.value
    return totals


def length_stats(lengths: list[int], reasons: list[str]) -> dict:
    ordered = sorted(lengths)
    return {
        "mean": round(sum(lengths) / max(len(lengths), 1), 1),
        "p50": ordered[len(ordered) // 2] if ordered else 0,
        "p90": ordered[int(len(ordered) * 0.9)] if ordered else 0,
        "truncated": round(reasons.count("length") / max(len(reasons), 1), 4),
    }


def worker(cfg, name: str, prompts_path: Path, out_dir: Path) -> None:
    """Generate the benchmark prompts with one variant and save outputs and timings."""
    from vllm.inputs import TokensPrompt

    variant = VARIANTS[name]
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base, revision=cfg.model.revision)
    eos, end_of_turn = end_tokens(cfg.model.base, tokenizer, cfg.model.revision)
    table = pq.read_table(prompts_path)
    prompts = table.column("prompt_ids").to_pylist()
    domains = table.column("domain").to_pylist()

    start = time.perf_counter()
    llm = build_engine(cfg, disable_log_stats=False, **variant.get("engine", {}))
    init_seconds = time.perf_counter() - start
    params = sampling_params(cfg, eos, detokenize=variant.get("detokenize", False))
    start = time.perf_counter()
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=p) for p in prompts], params, use_tqdm=False
    )
    seconds = time.perf_counter() - start
    generated = sum(len(o.outputs[0].token_ids) for o in outputs)
    ids, reasons = responses(outputs, eos, end_of_turn)
    lengths = [len(o.outputs[0].token_ids) for o in outputs]
    by_domain = {
        domain: length_stats(
            [n for n, d in zip(lengths, domains, strict=True) if d == domain],
            [r for r, d in zip(reasons, domains, strict=True) if d == domain],
        )
        for domain in sorted(set(domains))
    }
    result = {
        "variant": name,
        "settings": variant.get("engine", {}),
        "prompts": len(prompts),
        "init_seconds": round(init_seconds, 1),
        "generate_seconds": round(seconds, 1),
        "output_tokens": generated,
        "output_tokens_per_second": round(generated / seconds, 1),
        "lengths": length_stats(lengths, reasons),
        "lengths_by_domain": by_domain,
        "engine": engine_metrics(llm),
    }
    spec = result["engine"]
    if spec.get("vllm:spec_decode_num_drafts"):
        accepted = spec.get("vllm:spec_decode_num_accepted_tokens", 0)
        result["mean_acceptance_length"] = round(
            1 + accepted / spec["vllm:spec_decode_num_drafts"], 3
        )
        result["draft_acceptance_rate"] = round(
            accepted / max(spec.get("vllm:spec_decode_num_draft_tokens", 0), 1), 3
        )
    pq.write_table(
        pa.table(
            {
                "response_ids": pa.array(ids, type=pa.list_(pa.int32())),
                "finish_reason": reasons,
            }
        ),
        out_dir / f"{name}.parquet",
    )
    (out_dir / f"{name}.json").write_text(json.dumps(result), encoding="utf-8")


def agreement(responses_a: list[list[int]], responses_b: list[list[int]]) -> dict:
    """How many responses are token-identical, and where the others first differ."""
    identical, divergences, common, total = 0, [], 0, 0
    for a, b in zip(responses_a, responses_b, strict=True):
        prefix = next(
            (i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b))
        )
        if a == b:
            identical += 1
        else:
            divergences.append(prefix)
        common += prefix
        total += len(b)
    return {
        "identical_fraction": round(identical / max(len(responses_b), 1), 4),
        "mean_first_divergence": round(sum(divergences) / max(len(divergences), 1), 1),
        "common_prefix_token_fraction": round(common / max(total, 1), 4),
    }


def run_variant(cfg_argv, name, prompts_path, out_dir, timeout) -> tuple[dict | None, str | None]:
    """Run one variant in a child process; (result, error)."""
    for command in VARIANTS[name].get("build", []):
        built = subprocess.run([sys.executable, *command], capture_output=True, text=True)
        if built.returncode:
            return None, f"build failed: {(built.stderr or built.stdout)[-600:]}"
    env = {**os.environ, "VLLM_PLUGINS": ""}  # no vLLM plugins (Arctic patches vLLM 0.26 code)
    command = [
        sys.executable,
        "-m",
        "orthrus.genbench",
        "--worker",
        name,
        "--prompts",
        str(prompts_path),
        "--out",
        str(out_dir),
        *cfg_argv,
    ]
    try:
        code = subprocess.run(command, env=env, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        return None, f"timeout after {timeout} s"
    path = out_dir / f"{name}.json"
    if code or not path.exists():
        return None, f"worker exit code {code}"
    return json.loads(path.read_text(encoding="utf-8")), None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker")
    parser.add_argument("--prompts")
    parser.add_argument("--out")
    parser.add_argument("--prompts-per-domain", type=int, default=700)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--timeout-minutes", type=float, default=12)
    known, cfg_argv = parser.parse_known_args(argv)
    cfg = parse_args(__doc__, cfg_argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    if known.worker:
        worker(cfg, known.worker, Path(known.prompts), Path(known.out))
        return

    out_dir = Path(cfg.storage.local_dir) / "genbench"
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base, revision=cfg.model.revision)
    d = dataclasses.replace(cfg.datagen, samples_per_domain=known.prompts_per_domain)
    train, _, revision = sample_prompts(d, tokenizer)  # same order as the full run's first shard
    prompts_path = out_dir / "prompts.parquet"
    pq.write_table(
        pa.table(
            {
                "domain": [p["domain"] for p in train],
                "prompt_ids": pa.array([p["prompt_ids"] for p in train], type=pa.list_(pa.int32())),
            }
        ),
        prompts_path,
    )
    results: list = []
    report(
        results,
        variant="setup",
        prompts=len(train),
        source_revision=revision,
        max_new_tokens=cfg.datagen.max_new_tokens,
        max_num_seqs=cfg.datagen.max_num_seqs,
    )
    full_run_prompts = cfg.datagen.samples_per_domain * len(cfg.datagen.domains)
    baseline_ids, baseline_speed = None, None
    for name in known.variants.split(","):
        result, error = run_variant(
            cfg_argv, name, prompts_path, out_dir, int(known.timeout_minutes * 60)
        )
        if error:
            report(results, variant=name, error=error)
            continue
        ids = pq.read_table(out_dir / f"{name}.parquet").column("response_ids").to_pylist()
        if baseline_ids is None:
            baseline_ids, baseline_speed = ids, result["output_tokens_per_second"]
        else:
            result["vs_first_variant"] = agreement(ids, baseline_ids)
            result["speedup"] = round(result["output_tokens_per_second"] / baseline_speed, 3)
        tokens = result["lengths"]["mean"] * full_run_prompts
        result["full_run_hours_at_this_speed"] = round(
            tokens / result["output_tokens_per_second"] / 3600, 2
        )
        report(results, **result)

    out = out_dir / f"genbench-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    Bucket(cfg.storage.bucket).upload(out, f"bench/{out.name}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        exit_now(1)
    exit_now()  # vLLM workers can keep the interpreter alive (see datagen.exit_now)
