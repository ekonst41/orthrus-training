"""vLLM probe: the pinned CUDA 12.9 vLLM wheel on the DataSphere driver, Qwen3-0.6B throughput.

Usage: python -m probe_vllm [report.json]
Each engine configuration runs in a fresh subprocess; the first one that works is reported.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

MODEL = "Qwen/Qwen3-0.6B"
MAX_TOKENS = 1024
# (engine kwargs, env overrides); the fallback avoids JIT-compiled FlashInfer and cudagraphs.
ATTEMPTS = {
    "default": ({}, {}),
    "conservative": ({"enforce_eager": True}, {"VLLM_USE_FLASHINFER_SAMPLER": "0"}),
}


def build_prompts(n: int) -> list[str]:
    topics = (
        "photosynthesis, binary search, inflation, the French Revolution, black holes, recursion, "
        "vaccines, supply and demand, neural networks, plate tectonics, hash tables, "
        "the immune system, compound interest, climate change, TCP/IP"
    ).split(", ")
    templates = [
        "Explain {topic} to a high school student.",
        "Write a short essay about {topic}.",
        "Give three common misconceptions about {topic} and correct them.",
        "Solve step by step: a store sells {a} items at ${b} each with a {c}% discount. "
        "What is the total price?",
        "Write a Python function related to {topic}, with docstring and tests.",
        "Prove that the sum of the first {a} odd numbers equals {a} squared.",
        "Implement {topic} in Python and explain its time complexity.",
        "Summarize the key ideas of {topic} in a bulleted list.",
    ]
    # Unique prompts only: duplicates would hit vLLM's prefix cache and inflate throughput.
    styles = ["", " Be concise.", " Use examples.", " Answer in detail.", " Use simple words."]
    prompts = []
    for i in range(n):
        template = templates[i % len(templates)]
        topic = topics[(i // len(templates)) % len(topics)]
        style = styles[(i // (len(templates) * len(topics))) % len(styles)]
        text = template.format(topic=topic, a=7 + i % 23, b=3 + i % 17, c=5 + i % 40)
        prompts.append(text + style)
    return prompts


def worker(name: str, out_path: str) -> None:
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    engine_kwargs, _ = ATTEMPTS[name]
    report = {"attempt": name, "vllm": vllm.__version__, "torch": torch.__version__}
    start = time.perf_counter()
    llm = LLM(
        MODEL,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        seed=0,
        **engine_kwargs,
    )
    report["engine_init_s"] = round(time.perf_counter() - start, 1)

    messages = [[{"role": "user", "content": prompt}] for prompt in build_prompts(512)]
    params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
    start = time.perf_counter()
    outputs = llm.chat(
        messages, params, chat_template_kwargs={"enable_thinking": False}, use_tqdm=False
    )
    seconds = time.perf_counter() - start
    lengths = [len(output.outputs[0].token_ids) for output in outputs]
    report.update(
        prompts=len(lengths),
        generated_tokens=sum(lengths),
        generate_s=round(seconds, 1),
        output_tokens_per_s=round(sum(lengths) / seconds),
        mean_response_tokens=round(sum(lengths) / len(lengths)),
        hit_max_tokens=sum(length >= MAX_TOKENS for length in lengths),
        sample=outputs[0].outputs[0].text[:300],
    )
    Path(out_path).write_text(json.dumps(report, indent=2))


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(sys.argv[2], sys.argv[3])
        return
    report_path = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/probe_vllm.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    pip = [sys.executable, "-m", "pip", "install", "--no-deps", "-q"]
    subprocess.check_call([*pip, "-r", "requirements/datagen-urls.txt"])
    report = {"vllm_install_s": round(time.perf_counter() - start, 1), "attempts": {}}

    for name, (_, env_overrides) in ATTEMPTS.items():
        out = report_path.with_name(f"vllm_{name}.json")
        command = [sys.executable, "-m", "probe_vllm", "--worker", name, str(out)]
        try:
            code = subprocess.call(command, env={**os.environ, **env_overrides}, timeout=1200)
        except subprocess.TimeoutExpired:
            code = "timeout"
        result = json.loads(out.read_text()) if code == 0 and out.exists() else {"exit": code}
        report["attempts"][name] = result
        print(name, json.dumps(result), flush=True)
        if code == 0:
            break
    report_path.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
