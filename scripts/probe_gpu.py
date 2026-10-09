"""GPU probe: torch/CUDA stack, bf16 GEMM, compiled FlexAttention (Orthrus mask), HF Hub speed.

Usage: python scripts/probe_gpu.py [report.json]
"""

import json
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, create_mask, flex_attention


def timed(fn, iters):
    fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters


def stack_info():
    import transformers
    import triton

    props = torch.cuda.get_device_properties(0)
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "triton": triton.__version__,
        "transformers": transformers.__version__,
        "gcc": shutil.which("gcc"),
        "gpu": props.name,
        "capability": f"{props.major}.{props.minor}",
        "vram_gib": round(props.total_memory / 2**30, 1),
    }


def gemm_and_bandwidth():
    n = 8192
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    x = torch.empty(2**28, device="cuda")  # 1 GiB
    return {
        "bf16_gemm_tflops": round(2 * n**3 / timed(lambda: a @ b, 20) / 1e12, 1),
        "hbm_copy_gbs": round(2 * x.numel() * 4 / timed(x.clone, 20) / 1e9),
    }


def flex_attention_check():
    # Qwen3-0.6B attention shapes; paper setup: L=2048, K=32, 256 anchor blocks.
    seq_len, block, n_blocks, heads, kv_heads, dim = 2048, 32, 256, 16, 8, 128
    q_len, kv_len = n_blocks * block, seq_len + n_blocks * block
    anchors = (torch.randperm(seq_len - block, device="cuda")[:n_blocks] + 1).sort().values[None]

    def mask_mod(b, h, q, k):
        blk = q // block
        ar_visible = (k < seq_len) & (k < anchors[b, blk])
        same_block = (k >= seq_len) & ((k - seq_len) // block == blk)
        return ar_visible | same_block

    start = time.perf_counter()
    block_mask = create_block_mask(mask_mod, 1, None, q_len, kv_len, device="cuda")
    mask_s = time.perf_counter() - start

    opts = {"device": "cuda", "dtype": torch.bfloat16, "requires_grad": True}
    q = torch.randn(1, heads, q_len, dim, **opts)
    k = torch.randn(1, kv_heads, kv_len, dim, **opts)
    v = torch.randn_like(k, requires_grad=True)
    flex = torch.compile(flex_attention, dynamic=False)

    def step():
        flex(q, k, v, block_mask=block_mask, enable_gqa=True).float().sum().backward()

    start = time.perf_counter()
    step()
    torch.cuda.synchronize()
    compile_s = time.perf_counter() - start
    step_ms = timed(step, 10) * 1e3

    with torch.no_grad():
        out = flex(q, k, v, block_mask=block_mask, enable_gqa=True)
        dense = create_mask(mask_mod, 1, None, q_len, kv_len, device="cuda")
        ref = F.scaled_dot_product_attention(q, k, v, attn_mask=dense, enable_gqa=True)
        max_err = (out.float() - ref.float()).abs().max().item()
    return {
        "block_mask_s": round(mask_s, 3),
        "flex_compile_s": round(compile_s, 1),
        "flex_fwd_bwd_ms": round(step_ms, 2),
        "flex_vs_sdpa_max_abs_err": round(max_err, 5),
    }


def hub_download():
    from huggingface_hub import snapshot_download

    start = time.perf_counter()
    path = snapshot_download("Qwen/Qwen3-0.6B")
    size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    seconds = time.perf_counter() - start
    return {
        "hf_qwen3_0.6b_gb": round(size / 1e9, 2),
        "hf_download_s": round(seconds, 1),
        "hf_mb_per_s": round(size / 1e6 / seconds, 1),
    }


def main():
    report_path = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/probe_gpu.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available(), "CUDA is not available"
    report = {}
    for name, check in [
        ("stack", stack_info),
        ("compute", gemm_and_bandwidth),
        ("flex_attention", flex_attention_check),
        ("hf_hub", hub_download),
    ]:
        try:
            report[name] = check()
        except Exception as error:  # keep probing; the report shows what failed
            report[name] = {"error": f"{type(error).__name__}: {error}"}
        print(name, json.dumps(report[name]), flush=True)
    with open(report_path, "w") as file:
        json.dump(report, file, indent=2)


if __name__ == "__main__":
    main()
