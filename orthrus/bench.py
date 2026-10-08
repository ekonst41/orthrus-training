"""Training-step benchmark on one GPU: speed, memory and agreement of training-path variants.

    python -m orthrus.bench --config configs/smoke.yaml [--suite speedups|variants]

Forward + backward over real packed rows (no optimizer step), the same micro-batches for every
variant. Suite `variants` (first survey):
  reference  module path of modeling_orthrus.py + per-position KL (the code before the speed work)
  eager      functional views + shared-target KL, not compiled
  compiled   compiled decoder layers and fused KL
  then KL chunk sizes, FlexAttention kernel settings (on the real mask), the cross-entropy
  objective of the official code and a doubled micro-batch.
Suite `speedups` (default): skipping dK/dV of the AR keys, static-vocabulary KL kernels, doubled
micro-batch, and the gradient error of the old and new bf16 paths against fp32.
Every result is printed as a `BENCH {json}` line and uploaded to bench/ in the bucket.
"""

import argparse
import copy
import gc
import json
import logging
import time
from pathlib import Path

import torch
from torch.nn.attention.flex_attention import flex_attention

from orthrus import data
from orthrus.checkpoint import load_model
from orthrus.config import parse_args
from orthrus.modeling_orthrus import build_block_mask
from orthrus.objective import (
    FusedLinearForwardKL,
    _logits,
    model_states,
    orthrus_loss,
    reference_states,
    sample_anchors,
    supervision_mask,
)
from orthrus.storage import Bucket
from orthrus.train import anchor_rng

log = logging.getLogger("orthrus.bench")
RESULTS: list[dict] = []


def report(**result) -> None:
    RESULTS.append(result)
    print("BENCH " + json.dumps(result), flush=True)


def free() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def reference_loss(model, ids, mask, anchors, valid, chunk):
    """The objective as computed before the speed work: one teacher row per student row, eager."""
    k = model.config.block_size
    student, teacher = reference_states(model, ids, anchors)
    student = student.view(ids.shape[0], -1, k, student.shape[-1])[:, :, :-1]
    targets = anchors[:, :, None] + torch.arange(k - 1, device=ids.device)
    teacher = teacher.gather(1, targets.flatten(1)[..., None].expand(-1, -1, teacher.shape[-1]))
    keep = supervision_mask(mask, anchors, valid, k)
    rows = int(keep.sum())
    total, _ = FusedLinearForwardKL.apply(
        student[keep],
        teacher.view_as(student)[keep].to(student.dtype),
        torch.arange(rows, device=ids.device),
        model.lm_head.weight,
        chunk,
        False,
    )
    return total / max(rows, 1)


def ce_chunk(logits, labels, grad_dtype):
    lse = logits.logsumexp(-1)
    loss = (lse - logits.gather(1, labels[:, None]).squeeze(1)).sum()
    grad = (logits - lse[:, None]).exp()
    minus_one = torch.full((labels.shape[0], 1), -1.0, device=logits.device, dtype=grad.dtype)
    return loss, grad.scatter_add(1, labels[:, None], minus_one).to(grad_dtype)


_compiled_ce_chunk = torch.compile(ce_chunk, dynamic=True)


class FusedLinearCE(torch.autograd.Function):
    """Cross-entropy on the data tokens (official src/train/loss.py objective), chunked."""

    @staticmethod
    def forward(ctx, hidden, labels, weight, chunk_size):
        total = torch.zeros((), dtype=torch.float32, device=hidden.device)
        grad = torch.empty_like(hidden)
        for start in range(0, hidden.shape[0], chunk_size):
            rows = slice(start, start + chunk_size)
            chunk = _compiled_ce_chunk if hidden.is_cuda else ce_chunk
            loss, diff = chunk(_logits(hidden[rows], weight), labels[rows], weight.dtype)
            total += loss
            grad[rows] = (diff @ weight).to(grad.dtype)
        ctx.save_for_backward(grad)
        return total

    @staticmethod
    def backward(ctx, grad_total):
        (grad,) = ctx.saved_tensors
        return grad * grad_total.to(grad.dtype), None, None, None


def ce_loss(model, ids, mask, anchors, valid, chunk):
    k = model.config.block_size
    student, _ = model_states(model, ids, anchors, compiled=True)
    student = student.view(ids.shape[0], -1, k, student.shape[-1])[:, :, :-1]
    keep = supervision_mask(mask, anchors, valid, k)
    labels = ids.gather(1, (anchors[:, :, None] + torch.arange(1, k, device=ids.device)).flatten(1))
    total = FusedLinearCE.apply(
        student[keep], labels.view_as(keep)[keep], model.lm_head.weight, chunk
    )
    return total / keep.sum().clamp_min(1)


def make_batches(cfg, ids, mask, micro: int, count: int, device):
    k, slots = cfg.model.block_size, cfg.train.num_anchor_blocks
    batches = []
    for index in range(count):
        rows = slice(index * micro, (index + 1) * micro)
        batch_ids = torch.from_numpy(ids[rows].astype("int64")).to(device)
        batch_mask = torch.from_numpy(mask[rows].astype(bool)).to(device)
        rng = anchor_rng(cfg.train.seed, "bench", index, device=device)
        anchors, valid = sample_anchors(batch_mask, k, slots, rng)
        batches.append((batch_ids, batch_mask, anchors, valid))
    return batches


def run_variant(name, model, trainable, batches, loss_fn, warmup=3, steps=8, reference=None):
    """Time forward + backward per micro-batch; compare the gradient of batch 0 with reference."""
    free()
    try:
        model.zero_grad(set_to_none=True)
        start = time.perf_counter()
        for i in range(warmup):
            with torch.autocast("cuda", torch.bfloat16):
                loss = loss_fn(*batches[i % len(batches)])
            loss.backward()
        torch.cuda.synchronize()
        warmup_seconds = time.perf_counter() - start
        model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for i in range(steps):
            with torch.autocast("cuda", torch.bfloat16):
                loss = loss_fn(*batches[i % len(batches)])
            loss.backward()
        torch.cuda.synchronize()
        seconds = (time.perf_counter() - start) / steps
        peak = torch.cuda.max_memory_allocated() / 2**30

        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", torch.bfloat16):
            loss = loss_fn(*batches[0])
        loss.backward()
        grad = torch.cat([p.grad.float().flatten() for p in trainable])
        result = {
            "variant": name,
            "seconds_per_micro_batch": round(seconds, 4),
            "warmup_seconds": round(warmup_seconds, 1),
            "peak_memory_gb": round(peak, 2),
            "loss": round(loss.item(), 5),
        }
        if reference is not None:
            ref_loss, ref_grad = reference
            result["loss_rel_diff"] = abs(loss.item() - ref_loss) / abs(ref_loss)
            result["grad_rel_diff"] = ((grad - ref_grad).norm() / ref_grad.norm()).item()
            result["grad_cosine"] = torch.nn.functional.cosine_similarity(
                grad, ref_grad, dim=0
            ).item()
        report(**result)
        model.zero_grad(set_to_none=True)
        return result, (loss.item(), grad)
    except Exception as error:  # e.g. out of memory: report and continue with the next variant
        model.zero_grad(set_to_none=True)
        report(variant=name, error=f"{type(error).__name__}: {str(error)[:300]}")
        return None, None


def kernel_breakdown(name, model, batches, loss_fn, steps=2) -> None:
    """CUDA time per kernel category for `steps` micro-batches (profiler), plus the top kernels."""
    free()
    try:
        from torch.profiler import ProfilerActivity, profile

        with torch.autocast("cuda", torch.bfloat16):  # warm (and compile) outside the profile
            loss_fn(*batches[0]).backward()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for i in range(steps):
                with torch.autocast("cuda", torch.bfloat16):
                    loss = loss_fn(*batches[i % len(batches)])
                loss.backward()
            torch.cuda.synchronize()
        model.zero_grad(set_to_none=True)
        kernels = {}
        for event in prof.events():
            if event.device_type == torch.autograd.DeviceType.CUDA:
                ms = event.time_range.elapsed_us() / 1000 / steps
                kernels[event.name] = kernels.get(event.name, 0.0) + ms
        categories = {}
        for kernel, ms in kernels.items():
            categories[category(kernel)] = categories.get(category(kernel), 0.0) + ms
        top = sorted(kernels.items(), key=lambda item: -item[1])[:15]
        report(
            variant=f"profile/{name}",
            ms_per_micro_batch={k: round(v, 1) for k, v in sorted(categories.items())},
            total_ms=round(sum(categories.values()), 1),
            top_kernels=[[k[:90], round(v, 1)] for k, v in top],
        )
    except Exception as error:
        model.zero_grad(set_to_none=True)
        report(variant=f"profile/{name}", error=f"{type(error).__name__}: {str(error)[:300]}")


def category(kernel: str) -> str:
    name = kernel.lower()
    if "flex" in name or "triton_tem" in name:
        return "flex_attention"
    if "flash" in name or "fmha" in name or "attention" in name:
        return "sdpa_attention"
    if any(s in name for s in ("gemm", "cutlass", "xmma", "cublas", "sm80_", "ampere_")):
        return "gemm"
    if name.startswith("triton_"):
        return "fused_elementwise"
    if "memcpy" in name or "memset" in name:
        return "memcpy"
    return "eager_elementwise"


def flex_settings(anchors, seq_len, block_size, heads, kv_heads, head_dim, device) -> None:
    """fwd + bwd time of the diffusion attention on the real mask for kernel settings."""
    batch, slots = anchors.shape
    q_len = slots * block_size
    gen = torch.Generator(device=device).manual_seed(0)
    shape_q, shape_kv = (
        (batch, heads, q_len, head_dim),
        (batch, kv_heads, seq_len + q_len, head_dim),
    )
    q, k, v = (
        torch.randn(s, generator=gen, device=device, dtype=torch.bfloat16).requires_grad_()
        for s in (shape_q, shape_kv, shape_kv)
    )
    grad_out = torch.randn(shape_q, generator=gen, device=device, dtype=torch.bfloat16)
    mask = build_block_mask(anchors, seq_len, block_size)
    wide = {"fwd_num_warps": 8, "fwd_num_stages": 2}
    settings = {
        "default": ({}, None),
        "rows_safe": ({"ROWS_GUARANTEED_SAFE": True}, None),
        "fwd_128x128_w8_s2": ({"fwd_BLOCK_M": 128, "fwd_BLOCK_N": 128, **wide}, None),
        "fwd_64x64_w4_s3": (
            {"fwd_BLOCK_M": 64, "fwd_BLOCK_N": 64, "fwd_num_warps": 4, "fwd_num_stages": 3},
            None,
        ),
        "bwd_32x64_w4_s3": (
            {
                "bwd_BLOCK_M1": 32,
                "bwd_BLOCK_N1": 64,
                "bwd_BLOCK_M2": 64,
                "bwd_BLOCK_N2": 32,
                "bwd_num_warps": 4,
                "bwd_num_stages": 3,
            },
            None,
        ),  # fmt: skip
        "bwd_64x64_w8_s2": ({"bwd_num_warps": 8}, None),
        "bwd_64x64_w4_s3": ({"bwd_num_stages": 3}, None),
        "max_autotune": ({}, "max-autotune-no-cudagraphs"),
    }
    for name, (options, mode) in settings.items():
        free()
        try:

            def attend(q, k, v, mask, options=options):
                return flex_attention(
                    q, k, v, block_mask=mask, enable_gqa=True, kernel_options=options or None
                )

            fn = torch.compile(attend, dynamic=False, mode=mode)
            start = time.perf_counter()
            for _ in range(2):
                fn(q, k, v, mask).backward(grad_out)
            torch.cuda.synchronize()
            compile_seconds = time.perf_counter() - start
            times = []
            for _ in range(10):
                fwd_start, fwd_end, bwd_end = (torch.cuda.Event(enable_timing=True) for _ in "abc")
                fwd_start.record()
                out = fn(q, k, v, mask)
                fwd_end.record()
                out.backward(grad_out)
                bwd_end.record()
                torch.cuda.synchronize()
                times.append((fwd_start.elapsed_time(fwd_end), fwd_end.elapsed_time(bwd_end)))
            fwd = sorted(t[0] for t in times)[len(times) // 2]
            bwd = sorted(t[1] for t in times)[len(times) // 2]
            report(
                variant=f"flex/{name}",
                options=options,
                fwd_ms=round(fwd, 3),
                bwd_ms=round(bwd, 3),
                total_ms=round(fwd + bwd, 3),
                compile_seconds=round(compile_seconds, 1),
            )
        except Exception as error:
            report(variant=f"flex/{name}", error=f"{type(error).__name__}: {str(error)[:300]}")
        q.grad = k.grad = v.grad = None


def suite_variants(cfg, model, trainable, batches, big, default, reference_fn) -> None:
    """First survey: old vs new path, KL chunk sizes, FlexAttention settings, CE, micro-batch."""
    chunk = cfg.train.kl_chunk_size
    _, reference = run_variant("reference", model, trainable, batches, reference_fn, warmup=2)
    kernel_breakdown("reference", model, batches, reference_fn)
    run_variant("compiled", model, trainable, batches, default, reference=reference)
    kernel_breakdown("compiled", model, batches, default)
    run_variant(
        "eager",
        model,
        trainable,
        batches,
        lambda *b: default(*b, compiled=False),
        warmup=2,
        reference=reference,
    )
    for size in (4096, 8192):
        run_variant(
            f"compiled/kl_chunk_{size}",
            model,
            trainable,
            batches,
            lambda *b, size=size: default(*b, chunk=size),
            warmup=2,
            reference=reference,
        )
    config, attn = model.config, model.model.layers[0].self_attn
    flex_settings(
        batches[0][2],
        cfg.data.seq_len,
        config.block_size,
        config.num_attention_heads,
        config.num_key_value_heads,
        attn.head_dim,
        batches[0][0].device,
    )
    run_variant(
        "ce_objective",
        model,
        trainable,
        batches,
        lambda *b: ce_loss(model, *b, chunk),
        warmup=2,
    )
    if big:
        run_variant("compiled/micro_x2", model, trainable, big, default, warmup=2)


def suite_speedups(cfg, model, trainable, batches, big, default, reference_fn) -> None:
    """Second round: skipping dK/dV of the AR keys, static-vocabulary KL kernels, micro-batch,
    and the gradient error of the old and new bf16 paths against an fp32 run."""
    from orthrus import objective

    static_kl = objective._compiled_kl_chunk
    dynamic_kl = torch.compile(objective.kl_chunk, dynamic=True)  # the first round's setting

    def setting(skip: bool, kl):
        def loss(*batch):
            objective.SKIP_AR_KEY_GRADS, objective._compiled_kl_chunk = skip, kl
            return default(*batch)

        return loss

    _, previous = run_variant(
        "compiled/previous", model, trainable, batches, setting(False, dynamic_kl)
    )
    run_variant(
        "compiled/skip_ar_dkdv",
        model,
        trainable,
        batches,
        setting(True, dynamic_kl),
        reference=previous,
    )
    run_variant(
        "compiled/skip_ar_dkdv+static_kl",
        model,
        trainable,
        batches,
        setting(True, static_kl),
        reference=previous,
    )
    kernel_breakdown("new_default", model, batches, setting(True, static_kl))
    if big:
        run_variant("new_default/micro_x2", model, trainable, big, default, warmup=2)
    accuracy_vs_fp32(cfg, model, trainable, batches[0], default, reference_fn)


def accuracy_vs_fp32(cfg, model, trainable, batch, default, reference_fn) -> None:
    """Loss and gradient error of the bf16 paths against the module path in fp32, on one row."""
    free()
    try:
        row = tuple(t[:1] for t in batch)
        golden = copy.deepcopy(model).float()
        loss = reference_loss(golden, *row, cfg.train.kl_chunk_size)  # fp32, no autocast
        loss.backward()
        golden_loss = loss.item()
        golden_grad = torch.cat([p.grad.flatten() for p in golden.parameters() if p.requires_grad])
        del golden, loss
        free()
        result = {"variant": "accuracy_vs_fp32", "fp32_loss": golden_loss}
        for name, loss_fn in (("reference_bf16", reference_fn), ("new_default_bf16", default)):
            model.zero_grad(set_to_none=True)
            with torch.autocast("cuda", torch.bfloat16):
                loss = loss_fn(*row)
            loss.backward()
            grad = torch.cat([p.grad.float().flatten() for p in trainable])
            result[name] = {
                "loss_rel_err": abs(loss.item() - golden_loss) / abs(golden_loss),
                "grad_rel_err": ((grad - golden_grad).norm() / golden_grad.norm()).item(),
                "grad_cosine": torch.nn.functional.cosine_similarity(
                    grad, golden_grad, dim=0
                ).item(),
            }
        model.zero_grad(set_to_none=True)
        report(**result)
    except Exception as error:
        model.zero_grad(set_to_none=True)
        report(variant="accuracy_vs_fp32", error=f"{type(error).__name__}: {str(error)[:300]}")


SUITES = {"variants": suite_variants, "speedups": suite_speedups}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--suite", choices=sorted(SUITES), default="speedups")
    known, rest = parser.parse_known_args(argv)
    cfg = parse_args(__doc__, rest)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch._dynamo.config.recompile_limit = 64  # variants share the layer functions' code
    bucket = Bucket(cfg.storage.bucket)
    local_root = Path(cfg.storage.local_dir)
    raw_dir = data.fetch_raw(bucket, cfg.data.dataset, local_root)
    ids, mask, _ = data.load_packed(
        raw_dir, raw_dir.parent / "packed", cfg.data.seq_len, cfg.data.seed
    )
    model, _, trainable = load_model(cfg.model, device, torch.bfloat16)
    model.train()
    micro, chunk = cfg.train.micro_batch_size, cfg.train.kl_chunk_size
    batches = make_batches(cfg, ids, mask, micro, 6, device)
    big = make_batches(cfg, ids, mask, 2 * micro, 3, device) if len(ids) >= 6 * micro else []
    report(
        variant="setup",
        suite=known.suite,
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        micro_batch_size=micro,
        seq_len=cfg.data.seq_len,
        anchor_blocks=cfg.train.num_anchor_blocks,
        block_size=cfg.model.block_size,
    )

    def default(ids, mask, anchors, valid, chunk=chunk, compiled=True, options=None):
        return orthrus_loss(model, ids, mask, anchors, valid, chunk, compiled, options)[0]

    def reference_fn(*batch):
        return reference_loss(model, *batch, chunk)

    SUITES[known.suite](cfg, model, trainable, batches, big, default, reference_fn)
    out = local_root / "bench" / f"bench-{known.suite}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(RESULTS, indent=2), encoding="utf-8")
    bucket.upload(out, f"bench/{out.name}")


if __name__ == "__main__":
    main()
