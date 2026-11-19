"""One training loop for python (one GPU) and torchrun (several GPUs on one node)."""

import dataclasses
import hashlib
import json
import logging
import math
import signal
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import torch
import torch.distributed as dist
import yaml

from src.config import Config, parse_args
from src.dataset import data
from src.hub import Bucket
from src.test.benchmark import (
    balanced_prompts,
    eos_token_ids,
    generation_metrics,
    load_eval_prompts,
)
from src.train.checkpoint import CheckpointManager, export_model, load_checkpoint, load_model
from src.train.distributed import average_gradients, close, initialize, synchronize
from src.train.loss import draft_loss, sample_anchors

log = logging.getLogger(__name__)
REPORTED_OFFSETS = (1, 2, 4, 8, 16, 31)


def summed_stats(loss_type):
    return (
        f"{loss_type}_sum",
        f"weighted_{loss_type}_sum",
        "weight_sum",
        "tokens",
        "agree",
        f"{loss_type}_by_offset",
        "agree_by_offset",
        "tokens_by_offset",
    )


def lr_scale(step: int, warmup: int, total: int, min_ratio: float) -> float:
    """Linear warmup then cosine decay to min_ratio (official get_cosine_lr_scale)."""
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def anchor_rng(seed: int, *keys, device: torch.device) -> torch.Generator:
    """Generator keyed by (seed, epoch, micro-batch): anchors do not depend on resumes."""
    digest = hashlib.sha256(":".join(map(str, (seed, *keys))).encode()).digest()
    return torch.Generator(device=device).manual_seed(int.from_bytes(digest[:8], "little") >> 1)


def offset_metrics(prefix: str, sums: dict, loss_type: str) -> dict:
    """Per-position metrics and, for KL, the teacher-forced acceptance proxy."""
    tokens = sums["tokens_by_offset"].clamp_min(1).double()
    rate = (sums["agree_by_offset"].double() / tokens).cpu()
    losses = (sums[f"{loss_type}_by_offset"].double() / tokens).cpu()
    match = "agree" if loss_type == "kl" else "acc"
    metrics = {}
    for k in REPORTED_OFFSETS:
        if k <= len(rate):
            metrics[f"{prefix}/{match}_k{k}"] = rate[k - 1].item()
            metrics[f"{prefix}/{loss_type}_k{k}"] = losses[k - 1].item()
    if loss_type == "kl":
        metrics[f"{prefix}/acceptance_proxy"] = torch.cumprod(rate, 0).sum().item()
    return metrics


@torch.no_grad()
def evaluation_loss(model, ids, mask, cfg: Config, device: torch.device) -> dict:
    """Run the training objective on held-out rows with fixed anchors."""
    was_training = model.training
    model.eval()
    sums = defaultdict(lambda: torch.zeros((), device=device))
    micro = cfg.train.micro_batch_size
    loss_type = cfg.train.loss_type
    stat_keys = summed_stats(loss_type)
    for start in range(0, len(ids), micro):
        batch_ids = torch.from_numpy(ids[start : start + micro].astype("int64")).to(device)
        batch_mask = torch.from_numpy(mask[start : start + micro].astype(bool)).to(device)
        rng = anchor_rng(cfg.train.seed, "eval", start, device=device)
        anchors, valid = sample_anchors(
            batch_mask, model.config.block_size, cfg.train.num_anchor_blocks, rng
        )
        with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
            _, stats = draft_loss(
                model,
                batch_ids,
                batch_mask,
                anchors,
                valid,
                cfg.train.kl_chunk_size,
                cfg.train.compile,
                gamma=cfg.train.loss_gamma,
                loss_type=loss_type,
            )
        for key in stat_keys:
            sums[key] = sums[key] + stats[key]
    model.train(was_training)
    tokens = sums["tokens"].clamp_min(1)
    match = "top1_agreement" if loss_type == "kl" else "top1_accuracy"
    metrics = {
        f"eval/{loss_type}": (sums[f"{loss_type}_sum"] / tokens).item(),
        f"eval/{match}": (sums["agree"] / tokens).item(),
        **offset_metrics("eval", sums, loss_type),
    }
    if cfg.train.loss_gamma:
        norm = sums["weight_sum"].clamp_min(1e-6)
        metrics[f"eval/weighted_{loss_type}"] = (sums[f"weighted_{loss_type}_sum"] / norm).item()
    return metrics


def recipe(cfg, world_size):
    operational = {
        "resume",
        "log_every",
        "eval_every",
        "save_every",
        "keep_checkpoints",
        "compile",
        "activation_checkpointing",
    }
    result = {
        "model": dataclasses.asdict(cfg.model),
        "data": dataclasses.asdict(cfg.data),
        "train": {k: v for k, v in dataclasses.asdict(cfg.train).items() if k not in operational},
        "world_size": world_size,
        "trainer_version": 1,
    }
    # Match the saved recipe of pre-DFlash Orthrus/KL runs.
    if cfg.model.kind == "orthrus":
        result["model"].pop("kind")
        result["model"].pop("draft_layers")
    if cfg.train.loss_type == "kl":
        result["train"].pop("loss_type")
    return json.loads(json.dumps(result))


def check_teacher(cfg, manifest):
    if manifest.get("complete") is False:
        raise ValueError("data generation is incomplete; finish prepare.sh first")
    if manifest.get("model") and manifest["model"] != cfg.model.base:
        raise ValueError("dataset teacher differs from model.base")
    revision = manifest.get("model_revision")
    if revision and not Path(cfg.model.base).exists():
        if cfg.model.revision and cfg.model.revision != revision:
            raise ValueError("dataset teacher revision differs from model.revision")
        cfg.model.revision = revision


def train(cfg, ctx, stop):
    t, device = cfg.train, ctx.device
    loss_type = t.loss_type
    stat_keys = summed_stats(loss_type)
    match = "top1_agreement" if loss_type == "kl" else "top1_accuracy"
    unit = t.micro_batch_size * ctx.world_size
    if t.global_batch_size % unit:
        raise ValueError(f"global_batch_size must be divisible by {unit} (micro_batch_size * GPUs)")
    accum = t.global_batch_size // unit
    torch.manual_seed(t.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    bucket = Bucket(cfg.hf.bucket if ctx.primary else "", cfg.hf.upload)
    run = cfg.run_path
    remote = f"runs/{cfg.run_name}"

    def prepare():
        run.mkdir(parents=True, exist_ok=True)
        manifest = data.load_manifest(cfg, bucket)
        check_teacher(cfg, manifest)
        if not cfg.model.revision and not Path(cfg.model.base).exists():
            from huggingface_hub import HfApi

            cfg.model.revision = HfApi().model_info(cfg.model.base).sha
        raw = data.fetch_raw(cfg, bucket)
        files = sorted(raw.glob("*.parquet"))
        if manifest.get("shards") and len(files) != manifest["shards"]:
            raise ValueError("raw shard count differs from dataset manifest")
        for name, digest in manifest.get("files", {}).items():
            if data.file_sha256(raw / name) != digest:
                raise RuntimeError(f"dataset checksum mismatch: {name}")
        _, _, key = data.load_packed(raw, raw.parent / "packed", cfg.data.seq_len, cfg.data.seed)
        prompts = (
            balanced_prompts(
                load_eval_prompts(bucket, cfg.data.dataset, cfg.data_dir), cfg.eval.prompts
            )
            if cfg.eval.prompts
            else []
        )
        return str(raw.parent / "packed" / key), key, prompts, cfg.model.revision

    packed, data_key, prompts, cfg.model.revision = ctx.primary_call(prepare)
    import numpy as np

    ids, mask = ctx.each_call(
        lambda: (
            np.load(Path(packed) / "input_ids.npy", mmap_mode="r"),
            np.load(Path(packed) / "assistant_mask.npy", mmap_mode="r"),
        )
    )
    split = len(ids) - cfg.data.eval_rows
    if split < t.global_batch_size:
        raise ValueError("not enough packed training rows for one global batch")
    train_ids, train_mask = ids[:split], mask[:split]
    eval_ids, eval_mask = ids[split:], mask[split:]
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model, tokenizer, params = ctx.each_call(lambda: load_model(cfg.model, device, dtype))
    if t.activation_checkpointing:
        model.gradient_checkpointing_enable({"use_reentrant": False})
    model.train()
    if ctx.world_size > 1:
        for param in params:
            dist.broadcast(param.data, src=0)
    optimizer = torch.optim.AdamW(
        params, lr=t.learning_rate, betas=t.adam_betas, weight_decay=t.weight_decay
    )
    steps_per_epoch = len(train_ids) // t.global_batch_size
    total_steps = max(1, int(t.epochs * steps_per_epoch))
    if t.max_steps:
        total_steps = min(total_steps, t.max_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: lr_scale(s, int(total_steps * t.warmup_ratio), total_steps, t.min_lr_ratio),
    )
    manager = (
        CheckpointManager(run, bucket, f"{remote}/checkpoints", t.keep_checkpoints)
        if ctx.primary
        else None
    )
    latest = (
        ctx.primary_call(lambda: str(p) if (p := manager.latest()) else None) if t.resume else None
    )
    # No run may silently restart over existing checkpoints.
    if not t.resume and ctx.primary_call(
        lambda: bool(manager._local() or bucket.list(f"{remote}/checkpoints"))
    ):
        raise ValueError("existing run; choose a new run_name or enable train.resume")
    current_recipe = recipe(cfg, ctx.world_size)
    step = 0
    if latest:
        state = ctx.each_call(lambda: load_checkpoint(Path(latest), model, optimizer, scheduler))
        expected = {"data": data_key, "total_steps": total_steps, "recipe": current_recipe}
        changed = [key for key, value in expected.items() if state.get(key) != value]
        if changed:
            raise ValueError(f"cannot resume: changed {changed}; use a new run_name")
        step = state["step"]

    metrics_path = run / "train_metrics.jsonl"

    def start_run():
        if not metrics_path.exists():
            bucket.download_file(f"{remote}/train_metrics.jsonl", metrics_path)
        if metrics_path.exists():
            records = [json.loads(s) for s in metrics_path.read_text().splitlines() if s]
            metrics_path.write_text(
                "".join(json.dumps(r) + "\n" for r in records if r["step"] <= step)
            )
        (run / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
        info = {
            "config": cfg.to_dict(),
            "data_fingerprint": data_key,
            "world_size": ctx.world_size,
            "total_steps": total_steps,
            "resumed_at_step": step,
            "versions": {n: version(n) for n in ("torch", "transformers", "huggingface_hub")},
        }
        (run / "run.json").write_text(json.dumps(info, indent=2))
        if latest:
            from src.hub import upload_checkpoint

            upload_checkpoint(bucket, Path(latest), f"{remote}/checkpoints")
        log.info(
            "%d GPU/processes | global batch %d = %d x %d x %d | steps %d/%d",
            ctx.world_size,
            t.global_batch_size,
            ctx.world_size,
            t.micro_batch_size,
            accum,
            step,
            total_steps,
        )

    ctx.primary_call(start_run)

    def log_metrics(metrics):
        with metrics_path.open("a") as file:
            file.write(json.dumps({"step": step, **metrics}) + "\n")
        log.info("step %d/%d: %s", step, total_steps, json.dumps(metrics))

    def evaluate():
        result = evaluation_loss(model, eval_ids, eval_mask, cfg, device) if len(eval_ids) else {}
        if prompts:
            result.update(
                {
                    "generate/" + k: v
                    for k, v in generation_metrics(
                        model,
                        [p["prompt_ids"] for p in prompts],
                        cfg.eval.max_new_tokens,
                        eos_token_ids(model, tokenizer),
                        domains=[p["domain"] for p in prompts],
                    ).items()
                }
            )
        log_metrics(result)

    window = defaultdict(lambda: torch.zeros((), device=device))
    window_steps, seconds = 0, 0.0
    while step < total_steps:
        epoch, start_step = divmod(step, steps_per_epoch)
        batches = data.batches(
            train_ids,
            train_mask,
            t.micro_batch_size,
            t.seed,
            epoch,
            rank=ctx.rank,
            world_size=ctx.world_size,
            global_batch_size=t.global_batch_size,
            start_step=start_step,
        )
        for _ in range(start_step, steps_per_epoch):
            synchronize(device)
            started = perf_counter()
            for _ in range(accum):
                index, batch_ids, batch_mask = next(batches)
                batch_ids, batch_mask = batch_ids.to(device), batch_mask.to(device)
                anchors, valid = sample_anchors(
                    batch_mask,
                    model.config.block_size,
                    t.num_anchor_blocks,
                    anchor_rng(t.seed, epoch, index, device=device),
                )
                with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
                    loss, stats = draft_loss(
                        model,
                        batch_ids,
                        batch_mask,
                        anchors,
                        valid,
                        t.kl_chunk_size,
                        t.compile,
                        gamma=t.loss_gamma,
                        loss_type=loss_type,
                    )
                (loss / accum).backward()
                for key in stat_keys:
                    window[key] = window[key] + stats[key]
            average_gradients(params, ctx.world_size)
            grad_norm = torch.nn.utils.clip_grad_norm_(params, t.max_grad_norm)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(
                    f"non-finite gradient at step {step}; latest checkpoint retained"
                )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            synchronize(device)
            seconds += perf_counter() - started
            step += 1
            window_steps += 1
            window["grad_norm"] += grad_norm
            if step % t.log_every == 0 or step in (1, total_steps):
                elapsed = ctx.maximum(seconds)
                if ctx.world_size > 1:
                    for value in window.values():
                        dist.all_reduce(value)

                def report():
                    count = window["tokens"].clamp_min(1)
                    result = {
                        "train/loss": (window[f"{loss_type}_sum"] / count).item(),
                        f"train/{match}": (window["agree"] / count).item(),
                        "train/grad_norm": (
                            window["grad_norm"] / (window_steps * ctx.world_size)
                        ).item(),
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/epoch": step / steps_per_epoch,
                        "perf/seconds_per_step": elapsed / window_steps,
                        "perf/eta_hours": (total_steps - step) * elapsed / window_steps / 3600,
                        **offset_metrics("train", window, loss_type),
                    }
                    if t.loss_gamma:
                        result["train/weighted_loss"] = (
                            window[f"weighted_{loss_type}_sum"] / window["weight_sum"].clamp_min(1e-6)
                        ).item()
                    log_metrics(result)

                ctx.primary_call(report)
                window.clear()
                window_steps, seconds = 0, 0.0
            reason = ctx.primary_call(lambda: "STOP file" if (run / "STOP").exists() else "")
            reason = ctx.stop_reason(reason or ("signal" if stop else ""))
            if t.eval_every and step % t.eval_every == 0 and step < total_steps and not reason:
                ctx.primary_call(evaluate)
            if step % t.save_every == 0 or step == total_steps or reason:

                def save():
                    manager.save(
                        step,
                        model,
                        optimizer,
                        scheduler,
                        {
                            "step": step,
                            "data": data_key,
                            "total_steps": total_steps,
                            "recipe": current_recipe,
                        },
                    )
                    for name in ("train_metrics.jsonl", "config.yaml", "run.json"):
                        bucket.upload(run / name, f"{remote}/{name}")

                ctx.primary_call(save)
            if reason:
                ctx.primary_call(lambda: (run / "STOP").unlink(missing_ok=True))
                return
            if step >= total_steps:
                break
    ctx.primary_call(evaluate)

    def finish():
        export_model(model, tokenizer, run / "final", cfg.model)
        bucket.upload(run / "final", f"{remote}/final")
        bucket.upload(metrics_path, f"{remote}/train_metrics.jsonl")

    ctx.primary_call(finish)


def main(argv=None):
    cfg = parse_args(__doc__, argv)
    ctx = initialize()
    stop = []
    previous = {
        sig: signal.signal(sig, lambda signum, frame: stop.append(signum))
        for sig in (signal.SIGINT, signal.SIGTERM)
    }
    logging.basicConfig(
        level=logging.INFO if ctx.primary else logging.WARNING, format="%(asctime)s %(message)s"
    )
    try:
        train(cfg, ctx, stop)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        close()


if __name__ == "__main__":
    main()
