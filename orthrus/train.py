"""Train the Orthrus diffusion view on one GPU (paper recipe); resumable and budget-aware.

    python -m orthrus.train --config configs/qwen3-0.6b.yaml [section.key=value ...]

The run continues from the newest checkpoint of the same run_name (local or in the bucket), with
exactly the same data order and anchors as an uninterrupted run. It stops cleanly on SIGTERM
(DataSphere cancel) or when train.max_hours is about to run out, saving a checkpoint first.
"""

import hashlib
import logging
import math
import shutil
import signal
import time
from collections import defaultdict
from pathlib import Path

import torch

from orthrus import data
from orthrus.checkpoint import CheckpointManager, export_model, load_checkpoint, load_model
from orthrus.config import Config, parse_args
from orthrus.evaluate import eos_token_ids, generation_metrics, load_eval_prompts
from orthrus.objective import orthrus_loss, sample_anchors
from orthrus.storage import Bucket
from orthrus.tracking import Tracker, write_manifest

log = logging.getLogger("orthrus.train")
REPORTED_OFFSETS = (1, 2, 4, 8, 16, 31)  # block offsets k (token anchor+k) shown in the dashboard


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


def offset_metrics(prefix: str, agree: torch.Tensor, tokens: torch.Tensor) -> dict:
    """Top-1 agreement with the teacher per block offset, and the implied acceptance length if
    drafts were accepted while consecutive offsets agree (a teacher-forced, optimistic proxy)."""
    rate = (agree.double() / tokens.clamp_min(1).double()).cpu()
    metrics = {
        f"{prefix}/agree_k{k}": rate[k - 1].item() for k in REPORTED_OFFSETS if k <= len(rate)
    }
    metrics[f"{prefix}/acceptance_proxy"] = torch.cumprod(rate, 0).sum().item()
    return metrics


@torch.no_grad()
def evaluation_loss(model, ids, mask, cfg: Config, device: torch.device) -> dict:
    """Held-out KL and teacher agreement with fixed anchors (same objective as training)."""
    was_training = model.training
    model.eval()
    sums = defaultdict(lambda: torch.zeros((), device=device))
    micro = cfg.train.micro_batch_size
    for start in range(0, len(ids) - micro + 1, micro):
        batch_ids = torch.from_numpy(ids[start : start + micro].astype("int64")).to(device)
        batch_mask = torch.from_numpy(mask[start : start + micro].astype(bool)).to(device)
        rng = anchor_rng(cfg.train.seed, "eval", start, device=device)
        anchors, valid = sample_anchors(
            batch_mask, model.config.block_size, cfg.train.num_anchor_blocks, rng
        )
        with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
            _, stats = orthrus_loss(
                model,
                batch_ids,
                batch_mask,
                anchors,
                valid,
                cfg.train.kl_chunk_size,
                cfg.train.compile,
                cfg.train.flex_kernel_options,
            )
        for key in ("kl_sum", "tokens", "agree", "agree_by_offset", "tokens_by_offset"):
            sums[key] = sums[key] + stats[key]
    model.train(was_training)
    tokens = sums["tokens"].clamp_min(1)
    return {
        "eval/kl": (sums["kl_sum"] / tokens).item(),
        "eval/top1_agreement": (sums["agree"] / tokens).item(),
        **offset_metrics("eval", sums["agree_by_offset"], sums["tokens_by_offset"]),
    }


class StopFlag:
    """Set by SIGTERM/SIGINT (DataSphere cancel with graceful-shutdown); checked every step."""

    def __init__(self):
        self.reason = ""
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._handle)

    def _handle(self, signum, _frame):
        self.reason = signal.Signals(signum).name
        log.warning("received %s: stopping after the current step", self.reason)


def main(argv: list[str] | None = None) -> None:
    cfg = parse_args(__doc__, argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    started = time.time()
    torch.manual_seed(cfg.train.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    local_root = Path(cfg.storage.local_dir)
    run_dir = local_root / "runs" / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    remote = f"runs/{cfg.run_name}"
    bucket = Bucket(cfg.storage.bucket)

    # Data: packed rows (the last eval_rows are held out) and held-out generation prompts.
    raw_dir = data.fetch_raw(bucket, cfg.data.dataset, local_root)
    ids, mask, data_key = data.load_packed(
        raw_dir, raw_dir.parent / "packed", cfg.data.seq_len, cfg.data.seed
    )
    split = len(ids) - cfg.data.eval_rows
    if split < cfg.train.global_batch_size:
        raise ValueError(f"{len(ids)} packed rows leave {split} for training: fewer than one batch")
    train_ids, train_mask, eval_ids, eval_mask = (
        ids[:split],
        mask[:split],
        ids[split:],
        mask[split:],
    )
    eval_prompts = [
        p["prompt_ids"] for p in load_eval_prompts(bucket, cfg.data.dataset, local_root)
    ]
    eval_prompts = eval_prompts[: cfg.eval.prompts]

    # Model and optimizer (only the diffusion twins are trainable).
    model, tokenizer, trainable = load_model(cfg.model, device, dtype)
    if cfg.train.activation_checkpointing:
        model.gradient_checkpointing_enable({"use_reentrant": False})
    model.train()
    t = cfg.train
    optimizer = torch.optim.AdamW(
        trainable, lr=t.learning_rate, betas=t.adam_betas, weight_decay=t.weight_decay
    )
    micro, accum = t.micro_batch_size, t.global_batch_size // t.micro_batch_size
    steps_per_epoch = len(train_ids) // t.global_batch_size
    total_steps = max(1, int(t.epochs * steps_per_epoch))
    if t.max_steps:
        total_steps = min(total_steps, t.max_steps)
    warmup = int(total_steps * t.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_scale(s, warmup, total_steps, t.min_lr_ratio)
    )

    checkpoints = CheckpointManager(run_dir, bucket, f"{remote}/checkpoints", t.keep_checkpoints)
    step = 0
    latest = checkpoints.latest()
    if latest is not None:
        state = load_checkpoint(latest, model, optimizer, scheduler)
        expected = {
            "data": data_key,
            "global_batch_size": t.global_batch_size,
            "total_steps": total_steps,
        }
        changed = {k: (state.get(k), v) for k, v in expected.items() if state.get(k) != v}
        if changed:
            raise RuntimeError(f"cannot resume {latest}: settings changed {changed}")
        step = state["step"]
        log.info("resumed from %s at step %d", latest, step)

    trainable_count = sum(p.numel() for p in trainable)
    write_manifest(
        run_dir / "run.json",
        cfg,
        data=data_key,
        train_rows=len(train_ids),
        eval_rows=len(eval_ids),
        total_steps=total_steps,
        resumed_at_step=step,
        trainable_parameters=trainable_count,
    )
    tracker = Tracker(cfg, run_dir, "train")
    log.info(
        "rows %d (+%d eval) | %d steps of %d sequences (%d x %d) | %.1fM trainable params",
        len(train_ids),
        len(eval_ids),
        total_steps,
        t.global_batch_size,
        micro,
        accum,
        trainable_count / 1e6,
    )

    def evaluate(final: bool = False) -> None:
        metrics = evaluation_loss(model, eval_ids, eval_mask, cfg, device) if len(eval_ids) else {}
        if eval_prompts:
            gen = generation_metrics(
                model,
                eval_prompts,
                cfg.eval.max_new_tokens,
                eos_token_ids(model, tokenizer),
                compare_ar=cfg.eval.check_ar_parity or final,
            )
            metrics.update({f"generate/{k}": v for k, v in gen.items()})
        tracker.log(metrics, step)
        log.info("eval step %d: %s", step, {k: round(v, 4) for k, v in metrics.items()})

    def publish_logs() -> None:  # snapshot first: the live JSONL keeps growing while uploading
        for name in ("train_metrics.jsonl", "run.json"):
            if (run_dir / name).exists():
                snapshot = run_dir / f".{name}.snapshot"
                shutil.copy(run_dir / name, snapshot)
                bucket.submit(lambda s=snapshot, n=name: bucket.upload(s, f"{remote}/{n}"))

    stop = StopFlag()
    window = defaultdict(lambda: torch.zeros((), device=device))
    window_start, window_steps, step_seconds = time.perf_counter(), 0, 0.0
    last_saved = step
    while step < total_steps and not stop.reason:
        epoch, group = divmod(step, steps_per_epoch)
        batches = data.batches(train_ids, train_mask, micro, t.seed, epoch, start=group * accum)
        for _ in range(group, steps_per_epoch):
            step_start = time.perf_counter()
            for _ in range(accum):
                index, batch_ids, batch_mask = next(batches)
                batch_ids = batch_ids.to(device, non_blocking=True)
                batch_mask = batch_mask.to(device, non_blocking=True)
                rng = anchor_rng(t.seed, epoch, index, device=device)
                anchors, valid = sample_anchors(
                    batch_mask, model.config.block_size, t.num_anchor_blocks, rng
                )
                with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
                    loss, stats = orthrus_loss(
                        model,
                        batch_ids,
                        batch_mask,
                        anchors,
                        valid,
                        t.kl_chunk_size,
                        t.compile,
                        t.flex_kernel_options,
                    )
                (loss / accum).backward()
                for key in ("kl_sum", "tokens", "agree", "agree_by_offset", "tokens_by_offset"):
                    window[key] = window[key] + stats[key]
                window["valid_anchors"] += stats["valid_anchors"] / accum
                window["input_tokens"] += batch_ids.numel()

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, t.max_grad_norm)
            if torch.isfinite(grad_norm):
                optimizer.step()
            else:
                window["skipped_steps"] += 1
                log.warning("non-finite gradient at step %d: update skipped", step)
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            window["grad_norm"] += grad_norm.nan_to_num(0.0)
            step += 1
            window_steps += 1
            step_seconds = time.perf_counter() - step_start

            if step % t.log_every == 0 or step in (1, total_steps):
                elapsed = time.perf_counter() - window_start
                tokens = window["tokens"].clamp_min(1)
                metrics = {
                    "train/loss": (window["kl_sum"] / tokens).item(),
                    "train/top1_agreement": (window["agree"] / tokens).item(),
                    "train/grad_norm": (window["grad_norm"] / window_steps).item(),
                    "train/valid_anchors": (window["valid_anchors"] / window_steps).item(),
                    "train/skipped_steps": window["skipped_steps"].item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/epoch": step / steps_per_epoch,
                    "perf/seconds_per_step": elapsed / window_steps,
                    "perf/input_tokens_per_second": window["input_tokens"].item() / elapsed,
                    "perf/eta_hours": (total_steps - step) * elapsed / window_steps / 3600,
                    **offset_metrics(
                        "train", window["agree_by_offset"], window["tokens_by_offset"]
                    ),
                }
                if device.type == "cuda":
                    metrics["perf/max_memory_gb"] = torch.cuda.max_memory_allocated() / 2**30
                tracker.log(metrics, step)
                tracker.progress(step / total_steps, f"step {step}/{total_steps}")
                log.info(
                    "step %d/%d loss %.4f agree %.3f lr %.2e %.1fs/step eta %.1fh",
                    step,
                    total_steps,
                    metrics["train/loss"],
                    metrics["train/top1_agreement"],
                    metrics["train/lr"],
                    metrics["perf/seconds_per_step"],
                    metrics["perf/eta_hours"],
                )
                window.clear()
                window_start, window_steps = time.perf_counter(), 0

            if t.eval_every and step % t.eval_every == 0 and step < total_steps:
                evaluate()
            budget_left = t.max_hours * 3600 - (time.time() - started) if t.max_hours else math.inf
            if budget_left < 3 * step_seconds + 900:  # keep 15 min for saving and uploading
                stop.reason = stop.reason or "time budget"
            if step % t.save_every == 0 or stop.reason or step == total_steps:
                state = {
                    "step": step,
                    "data": data_key,
                    "global_batch_size": t.global_batch_size,
                    "total_steps": total_steps,
                }
                checkpoints.save(
                    step,
                    model,
                    optimizer,
                    scheduler,
                    state,
                    wait=bool(stop.reason) or step == total_steps,
                )
                last_saved = step
                publish_logs()
            if stop.reason or step >= total_steps:
                break

    if step >= total_steps:
        evaluate(final=True)
        final_dir = run_dir / "final"
        export_model(model, tokenizer, final_dir)
        bucket.submit(lambda: bucket.upload(final_dir, f"{remote}/final"))
        log.info("training complete: model exported to %s", final_dir)
    else:
        log.info("stopped at step %d/%d (%s); rerun to resume", step, total_steps, stop.reason)
        if last_saved != step:  # defensive: the loop saves before breaking
            log.warning("last checkpoint is at step %d", last_saved)
    publish_logs()
    tracker.close()
    bucket.wait()


if __name__ == "__main__":
    main()
