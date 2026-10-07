import os

import math
from pathlib import Path
import random
import time
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer

from src.configuration import OrthrusConfig
from src.model import OrthrusDecoderLayer
from src.train.checkpoint import (
    latest_checkpoint_dir, load_trainer_state, resume_batch_idx_is_valid,
    save_checkpoint, save_final,
)
from src.train.qwen3.model import (
    OrthrusQwen3ForTraining, configure_flex_backend,
    copy_diff_from_ar, freeze_to_diffusion,
)
from src.utils import distributed as dist_utils
from src.utils.data_utils import create_packed_dataloader
from src.utils.logging import (
    configure_run_logger, format_human_count, get_cosine_lr_scale,
    log_main, mix_seed,
)

try:
    import wandb
except ImportError:
    wandb = None
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

CONFIG_PATH = "/home/konstantinova.e87/yandex_contest/orthrus/orthrus_repo/configs/train_qwen3.yaml"


def sample_batch_anchors(input_ids, block_size, num_anchors,
                         generator=None, assistant_mask=None):
    batch_size, seq_len = input_ids.shape
    num_slots = min(num_anchors, seq_len - block_size)
    anchors = torch.zeros((batch_size, num_slots), dtype=torch.long,
                          device=input_ids.device)
    anchor_valid = torch.zeros_like(anchors, dtype=torch.bool)
    for batch in range(batch_size):
        candidates = torch.arange(1, seq_len - block_size + 1, device=input_ids.device)
        if assistant_mask is not None:
            candidates = candidates[
                assistant_mask[batch, candidates].bool()
                & assistant_mask[batch, candidates + 1].bool()
            ]
        if not candidates.numel():
            anchors[batch] = 1
            continue
        picked = candidates[torch.randperm(
            candidates.numel(), generator=generator, device=candidates.device
        )[:num_slots]].sort().values
        count = picked.numel()
        anchors[batch, :count] = picked
        anchors[batch, count:] = picked[-1]
        anchor_valid[batch, :count] = True
    return anchors, anchor_valid


def load_model(weights_dir, tokenizer, dtype, args, *, is_resume=False):
    raw_config, _ = OrthrusConfig.get_config_dict(weights_dir)
    if raw_config.get("model_type") not in ("qwen3", "orthrus"):
        raise ValueError("Expected dense Qwen3 or Orthrus-Qwen3 weights.")
    cfg = OrthrusConfig.from_dict(raw_config)
    if any(kind != "full_attention" for kind in cfg.layer_types):
        raise ValueError("This trainer requires full-attention Qwen3 layers.")
    if is_resume and cfg.block_size != args.block_size:
        raise ValueError("block_size differs from the resume checkpoint.")
    cfg.block_size = args.block_size
    if args.mask_token_id is not None:
        cfg.mask_token_id = args.mask_token_id
    if cfg.mask_token_id is None:
        cfg.mask_token_id = tokenizer.get_vocab().get("<|fim_pad|>")
    if cfg.mask_token_id is None or not 0 <= cfg.mask_token_id < cfg.vocab_size:
        raise ValueError("Set mask_token_id to an existing reserved vocabulary token.")
    cfg.use_cache = True
    model, info = OrthrusQwen3ForTraining.from_pretrained(
        weights_dir, config=cfg, dtype=dtype,
        attn_implementation=args.attn_implementation, output_loading_info=True,
        activation_checkpointing=args.activation_checkpointing,
        kl_chunk_size=args.kl_chunk_size, temperature=args.temperature,
    )
    is_diff = lambda name: any(part.endswith("_diff") for part in name.split("."))
    if (any(not is_diff(name) for name in info["missing_keys"])
            or info["unexpected_keys"] or info.get("mismatched_keys")):
        raise RuntimeError(f"Qwen3 AR weights did not fully match: {info}")
    warm_start = bool(info["missing_keys"])
    if warm_start:
        if raw_config["model_type"] == "orthrus":
            raise RuntimeError("Orthrus checkpoint is missing diffusion weights.")
        copy_diff_from_ar(model)
    return model, cfg, warm_start


def save_config(directory, config):
    with open(Path(directory) / "config.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)


@record
def train(args, config):
    distributed, rank, local_rank, world_size = dist_utils.setup_distributed(
        args.dist_timeout_minutes
    )
    is_main = rank == 0
    try:
        if is_main:
            os.makedirs(args.output_dir, exist_ok=True)
        configure_run_logger(is_main, str(Path(args.output_dir) / "train.log")
                             if is_main else None)
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        resume_dir = args.resume_from or (
            latest_checkpoint_dir(args.output_dir) if args.auto_resume else None
        )
        if resume_dir is not None and not os.path.isdir(resume_dir):
            raise FileNotFoundError(resume_dir)
        weights_dir = resume_dir or args.model_dir
        tokenizer = AutoTokenizer.from_pretrained(weights_dir, trust_remote_code=True)
        model, cfg, warm_start = load_model(
            weights_dir, tokenizer, dtype, args, is_resume=bool(resume_dir)
        )
        if warm_start:
            log_main(is_main, "Warm-started diffusion attention from frozen AR weights.")
        n_train, n_total = freeze_to_diffusion(model)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        configure_flex_backend(args.flex_backend)
        model = model.to(device)
        if args.fsdp2:
            if not distributed:
                raise ValueError("fsdp2 requires torchrun with WORLD_SIZE > 1.")
            if dtype != torch.bfloat16:
                raise ValueError("The shared FSDP2 helper requires bfloat16.")
            dist_utils.apply_fsdp2(model, modules_to_shard=[OrthrusDecoderLayer],
                                  activation_checkpointing=False)
        elif distributed:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                        broadcast_buffers=False)
        base_model = model.module if distributed and not args.fsdp2 else model

        wandb_run = None
        if args.wandb and is_main:
            if wandb is None:
                raise ImportError("Install wandb or set logging.wandb: false.")
            wandb_run = wandb.init(project=args.wandb_project,
                                   name=args.wandb_run_name, config=config)
        log_main(is_main, f"Orthrus-Qwen3 | model={args.model_dir} | "
                         f"world_size={world_size} | dtype={args.dtype}")
        log_main(is_main, f"seq_len={args.seq_len} | block_size={cfg.block_size} | "
                         f"anchors={args.num_anchor_blocks} | "
                         f"micro_bsz={args.micro_batch_size} | "
                         f"grad_accum={args.grad_accum_steps} | lr={args.lr:.2e}")
        log_main(is_main, f"Params | total={format_human_count(total_params)} | "
                         f"trainable={format_human_count(trainable_params)} | "
                         f"{n_train}/{n_total} tensors | full forward KL")

        loader = create_packed_dataloader(
            cache_path=args.packed_cache_path, seq_len=args.seq_len,
            micro_batch_size=args.micro_batch_size, num_workers=args.num_workers,
            distributed=distributed, shuffle=True, seed=args.seed,
            require_assistant_mask=not args.allow_missing_assistant_mask,
        )
        if not len(loader):
            raise ValueError("Packed dataset is too small for this batch/world size.")
        sampler = loader.sampler
        loader_gen = torch.Generator()
        loader.generator = loader_gen
        if not distributed:
            sampler.generator = loader_gen
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay,
        )
        total_steps = math.ceil(len(loader) / args.grad_accum_steps) * args.epochs
        if args.max_steps > 0:
            total_steps = min(total_steps, args.max_steps)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda cur: get_cosine_lr_scale(cur, warmup_steps, total_steps)
        )
        step, start_epoch, resume_batch_idx = 0, 0, -1
        if resume_dir:
            step, start_epoch, resume_batch_idx, saved_world, saved_ga = load_trainer_state(
                resume_dir, model, optimizer, scheduler, args.fsdp2, distributed
            )
            if not resume_batch_idx_is_valid(
                saved_world, saved_ga, world_size, args.grad_accum_steps
            ):
                raise ValueError("Resume world size / grad accumulation differs.")
            log_main(is_main, f"Resumed {resume_dir} | step={step}")
        if is_main:
            save_config(args.output_dir, config)

        amp_context = ((lambda: torch.autocast("cuda", dtype=dtype))
                       if dtype == torch.bfloat16 else nullcontext)
        anchor_gen = torch.Generator(device=device)
        optimizer.zero_grad(set_to_none=True)
        train_start = last_log = time.perf_counter()
        running_loss, logged_batches, logged_tokens = 0.0, 0, 0
        for epoch in range(start_epoch, args.epochs):
            if step >= total_steps:
                break
            if distributed:
                sampler.set_epoch(epoch)
            loader_gen.manual_seed(args.seed + epoch)
            model.train()
            epoch_iter = (tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
                          if tqdm is not None and is_main else loader)
            for batch_idx, batch in enumerate(epoch_iter):
                if epoch == start_epoch and batch_idx <= resume_batch_idx:
                    continue
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                assistant_mask = batch.get("assistant_mask")
                if assistant_mask is not None:
                    assistant_mask = assistant_mask.to(device, non_blocking=True)
                group_start = batch_idx // args.grad_accum_steps * args.grad_accum_steps
                group_size = min(args.grad_accum_steps, len(loader) - group_start)
                is_sync_step = ((batch_idx + 1) % args.grad_accum_steps == 0
                                or batch_idx + 1 == len(loader))
                sync_ctx = nullcontext()
                if distributed and args.fsdp2:
                    model.set_requires_gradient_sync(is_sync_step)
                elif distributed and not is_sync_step:
                    sync_ctx = model.no_sync()
                # DDP no_sync must include both forward and backward.
                with sync_ctx:
                    with amp_context():
                        anchor_gen.manual_seed(mix_seed(args.seed, epoch, rank, batch_idx))
                        anchors, anchor_valid = sample_batch_anchors(
                            input_ids, cfg.block_size, args.num_anchor_blocks,
                            generator=anchor_gen, assistant_mask=assistant_mask,
                        )
                        loss, _hidden, _keep = model(
                            input_ids=input_ids, anchors=anchors, anchor_valid=anchor_valid,
                            supervise_mask=assistant_mask,
                        )
                    (loss / group_size).backward()
                running_loss += loss.detach().float().item()
                logged_batches += 1
                logged_tokens += input_ids.numel() * world_size
                if not is_sync_step:
                    continue
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % args.log_every == 0 or step == total_steps:
                    mean_loss = torch.tensor(running_loss / logged_batches, device=device)
                    if distributed:
                        dist.all_reduce(mean_loss)
                        mean_loss /= world_size
                    now = time.perf_counter()
                    tok_s = logged_tokens / max(now - last_log, 1e-6)
                    last_log = now
                    metrics = {"train/loss": mean_loss.item(),
                               "train/lr": optimizer.param_groups[0]["lr"],
                               "train/tokens_per_sec": tok_s,
                               "train/anchor_valid": anchor_valid.float().mean().item()}
                    log_main(is_main, f"[train] epoch={epoch + 1}/{args.epochs} step={step} "
                                     f"loss={metrics['train/loss']:.4f} "
                                     f"lr={metrics['train/lr']:.2e} tok/s={tok_s:,.0f}")
                    if wandb_run is not None:
                        wandb_run.log(metrics, step=step)
                    running_loss, logged_batches, logged_tokens = 0.0, 0, 0
                if args.save_every > 0 and step % args.save_every == 0:
                    save_dir = str(Path(args.output_dir) / f"step-{step}")
                    save_checkpoint(save_dir, model, base_model, tokenizer,
                                    optimizer, scheduler, step, epoch, batch_idx,
                                    args, distributed, is_main)
                    if is_main:
                        save_config(save_dir, config)
                    if distributed:
                        dist.barrier()
                if step >= total_steps:
                    break
            resume_batch_idx = -1

        final_dir = save_final(args.output_dir, model, base_model, tokenizer,
                               args.fsdp2, distributed, is_main)
        if is_main:
            save_config(final_dir, config)
            log_main(is_main, f"Training complete | steps={step} | "
                             f"{(time.perf_counter() - train_start) / 60:.2f} min | {final_dir}")
            if wandb_run is not None:
                wandb_run.finish()
        if distributed:
            dist.barrier()
    finally:
        dist_utils.cleanup_distributed(distributed)


def main():
    with open(CONFIG_PATH, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    args = SimpleNamespace(**{key: value for section in config.values()
                              for key, value in section.items()})
    train(args, config)


if __name__ == "__main__":
    main()
