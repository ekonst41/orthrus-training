"""Model I/O: build Orthrus from a Qwen3 checkpoint, save/restore training state, export the model.

A training checkpoint holds only what changes: the diffusion weights, optimizer and scheduler
state and progress counters (~1 GB for Qwen3-0.6B instead of the full model). The final export is
a regular HF model directory in the official format (Qwen3 config + block_size/mask_token_id,
auto_map to modeling_orthrus.OrthrusLM), loadable with trust_remote_code=True.
"""

import json
import logging
import os
import re
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoTokenizer

from orthrus import modeling_orthrus
from orthrus.config import ModelConfig
from orthrus.modeling_orthrus import OrthrusLM, copy_diff_from_ar, freeze_to_diffusion
from orthrus.storage import Bucket

log = logging.getLogger(__name__)

WEIGHTS, OPTIMIZER, STATE = "trainable.safetensors", "optimizer.pt", "trainer_state.json"


def load_model(cfg: ModelConfig, device: torch.device, dtype: torch.dtype = torch.bfloat16):
    """Orthrus from a plain Qwen3 checkpoint (diffusion twins warm-started from the AR weights) or
    from an exported Orthrus model (twins loaded). Returns (model, tokenizer, trainable params)."""
    tokenizer = AutoTokenizer.from_pretrained(cfg.base)
    config = AutoConfig.from_pretrained(cfg.base)
    if config.model_type != "qwen3":
        raise ValueError(f"{cfg.base} is a {config.model_type} model; this code supports Qwen3")
    config.block_size = getattr(config, "block_size", None) or cfg.block_size
    if config.block_size != cfg.block_size:
        raise ValueError(f"{cfg.base} was trained with block_size={config.block_size}")
    mask_token_id = cfg.mask_token_id if cfg.mask_token_id is not None else len(tokenizer)
    config.mask_token_id = getattr(config, "mask_token_id", None) or mask_token_id
    if not 0 <= config.mask_token_id < config.vocab_size:
        raise ValueError(f"mask_token_id {config.mask_token_id} is outside the vocabulary")

    model, info = OrthrusLM.from_pretrained(
        cfg.base,
        config=config,
        dtype=dtype,
        attn_implementation=cfg.attn_implementation,
        output_loading_info=True,
    )
    unexpected = info["unexpected_keys"]
    tied = {"lm_head.weight"} if config.tie_word_embeddings else set()
    foreign = [k for k in info["missing_keys"] if "_diff." not in k and k not in tied]
    if unexpected or foreign:
        raise RuntimeError(f"weights do not match: missing={foreign} unexpected={unexpected}")
    if any("_diff." in k for k in info["missing_keys"]):  # plain Qwen3: warm-start from AR
        copy_diff_from_ar(model)
    trainable = freeze_to_diffusion(model)
    if cfg.trainable_dtype == "float32":  # optional fp32 master weights (paper: bf16)
        for param in trainable:
            param.data = param.data.float()
    return model.to(device), tokenizer, trainable


def save_checkpoint(path: Path, model, optimizer, scheduler, state: dict) -> None:
    """Write atomically: files go to <path>.tmp, renamed once complete."""
    tmp = path.with_name(path.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    weights = {n: p.detach().contiguous() for n, p in model.named_parameters() if p.requires_grad}
    save_file(weights, tmp / WEIGHTS)
    torch.save(
        {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}, tmp / OPTIMIZER
    )
    (tmp / STATE).write_text(json.dumps(state, indent=2), encoding="utf-8")
    shutil.rmtree(path, ignore_errors=True)
    os.replace(tmp, path)


def load_checkpoint(path: Path, model, optimizer=None, scheduler=None) -> dict:
    device = next(model.parameters()).device
    weights = load_file(path / WEIGHTS, device=str(device))
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    if set(weights) != trainable:
        raise RuntimeError(f"checkpoint {path} does not match the trainable parameters")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in weights:
                param.copy_(weights[name])
    if optimizer is not None:
        blob = torch.load(path / OPTIMIZER, map_location=device, weights_only=False)
        optimizer.load_state_dict(blob["optimizer"])
        scheduler.load_state_dict(blob["scheduler"])
    return json.loads((path / STATE).read_text(encoding="utf-8"))


def export_model(model: OrthrusLM, tokenizer, out_dir: Path) -> None:
    """HF model directory in the official Orthrus format (bf16 weights + modeling_orthrus.py)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.config.auto_map = {"AutoModelForCausalLM": "modeling_orthrus.OrthrusLM"}
    state = {k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}
    model.save_pretrained(out_dir, state_dict=state)
    tokenizer.save_pretrained(out_dir)
    shutil.copy(modeling_orthrus.__file__, out_dir / "modeling_orthrus.py")


class CheckpointManager:
    """Local checkpoints mirrored to the bucket; the newest `keep` survive in both places."""

    STEP = re.compile(r"step-(\d+)$")

    def __init__(self, run_dir: Path, bucket: Bucket, remote: str, keep: int):
        self.root, self.bucket, self.remote, self.keep = (
            run_dir / "checkpoints",
            bucket,
            remote,
            keep,
        )
        self.root.mkdir(parents=True, exist_ok=True)

    def _local(self) -> list[Path]:
        return sorted(p for p in self.root.iterdir() if p.is_dir() and self.STEP.search(p.name))

    def save(self, step: int, model, optimizer, scheduler, state: dict, wait: bool = False) -> Path:
        path = self.root / f"step-{step:07d}"
        save_checkpoint(path, model, optimizer, scheduler, state)
        stale = self._local()[: -self.keep]

        def publish() -> None:  # upload first, then drop old copies: never zero good checkpoints
            for name in (WEIGHTS, OPTIMIZER, STATE):  # trainer_state.json last: marks completeness
                self.bucket.upload(path / name, f"{self.remote}/{path.name}/{name}")
            for old in stale:
                self.bucket.delete(f"{self.remote}/{old.name}")
                shutil.rmtree(old, ignore_errors=True)

        if self.bucket.enabled:
            self.bucket.submit(publish)
            if wait:
                self.bucket.wait()
        else:
            for old in stale:
                shutil.rmtree(old, ignore_errors=True)
        log.info("checkpoint step %d saved to %s", step, path)
        return path

    def latest(self) -> Path | None:
        """Newest complete checkpoint, downloading it from the bucket if it is newer than local."""
        local = self._local()
        remote_steps = {
            int(m.group(1))
            for p in self.bucket.list(self.remote)
            if (m := self.STEP.search(Path(p).parent.name)) and Path(p).name == STATE
        }
        best_local = int(self.STEP.search(local[-1].name).group(1)) if local else -1
        best_remote = max(remote_steps, default=-1)
        if best_remote > best_local:
            path = self.root / f"step-{best_remote:07d}"
            self.bucket.download(f"{self.remote}/{path.name}", path)
            return path
        return local[-1] if local else None
