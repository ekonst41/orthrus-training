"""Model I/O: build Orthrus from a Qwen3 checkpoint, save/restore training state, export the model.

A training checkpoint holds only what changes: the diffusion weights, optimizer and scheduler
state and progress counters (~1 GB for Qwen3-0.6B instead of the full model). The final export is
a regular HF model directory in the official format (Qwen3 config + block_size/mask_token_id,
auto_map to modeling_orthrus.OrthrusLM), loadable with trust_remote_code=True.
"""

import json
import logging
import os
import shutil
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoTokenizer, Qwen3ForCausalLM

from src import model as modeling_orthrus
from src.config import ModelConfig
from src.hub import Bucket, upload_checkpoint
from src.model import OrthrusLM, copy_diff_from_ar, freeze_to_diffusion
from src.model_dflash import DFlashLM

log = logging.getLogger(__name__)

WEIGHTS, OPTIMIZER, STATE = "trainable.safetensors", "optimizer.pt", "trainer_state.json"
DFLASH_CONFIG = "dflash_config.json"


def load_model(cfg: ModelConfig, device: torch.device, dtype: torch.dtype = torch.bfloat16):
    """Load the selected drafter and return (model, tokenizer, trainable params)."""
    dflash_export = Path(cfg.base) / DFLASH_CONFIG
    if cfg.kind == "dflash" and dflash_export.is_file():
        spec = ModelConfig(**json.loads(dflash_export.read_text(encoding="utf-8")))
        model, tokenizer, trainable = load_model(spec, device, dtype)
        load_trainable(dflash_export.parent, model)
        return model, tokenizer, trainable

    revision = None if Path(cfg.base).exists() else cfg.revision  # Hub commit (local dirs: None)
    tokenizer = AutoTokenizer.from_pretrained(cfg.base, revision=revision)
    config = AutoConfig.from_pretrained(cfg.base, revision=revision)
    if config.model_type != "qwen3":
        raise ValueError(f"{cfg.base} is a {config.model_type} model; this code supports Qwen3")
    config.block_size = getattr(config, "block_size", None) or cfg.block_size
    if config.block_size != cfg.block_size:
        raise ValueError(f"{cfg.base} was trained with block_size={config.block_size}")
    mask_token_id = cfg.mask_token_id if cfg.mask_token_id is not None else len(tokenizer)
    config.mask_token_id = getattr(config, "mask_token_id", None) or mask_token_id
    if not 0 <= config.mask_token_id < config.vocab_size:
        raise ValueError(f"mask_token_id {config.mask_token_id} is outside the vocabulary")

    if cfg.kind == "dflash":
        target = Qwen3ForCausalLM.from_pretrained(
            cfg.base,
            config=config,
            revision=revision,
            dtype=dtype,
            attn_implementation=cfg.attn_implementation,
        )
        model = DFlashLM(target, cfg.draft_layers).to(device)
        trainable = [p for p in model.parameters() if p.requires_grad]
        trainable_dtype = torch.float32 if cfg.trainable_dtype == "float32" else dtype
        for param in trainable:
            param.data = param.data.to(trainable_dtype)
        return model, tokenizer, trainable

    model, info = OrthrusLM.from_pretrained(
        cfg.base,
        revision=revision,
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
    missing_diff = {k for k in info["missing_keys"] if "_diff." in k}
    expected_diff = {k for k in model.state_dict() if "_diff." in k}
    if missing_diff and missing_diff != expected_diff:
        raise RuntimeError("partially missing diffusion weights; export is incomplete")
    if missing_diff:
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


def load_trainable(path: Path, model) -> None:
    device = next(model.parameters()).device
    weights = load_file(path / WEIGHTS, device=str(device))
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    if set(weights) != trainable:
        raise RuntimeError(f"checkpoint {path} does not match the trainable parameters")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in weights:
                param.copy_(weights[name])


def load_checkpoint(path: Path, model, optimizer=None, scheduler=None) -> dict:
    load_trainable(path, model)
    if optimizer is not None:
        device = next(model.parameters()).device
        blob = torch.load(path / OPTIMIZER, map_location=device, weights_only=False)
        optimizer.load_state_dict(blob["optimizer"])
        scheduler.load_state_dict(blob["scheduler"])
    return json.loads((path / STATE).read_text(encoding="utf-8"))


def export_model(
    model: OrthrusLM | DFlashLM, tokenizer, out_dir: Path, model_cfg: ModelConfig | None = None
) -> None:
    """Export the full Orthrus model or the DFlash drafter and its target reference."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(model, DFlashLM):
        if model_cfg is None:
            raise ValueError("DFlash export requires its target model config")
        weights = {
            name: param.detach().cpu().contiguous()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        save_file(weights, out_dir / WEIGHTS)
        spec = asdict(model_cfg) | {"mask_token_id": model.config.mask_token_id}
        (out_dir / DFLASH_CONFIG).write_text(json.dumps(spec, indent=2), encoding="utf-8")
        return
    model.config.auto_map = {"AutoModelForCausalLM": "modeling_orthrus.OrthrusLM"}
    state = {k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}
    model.save_pretrained(out_dir, state_dict=state)
    tokenizer.save_pretrained(out_dir)
    shutil.copy(modeling_orthrus.__file__, out_dir / "modeling_orthrus.py")
    root = Path(__file__).resolve().parents[2]
    for name in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
        if (root / name).exists():
            shutil.copy(root / name, out_dir / name)


class CheckpointManager:
    """Atomic local saves; synchronous upload; keep the newest local checkpoints."""

    def __init__(self, run_dir: Path, bucket: Bucket, remote: str, keep: int):
        self.root = Path(run_dir) / "checkpoints"
        self.root.mkdir(parents=True, exist_ok=True)
        self.bucket, self.remote, self.keep = bucket, remote, keep

    def _local(self):
        return sorted(
            p
            for p in self.root.glob("step-*")
            if p.name.removeprefix("step-").isdigit() and (p / STATE).exists()
        )

    def get(self, name: str):
        if not name.startswith("step-") or not name[5:].isdigit():
            raise ValueError("checkpoint must be step-0001000")
        path = self.root / name
        if not all((path / n).exists() for n in (WEIGHTS, OPTIMIZER, STATE)):
            if not self.bucket.download(f"{self.remote}/{name}", path):
                raise FileNotFoundError(path)
        if not all((path / n).exists() for n in (WEIGHTS, OPTIMIZER, STATE)):
            raise RuntimeError(f"incomplete checkpoint: {path}")
        return path

    def latest(self):
        names = {p.name for p in self._local()}
        for file in self.bucket.list(self.remote):
            p = Path(file)
            if (
                p.name == STATE
                and p.parent.name.startswith("step-")
                and p.parent.name[5:].isdigit()
            ):
                names.add(p.parent.name)
        return self.get(max(names, key=lambda n: int(n[5:]))) if names else None

    def save(self, step, model, optimizer, scheduler, state):
        path = self.root / f"step-{step:07d}"
        save_checkpoint(path, model, optimizer, scheduler, state)
        upload_checkpoint(self.bucket, path, self.remote)
        # Remote history is not automatically deleted.
        for old in self._local()[: -self.keep] if self.keep else []:
            shutil.rmtree(old)
        return path
