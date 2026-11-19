"""One YAML per experiment; CLI overrides use section.key=value."""

import argparse
import dataclasses
import os
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    base: str = "Qwen/Qwen3-0.6B"
    revision: str | None = None
    block_size: int = 32  # anchor + K-1 drafted tokens
    mask_token_id: int | None = None
    attn_implementation: str = "sdpa"
    trainable_dtype: str = "bfloat16"
    kind: str = "orthrus"
    draft_layers: int = 5


@dataclass
class DataConfig:
    dataset: str = "nemotron-v2-qwen3-0.6b"
    seq_len: int = 2048
    eval_rows: int = 256
    seed: int = 42


@dataclass
class TrainConfig:
    epochs: float = 2
    global_batch_size: int = 128
    micro_batch_size: int = 4  # per GPU
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    min_lr_ratio: float = 0.1
    adam_betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    num_anchor_blocks: int = 256
    loss_gamma: float = 0.0  # 0: uniform; >0: exp(-position/gamma)
    kl_chunk_size: int = 2048
    activation_checkpointing: bool = False
    compile: bool = True
    seed: int = 42
    max_steps: int = 0  # 0: all epochs
    log_every: int = 10
    eval_every: int = 500  # 0: only final evaluation
    save_every: int = 100
    keep_checkpoints: int = 2
    resume: bool = True
    loss_type: str = "kl"


@dataclass
class EvalConfig:
    suites: tuple[str, ...] = ("heldout", "gsm8k", "math500", "humaneval", "mbpp")
    prompts: int = 32  # per suite; 0: all (disables generation during training)
    max_new_tokens: int = 512
    dtype: str = "float32"
    checkpoint: str = ""  # final export, "latest", or "step-0001000"


@dataclass
class DatagenConfig:
    source: str = "nvidia/Nemotron-Post-Training-Dataset-v2"
    source_revision: str | None = None
    domains: tuple[str, ...] = ("math", "code", "chat")
    samples_per_domain: int = 175000
    eval_prompts_per_domain: int = 64
    max_prompt_tokens: int = 2048
    max_new_tokens: int = 2048
    shard_size: int = 20000
    gpu_memory_utilization: float = 0.9
    max_num_seqs: int = 512
    seed: int = 42


@dataclass
class HFConfig:
    bucket: str | None = None  # None: ORTHRUS_BUCKET; "": local only
    upload: bool = True


@dataclass
class Config:
    run_name: str = "qwen3-0.6b"
    data_dir: str = "data"
    output_dir: str = "outputs"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    datagen: DatagenConfig = field(default_factory=DatagenConfig)
    hf: HFConfig = field(default_factory=HFConfig)

    def __post_init__(self):
        if self.hf.bucket is None:
            self.hf.bucket = os.environ.get("ORTHRUS_BUCKET", "")
        for name, value in [("run_name", self.run_name), ("data.dataset", self.data.dataset)]:
            if not value or value in (".", "..") or "/" in value or "\\" in value:
                raise ValueError(f"{name} must be one directory name")
        t, d, g = self.train, self.data, self.datagen
        if self.model.kind not in ("orthrus", "dflash") or self.model.draft_layers < 1:
            raise ValueError("require model.kind=orthrus|dflash and model.draft_layers >= 1")
        if t.loss_type not in ("kl", "ce"):
            raise ValueError("train.loss_type must be kl or ce")
        if self.model.block_size < 2 or d.seq_len <= self.model.block_size:
            raise ValueError("require 2 <= model.block_size < data.seq_len")
        if (
            min(
                t.global_batch_size,
                t.micro_batch_size,
                t.num_anchor_blocks,
                t.kl_chunk_size,
                t.log_every,
                t.save_every,
                t.keep_checkpoints,
            )
            < 1
        ):
            raise ValueError(
                "batch sizes, anchor/chunk counts and save/log intervals must be positive"
            )
        if t.global_batch_size % t.micro_batch_size:
            raise ValueError("global_batch_size must be divisible by micro_batch_size")
        if (
            t.epochs <= 0
            or t.max_steps < 0
            or t.eval_every < 0
            or t.loss_gamma < 0
            or d.eval_rows < 0
        ):
            raise ValueError("invalid training duration, gamma or validation size")
        if not 0 <= t.warmup_ratio <= 1 or not 0 <= t.min_lr_ratio <= 1:
            raise ValueError("LR ratios must be between 0 and 1")
        if t.learning_rate <= 0 or t.max_grad_norm <= 0:
            raise ValueError("learning_rate and max_grad_norm must be positive")
        if self.model.trainable_dtype not in ("bfloat16", "float32"):
            raise ValueError("model.trainable_dtype must be bfloat16 or float32")
        if self.eval.dtype not in ("bfloat16", "float32") or self.eval.max_new_tokens < 1:
            raise ValueError("invalid evaluation dtype or token limit")
        known = {"heldout", "gsm8k", "math500", "humaneval", "mbpp"}
        if self.eval.prompts < 0 or not self.eval.suites or set(self.eval.suites) - known:
            raise ValueError("invalid evaluation suites or prompt count")
        if (
            not g.domains
            or len(set(g.domains)) != len(g.domains)
            or min(
                g.samples_per_domain,
                g.shard_size,
                g.max_prompt_tokens,
                g.max_new_tokens,
                g.max_num_seqs,
            )
            < 1
            or g.eval_prompts_per_domain < 0
            or not 0 < g.gpu_memory_utilization < 1
        ):
            raise ValueError("invalid datagen settings")

    @property
    def dataset_path(self) -> Path:
        return Path(self.data_dir) / self.data.dataset

    @property
    def run_path(self) -> Path:
        return Path(self.output_dir) / self.run_name

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _coerce(value, annotation):
    """Convert a YAML or command-line value to the annotated field type."""
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        if value is None:
            return None
        inner = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
        return _coerce(value, inner[0])
    if origin is tuple:
        items = value if isinstance(value, (list, tuple)) else yaml.safe_load(str(value))
        args = typing.get_args(annotation)
        item_type = args[0] if len(args) == 2 and args[1] is Ellipsis else None
        return tuple(_coerce(v, item_type or args[i]) for i, v in enumerate(items))
    if annotation is bool and isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    if annotation in (int, float, str) and not isinstance(value, annotation):
        if annotation is int and isinstance(value, str):
            return int(float(value)) if "e" in value.lower() else int(value)
        return annotation(value)
    return value


def _build(cls, values: dict):
    hints = typing.get_type_hints(cls)
    unknown = set(values) - {f.name for f in dataclasses.fields(cls)}
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in values:
            continue
        value = values[f.name]
        if dataclasses.is_dataclass(hints[f.name]):
            kwargs[f.name] = _build(hints[f.name], value or {})
        else:
            kwargs[f.name] = _coerce(value, hints[f.name])
    return cls(**kwargs)


def load_config(path=None, overrides=()) -> Config:
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) if path else {}
    values = values or {}
    for override in overrides:
        key, sep, raw = override.partition("=")
        if not sep:
            raise ValueError(f"expected section.key=value, got {override!r}")
        *sections, name = key.split(".")
        node = values
        for section in sections:
            node = node.setdefault(section, {})
        node[name] = yaml.safe_load(raw) if raw else ""
    return _build(Config, values)


def parse_args(description=None, argv=None) -> Config:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*", help="section.key=value")
    args = parser.parse_args(argv)
    return load_config(args.config, args.overrides)
