"""Experiment configuration: one YAML file per experiment, overridable from the command line.

    python -m orthrus.train --config configs/qwen3-0.6b.yaml train.max_steps=20 run_name=smoke

Defaults follow the paper (arXiv:2605.12825, Table 3); details the paper does not state follow the
official code (github.com/chiennv2000/orthrus). The source of every value is noted next to it.
"""

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
    block_size: int = 32  # K, paper
    mask_token_id: int | None = None  # None: len(tokenizer), official init_model.py
    attn_implementation: str = "sdpa"  # AR view; the diffusion view always uses FlexAttention
    trainable_dtype: str = "bfloat16"  # paper trains the diffusion weights in bf16


@dataclass
class DataConfig:
    dataset: str = "nemotron-v2-qwen3-0.6b"  # bucket prefix data/<dataset>/ written by datagen
    seq_len: int = 2048  # packed row length, paper
    eval_rows: int = 256  # held-out packed rows for the evaluation loss
    seed: int = 42


@dataclass
class TrainConfig:
    epochs: float = 2  # paper
    global_batch_size: int = 128  # sequences per optimizer step, paper
    micro_batch_size: int = 4  # per forward pass; global / micro = gradient accumulation
    learning_rate: float = 2e-4  # paper
    warmup_ratio: float = 0.05  # paper
    min_lr_ratio: float = 0.1  # cosine floor, official code
    adam_betas: tuple[float, float] = (0.9, 0.95)  # official code
    weight_decay: float = 0.0  # official code
    max_grad_norm: float = 1.0  # paper
    num_anchor_blocks: int = 256  # blocks per sequence, paper
    kl_chunk_size: int = 2048  # rows per vocabulary projection chunk in the KL loss
    activation_checkpointing: bool = False
    compile: bool = True  # GPU: torch.compile each decoder layer and the fused KL (same math)
    flex_kernel_options: dict = field(default_factory=dict)  # FlexAttention kernel settings
    seed: int = 42
    max_steps: int = 0  # stop after this many optimizer steps (0: run all epochs)
    max_hours: float = 0  # stop and save before this wall-clock budget (0: no limit)
    log_every: int = 10
    eval_every: int = 500  # optimizer steps between evaluations (0: only at the end)
    save_every: int = 250  # optimizer steps between checkpoints
    keep_checkpoints: int = 2  # newest checkpoints kept locally and in the bucket


@dataclass
class EvalConfig:
    prompts: int = 32  # held-out prompts for generation metrics during training
    max_new_tokens: int = 256
    check_ar_parity: bool = False  # also decode with the AR view and compare tokens (slower)
    dtype: str = "bfloat16"  # evaluate.py on GPU; "float32" checks AR parity without bf16 rounding


@dataclass
class DatagenConfig:
    source: str = "nvidia/Nemotron-Post-Training-Dataset-v2"  # paper
    domains: tuple[str, ...] = ("math", "code", "chat")  # 1:1:1, paper
    samples_per_domain: int = 175_000  # code has 175k prompts: largest strictly balanced set
    eval_prompts_per_domain: int = 64  # held out from training for generation metrics
    max_prompt_tokens: int = 2048
    max_new_tokens: int = 2048
    temperature: float = 0.0  # greedy, like the paper's main evaluation
    shard_size: int = 20_000  # prompts per generated shard (one upload per shard)
    gpu_memory_utilization: float = 0.90
    max_num_seqs: int = 512  # concurrent sequences in vLLM: a 0.6B model needs a large batch
    speculative: dict = field(default_factory=dict)  # vLLM speculative_config ({}: off)
    max_hours: float = 0  # stop before this wall-clock budget; rerun resumes (0: no limit)
    seed: int = 42


@dataclass
class StorageConfig:
    bucket: str | None = None  # HF Storage Bucket "user/name"; None: $ORTHRUS_BUCKET; "" disables
    local_dir: str = "runs"  # working directory for data, checkpoints and logs


@dataclass
class TrackingConfig:
    enabled: bool = True  # False: metrics only in the local JSONL file
    project: str = "orthrus"
    space_id: str | None = None  # Trackio dashboard Space; None: $ORTHRUS_TRACKIO_SPACE; "" local


@dataclass
class Config:
    run_name: str = "qwen3-0.6b"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    datagen: DatagenConfig = field(default_factory=DatagenConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    def __post_init__(self) -> None:
        if self.storage.bucket is None:
            self.storage.bucket = os.environ.get("ORTHRUS_BUCKET") or ""
        if self.tracking.space_id is None:
            self.tracking.space_id = os.environ.get("ORTHRUS_TRACKIO_SPACE") or ""
        if self.train.global_batch_size % self.train.micro_batch_size:
            raise ValueError("train.global_batch_size must be a multiple of train.micro_batch_size")

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


def load_config(path: str | Path | None = None, overrides: list[str] | tuple = ()) -> Config:
    """Read a YAML config and apply `section.key=value` overrides."""
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) if path else {}
    values = values or {}
    for override in overrides:
        key, sep, raw = override.partition("=")
        if not sep:
            raise ValueError(f"override must look like section.key=value, got {override!r}")
        *sections, name = key.split(".")
        node = values
        for section in sections:
            node = node.setdefault(section, {})
        node[name] = yaml.safe_load(raw) if raw else ""
    return _build(Config, values)


def parse_args(description: str, argv: list[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config", required=True, help="experiment YAML, e.g. configs/qwen3-0.6b.yaml"
    )
    parser.add_argument("overrides", nargs="*", help="section.key=value overrides")
    args = parser.parse_args(argv)
    return load_config(args.config, args.overrides)
