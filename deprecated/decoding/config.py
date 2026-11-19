"""Configuration retained only for the archived decoding experiments."""
import os
from dataclasses import dataclass, field
from pathlib import Path
import yaml
from src.config import ModelConfig, DataConfig, _build

@dataclass
class DecodeConfig:
    """Decoding-cycle experiments (orthrus.decode_exp, orthrus.decode_report)."""

    models: list = field(default_factory=list)  # [{tag, base, revision}, ...] overriding model
    # per checkpoint, all decoded in one job ([]: the run's model); base "bucket:runs/<run>/final"
    # takes an export of our own run from the bucket
    prompt_set: str = "decode-v1"  # bucket experiments/prompts/<name>.jsonl (--build-prompts)
    prompt_set_sha256: str = ""  # pin: a run refuses a prompt set with another hash
    suites: tuple[str, ...] = ("gsm8k", "math500", "humaneval", "mbpp", "chat")
    prompts_per_suite: int = 64  # rounds at most: round r decodes prompt r of every suite
    max_minutes: float = 0  # no new round after this many minutes of the job (0: no limit)
    max_new_tokens: int = 512
    dtype: str = "float32"  # fp32: every strategy reproduces the AR output exactly (bf16 ties)
    strategies: list = field(default_factory=lambda: [{"name": "base"}])  # decoding.Strategy
    # fields, replayed on every prompt; the first must be kind base (it decodes the reference)
    check_rounds: int = 1  # rounds in which check_strategies also decode for real
    check_strategies: list = field(default_factory=list)  # names (real decodes besides base)
    ar_prompts: int = 1  # prompts per check round also decoded with plain AR (losslessness)
    chunk_blocks: int = 32  # blocks per pass of the draft table


@dataclass
class StorageConfig:
    bucket: str | None = None
    local_dir: str = "."  # data/ and experiments/ under this directory

@dataclass
class TrainConfig:
    stall_minutes: float = 0

@dataclass
class Config:
    run_name: str = "decode-tail-v1"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self):
        if self.storage.bucket is None:
            self.storage.bucket = os.environ.get("ORTHRUS_BUCKET", "")

    def to_dict(self):
        import dataclasses
        return dataclasses.asdict(self)


def load_config(path=None, overrides=()):
    values = yaml.safe_load(Path(path).read_text()) if path else {}
    values = values or {}
    for override in overrides:
        key, _, value = override.partition("=")
        *sections, name = key.split(".")
        node = values
        for section in sections:
            node = node.setdefault(section, {})
        node[name] = yaml.safe_load(value) if value else ""
    return _build(Config, values)


def parse_args(description=None, argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)
    return load_config(args.config, args.overrides)
