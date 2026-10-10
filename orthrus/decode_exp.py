"""Decoding-cycle experiments: does verifying the rejected tail of a draft block raise TPF?

    python -m orthrus.decode_exp --config configs/decode/tail-v1.yaml [decode.max_minutes=10]
    python -m orthrus.decode_exp --config configs/decode/tail-v1.yaml --build-prompts

One job covers every checkpoint of decode.models (official Orthrus Qwen3 models at pinned Hub
commits; empty: the run's model): weights download in the background while the first model
already decodes, and all models stay on the GPU. Work goes in rounds: round r decodes prompt r of
every suite on one model, always on the model with the fewest rounds, so the results stay balanced
across models and suites whenever the job stops; decode.max_minutes bounds the job (no round
starts after it). Per prompt, greedy, batch 1, in decode.dtype:
  1. base (the paper's cycle, the first strategy) decodes the reference output;
  2. the draft table: the diffusion drafts at every anchor of that output (decoding.draft_table),
     from which every strategy of decode.strategies is replayed exactly on the CPU
     (decoding.OracleBackend; orthrus.decode_report);
  3. in the first decode.check_rounds rounds, decode.check_strategies also decode for real: their
     output must equal the reference ("match_base"), the drafts they met must equal the table
     ("draft_mismatch") and the replay must give the same passes ("sim_match"); the first
     decode.ar_prompts prompts of those rounds also decode with plain AR ("match_ar").
Base itself is checked the same way on every prompt.

Results: experiments/<run_name>/ in the bucket (locally <storage.local_dir>/experiments/):
run.json (config, code, environment, one segment per job), <model>/model.json (checkpoint, EOS
ids), <model>/round-NNNN.jsonl (one line per real decode) and round-NNNN.npz (reference outputs and
draft tables), uploaded after every round. A rerun continues with the first missing round of each
model. The job log shows the replayed TPF of every strategy after each round; the full analysis:
python -m orthrus.decode_report.

--build-prompts writes the prompt set decode.prompt_set: up to 128 prompts per suite from pinned
benchmark revisions and the held-out chat prompts of data/<data.dataset>/eval_prompts.jsonl (never
trained on by our models), ranked by sha256(seed:suite:id); ranks alternate between the dev and
test splits. It uploads experiments/prompts/<name>.jsonl and prints the sha256 to pin as
decode.prompt_set_sha256.
"""

import argparse
import dataclasses
import gc
import hashlib
import json
import logging
import re
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import torch

from orthrus.config import Config, ModelConfig, parse_args
from orthrus.decoding import (
    ModelBackend,
    OracleBackend,
    ar_decode,
    decode,
    draft_table,
    parse_strategies,
)
from orthrus.storage import Bucket

log = logging.getLogger("orthrus.decode_exp")

# suite: (Hub dataset, config, split, prompt field); the same prompts as orthrus.evaluate.BENCHMARKS
# (zero-shot, the question as the user turn), at commits pinned on 2026-10-10
BENCHMARKS = {
    "gsm8k": ("openai/gsm8k", "main", "test", "question"),
    "math500": ("HuggingFaceH4/MATH-500", None, "test", "problem"),
    "humaneval": ("openai/openai_humaneval", None, "test", "prompt"),
    "mbpp": ("google-research-datasets/mbpp", "sanitized", "test", "prompt"),
}
REVISIONS = {
    "gsm8k": "740312add88f781978c0658806c59bc2815b9866",
    "math500": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
    "humaneval": "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
    "mbpp": "4bb6404fdc6cacfda99d4ac4205087b89d32030c",
}
SET_SIZE = 128  # prompts stored per suite; a run takes the first decode.prompts_per_suite
SET_SEED = 42
TURN = re.compile(r"<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>\n", re.S)


def chat_ids(tokenizer, messages: list[dict]) -> list[int]:
    """Inference format of the training data and the official README: no thinking."""
    encoded = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=False, tokenize=True, return_dict=True
    )
    return list(encoded["input_ids"])


def messages_from_ids(tokenizer, ids: list[int]) -> list[dict]:
    """The chat turns of a templated prompt (held-out prompts are stored as token ids only)."""
    text = tokenizer.decode(ids, skip_special_tokens=False)
    return [{"role": role, "content": content} for role, content in TURN.findall(text)]


def _rank(suite: str, source_id) -> str:
    return hashlib.sha256(f"{SET_SEED}:{suite}:{source_id}".encode()).hexdigest()


def prompt_set_paths(cfg: Config) -> tuple[Path, str]:
    name = cfg.decode.prompt_set
    return Path(cfg.storage.local_dir) / "experiments" / "prompts" / f"{name}.jsonl", (
        f"experiments/prompts/{name}.jsonl"
    )


def build_prompt_set(cfg: Config) -> str:
    """Write and upload the prompt set (module doc); returns its sha256."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from orthrus.evaluate import load_eval_prompts

    revision = None if Path(cfg.model.base).exists() else cfg.model.revision
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base, revision=revision)
    candidates: dict[str, list[tuple[str, str, str, list[dict]]]] = {}
    sources = {}
    for suite, (repo, config, split, column) in BENCHMARKS.items():
        commit = REVISIONS[suite]
        rows = load_dataset(repo, config, split=split, revision=commit)
        sources[suite] = f"{repo}@{commit} {config or ''}:{split}:{column}"
        candidates[suite] = [
            (
                _rank(suite, i),
                str(i),
                sources[suite],
                [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": row[column]},
                ],
            )
            for i, row in enumerate(rows)
        ]
    bucket = Bucket(cfg.storage.bucket)
    held_out = load_eval_prompts(bucket, cfg.data.dataset, Path(cfg.storage.local_dir))
    sources["chat"] = f"bucket data/{cfg.data.dataset}/eval_prompts.jsonl (domain chat)"
    candidates["chat"] = []
    for prompt in held_out:
        if prompt["domain"] != "chat":
            continue
        messages = messages_from_ids(tokenizer, prompt["prompt_ids"])
        if chat_ids(tokenizer, messages) != prompt["prompt_ids"]:
            raise ValueError(f"chat prompt {prompt['uuid']} does not survive decoding")
        candidates["chat"].append(
            (_rank("chat", prompt["uuid"]), prompt["uuid"], sources["chat"], messages)
        )
    lines = []
    for suite, items in candidates.items():
        for rank, (_, source_id, source, messages) in enumerate(sorted(items)[:SET_SIZE]):
            lines.append(
                {
                    "suite": suite,
                    "rank": rank,
                    "split": "dev" if rank % 2 == 0 else "test",
                    "source": source,
                    "source_id": source_id,
                    "messages": messages,
                    "prompt_ids": chat_ids(tokenizer, messages),
                }
            )
    path, remote = prompt_set_paths(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines).encode("utf-8")
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    meta = {
        "name": cfg.decode.prompt_set,
        "sha256": digest,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tokenizer": f"{cfg.model.base}@{revision}",
        "seed": SET_SEED,
        "sources": sources,
        "counts": {s: sum(line["suite"] == s for line in lines) for s in candidates},
    }
    meta_path = path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    bucket.upload(path, remote)
    bucket.upload(meta_path, remote.replace(".jsonl", ".json"))
    log.info("prompt set %s: %s, sha256 %s", cfg.decode.prompt_set, meta["counts"], digest)
    print(f"decode.prompt_set_sha256: {digest}")
    return digest


def load_prompt_set(cfg: Config, bucket: Bucket) -> tuple[list[dict], str]:
    path, remote = prompt_set_paths(cfg)
    if not path.exists() and not bucket.download_file(remote, path):
        raise FileNotFoundError(f"no prompt set {remote}: build it with --build-prompts")
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    pinned = cfg.decode.prompt_set_sha256
    if pinned and digest != pinned:
        raise ValueError(f"prompt set {path} has sha256 {digest}, the config pins {pinned}")
    if not pinned:
        log.warning("decode.prompt_set_sha256 is not set: the prompt set is not pinned")
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line], digest


def _timed(device: torch.device, fn):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    begin = time.perf_counter()
    result = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return result, time.perf_counter() - begin


def generation_fields(gen) -> dict:
    return {
        "new_tokens": len(gen.tokens),
        "eos": gen.eos,
        "passes": gen.forward_passes,
        "passes_by_kind": gen.passes,
        "input_tokens": gen.input_tokens,
        "pass_seconds": {k: round(v, 6) for k, v in gen.seconds.items()},
        "accepted": gen.accepted,
        "branch": gen.branch,
        "tails": [list(t) for t in gen.tails],
    }


def same_trace(a, b) -> bool:
    return (a.tokens, a.passes, a.accepted, a.branch, a.tails) == (
        b.tokens,
        b.passes,
        b.accepted,
        b.branch,
        b.tails,
    )


@torch.inference_mode()
def run_prompt(model, prompt: dict, strategies, eos: set[int], cfg: Config, with_ar: bool):
    """All strategies (and optionally AR) on one prompt: (records, trace)."""
    d = cfg.decode
    device = next(model.parameters()).device
    ids, new = prompt["prompt_ids"], d.max_new_tokens
    common = {
        "suite": prompt["suite"],
        "rank": prompt["rank"],
        "split": prompt["split"],
        "prompt_tokens": len(ids),
    }
    runs = []
    for strategy in strategies:
        backend = ModelBackend(model, ids, record_drafts=True)
        gen, seconds = _timed(
            device, lambda b=backend, s=strategy: decode(b, s, len(ids), new, eos)
        )
        runs.append((strategy, gen, seconds, backend.drafts))
    reference = runs[0][1].tokens
    trace = {
        "suite": prompt["suite"],
        "rank": prompt["rank"],
        "prompt_len": len(ids),
        "reference": reference,
    }
    (drafts, confidence), seconds = _timed(
        device, lambda: draft_table(model, ids, reference, new, d.chunk_blocks)
    )
    trace.update(drafts=drafts, confidence=confidence, seconds=seconds)
    rows = [row[row >= 0].tolist() for row in drafts]
    confidence_rows = confidence.tolist()
    ar = None
    if with_ar:
        backend = ModelBackend(model, ids)
        ar, ar_seconds = _timed(device, lambda: ar_decode(backend, len(ids), new, eos))
    records = []
    for strategy, gen, seconds, visited in runs:
        record = {**common, "strategy": strategy.name, **generation_fields(gen)}
        record.update(seconds=round(seconds, 6), match_base=gen.tokens == reference)
        if not record["match_base"]:
            record["tokens"] = gen.tokens
        if ar is not None:
            record["match_ar"] = gen.tokens == ar.tokens
        mismatch = sum(
            drafts[start - len(ids), : len(tokens)].tolist() != tokens
            for start, tokens in visited.items()
        )
        record.update(draft_checked=len(visited), draft_mismatch=mismatch, sim_match=None)
        if record["match_base"]:
            oracle = OracleBackend(
                len(ids), reference, rows, confidence_rows, model.config.block_size
            )
            replay = decode(oracle, strategy, len(ids), new, eos)
            record["sim_match"] = same_trace(gen, replay)
        records.append(record)
    if ar is not None:
        record = {
            **common,
            "strategy": "ar",
            **generation_fields(ar),
            "seconds": round(ar_seconds, 6),
        }
        record.update(match_base=ar.tokens == reference, match_ar=True)
        records.append(record)
    return records, trace


def write_traces(path: Path, traces: list[dict], block_size: int, max_new_tokens: int) -> None:
    """Concatenated per-prompt arrays; prompt i's rows are offsets[i]:offsets[i + 1]."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lengths = [len(t["reference"]) for t in traces]
    arrays = {
        "suite": np.array([t["suite"] for t in traces]),
        "rank": np.array([t["rank"] for t in traces], dtype=np.int32),
        "prompt_len": np.array([t["prompt_len"] for t in traces], dtype=np.int32),
        "offsets": np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
        "reference": np.array([x for t in traces for x in t["reference"]], dtype=np.int32),
        "block_size": np.array(block_size),
        "max_new_tokens": np.array(max_new_tokens),
    }
    arrays["drafts"] = np.concatenate([t["drafts"] for t in traces])
    arrays["confidence"] = np.concatenate([t["confidence"] for t in traces])
    arrays["table_seconds"] = np.array([t["seconds"] for t in traces])
    np.savez_compressed(path, **arrays)


def _warm_up(model, prompt: dict, strategies, eos: set[int], cfg: Config) -> None:
    """Kernel selection and allocator growth outside the timed decodes."""
    ids = prompt["prompt_ids"]
    for strategy in strategies:
        decode(ModelBackend(model, ids), strategy, len(ids), 48, eos)
    out = ar_decode(ModelBackend(model, ids), len(ids), 8, eos)
    draft_table(model, ids, out.tokens, cfg.decode.max_new_tokens, cfg.decode.chunk_blocks)


def model_specs(cfg: Config) -> list[tuple[str, ModelConfig]]:
    """(tag, model settings) per checkpoint of the run: decode.models entries override model."""
    if not cfg.decode.models:
        return [(Path(cfg.model.base).name.lower(), cfg.model)]
    specs = []
    for entry in cfg.decode.models:
        fields = dict(entry)
        specs.append((fields.pop("tag"), dataclasses.replace(cfg.model, **fields)))
    if len({tag for tag, _ in specs}) != len(specs):
        raise ValueError("decode.models tags must be unique")
    return specs


class Prefetcher:
    """Downloads Hub checkpoints one at a time in a background thread, so the first model already
    decodes while the next one arrives. A DataSphere job has ~34 GB of disk, less than the
    environment plus all three official checkpoints (4.1 + 9.9 + 19.4 GB), so: the next download
    starts only after the previous model is in memory and its files are deleted; files go to
    /dev/shm (RAM) when it has room; a checkpoint that does not fit anywhere is skipped with an
    error instead of failing the job (the other models go on)."""

    def __init__(self, specs: list[tuple[str, ModelConfig]]):
        self.ready = {tag: threading.Event() for tag, _ in specs}
        self.local: dict[str, Path] = {}  # tag -> downloaded snapshot directory
        self.failed: dict[str, str] = {}
        self._released = {tag: threading.Event() for tag, _ in specs}
        threading.Thread(target=self._work, args=(specs,), daemon=True).start()

    @staticmethod
    def _where(needed: int) -> str | None:
        """Cache directory for `needed` bytes: /dev/shm if it fits there, else the default HF
        cache if the disk has room, else None."""
        from huggingface_hub import constants

        margin = 2 * 2**30
        if Path("/dev/shm").is_dir() and shutil.disk_usage("/dev/shm").free > needed + margin:
            return "/dev/shm/hf-cache"
        root = Path(constants.HF_HUB_CACHE)
        root.mkdir(parents=True, exist_ok=True)
        return str(root) if shutil.disk_usage(root).free > needed + margin else None

    def _work(self, specs) -> None:
        from huggingface_hub import HfApi, snapshot_download

        skip = ["*.png", "*.md"]
        for tag, spec in specs:
            try:
                if not Path(spec.base).exists():
                    info = HfApi().model_info(
                        spec.base, revision=spec.revision, files_metadata=True
                    )
                    needed = sum(f.size or 0 for f in info.siblings if not f.rfilename.endswith(
                        (".png", ".md")))  # fmt: skip
                    cache = self._where(needed)
                    if cache is None:
                        raise OSError(f"no room for {needed / 2**30:.1f} GiB")
                    log.info("downloading %s (%.1f GiB) to %s", spec.base, needed / 2**30, cache)
                    self.local[tag] = Path(
                        snapshot_download(
                            spec.base, revision=spec.revision, cache_dir=cache, ignore_patterns=skip
                        )
                    )
            except Exception as error:
                log.error("%s (%s) skipped: %s", tag, spec.base, error)
                self.failed[tag] = str(error)
            self.ready[tag].set()
            if tag not in self.failed:
                self._released[tag].wait()

    def release(self, tag: str) -> None:
        """The model is in memory: delete its downloaded files, start the next download."""
        gc.collect()  # drop memory maps of the checkpoint files
        if tag in self.local:
            shutil.rmtree(self.local[tag].parents[1], ignore_errors=True)  # models--<org>--<name>
        self._released[tag].set()


@dataclasses.dataclass
class _Model:
    tag: str
    spec: ModelConfig
    model: object = None
    eos: set | None = None
    next_round: int = 0
    replayed: dict = dataclasses.field(default_factory=dict)  # (strategy, suite): [tokens, passes]


def _replay_round(m: _Model, traces: list[dict], strategies, max_new_tokens: int) -> None:
    for trace in traces:
        rows = [row[row >= 0].tolist() for row in trace["drafts"]]
        confidence = trace["confidence"].tolist()
        for strategy in strategies:
            oracle = OracleBackend(
                trace["prompt_len"], trace["reference"], rows, confidence, m.model.config.block_size
            )
            gen = decode(oracle, strategy, trace["prompt_len"], max_new_tokens, m.eos)
            totals = m.replayed.setdefault((strategy.name, trace["suite"]), [0, 0])
            totals[0] += len(gen.tokens)
            totals[1] += gen.forward_passes


def _live_table(models: list[_Model], strategies, suites) -> str:
    """Replayed TPF so far: per model, the mean over suites of tokens / passes."""
    lines = [f"{'strategy':<18}" + "".join(f"{m.tag:>14}" for m in models)]
    for strategy in strategies:
        cells = []
        for m in models:
            totals = [m.replayed.get((strategy.name, s)) for s in suites]
            tpfs = [t[0] / t[1] for t in totals if t]
            cells.append(f"{sum(tpfs) / len(tpfs):>14.3f}" if tpfs else f"{'-':>14}")
        lines.append(f"{strategy.name:<18}" + "".join(cells))
    rounds = ", ".join(f"{m.tag} {m.next_round}" for m in models)
    return f"replayed TPF (mean over suites; rounds: {rounds})\n" + "\n".join(lines)


def _load(m: _Model, local: Path | None, warm_up: list, cfg: Config, device, dtype, run_dir, bucket,
          by_suite: dict):  # fmt: skip
    from orthrus.checkpoint import load_model
    from orthrus.evaluate import eos_token_ids

    spec = dataclasses.replace(m.spec, base=str(local)) if local else m.spec  # the downloaded files
    model, tokenizer, _ = load_model(spec, device, dtype)
    m.model, m.eos = model.eval(), eos_token_ids(model, tokenizer)
    for p in (p for items in by_suite.values() for p in items):
        if chat_ids(tokenizer, p["messages"]) != p["prompt_ids"]:
            raise ValueError(f"{m.tag}: the tokenization of {p['suite']}/{p['rank']} differs")
    info = {
        "tag": m.tag,
        **dataclasses.asdict(m.spec),
        "commit": getattr(model.config, "_commit_hash", None),
        "eos_token_ids": sorted(m.eos),
        "block_size": model.config.block_size,
        "mask_token_id": model.config.mask_token_id,
        "dtype": str(dtype),
    }
    path = run_dir / m.tag / "model.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    bucket.upload(path, f"experiments/{cfg.run_name}/{m.tag}/model.json")
    _warm_up(model, by_suite[cfg.decode.suites[0]][0], warm_up, m.eos, cfg)


def run(cfg: Config) -> None:
    from orthrus.tracking import Tracker, Watchdog, end_segment, write_manifest

    started = time.time()
    d = cfg.decode
    strategies = parse_strategies(d.strategies)
    if strategies[0].kind != "base":
        raise ValueError("the first strategy must be kind base: it decodes the reference")
    by_name = {s.name: s for s in strategies}
    checks = [by_name[name] for name in d.check_strategies]
    if any(s.oracle_min_tokens or s.kind == "base" for s in checks):
        raise ValueError("check_strategies: real decodes besides base (no replay-only gates)")
    specs = model_specs(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, d.dtype) if device.type == "cuda" else torch.float32
    torch.backends.cuda.matmul.allow_tf32 = False  # exact fp32 (the defaults, made explicit)
    torch.backends.cudnn.allow_tf32 = False
    bucket = Bucket(cfg.storage.bucket)
    run_dir = Path(cfg.storage.local_dir) / "experiments" / cfg.run_name
    remote = f"experiments/{cfg.run_name}"
    models = [_Model(tag, spec) for tag, spec in specs]
    for m in models:  # continue after the last complete round of an earlier job
        done = {Path(p).name for p in bucket.list(f"{remote}/{m.tag}")}
        done |= {p.name for p in (run_dir / m.tag).glob("round-*.jsonl")}
        while f"round-{m.next_round:04d}.jsonl" in done:
            m.next_round += 1
    prompts, digest = load_prompt_set(cfg, bucket)
    by_suite = {s: [p for p in prompts if p["suite"] == s][: d.prompts_per_suite] for s in d.suites}
    rounds = min(len(v) for v in by_suite.values())
    if rounds == 0:
        raise ValueError(f"the prompt set lacks some of the suites {d.suites}")
    prefetch = Prefetcher([(m.tag, m.spec) for m in models if m.next_round < rounds])
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = run_dir / "run.json"
    if not manifest.exists():
        bucket.download_file(f"{remote}/run.json", manifest)
    write_manifest(
        manifest,
        cfg,
        prompt_set_sha256=digest,
        strategies=[s.to_dict() for s in strategies],
        models={tag: dataclasses.asdict(spec) for tag, spec in specs},
    )
    bucket.upload(manifest, f"{remote}/run.json")
    tracker = Tracker(cfg, run_dir, "decode")
    watchdog = Watchdog(cfg.train.stall_minutes, "decoding progress")

    completed = []
    deadline = started + d.max_minutes * 60 if d.max_minutes else None
    while True:
        for m in models:  # load what has arrived (between rounds: loading blocks the GPU)
            if m.model is None and m.next_round < rounds and prefetch.ready[m.tag].is_set():
                if m.tag in prefetch.failed:
                    m.next_round = rounds  # skipped (logged by the prefetcher)
                    continue
                warm = [strategies[0], *checks]
                local = prefetch.local.get(m.tag)
                _load(m, local, warm, cfg, device, dtype, run_dir, bucket, by_suite)
                prefetch.release(m.tag)
                log.info("disk free: %.1f GiB", shutil.disk_usage(run_dir).free / 2**30)
                log.info("%s loaded (%.1f min since start)", m.tag, (time.time() - started) / 60)
                watchdog.beat()
        pending = [m for m in models if m.next_round < rounds]
        if not pending:
            break
        if deadline is not None and time.time() > deadline:
            log.info("decode.max_minutes reached: no new round")
            break
        loaded = [m for m in pending if m.model is not None]
        if not loaded:
            next(prefetch.ready[m.tag] for m in pending if m.model is None).wait(timeout=10)
            watchdog.beat()
            continue
        m = min(loaded, key=lambda x: (x.next_round, models.index(x)))
        r, check = m.next_round, m.next_round < d.check_rounds
        records, traces = [], []
        for i, suite in enumerate(d.suites):
            new_records, trace = run_prompt(
                m.model,
                by_suite[suite][r],
                [strategies[0], *checks] if check else [strategies[0]],
                m.eos,
                cfg,
                with_ar=check and i < d.ar_prompts,
            )
            for record in new_records:
                record.update(model=m.tag, round=r)
            records += new_records
            traces.append(trace)
            watchdog.beat()
        failed = [
            f"{x['suite']}/{x['strategy']}"
            for x in records
            if not x["match_base"]
            or x.get("sim_match") is False
            or x.get("match_ar") is False
            or x.get("draft_mismatch")
        ]
        stem = run_dir / m.tag / f"round-{r:04d}"
        write_traces(stem.with_suffix(".npz"), traces, m.model.config.block_size, d.max_new_tokens)
        stem.with_suffix(".jsonl").write_text(
            "".join(json.dumps(x) + "\n" for x in records), encoding="utf-8"
        )
        for suffix in (".npz", ".jsonl"):  # uploaded in order: the .jsonl marks a full round
            path = stem.with_suffix(suffix)
            target = f"{remote}/{m.tag}/{path.name}"  # bound now: the upload runs later
            bucket.submit(lambda path=path, target=target: bucket.upload(path, target))
        _replay_round(m, traces, strategies, d.max_new_tokens)
        m.next_round += 1
        completed.append(f"{m.tag}/{r}")
        log.info(
            "%s round %d done (%.1f min)%s\n%s",
            m.tag,
            r,
            (time.time() - started) / 60,
            f"  CHECK FAILED: {failed}" if failed else "",
            _live_table(models, strategies, d.suites),
        )
        tracker.progress(
            sum(x.next_round for x in models) / (rounds * len(models)), f"{m.tag} round {r}"
        )
    bucket.wait()
    end_segment(manifest, rounds_completed=completed, minutes=(time.time() - started) / 60)
    bucket.upload(manifest, f"{remote}/run.json")
    tracker.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--build-prompts", action="store_true")
    known, rest = parser.parse_known_args(argv)
    cfg = parse_args(__doc__, rest)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    if known.build_prompts:
        build_prompt_set(cfg)
    else:
        run(cfg)


if __name__ == "__main__":
    main()
