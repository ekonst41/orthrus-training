"""Decoding-cycle experiments: does verifying the rejected tail of a draft block raise TPF?

    python -m orthrus.decode_exp --config configs/decode/qwen3-1.7b.yaml [decode.max_new_tokens=64]
    python -m orthrus.decode_exp --config configs/decode/qwen3-1.7b.yaml --build-prompts

Runs every strategy of decode.strategies (orthrus/decoding.py) on a frozen prompt set with an
Orthrus model (model.base: an export directory or a Hub id such as chiennv/Orthrus-Qwen3-4B, at a
pinned model.revision), greedy, batch 1, in decode.dtype. Per prompt:
  1. the first strategy (kind base: the paper's cycle) decodes the reference output;
  2. the draft table: the diffusion drafts at every anchor of that output (decoding.draft_table);
  3. every other strategy decodes for real; its output must equal the reference ("match_base"),
     the drafts it met must equal the table ("draft_mismatch"), and replaying it from the table
     (decoding.OracleBackend) must give the same passes ("sim_match");
  4. the first decode.ar_reference_prompts prompts of each suite are also decoded with plain AR:
     losslessness ("match_ar") and the AR wall-clock time.
Strategies run one after another on the same prompt, so slow drifts of the machine hit all alike.

Results go to experiments/<run_name>/ in the bucket (locally <storage.local_dir>/experiments/):
run.json (config, code, environment, one segment per job), suites/<suite>/traces.npz (reference
outputs and draft tables) and suites/<suite>/records.jsonl (one line per prompt and strategy,
uploaded last: it marks a complete suite), summary.json. A rerun continues with the first missing
suite. Analysis and the offline policy search: python -m orthrus.decode_report.

--build-prompts writes the prompt set decode.prompt_set: up to 128 prompts per suite from pinned
benchmark revisions and the held-out chat prompts of data/<data.dataset>/eval_prompts.jsonl (never
trained on by our models), ranked by sha256(seed:suite:id); ranks alternate between the dev and
test splits. It uploads experiments/prompts/<name>.jsonl and prints the sha256 to pin as
decode.prompt_set_sha256.
"""

import argparse
import hashlib
import json
import logging
import re
import time
from pathlib import Path

import numpy as np
import torch

from orthrus.config import Config, parse_args
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
        backend = ModelBackend(model, ids, record_drafts=d.draft_table)
        gen, seconds = _timed(
            device, lambda b=backend, s=strategy: decode(b, s, len(ids), new, eos)
        )
        runs.append((strategy, gen, seconds, backend.drafts))
    reference = runs[0][1].tokens
    trace = {"rank": prompt["rank"], "prompt_len": len(ids), "reference": reference}
    if d.draft_table:
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
        if d.draft_table:
            mismatch = sum(
                drafts[start - len(ids), : len(tokens)].tolist() != tokens
                for start, tokens in visited.items()
            )
            record.update(draft_checked=len(visited), draft_mismatch=mismatch)
            record["sim_match"] = None
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
        "rank": np.array([t["rank"] for t in traces], dtype=np.int32),
        "prompt_len": np.array([t["prompt_len"] for t in traces], dtype=np.int32),
        "offsets": np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
        "reference": np.array([x for t in traces for x in t["reference"]], dtype=np.int32),
        "block_size": np.array(block_size),
        "max_new_tokens": np.array(max_new_tokens),
    }
    if traces and "drafts" in traces[0]:
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
    if cfg.decode.draft_table:
        draft_table(model, ids, out.tokens, cfg.decode.max_new_tokens, cfg.decode.chunk_blocks)


def run(cfg: Config) -> None:
    from orthrus.checkpoint import load_model
    from orthrus.decode_report import summarize, summary_table
    from orthrus.evaluate import eos_token_ids
    from orthrus.tracking import Tracker, Watchdog, end_segment, write_manifest

    d = cfg.decode
    strategies = parse_strategies(d.strategies)
    if strategies[0].kind != "base":
        raise ValueError("the first strategy must be kind base: it decodes the reference")
    if any(s.oracle_min_tokens for s in strategies):
        raise ValueError("oracle_min_tokens strategies are replay-only (orthrus.decode_report)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, d.dtype) if device.type == "cuda" else torch.float32
    torch.backends.cuda.matmul.allow_tf32 = False  # exact fp32 (the defaults, made explicit)
    torch.backends.cudnn.allow_tf32 = False
    bucket = Bucket(cfg.storage.bucket)
    prompts, digest = load_prompt_set(cfg, bucket)
    model, tokenizer, _ = load_model(cfg.model, device, dtype)
    model.eval()
    eos = eos_token_ids(model, tokenizer)
    for prompt in prompts:  # the model's own chat template must give the stored ids
        if (
            prompt["suite"] in d.suites
            and chat_ids(tokenizer, prompt["messages"]) != (prompt["prompt_ids"])
        ):
            raise ValueError(f"{prompt['suite']}/{prompt['rank']}: tokenization differs")

    run_dir = Path(cfg.storage.local_dir) / "experiments" / cfg.run_name
    remote = f"experiments/{cfg.run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = run_dir / "run.json"
    if not manifest.exists():
        bucket.download_file(f"{remote}/run.json", manifest)
    write_manifest(
        manifest,
        cfg,
        prompt_set_sha256=digest,
        strategies=[s.to_dict() for s in strategies],
        eos_token_ids=sorted(eos),
        model_commit=getattr(model.config, "_commit_hash", None),
    )
    bucket.upload(manifest, f"{remote}/run.json")
    tracker = Tracker(cfg, run_dir, "decode")
    watchdog = Watchdog(cfg.train.stall_minutes, "decoding progress")
    started, finished = time.time(), []

    selected = {s: [p for p in prompts if p["suite"] == s][: d.prompts_per_suite] for s in d.suites}
    total = sum(len(v) for v in selected.values())
    _warm_up(model, next(iter(selected.values()))[0], strategies, eos, cfg)
    done_prompts = 0
    for suite, items in selected.items():
        if not items:
            raise ValueError(f"the prompt set has no prompts for suite {suite}")
        records_path = run_dir / "suites" / suite / "records.jsonl"
        if records_path.exists() or bucket.exists(f"{remote}/suites/{suite}/records.jsonl"):
            log.info("%s: complete in an earlier job, skipped", suite)
            done_prompts += len(items)
            continue
        records, traces = [], []
        for i, prompt in enumerate(items):
            new_records, trace = run_prompt(
                model, prompt, strategies, eos, cfg, i < d.ar_reference_prompts
            )
            records += new_records
            traces.append(trace)
            done_prompts += 1
            line = " | ".join(
                f"{r['strategy']} {r['new_tokens'] / r['passes']:.2f}"
                for r in new_records
                if r["strategy"] != "ar"
            )
            log.info(
                "%s %d/%d: %d+%d tokens, TPF %s%s",
                suite,
                i + 1,
                len(items),
                len(prompt["prompt_ids"]),
                len(trace["reference"]),
                line,
                ""
                if all(r.get("sim_match") is not False and r["match_base"] for r in new_records)
                else "  [CHECK FAILED]",
            )
            tracker.progress(done_prompts / total, f"{suite} {i + 1}/{len(items)}")
            watchdog.beat()
        write_traces(
            run_dir / "suites" / suite / "traces.npz",
            traces,
            model.config.block_size,
            d.max_new_tokens,
        )
        records_path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        bucket.upload(
            run_dir / "suites" / suite / "traces.npz", f"{remote}/suites/{suite}/traces.npz"
        )
        bucket.upload(records_path, f"{remote}/suites/{suite}/records.jsonl")  # marks completion
        finished.append(suite)
        log.info("%s complete:\n%s", suite, summary_table(summarize(records)))

    all_records = []
    for suite in d.suites:
        path = run_dir / "suites" / suite / "records.jsonl"
        if not path.exists():
            bucket.download_file(f"{remote}/suites/{suite}/records.jsonl", path)
        all_records += [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
    summary = summarize(all_records)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    bucket.upload(run_dir / "summary.json", f"{remote}/summary.json")
    log.info("all suites:\n%s", summary_table(summary))
    end_segment(manifest, suites_completed=finished, hours_decoding=(time.time() - started) / 3600)
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
