"""Generate the training data (paper, App. A) with vLLM.

Prompts come from Nemotron-Post-Training-Dataset-v2 (math, code, chat, 1:1:1) and are answered by
the frozen AR model itself (greedy, thinking disabled). The prompt selection is cached as
data/<dataset>/prompts.parquet, so a rerun answers the same prompts in the same order. Each shard of
token ids is uploaded to data/<dataset>/raw/ as soon as it is ready; a rerun continues with the
first missing shard. Speed settings (vLLM speculative decoding etc.): see orthrus/genbench.py.

    python -m orthrus.datagen --config configs/qwen3-0.6b.yaml [datagen.samples_per_domain=200]
"""

import hashlib
import itertools
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem
from transformers import AutoTokenizer, GenerationConfig

from orthrus.config import DatagenConfig, parse_args
from orthrus.storage import Bucket
from orthrus.tracking import Tracker, Watchdog, restore_history, write_manifest

log = logging.getLogger("orthrus.datagen")


def prompt_turns(messages: list[dict]) -> list[dict] | None:
    """Turns before the first assistant reply; the empty system turn is kept, as in the data."""
    turns = []
    for message in messages:
        if message["role"] == "assistant":
            break
        turns.append({"role": message["role"], "content": message["content"]})
    return turns if any(t["role"] == "user" for t in turns) else None


def encode(tokenizer, conversations: list[list[dict]]) -> list[list[int]]:
    """Chat-template prompts exactly as at inference: generation prompt, thinking disabled."""
    encoded = tokenizer.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_dict=True,
    )
    return [list(ids) for ids in encoded["input_ids"]]


def sample_prompts(cfg: DatagenConfig, tokenizer) -> tuple[list[dict], list[dict], str]:
    """Deterministic, balanced prompt sample. Returns (train, eval, dataset revision).

    Per domain, prompts are ranked by sha256(seed:uuid), tokenized, and kept if they fit
    max_prompt_tokens; the first eval_prompts_per_domain are held out. Training prompts are
    interleaved across domains, so every prefix of the shards stays 1:1:1.
    """
    api, fs = HfApi(), HfFileSystem()
    revision = cfg.source_revision or api.dataset_info(cfg.source).sha
    files = [
        f
        for f in api.list_repo_files(cfg.source, repo_type="dataset", revision=revision)
        if f.startswith("data/") and f.endswith(".parquet")
    ]
    need = cfg.samples_per_domain + cfg.eval_prompts_per_domain
    selected: dict[str, list[dict]] = {}
    for domain in cfg.domains:
        ranked = []
        for name in sorted(f for f in files if f.split("/")[1].split("-")[0] == domain):
            with fs.open(f"datasets/{cfg.source}@{revision}/{name}", "rb") as file:
                table = pq.read_table(file, columns=["uuid", "messages"])
            heads = pc.list_slice(table.column("messages"), 0, 2).to_pylist()  # skip long replies
            for uuid, messages in zip(table.column("uuid").to_pylist(), heads, strict=True):
                turns = prompt_turns(messages)
                if turns:
                    key = hashlib.sha256(f"{cfg.seed}:{uuid}".encode()).hexdigest()
                    ranked.append((key, uuid, turns))
        ranked.sort()
        chosen: list[dict] = []
        for start in range(0, len(ranked), 4096):
            chunk = ranked[start : start + 4096]
            for (_, uuid, _), ids in zip(
                chunk, encode(tokenizer, [t for _, _, t in chunk]), strict=True
            ):
                if len(ids) <= cfg.max_prompt_tokens:
                    chosen.append({"uuid": uuid, "domain": domain, "prompt_ids": ids})
            if len(chosen) >= need:
                break
        if len(chosen) < need:
            log.warning("%s: only %d prompts available (%d requested)", domain, len(chosen), need)
        selected[domain] = chosen[:need]
        log.info("%s: %d prompts selected from %d", domain, len(selected[domain]), len(ranked))
    held_out = cfg.eval_prompts_per_domain
    eval_prompts = [p for prompts in selected.values() for p in prompts[:held_out]]
    train_lists = [prompts[held_out:] for prompts in selected.values()]
    train = [p for group in itertools.zip_longest(*train_lists) for p in group if p is not None]
    return train, eval_prompts, revision


def end_tokens(model: str, tokenizer, revision: str | None = None) -> tuple[set[int], int]:
    """(ids that end a response, the end-of-turn id <|im_end|>)."""
    eos = set(GenerationConfig.from_pretrained(model, revision=revision).eos_token_id or [])
    end_of_turn = tokenizer.convert_tokens_to_ids("<|im_end|>")
    return eos | {end_of_turn}, end_of_turn


def selected_prompts(cfg, tokenizer, bucket: Bucket, out_dir: Path) -> tuple[list, list, str]:
    """(train prompts, eval prompts, source revision), from the cached selection when it was made
    with the same settings, else sampled and cached (in the bucket too)."""
    d, path = cfg.datagen, out_dir / "prompts.parquet"
    key = {
        "source": d.source,
        "source_revision": d.source_revision,
        "domains": list(d.domains),
        "samples_per_domain": d.samples_per_domain,
        "eval_prompts_per_domain": d.eval_prompts_per_domain,
        "max_prompt_tokens": d.max_prompt_tokens,
        "seed": d.seed,
        "tokenizer": cfg.model.base,
        "tokenizer_revision": cfg.model.revision,
    }
    remote = f"data/{cfg.data.dataset}/prompts.parquet"
    if not path.exists():
        bucket.download_file(remote, path)
    if path.exists():
        table = pq.read_table(path)
        meta = json.loads((table.schema.metadata or {}).get(b"orthrus", b"{}"))
        if meta.get("key") == key:
            rows = table.to_pylist()
            splits = {"train": [], "eval": []}
            for row in rows:
                splits[row.pop("split")].append(row)
            log.info("reusing %d cached prompts (%s)", len(rows), remote)
            return splits["train"], splits["eval"], meta["revision"]
        log.info("cached prompts were selected with other settings: sampling again")
    train, eval_prompts, revision = sample_prompts(d, tokenizer)
    rows = [{**p, "split": "train"} for p in train] + [{**p, "split": "eval"} for p in eval_prompts]
    table = pa.table(
        {
            "uuid": [r["uuid"] for r in rows],
            "domain": [r["domain"] for r in rows],
            "prompt_ids": pa.array([r["prompt_ids"] for r in rows], type=pa.list_(pa.int32())),
            "split": [r["split"] for r in rows],
        }
    )
    meta = {"key": key, "revision": revision}
    pq.write_table(table.replace_schema_metadata({"orthrus": json.dumps(meta)}), path)
    bucket.upload(path, remote)
    return train, eval_prompts, revision


def build_engine(cfg, **overrides):
    """vLLM engine with the datagen settings; overrides are extra or replaced LLM arguments."""
    # The DataSphere image ships nvcc 11.8, which cannot JIT-compile FlashInfer's sampler.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM

    d = cfg.datagen
    kwargs = {
        "model": cfg.model.base,
        "revision": cfg.model.revision,
        "tokenizer_revision": cfg.model.revision,
        "dtype": "bfloat16",
        "seed": d.seed,
        "max_model_len": d.max_prompt_tokens + d.max_new_tokens,
        "gpu_memory_utilization": d.gpu_memory_utilization,
        "max_num_seqs": d.max_num_seqs,
    }
    if d.speculative:
        kwargs["speculative_config"] = dict(d.speculative)
    kwargs.update(overrides)
    return LLM(**kwargs)


def sampling_params(cfg, eos: set[int], detokenize: bool = False):
    """Greedy (paper) sampling; text is never needed, so detokenization is off by default."""
    from vllm import SamplingParams

    d = cfg.datagen
    return SamplingParams(
        temperature=d.temperature,
        max_tokens=d.max_new_tokens,
        seed=d.seed,
        stop_token_ids=sorted(eos),
        detokenize=detokenize,
    )


ENGINE_COUNTERS = (
    "vllm:num_preemptions",
    "vllm:generation_tokens",
    "vllm:prompt_tokens",
    "vllm:spec_decode_num_drafts",
    "vllm:spec_decode_num_draft_tokens",
    "vllm:spec_decode_num_accepted_tokens",
)


def engine_counters(llm) -> dict:
    """Cumulative vLLM counters (needs disable_log_stats=False), summed over label sets."""
    totals: dict = {}
    for metric in llm.get_metrics():
        if metric.name in ENGINE_COUNTERS and hasattr(metric, "value"):
            totals[metric.name] = totals.get(metric.name, 0) + metric.value
    return totals


def counter_metrics(before: dict, after: dict) -> dict:
    """Preemptions and speculative-decoding acceptance between two counter snapshots."""
    delta = {k: after.get(k, 0) - before.get(k, 0) for k in after}
    metrics = {"datagen/preemptions": delta.get("vllm:num_preemptions", 0)}
    drafts = delta.get("vllm:spec_decode_num_drafts", 0)
    if drafts:
        accepted = delta.get("vllm:spec_decode_num_accepted_tokens", 0)
        metrics["datagen/mean_acceptance_length"] = 1 + accepted / drafts
        metrics["datagen/draft_acceptance_rate"] = accepted / max(
            delta.get("vllm:spec_decode_num_draft_tokens", 0), 1
        )
    return metrics


def responses(outputs, eos: set[int], end_of_turn: int) -> tuple[list[list[int]], list[str]]:
    """Token ids and finish reasons of vLLM outputs; a finished turn always ends with an end id."""
    ids_list, reasons = [], []
    for output in outputs:
        ids, reason = list(output.outputs[0].token_ids), output.outputs[0].finish_reason
        if reason == "stop" and (not ids or ids[-1] not in eos):
            ids.append(end_of_turn)  # the turn ended: make the end-of-turn token explicit
        ids_list.append(ids)
        reasons.append(reason)
    return ids_list, reasons


def main(argv: list[str] | None = None) -> None:
    cfg = parse_args(__doc__, argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    started, d = time.time(), cfg.datagen
    bucket = Bucket(cfg.storage.bucket)
    out_dir = Path(cfg.storage.local_dir) / "data" / cfg.data.dataset
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    remote = f"data/{cfg.data.dataset}"

    watchdog = Watchdog(d.stall_minutes, "finished shard")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base, revision=cfg.model.revision)
    eos, end_of_turn = end_tokens(cfg.model.base, tokenizer, cfg.model.revision)
    train, eval_prompts, revision = selected_prompts(cfg, tokenizer, bucket, out_dir)
    with (out_dir / "eval_prompts.jsonl").open("w", encoding="utf-8") as file:
        file.writelines(json.dumps(p) + "\n" for p in eval_prompts)
    bucket.upload(out_dir / "eval_prompts.jsonl", f"{remote}/eval_prompts.jsonl")
    shards = (len(train) + d.shard_size - 1) // d.shard_size
    write_manifest(
        out_dir / "manifest.json",
        cfg,
        lock="datagen",
        kind="datagen",
        source=d.source,
        source_revision=revision,
        model=cfg.model.base,
        model_revision=cfg.model.revision or HfApi().model_info(cfg.model.base).sha,
        train_prompts=len(train),
        eval_prompts=len(eval_prompts),
        shards=shards,
    )
    bucket.upload(out_dir / "manifest.json", f"{remote}/manifest.json")

    local = {p.name: p for p in raw_dir.glob("shard-*.parquet")}
    if bucket.enabled:  # the bucket is the source of truth; finish uploads of earlier runs
        done = {Path(p).name for p in bucket.list(f"{remote}/raw")}
        for name in sorted(set(local) - done):
            bucket.upload(local[name], f"{remote}/raw/{name}")
            done.add(name)
    else:
        done = set(local)
    todo = [i for i in range(shards) if f"shard-{i:05d}.parquet" not in done]
    log.info("%d training prompts in %d shards, %d to generate", len(train), shards, len(todo))
    if not todo:
        return

    from vllm.inputs import TokensPrompt

    llm = build_engine(cfg, disable_log_stats=False)  # counters for preemptions / acceptance
    params = sampling_params(cfg, eos)
    restore_history(bucket, f"{remote}/datagen_metrics.jsonl", out_dir / "datagen_metrics.jsonl")
    tracker = Tracker(cfg, out_dir, "datagen")
    for index in todo:
        watchdog.beat()
        if d.max_hours and time.time() - started > d.max_hours * 3600:
            log.info("time budget reached; rerun to continue")
            break
        if bucket.exists(f"{remote}/STOP"):  # scripts/ds.py stop data/<dataset>
            log.warning("stop requested (%s/STOP); rerun to continue", remote)
            bucket.delete(f"{remote}/STOP")
            break
        batch = train[index * d.shard_size : (index + 1) * d.shard_size]
        start, counters = time.perf_counter(), engine_counters(llm)
        outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=p["prompt_ids"]) for p in batch], params, use_tqdm=False
        )
        seconds = time.perf_counter() - start
        response_ids, reasons = responses(outputs, eos, end_of_turn)
        name = f"shard-{index:05d}.parquet"
        table = pa.table(
            {
                "uuid": [p["uuid"] for p in batch],
                "domain": [p["domain"] for p in batch],
                "prompt_ids": pa.array([p["prompt_ids"] for p in batch], type=pa.list_(pa.int32())),
                "response_ids": pa.array(response_ids, type=pa.list_(pa.int32())),
                "finish_reason": reasons,
            }
        )
        pq.write_table(table, raw_dir / name)
        bucket.upload(raw_dir / name, f"{remote}/raw/{name}")
        generated = sum(len(r) for r in response_ids)
        tracker.log(
            {
                "datagen/output_tokens_per_second": generated / seconds,
                "datagen/mean_response_tokens": generated / len(batch),
                "datagen/truncated_fraction": reasons.count("length") / len(batch),
                "datagen/shard_minutes": seconds / 60,
                **counter_metrics(counters, engine_counters(llm)),
            },
            step=index,
        )
        tracker.progress((index + 1) / shards, f"shard {index + 1}/{shards}")
        snapshot = out_dir / ".datagen_metrics.snapshot"
        snapshot.write_bytes((out_dir / "datagen_metrics.jsonl").read_bytes())
        bucket.upload(snapshot, f"{remote}/datagen_metrics.jsonl")
        log.info(
            "%s: %d prompts, %.0f tok/s, %.1f min",
            name,
            len(batch),
            generated / seconds,
            seconds / 60,
        )
    tracker.close()


def exit_now(code: int = 0) -> None:
    """Exit without interpreter teardown, which hung on DataSphere after the last shard (vLLM's
    engine process kept the job alive with the GPU idle). Everything is uploaded by now."""
    import psutil

    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    for child in psutil.Process().children(recursive=True):
        try:
            child.kill()
        except psutil.Error:
            pass
    os._exit(code)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        exit_now(1)
    exit_now()
