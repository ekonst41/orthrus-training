"""Select balanced Nemotron prompts and generate token IDs with the frozen AR model."""

import hashlib
import itertools
import json
import logging
import signal
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem
from transformers import AutoTokenizer, GenerationConfig

from src.config import DatagenConfig, parse_args
from src.dataset.data import file_sha256
from src.hub import Bucket

log = logging.getLogger(__name__)


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
    count = min(len(prompts) - held_out for prompts in selected.values())
    if count < 1:
        raise ValueError("not enough prompts after reserving evaluation examples")
    train_lists = [prompts[held_out : held_out + count] for prompts in selected.values()]
    train = [p for group in itertools.zip_longest(*train_lists) for p in group if p is not None]
    return train, eval_prompts, revision


def end_tokens(model: str, tokenizer, revision: str | None = None) -> tuple[set[int], int]:
    """(ids that end a response, the end-of-turn id <|im_end|>)."""
    ids = GenerationConfig.from_pretrained(model, revision=revision).eos_token_id
    eos = set(ids if isinstance(ids, (list, tuple)) else [ids] if ids is not None else [])
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


def write_shard(path, prompts, llm, params, eos, end_of_turn, batch_size):
    from vllm.inputs import TokensPrompt

    token_ids = pa.list_(pa.int32())
    schema = pa.schema(
        [
            ("uuid", pa.array([p["uuid"] for p in prompts]).type),
            ("domain", pa.string()),
            ("prompt_ids", token_ids),
            ("response_ids", token_ids),
            ("finish_reason", pa.string()),
        ]
    )
    tmp = path.with_suffix(".tmp")
    with pq.ParquetWriter(tmp, schema) as writer:
        for batch in itertools.batched(prompts, batch_size):
            requests = [TokensPrompt(prompt_token_ids=p["prompt_ids"]) for p in batch]
            rows = []
            for prompt, output in zip(batch, llm.generate(requests, params), strict=True):
                result = output.outputs[0]
                ids = list(result.token_ids)
                if result.finish_reason == "stop" and (not ids or ids[-1] not in eos):
                    ids.append(end_of_turn)
                rows.append({**prompt, "response_ids": ids, "finish_reason": result.finish_reason})
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))
    tmp.replace(path)


def main(argv=None):
    cfg = parse_args(__doc__, argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    bucket = Bucket(cfg.hf.bucket, cfg.hf.upload)
    out = cfg.dataset_path
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    remote = f"data/{cfg.data.dataset}"
    manifest_path = out / "manifest.json"
    if not manifest_path.exists():
        bucket.download_file(f"{remote}/manifest.json", manifest_path)
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if not Path(cfg.model.base).exists():
        cfg.model.revision = cfg.model.revision or HfApi().model_info(cfg.model.base).sha
    cfg.datagen.source_revision = (
        cfg.datagen.source_revision or HfApi().dataset_info(cfg.datagen.source).sha
    )
    d = cfg.datagen
    recipe = {
        "model": cfg.model.base,
        "model_revision": cfg.model.revision,
        "source": d.source,
        "source_revision": d.source_revision,
        "domains": list(d.domains),
        "samples_per_domain": d.samples_per_domain,
        "eval_prompts_per_domain": d.eval_prompts_per_domain,
        "max_prompt_tokens": d.max_prompt_tokens,
        "max_new_tokens": d.max_new_tokens,
        "shard_size": d.shard_size,
        "seed": d.seed,
        "thinking": False,
        "temperature": 0.0,
    }
    if previous and previous.get("generation_recipe") != recipe:
        raise ValueError(
            "dataset settings changed (or legacy manifest): use a new data.dataset name"
        )
    if list(raw.glob("*.parquet")) and not previous:
        raise ValueError("raw shards without a manifest: use a new data.dataset name")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base, revision=cfg.model.revision)
    eos, end_of_turn = end_tokens(cfg.model.base, tokenizer, cfg.model.revision)
    train, heldout, revision = selected_prompts(cfg, tokenizer, bucket, out)
    (out / "eval_prompts.jsonl").write_text("".join(json.dumps(p) + "\n" for p in heldout))
    shards = (len(train) + d.shard_size - 1) // d.shard_size
    manifest = {
        "generation_recipe": recipe,
        "model": cfg.model.base,
        "model_revision": cfg.model.revision,
        "source_revision": revision,
        "train_prompts": len(train),
        "eval_prompts": len(heldout),
        "shards": shards,
        "complete": False,
        "files": previous.get("files", {}),
    }

    def save_manifest():
        tmp = manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, indent=2))
        tmp.replace(manifest_path)
        bucket.upload(manifest_path, f"{remote}/manifest.json")

    save_manifest()
    bucket.upload(out / "eval_prompts.jsonl", f"{remote}/eval_prompts.jsonl")
    # Existing completed shards are reused and locally missing shards are downloaded.
    for name in bucket.list(f"{remote}/raw"):
        path = raw / Path(name).name
        if not path.exists():
            bucket.download_file(name, path)
    todo = []
    for index in range(shards):
        path = raw / f"shard-{index:05d}.parquet"
        if path.exists():
            expected = manifest["files"].get(path.name)
            if expected and file_sha256(path) != expected:
                raise RuntimeError(f"checksum mismatch: {path}")
            # Retry any upload interrupted after a local save.
            bucket.upload(path, f"{remote}/raw/{path.name}")
            manifest["files"][path.name] = file_sha256(path)
        else:
            todo.append(index)
    if todo:
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=cfg.model.base,
            revision=cfg.model.revision,
            tokenizer_revision=cfg.model.revision,
            dtype="bfloat16",
            seed=d.seed,
            max_model_len=d.max_prompt_tokens + d.max_new_tokens,
            gpu_memory_utilization=d.gpu_memory_utilization,
            max_num_seqs=d.max_num_seqs,
        )
        params = SamplingParams(
            temperature=0,
            max_tokens=d.max_new_tokens,
            seed=d.seed,
            stop_token_ids=sorted(eos),
            detokenize=False,
        )
        stopping = []

        def request_stop(signum, frame):
            stopping.append(signum)

        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, request_stop)
        for index in todo:
            if stopping:
                break
            shard = train[index * d.shard_size : (index + 1) * d.shard_size]
            path = raw / f"shard-{index:05d}.parquet"
            write_shard(path, shard, llm, params, eos, end_of_turn, 4 * d.max_num_seqs)
            bucket.upload(path, f"{remote}/raw/{path.name}")
            manifest["files"][path.name] = file_sha256(path)
            save_manifest()
            log.info("shard %d/%d saved: %d examples", index + 1, shards, len(shard))
    manifest["complete"] = len(manifest["files"]) == shards
    save_manifest()
    log.info("dataset %s: complete=%s", out, manifest["complete"])


if __name__ == "__main__":
    main()
