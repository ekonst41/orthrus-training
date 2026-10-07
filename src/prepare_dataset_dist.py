import argparse
import json
import os
import shutil
from datetime import timedelta
from itertools import islice
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

CONFIG_PATH = Path("/home/konstantinova.e87/yandex_contest/orthrus/orthrus_repo/configs/train_qwen3.yaml")

def merge_shards(config, world_size):
    prep = config["preparation"]
    train_jsonl = Path(prep["train_jsonl"])
    packed_jsonl = Path(prep["packed_jsonl"])
    packed_cache = Path(config["data"]["packed_cache_path"])
    sources = {}
    for target in (train_jsonl, packed_jsonl):
        sources[target] = [
            target.with_name(f"{target.stem}_rank{rank}{target.suffix}")
            for rank in range(world_size)
        ]
        for source in sources[target]:
            if not source.is_file():
                raise FileNotFoundError(source)
    if packed_cache.exists():
        raise FileExistsError(
            f"Cache already exists: {packed_cache}. Choose a new packed_cache_path."
        )
    for target, shards in sources.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as output:
            for shard in shards:
                with shard.open(encoding="utf-8") as source:
                    shutil.copyfileobj(source, output)
    dataset = load_dataset("json", data_files=str(packed_jsonl), split="train")
    if not len(dataset):
        raise ValueError("No full packed rows across shards.")
    dataset.save_to_disk(str(packed_cache))
    print(f"Merged {world_size} shards: {len(dataset)} packed rows -> {packed_cache}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--merge-shards", type=int, default=0)
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if args.merge_shards:
        if args.merge_shards < 1:
            raise ValueError("--merge-shards must be positive.")
        merge_shards(config, args.merge_shards)
        return
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        # CPU synchronization only: each GPU generates independently.
        dist.init_process_group(
            backend="gloo",
            timeout=timedelta(hours=24),
        )
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    prep = config["preparation"]
    model_dir = config["model"]["model_dir"]
    source = prep["source"]
    domains = prep["domains"]
    n_per_domain = prep["n_per_domain"]
    max_new_tokens = prep["max_new_tokens"]
    seq_len = config["training"]["seq_len"]
    train_jsonl = Path(prep["train_jsonl"])
    packed_jsonl = Path(prep["packed_jsonl"])
    packed_cache = Path(config["data"]["packed_cache_path"])
    if world_size > 1:
        train_jsonl = train_jsonl.with_name(
            f"{train_jsonl.stem}_rank{rank}{train_jsonl.suffix}"
        )
        packed_jsonl = packed_jsonl.with_name(
            f"{packed_jsonl.stem}_rank{rank}{packed_jsonl.suffix}"
        )
    if packed_cache.exists():
        raise FileExistsError(f"Cache already exists: {packed_cache}. Choose a new path.")
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[config["training"]["dtype"]]
    seed = config["training"]["seed"]
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=dtype, attn_implementation=config["model"]["attn_implementation"],
    ).to(device).eval()
    streams = {
        # All ranks shuffle identically; islice selects disjoint positions.
        domain: islice(iter(load_dataset(
            source, split=domain, streaming=True, token=True,
        ).shuffle(seed=seed, buffer_size=prep["shuffle_buffer_size"])),
            rank, n_per_domain, world_size)
        for domain in domains
    }

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    batch_size = int(prep.get("batch_size", 8))
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    train_jsonl.parent.mkdir(parents=True, exist_ok=True)
    packed_jsonl.parent.mkdir(parents=True, exist_ok=True)
    ids_buffer, mask_buffer = [], []
    packed_rows = 0

    n_examples = len(range(rank, n_per_domain, world_size))

    with train_jsonl.open("w", encoding="utf-8") as train_file, \
        packed_jsonl.open("w", encoding="utf-8") as packed_file:

        with tqdm(
            total=n_examples * len(domains),
            desc=f"Generating on {model.device}",
            unit="answer",
            position=local_rank,
        ) as progress:

            for start in range(0, n_examples, batch_size):
                current_batch_size = min(batch_size, n_examples - start)

                for domain, stream in streams.items():
                    batch_messages = []
                    batch_prompt_ids = []

                    for _ in range(current_batch_size):
                        messages = []

                        for message in next(stream)["messages"]:
                            if message["role"] == "assistant":
                                break
                            messages.append({
                                "role": message["role"],
                                "content": message["content"],
                            })

                        encoded = tokenizer.apply_chat_template(
                            messages,
                            tokenize=True,
                            add_generation_prompt=True,
                            enable_thinking=prep["enable_thinking"],
                            return_dict=True,
                        )

                        prompt_ids = encoded["input_ids"]

                        batch_messages.append(messages)
                        batch_prompt_ids.append(prompt_ids)

                    inputs = tokenizer.pad(
                        {
                            "input_ids": batch_prompt_ids,
                            "attention_mask": [
                                [1] * len(ids) for ids in batch_prompt_ids
                            ],
                        },
                        padding=True,
                        return_tensors="pt",
                    ).to(model.device)

                    with torch.inference_mode():
                        generated = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            num_return_sequences=1,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )

                    input_width = inputs["input_ids"].shape[1]
                    batch_answers = generated[:, input_width:].tolist()

                    for messages, prompt_ids, answer_ids in zip(
                        batch_messages,
                        batch_prompt_ids,
                        batch_answers,
                    ):
                        if tokenizer.eos_token_id in answer_ids:
                            end = answer_ids.index(tokenizer.eos_token_id)
                            answer_ids = answer_ids[:end + 1]

                        ended_with_eos = (
                            bool(answer_ids)
                            and answer_ids[-1] == tokenizer.eos_token_id
                        )

                        text_ids = (
                            answer_ids[:-1] if ended_with_eos else answer_ids
                        )
                        answer = tokenizer.decode(
                            text_ids,
                            skip_special_tokens=False,
                        )

                        train_file.write(json.dumps({
                            "domain": domain,
                            "messages": messages + [{
                                "role": "assistant",
                                "content": answer,
                            }],
                            "truncated": not ended_with_eos,
                        }, ensure_ascii=False) + "\n")

                        packed_answer_ids = list(answer_ids)
                        if not ended_with_eos:
                            packed_answer_ids.append(tokenizer.eos_token_id)

                        ids_buffer.extend(prompt_ids + packed_answer_ids)
                        mask_buffer.extend(
                            [0] * len(prompt_ids)
                            + [1] * len(packed_answer_ids)
                        )

                        while len(ids_buffer) >= seq_len:
                            packed_file.write(json.dumps({
                                "input_ids": ids_buffer[:seq_len],
                                "assistant_mask": mask_buffer[:seq_len],
                            }) + "\n")

                            del ids_buffer[:seq_len]
                            del mask_buffer[:seq_len]
                            packed_rows += 1

                    train_file.flush()
                    packed_file.flush()
                    progress.update(current_batch_size)

    print(f"Rank {rank}: saved {len(domains) * n_examples} conversations to {train_jsonl}")
    print(f"Rank {rank}: {packed_rows} packed rows; dropped {len(ids_buffer)} trailing tokens", flush=True)

    # Generation is finished; release GPU memory before building the shared cache.
    del model
    torch.cuda.empty_cache()
    if world_size > 1:
        # Every rank has closed its output files before rank 0 starts merging.
        dist.barrier()
        status = [None]
        if rank == 0:
            try:
                merge_shards(config, world_size)
            except Exception as error:
                status[0] = f"{type(error).__name__}: {error}"
        # Keep other ranks alive until merging finishes and propagate errors.
        dist.broadcast_object_list(status, src=0)
        if status[0] is not None:
            raise RuntimeError(f"Automatic dataset merge failed: {status[0]}")
    else:
        if not packed_rows:
            raise ValueError("No full packed rows: increase n_per_domain or reduce seq_len.")
        load_dataset("json", data_files=str(packed_jsonl), split="train").save_to_disk(
            str(packed_cache)
        )
        print(f"Saved {packed_rows} packed rows to {packed_cache}")


if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
