import json
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

CONFIG_PATH = Path("/home/konstantinova.e87/yandex_contest/orthrus/orthrus_repo/configs/train_qwen3.yaml")

def main():
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
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
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[config["training"]["dtype"]]
    seed = config["training"]["seed"]
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=dtype, attn_implementation=config["model"]["attn_implementation"],
    ).to("cuda").eval()
    streams = {
        domain: iter(load_dataset(
            source, split=domain, streaming=True, token=True,
        ).shuffle(seed=seed, buffer_size=prep["shuffle_buffer_size"]))
        for domain in domains
    }

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    batch_size = int(prep.get("batch_size", 8))

    train_jsonl.parent.mkdir(parents=True, exist_ok=True)
    packed_jsonl.parent.mkdir(parents=True, exist_ok=True)
    ids_buffer, mask_buffer = [], []
    packed_rows = 0

    n_examples = n_per_domain

    with train_jsonl.open("w", encoding="utf-8") as train_file, \
        packed_jsonl.open("w", encoding="utf-8") as packed_file:

        with tqdm(
            total=n_examples * len(domains),
            desc=f"Generating on {model.device}",
            unit="answer",
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

    if not packed_rows:
        raise ValueError("No full packed rows: increase n_per_domain or reduce seq_len.")
    load_dataset("json", data_files=str(packed_jsonl), split="train").save_to_disk(
        str(packed_cache)
    )
    print(f"Saved {len(domains) * n_per_domain} conversations to {train_jsonl}")
    print(f"Saved {packed_rows} packed rows to {packed_cache}; dropped {len(ids_buffer)} trailing tokens")


if __name__ == "__main__":
    main()
