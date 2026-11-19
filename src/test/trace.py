"""Record greedy decoding blocks for notebooks. Hooks are removed even if generation fails."""

import torch


@torch.inference_mode()
def trace_orthrus(model, input_ids, max_new_tokens=128, eos_token_id=None, top_k=4):
    blocks = []

    def capture(module, args, kwargs, result):
        positions = kwargs.get("position_ids")
        if positions is None:
            return  # AR prefill
        if kwargs.get("is_diffusion_pass", False):
            logits = result.logits[0, :-1].float()
            log_probs = logits.log_softmax(-1)
            top, ids = log_probs.topk(min(top_k, logits.shape[-1]), dim=-1)
            blocks.append(
                {
                    "start": int(positions[0, 0]),
                    "anchor": int(kwargs["input_ids"][0, 0]),
                    "draft": logits.argmax(-1).cpu().tolist(),
                    "top_ids": ids.cpu().tolist(),
                    "top_probs": top.exp().cpu().tolist(),
                }
            )
        else:
            block = blocks[-1]
            if kwargs["input_ids"][0].tolist() != [block["anchor"], *block["draft"]]:
                raise RuntimeError("unexpected draft/verification alignment")
            block["ar"] = result.logits[0].argmax(-1).cpu().tolist()
            for j, (draft, expected) in enumerate(zip(block["draft"], block["ar"])):
                if draft != expected:
                    log_probs = result.logits[0, j].float().log_softmax(-1)
                    block.update(
                        p_rejected=float(log_probs[draft].exp()),
                        p_expected=float(log_probs[expected].exp()),
                        ratio=float((log_probs[draft] - log_probs[expected]).exp()),
                    )
                    break

    was_training = model.training
    model.eval()
    hook = model.register_forward_hook(capture, with_kwargs=True)
    try:
        output, stats = model.diffusion_generate(
            input_ids, max_new_tokens, temperature=0, eos_token_id=eos_token_id
        )
    finally:
        hook.remove()
        model.train(was_training)
    full_ids = output[0].cpu().tolist()
    for block, accepted in zip(blocks, stats.accepted, strict=True):
        block["accepted"] = accepted
        position = block["start"] + accepted + 1
        block["rejected"] = (
            accepted < len(block["draft"])
            and position < len(full_ids)
            and block["draft"][accepted] != block["ar"][accepted]
        )
    return full_ids, blocks, stats


def rejection_rows(blocks, full_ids, tokenizer, eos=()):
    """Overlapping heuristic flags, not mutually exclusive error classes."""
    rows = []

    def text(token):
        return tokenizer.decode([token], skip_special_tokens=False)

    for number, block in enumerate(blocks, 1):
        if not block["rejected"]:
            continue
        j = block["accepted"]
        draft = block["draft"]
        rejected, expected = draft[j], block["ar"][j]
        position = block["start"] + j + 1
        previous = draft[j - 1] if j else block["anchor"]
        following = draft[j + 1] if j + 1 < len(draft) else None
        ar_after = (
            full_ids[position + 1] if position + 1 < len(full_ids) and expected not in eos else None
        )
        candidates = block["top_ids"][j]
        rank = candidates.index(expected) + 1 if expected in candidates else None
        repetition = rejected in (previous, following)
        flags = {
            "repetition": repetition,
            "skip": ar_after is not None and rejected == ar_after,
            "first_rejected": j == 0,
            "part_of_token": (
                rejected not in tokenizer.all_special_ids
                and expected not in tokenizer.all_special_ids
                and bool(text(rejected))
                and text(rejected) != text(expected)
                and text(rejected) in text(expected)
            ),
        }
        rows.append(
            {
                "block": number,
                "position": j + 1,
                "rejected": repr(text(rejected)),
                "expected": repr(text(expected)),
                "correct_rank_top4": rank,
                "p_rejected": block["p_rejected"],
                "p_expected": block["p_expected"],
                "ratio": block["ratio"],
                **flags,
                "second_is_correct": repetition and rank == 2,
                "other": not any(flags.values()),
            }
        )
    return rows
