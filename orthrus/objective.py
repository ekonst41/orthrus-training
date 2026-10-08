"""Orthrus training objective (paper, Sec. 3.2): dual-pass block masking, forward KL to AR view.

For each packed row the frozen AR view runs once over the clean sequence and fills the KV cache.
Anchor blocks (anchor token + K-1 <mask> tokens) then run through the diffusion view against that
cache, and every masked position is distilled to the AR distribution at the same position:
    KL( p_AR(. | x_<=a+i) || p_diff(. | x_<a, block) ),  i = 0 .. K-2
(the last block position has no target inside the block, as in the official code and inference).
"""

import torch
import torch.nn.functional as F

from orthrus.modeling_orthrus import OrthrusLM, build_block_mask


def sample_anchors(
    assistant_mask: torch.Tensor, block_size: int, num_anchors: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sorted anchor positions per row, drawn without replacement from assistant tokens.

    Official code (src/utils/init_model.py): an anchor a lies on an assistant token with
    1 <= a <= seq_len - block_size. Rows with fewer candidates repeat their last anchor in the
    spare slots, marked invalid; rows without assistant tokens get arbitrary, all-invalid anchors.
    Returns (anchors, valid), both [batch, slots].
    """
    batch, seq_len = assistant_mask.shape
    slots = min(num_anchors, seq_len - block_size)
    device = assistant_mask.device
    positions = torch.arange(1, seq_len - block_size + 1, device=device)
    anchors = torch.zeros((batch, slots), dtype=torch.long, device=device)
    valid = torch.zeros((batch, slots), dtype=torch.bool, device=device)
    for row in range(batch):
        candidates = positions[assistant_mask[row, positions]]
        has_targets = candidates.numel() > 0
        if not has_targets:
            candidates = positions
        order = torch.randperm(candidates.numel(), generator=generator, device=device)
        picked = candidates[order[:slots]].sort().values
        anchors[row, : len(picked)] = picked
        anchors[row, len(picked) :] = picked[-1]
        valid[row, : len(picked)] = has_targets
    return anchors, valid


def supervision_mask(
    assistant_mask: torch.Tensor, anchors: torch.Tensor, valid: torch.Tensor, block_size: int
) -> torch.Tensor:
    """keep[b, a, i]: block position anchor+i is trained iff its target anchor+i+1 is an assistant
    token (official block_local_labels) and the anchor slot is valid. Shape [batch, slots, K-1]."""
    targets = anchors[:, :, None] + torch.arange(1, block_size, device=anchors.device)
    keep = assistant_mask.gather(1, targets.flatten(1)).view_as(targets)
    return keep & valid[:, :, None]


def diffusion_states(
    model: OrthrusLM, input_ids: torch.Tensor, anchors: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Final hidden states [batch, slots, K-1, hidden] of the diffusion view (student) and of the
    frozen AR view at the matching positions (teacher)."""
    block_size, mask_token_id = model.config.block_size, model.config.mask_token_id
    batch, seq_len = input_ids.shape
    slots = anchors.shape[1]

    training = model.training
    model.eval()
    with torch.no_grad():  # teacher: one causal pass over the clean row, keeps the KV cache
        teacher = model.model(input_ids=input_ids, use_cache=True)
    model.train(training)

    positions = anchors[:, :, None] + torch.arange(block_size, device=anchors.device)
    block_ids = torch.full_like(positions, mask_token_id)
    block_ids[:, :, 0] = input_ids.gather(1, anchors)
    student = model.model(
        input_ids=block_ids.flatten(1),
        position_ids=positions.flatten(1),
        past_key_values=teacher.past_key_values,
        use_cache=False,
        is_diffusion_pass=True,
        ar_seq_len=seq_len,
        flex_block_mask=build_block_mask(anchors, seq_len, block_size),
    ).last_hidden_state
    hidden = student.shape[-1]
    student = student.view(batch, slots, block_size, hidden)[:, :, :-1]
    index = positions[:, :, :-1].flatten(1)[..., None].expand(-1, -1, hidden)
    teacher = teacher.last_hidden_state.gather(1, index).view(batch, slots, block_size - 1, hidden)
    return student, teacher


def _logits(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Vocabulary logits in fp32. On GPU bf16 inputs accumulate and land in fp32 directly (no bf16
    output rounding of the distillation targets); CPU tests run in fp32 anyway."""
    if hidden.is_cuda and hidden.dtype in (torch.bfloat16, torch.float16):
        return torch.mm(hidden, weight.t(), out_dtype=torch.float32)
    return (hidden @ weight.t()).float()


class FusedLinearForwardKL(torch.autograd.Function):
    """sum_i KL(softmax(W h_teacher_i) || softmax(W h_student_i)) over rows, chunk by chunk.

    The gradient w.r.t. student hidden states, (p_student - p_teacher) @ W, is computed in the
    forward pass, so full-vocabulary logits are never stored or recomputed (3 matmuls per chunk
    instead of 5 with activation checkpointing). Also returns per-row top-1 agreement.
    """

    @staticmethod
    def forward(ctx, student, teacher, weight, chunk_size: int):
        total = torch.zeros((), dtype=torch.float32, device=student.device)
        agree = torch.empty(student.shape[0], dtype=torch.bool, device=student.device)
        grad = torch.empty_like(student) if ctx.needs_input_grad[0] else None
        for start in range(0, student.shape[0], chunk_size):
            rows = slice(start, start + chunk_size)
            log_pt = F.log_softmax(_logits(teacher[rows], weight), dim=-1)
            log_ps = F.log_softmax(_logits(student[rows], weight), dim=-1)
            agree[rows] = log_ps.argmax(dim=-1) == log_pt.argmax(dim=-1)
            p_t = log_pt.exp()
            total += (p_t * log_pt).sum() - (p_t * log_ps).sum()
            del log_pt
            if grad is not None:
                diff = log_ps.exp_().sub_(p_t)  # p_student - p_teacher, reusing the log_ps buffer
                grad[rows] = (diff.to(weight.dtype) @ weight).to(grad.dtype)
        ctx.save_for_backward(grad)
        ctx.mark_non_differentiable(agree)
        return total, agree

    @staticmethod
    def backward(ctx, grad_total, _grad_agree):
        (grad,) = ctx.saved_tensors
        return grad * grad_total.to(grad.dtype), None, None, None


def orthrus_loss(
    model: OrthrusLM,
    input_ids: torch.Tensor,
    assistant_mask: torch.Tensor,
    anchors: torch.Tensor,
    valid: torch.Tensor,
    kl_chunk_size: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Mean forward KL over supervised block positions of a micro-batch (official: token mean).

    Returns the loss and detached statistics: supervised token count, top-1 agreement with the
    teacher (overall and per block offset) and the fraction of valid anchor slots.
    """
    block_size = model.config.block_size
    student, teacher = diffusion_states(model, input_ids, anchors)
    keep = supervision_mask(assistant_mask, anchors, valid, block_size)
    weight = model.lm_head.weight
    total, agree = FusedLinearForwardKL.apply(
        student[keep], teacher[keep].to(student.dtype), weight, kl_chunk_size
    )
    count = keep.sum()
    offsets = torch.arange(block_size - 1, device=keep.device).expand_as(keep)[keep]
    stats = {
        "tokens": count.detach(),
        "kl_sum": total.detach(),
        "agree": agree.sum(),
        "agree_by_offset": torch.bincount(offsets[agree], minlength=block_size - 1),
        "tokens_by_offset": torch.bincount(offsets, minlength=block_size - 1),
        "valid_anchors": valid.float().mean(),
    }
    return total / count.clamp_min(1), stats
