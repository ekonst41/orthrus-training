"""Shared draft losses and the functional Orthrus training forward pass.

For each packed row the frozen AR view runs once over the clean sequence and keeps its keys and
values. Anchor blocks (anchor token + K-1 <mask> tokens) then run through the diffusion view against
them, and every masked position is distilled to the AR distribution at the same position:
    KL( p_AR(. | x_<=a+i) || p_diff(. | x_<a, block) ),  i = 0 .. K-2
(the last block position has no target inside the block, as in the official code and inference).

Optional position weighting (ablation, off by default; the paper and official code use a plain
token mean): position i is weighted by w_i = exp(-i / gamma), so the first drafted token has
weight 1, and the loss is the weighted mean sum(w * KL) / sum(w). Same form as DFlash (arXiv
2602.06036, Sec. 4.2): an early mistake rejects the whole tail of the draft, so early positions
matter most for the acceptance length.

Speed (same math, tests compare against the module path in tests/reference.py):
- both passes run as plain functions over the modules' weights, compiled per decoder layer on GPU,
  so norms, RoPE, SwiGLU and FlexAttention fuse into one graph per layer;
- overlapping anchor blocks share target positions, so the teacher's vocabulary projection runs
  once per distinct position (~4x fewer rows with 256 blocks of 32 in 2048 tokens);
- softmax, KL, top-1 agreement and the gradient (p_student - p_teacher) are one fused pass per
  chunk of rows, so vocabulary-sized tensors are read a few times instead of ~15.
"""

import logging
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from torch.utils.checkpoint import checkpoint as checkpoint_fn
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from src.model import OrthrusLM, build_block_mask, diffusion_attention
from src.model_dflash import DFlashLM

log = logging.getLogger(__name__)


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


def position_weights(block_size: int, gamma: float, device=None) -> torch.Tensor | None:
    """Loss weight of block position i = 0 .. K-2 (the token i+1 after the anchor): exp(-i/gamma),
    DFlash's w_k = exp(-(k-1)/gamma) with k = i+1. None for gamma <= 0 (uniform, paper)."""
    if gamma <= 0:
        return None
    return torch.exp(-torch.arange(block_size - 1, device=device, dtype=torch.float32) / gamma)


# --- Both views as functions of the weights (Qwen3 math, as in model.py) -------------


class Attention(NamedTuple):
    q: torch.Tensor
    q_bias: torch.Tensor | None
    k: torch.Tensor
    k_bias: torch.Tensor | None
    v: torch.Tensor
    v_bias: torch.Tensor | None
    o: torch.Tensor
    o_bias: torch.Tensor | None
    q_norm: torch.Tensor
    k_norm: torch.Tensor


class Layer(NamedTuple):
    input_norm: torch.Tensor
    post_norm: torch.Tensor
    ar: Attention
    diff: Attention
    mlp: tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # gate, up, down


def layer_weights(layer) -> Layer:
    attn, mlp = layer.self_attn, layer.mlp

    def view(suffix: str) -> Attention:
        q, k, v, o = (getattr(attn, f"{n}_proj{suffix}") for n in "qkvo")
        q_norm, k_norm = getattr(attn, f"q_norm{suffix}"), getattr(attn, f"k_norm{suffix}")
        return Attention(
            q.weight, q.bias, k.weight, k.bias, v.weight, v.bias, o.weight, o.bias,
            q_norm.weight, k_norm.weight,
        )  # fmt: skip

    return Layer(
        layer.input_layernorm.weight,
        layer.post_attention_layernorm.weight,
        view(""),
        view("_diff"),
        (mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight),
    )


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen3RMSNorm."""
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return weight * x.to(dtype)


def _attention_inputs(h, w: Attention, cos, sin, eps: float, head_dim: int):
    """Queries, keys, values [batch, heads, tokens, head_dim] after q/k norms and RoPE."""
    shape = (*h.shape[:-1], -1, head_dim)
    q = _rms_norm(F.linear(h, w.q, w.q_bias).view(shape), w.q_norm, eps).transpose(1, 2)
    k = _rms_norm(F.linear(h, w.k, w.k_bias).view(shape), w.k_norm, eps).transpose(1, 2)
    v = F.linear(h, w.v, w.v_bias).view(shape).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    return q, k, v


def _mlp_residual(x, w: Layer, eps: float):
    gate, up, down = w.mlp
    h = _rms_norm(x, w.post_norm, eps)
    return x + F.linear(F.silu(F.linear(h, gate)) * F.linear(h, up), down)


def ar_layer(x, cos, sin, w: Layer, eps: float, head_dim: int):
    """Frozen causal decoder layer; also returns its keys and values (the AR cache)."""
    q, k, v = _attention_inputs(_rms_norm(x, w.input_norm, eps), w.ar, cos, sin, eps, head_dim)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    x = x + F.linear(out.transpose(1, 2).flatten(2), w.ar.o, w.ar.o_bias)
    return _mlp_residual(x, w, eps), k, v


def diffusion_layer(x, cos, sin, ar_k, ar_v, block_mask, w: Layer, eps, head_dim, attention):
    """Diffusion decoder layer: twin attention over cat([AR cache, block keys]), shared MLP."""
    q, k, v = _attention_inputs(_rms_norm(x, w.input_norm, eps), w.diff, cos, sin, eps, head_dim)
    out = attention(q, torch.cat([ar_k, k], dim=2), torch.cat([ar_v, v], dim=2), block_mask)
    x = x + F.linear(out.transpose(1, 2).flatten(2), w.diff.o, w.diff.o_bias)
    return _mlp_residual(x, w, eps)


def skip_ar_key_grads(mask: BlockMask, ar_len: int) -> BlockMask:
    """The same mask, but the backward skips dK/dV of the AR keys: they come from the frozen
    teacher pass and get no gradient, yet are almost all of the dK/dV work (every block attends
    its whole prefix). The dK/dV programs of a key block iterate over its q-block lists, so
    emptying them for the AR key blocks makes those programs no-ops; the forward and dQ read the
    kv-block lists, which are unchanged. Gradients that are used stay bit-identical."""
    ar_blocks = ar_len // mask.BLOCK_SIZE[1]

    def emptied(counts):
        if counts is None:
            return None
        counts = counts.clone()
        counts[..., :ar_blocks] = 0
        return counts

    return BlockMask(
        mask.seq_lengths,
        mask.kv_num_blocks,
        mask.kv_indices,
        mask.full_kv_num_blocks,
        mask.full_kv_indices,
        emptied(mask.q_num_blocks),
        mask.q_indices,
        emptied(mask.full_q_num_blocks),
        mask.full_q_indices,
        BLOCK_SIZE=mask.BLOCK_SIZE,
        mask_mod=mask.mask_mod,
    )


SKIP_AR_KEY_GRADS = True  # switch for benchmarks; the result is the same either way
_LAYER_FUNCTIONS: dict = {}


def layer_functions(compiled: bool, kernel_options: tuple = ()):
    """(ar_layer, diffusion_layer), compiled once per setting: every decoder layer reuses the graph,
    since the weights are inputs. kernel_options: FlexAttention kernel settings, as (key, value)
    pairs."""
    key = (compiled, kernel_options)
    if key not in _LAYER_FUNCTIONS:
        if compiled:
            options = dict(kernel_options) or None

            def attention(q, k, v, block_mask):
                return flex_attention(
                    q, k, v, block_mask=block_mask, enable_gqa=True, kernel_options=options
                )
        else:
            attention = diffusion_attention  # separately compiled flex on GPU, dense SDPA on CPU

        def diffusion(x, cos, sin, ar_k, ar_v, block_mask, w, eps, head_dim):
            return diffusion_layer(x, cos, sin, ar_k, ar_v, block_mask, w, eps, head_dim, attention)

        functions = (ar_layer, diffusion)
        if compiled:
            functions = tuple(torch.compile(f, dynamic=False) for f in functions)
        _LAYER_FUNCTIONS[key] = functions
    return _LAYER_FUNCTIONS[key]


def model_states(
    model: OrthrusLM,
    input_ids: torch.Tensor,
    anchors: torch.Tensor,
    compiled: bool = False,
    kernel_options: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Final hidden states of the diffusion view over all anchor blocks [batch, slots*K, hidden]
    and of the frozen AR view over the clean rows [batch, seq_len, hidden]."""
    config, backbone = model.config, model.model
    if any(t != "full_attention" for t in config.layer_types[: config.num_hidden_layers]):
        raise ValueError("model_states supports full attention layers only (no sliding window)")
    eps, head_dim = config.rms_norm_eps, backbone.layers[0].self_attn.head_dim
    block_size = config.block_size
    batch, seq_len = input_ids.shape
    ar, diffusion = layer_functions(
        compiled and input_ids.is_cuda, tuple(sorted((kernel_options or {}).items()))
    )
    weights = [layer_weights(layer) for layer in backbone.layers[: config.num_hidden_layers]]

    with torch.no_grad():  # teacher: one causal pass over the clean rows, keeps keys and values
        x = backbone.embed_tokens(input_ids)
        cos, sin = backbone.rotary_emb(x, torch.arange(seq_len, device=x.device)[None])
        cache = []
        for w in weights:
            x, k, v = ar(x, cos, sin, w, eps, head_dim)
            cache.append((k, v))
        teacher = backbone.norm(x)

    positions = anchors[:, :, None] + torch.arange(block_size, device=anchors.device)
    block_ids = torch.full_like(positions, config.mask_token_id)
    block_ids[:, :, 0] = input_ids.gather(1, anchors)
    x = backbone.embed_tokens(block_ids.flatten(1))
    cos, sin = backbone.rotary_emb(x, positions.flatten(1))
    block_mask = build_block_mask(anchors, seq_len, block_size)
    if SKIP_AR_KEY_GRADS:
        block_mask = skip_ar_key_grads(block_mask, seq_len)
    checkpoint = model.is_gradient_checkpointing and torch.is_grad_enabled()
    for w, (k, v) in zip(weights, cache, strict=True):
        args = (x, cos, sin, k, v, block_mask, w, eps, head_dim)
        if checkpoint:
            x = torch.utils.checkpoint.checkpoint(diffusion, *args, use_reentrant=False)
        else:
            x = diffusion(*args)
    return backbone.norm(x), teacher


# --- Fused vocabulary projection + forward KL --------------------------------------------------

_FP32_OUTPUT = True  # cleared if this torch/GPU lacks bf16 -> fp32 matmul output


def _logits(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Vocabulary logits in fp32. On GPU bf16 inputs accumulate and land in fp32 directly (no bf16
    output rounding of the distillation targets); CPU tests run in fp32 anyway."""
    global _FP32_OUTPUT
    if _FP32_OUTPUT and hidden.is_cuda and hidden.dtype in (torch.bfloat16, torch.float16):
        try:
            return torch.mm(hidden, weight.t(), out_dtype=torch.float32)
        except (RuntimeError, NotImplementedError) as error:
            _FP32_OUTPUT = False
            log.warning("fp32-output matmul unavailable (%s): using bf16 logits", error)
    return (hidden @ weight.t()).float()


def kl_chunk(teacher_logits, student_logits, index, row_weight, grad_dtype: torch.dtype | None):
    """KL(p_teacher[index[r]] || p_student[r]) for every row r of one chunk, top-1 agreement per
    row and, if grad_dtype is set, w_r * (p_student - p_teacher) (the gradient of the weighted KL
    sum w.r.t. the student logits; w_r = 1 if row_weight is None). One fused pass per tensor
    under torch.compile."""
    log_pt = teacher_logits.log_softmax(-1)[index]
    log_ps = student_logits.log_softmax(-1)
    p_t = log_pt.exp()
    kl = (p_t * (log_pt - log_ps)).sum(-1)
    agree = student_logits.argmax(-1) == teacher_logits.argmax(-1)[index]
    diff = None
    if grad_dtype is not None:
        diff = log_ps.exp() - p_t
        if row_weight is not None:
            diff = diff * row_weight[:, None]
        diff = diff.to(grad_dtype)
    return kl, agree, diff


_compiled_kl_chunk = torch.compile(kl_chunk)  # static vocabulary; row counts marked dynamic


class FusedLinearForwardKL(torch.autograd.Function):
    """sum_r w_r KL(softmax(W teacher[index[r]]) || softmax(W student[r])) over student rows
    (w_r = row_weight[r], or 1 if row_weight is None).

    index maps every student row to its teacher row and must be non-decreasing, so each chunk of
    student rows needs one contiguous slice of teacher rows: a teacher row shared by several
    blocks is projected once (twice at most, on a chunk boundary). The gradient w.r.t. the
    student rows, w_r (p_student - p_teacher) @ W, is computed in the forward pass, so vocabulary
    logits are never stored for backward. Also returns the unweighted KL and the top-1
    agreement of every row.
    """

    @staticmethod
    def forward(ctx, student, teacher, index, weight, chunk_size: int, compiled: bool, row_weight):
        rows = student.shape[0]
        total = torch.zeros((), dtype=torch.float32, device=student.device)
        kl_rows = torch.zeros(rows, dtype=torch.float32, device=student.device)
        agree = torch.zeros(rows, dtype=torch.bool, device=student.device)
        grad = torch.empty_like(student) if ctx.needs_input_grad[0] else None
        chunk = _compiled_kl_chunk if compiled and student.is_cuda else kl_chunk
        starts = list(range(0, rows, chunk_size))
        ends = [min(start + chunk_size, rows) for start in starts]
        edges = (
            index[torch.tensor(starts + [end - 1 for end in ends], dtype=torch.long)]
            if rows
            else []
        )
        edges = edges.tolist() if rows else []  # one device sync: teacher slice per chunk
        for start, end, first, last in zip(
            starts, ends, edges[: len(starts)], edges[len(starts) :], strict=True
        ):
            chunk_weight = row_weight[start:end] if row_weight is not None else None
            inputs = (
                _logits(teacher[first : last + 1], weight),
                _logits(student[start:end], weight),
                index[start:end] - first,
                chunk_weight,
            )
            if chunk is not kl_chunk:  # one graph for all chunks, specialized to the vocabulary
                for tensor in inputs:
                    if tensor is not None:
                        torch._dynamo.maybe_mark_dynamic(tensor, 0)
            kl, agree[start:end], diff = chunk(*inputs, weight.dtype if grad is not None else None)
            kl_rows[start:end] = kl
            total += (kl * chunk_weight).sum() if chunk_weight is not None else kl.sum()
            if grad is not None:
                grad[start:end] = (diff @ weight).to(grad.dtype)
            del inputs, diff  # Release vocabulary-sized tensors before projecting the next chunk.
        ctx.save_for_backward(grad)
        ctx.mark_non_differentiable(kl_rows, agree)
        return total, kl_rows, agree

    @staticmethod
    def backward(ctx, grad_total, _grad_kl_rows, _grad_agree):
        (grad,) = ctx.saved_tensors
        return grad * grad_total.to(grad.dtype), None, None, None, None, None, None


def loss_kl(student, teacher, teacher_positions, weight, chunk_size, compiled, row_weight):
    """Forward KL to a frozen teacher. Overlapping blocks share vocabulary projections."""
    positions, index = torch.unique(teacher_positions, return_inverse=True)
    order = torch.argsort(index, stable=True)
    ordered_weight = row_weight[order] if row_weight is not None else None
    total, rows, agree = FusedLinearForwardKL.apply(
        student[order],
        teacher[positions].to(student.dtype),
        index[order],
        weight,
        chunk_size,
        compiled,
        ordered_weight,
    )
    losses = torch.empty_like(rows).scatter_(0, order, rows)
    matches = torch.empty_like(agree).scatter_(0, order, agree)
    return total, losses, matches


def _ce_chunk(hidden, labels, weight):
    logits = F.linear(hidden, weight).float()
    rows = F.cross_entropy(logits, labels, reduction="none")
    return rows, logits.detach().argmax(-1).eq(labels)


def loss_ce(student, labels, weight, chunk_size, row_weight):
    """Token cross-entropy. Recompute chunk logits in backward to bound vocabulary memory."""
    if student.shape[0] == 0:
        empty = student.new_empty((0,), dtype=torch.float32)
        return student.float().sum(), empty, empty.bool()
    losses, matches = [], []
    for start in range(0, student.shape[0], chunk_size):
        end = start + chunk_size
        args = (student[start:end], labels[start:end], weight)
        if torch.is_grad_enabled() and student.requires_grad:
            rows, agree = checkpoint_fn(_ce_chunk, *args, use_reentrant=False)
        else:
            rows, agree = _ce_chunk(*args)
        losses.append(rows)
        matches.append(agree)
    rows = torch.cat(losses)
    total = (rows * row_weight).sum() if row_weight is not None else rows.sum()
    return total, rows.detach(), torch.cat(matches)


def draft_loss(
    model: OrthrusLM | DFlashLM,
    input_ids: torch.Tensor,
    assistant_mask: torch.Tensor,
    anchors: torch.Tensor,
    valid: torch.Tensor,
    kl_chunk_size: int,
    compiled: bool = False,
    kernel_options: dict | None = None,
    gamma: float = 0.0,
    loss_type: str = "kl",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Align the draft states to the same labels, then apply the selected token objective."""
    block_size = model.config.block_size
    batch, seq_len = input_ids.shape
    shape = (batch, anchors.shape[1], block_size, -1)
    if isinstance(model, DFlashLM):
        student, teacher = model.training_states(input_ids, anchors)
        student = student.reshape(shape)[:, :, 1:]
    else:
        student, teacher = model_states(model, input_ids, anchors, compiled, kernel_options)
        student = student.reshape(shape)[:, :, :-1]
    keep = supervision_mask(assistant_mask, anchors, valid, block_size)
    positions = torch.arange(block_size - 1, device=anchors.device)
    offsets = positions.expand_as(keep)[keep]
    weights = position_weights(block_size, gamma, keep.device)
    row_weight = weights[offsets] if weights is not None else None
    weight = model.lm_head.weight
    supervised = student[keep].to(weight.dtype)

    if loss_type == "kl":
        teacher_pos = anchors[:, :, None] + positions + (
            seq_len * torch.arange(batch, device=anchors.device)[:, None, None]
        )
        total, rows, agree = loss_kl(
            supervised,
            teacher.flatten(0, 1),
            teacher_pos[keep],
            weight,
            kl_chunk_size,
            compiled,
            row_weight,
        )
    elif loss_type == "ce":
        label_pos = anchors[:, :, None] + positions + 1
        labels = input_ids.gather(1, label_pos.flatten(1)).reshape_as(label_pos)[keep]
        total, rows, agree = loss_ce(supervised, labels, weight, kl_chunk_size, row_weight)
    else:
        raise ValueError(f"unknown draft loss: {loss_type}")

    count = keep.sum()
    norm = row_weight.sum() if row_weight is not None else count.float()
    stats = {
        "tokens": count.detach(),
        f"{loss_type}_sum": rows.sum(),
        f"weighted_{loss_type}_sum": total.detach(),
        "weight_sum": norm.detach(),
        "agree": agree.sum(),
        f"{loss_type}_by_offset": rows.new_zeros(block_size - 1).index_add_(0, offsets, rows),
        "agree_by_offset": torch.bincount(offsets[agree], minlength=block_size - 1),
        "tokens_by_offset": torch.bincount(offsets, minlength=block_size - 1),
        "valid_anchors": valid.float().mean(),
    }
    return total / norm.clamp_min(1e-6), stats


# Public name used by existing notebooks and tests.
orthrus_loss = draft_loss
