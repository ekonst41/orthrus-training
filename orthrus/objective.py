"""Orthrus training objective (paper, Sec. 3.2): dual-pass block masking, forward KL to AR view.

For each packed row the frozen AR view runs once over the clean sequence and keeps its keys and
values. Anchor blocks (anchor token + K-1 <mask> tokens) then run through the diffusion view against
them, and every masked position is distilled to the AR distribution at the same position:
    KL( p_AR(. | x_<=a+i) || p_diff(. | x_<a, block) ),  i = 0 .. K-2
(the last block position has no target inside the block, as in the official code and inference).

Speed (same math, tests compare against the module path `reference_states`):
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
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from orthrus.modeling_orthrus import OrthrusLM, build_block_mask, diffusion_attention

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


# --- Both views as functions of the weights (Qwen3 math, as in modeling_orthrus.py) -------------


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


def reference_states(
    model: OrthrusLM, input_ids: torch.Tensor, anchors: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """model_states through the modules of modeling_orthrus.py (as at inference): the reference."""
    block_size, mask_token_id = model.config.block_size, model.config.mask_token_id
    seq_len = input_ids.shape[1]
    with torch.no_grad():
        teacher = model.model(input_ids=input_ids, use_cache=True)
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
    return student, teacher.last_hidden_state


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


def kl_chunk(teacher_logits, student_logits, index, grad_dtype: torch.dtype | None):
    """KL(p_teacher[index[r]] || p_student[r]) summed over the rows r of one chunk, top-1
    agreement per row and, if grad_dtype is set, p_student - p_teacher (the gradient of the KL
    sum w.r.t. the student logits). One fused pass per tensor under torch.compile."""
    log_pt = teacher_logits.log_softmax(-1)[index]
    log_ps = student_logits.log_softmax(-1)
    p_t = log_pt.exp()
    kl = (p_t * (log_pt - log_ps)).sum()
    agree = student_logits.argmax(-1) == teacher_logits.argmax(-1)[index]
    diff = (log_ps.exp() - p_t).to(grad_dtype) if grad_dtype is not None else None
    return kl, agree, diff


_compiled_kl_chunk = torch.compile(kl_chunk)  # static vocabulary; row counts marked dynamic


class FusedLinearForwardKL(torch.autograd.Function):
    """sum_r KL(softmax(W teacher[index[r]]) || softmax(W student[r])) over student rows.

    index maps every student row to its teacher row and must be non-decreasing, so each chunk of
    student rows needs one contiguous slice of teacher rows: a teacher row shared by several
    blocks is projected once (twice at most, on a chunk boundary). The gradient w.r.t. the
    student rows, (p_student - p_teacher) @ W, is computed in the forward pass, so vocabulary
    logits are never stored for backward. Also returns per-row top-1 agreement.
    """

    @staticmethod
    def forward(ctx, student, teacher, index, weight, chunk_size: int, compiled: bool):
        rows = student.shape[0]
        total = torch.zeros((), dtype=torch.float32, device=student.device)
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
            inputs = (
                _logits(teacher[first : last + 1], weight),
                _logits(student[start:end], weight),
                index[start:end] - first,
            )
            if chunk is not kl_chunk:  # one graph for all chunks, specialized to the vocabulary
                for tensor in inputs:
                    torch._dynamo.maybe_mark_dynamic(tensor, 0)
            kl, agree[start:end], diff = chunk(*inputs, weight.dtype if grad is not None else None)
            total += kl
            if grad is not None:
                grad[start:end] = (diff @ weight).to(grad.dtype)
        ctx.save_for_backward(grad)
        ctx.mark_non_differentiable(agree)
        return total, agree

    @staticmethod
    def backward(ctx, grad_total, _grad_agree):
        (grad,) = ctx.saved_tensors
        return grad * grad_total.to(grad.dtype), None, None, None, None, None


def orthrus_loss(
    model: OrthrusLM,
    input_ids: torch.Tensor,
    assistant_mask: torch.Tensor,
    anchors: torch.Tensor,
    valid: torch.Tensor,
    kl_chunk_size: int,
    compiled: bool = False,
    kernel_options: dict | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Mean forward KL over supervised block positions of a micro-batch (official: token mean).

    Returns the loss and detached statistics: supervised token count, top-1 agreement with the
    teacher (overall and per block offset) and the fraction of valid anchor slots.
    """
    block_size = model.config.block_size
    batch, seq_len = input_ids.shape
    student, teacher = model_states(model, input_ids, anchors, compiled, kernel_options)
    hidden = student.shape[-1]
    student = student.view(batch, -1, block_size, hidden)[:, :, :-1]
    keep = supervision_mask(assistant_mask, anchors, valid, block_size)

    # Block position i of anchor a is distilled to the AR distribution at a+i. Blocks overlap, so
    # the student rows are sorted by that position and each position is projected once.
    targets = anchors[:, :, None] + torch.arange(block_size - 1, device=anchors.device)
    targets = targets + seq_len * torch.arange(batch, device=anchors.device)[:, None, None]
    positions, index = torch.unique(targets[keep], return_inverse=True)
    order = torch.argsort(index, stable=True)
    total, agree = FusedLinearForwardKL.apply(
        student[keep][order],
        teacher.flatten(0, 1)[positions].to(student.dtype),
        index[order],
        model.lm_head.weight,
        kl_chunk_size,
        compiled,
    )
    count = keep.sum()
    offsets = torch.arange(block_size - 1, device=keep.device).expand_as(keep)[keep][order]
    stats = {
        "tokens": count.detach(),
        "kl_sum": total.detach(),
        "agree": agree.sum(),
        "agree_by_offset": torch.bincount(offsets[agree], minlength=block_size - 1),
        "tokens_by_offset": torch.bincount(offsets, minlength=block_size - 1),
        "valid_anchors": valid.float().mean(),
    }
    return total / count.clamp_min(1), stats
