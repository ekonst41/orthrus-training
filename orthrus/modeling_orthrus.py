# Orthrus dual-view diffusion model on a frozen Qwen3 backbone.
#
# Adapted from the official implementation https://github.com/chiennv2000/orthrus (src/model.py),
# MIT License, Copyright (c) 2026 Chien Nguyen (full text in THIRD_PARTY_NOTICES.md).
# Changes: anchor-based dual-pass mask with a compiled builder, Triton FlexAttention (no
# FlashAttention-4 on A100) with a dense SDPA fallback on CPU, cache truncation compatible with
# transformers >= 5.18, several EOS ids, generation statistics.
#
# The file is self-contained (torch + transformers only): exported checkpoints ship it as
# modeling_orthrus.py and load with AutoModelForCausalLM.from_pretrained(path,
# trust_remote_code=True).
# The config is a plain Qwen3Config with two extra fields, as in the official checkpoints:
#   block_size     size K of a parallel block (anchor token + K-1 <mask> tokens)
#   mask_token_id  id of the <mask> token (official: len(tokenizer), an unused embedding row)

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    create_mask,
    flex_attention,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    GradientCheckpointingLayer,
    Qwen3Attention,
    Qwen3Config,
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

OrthrusConfig = Qwen3Config  # plus `block_size` and `mask_token_id`, like the official checkpoints
DIFF_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "q_norm",
    "k_norm",
)  # trainable twins: *_diff

_compiled_flex_attention = torch.compile(flex_attention, dynamic=False)
_compiled_create_block_mask = torch.compile(create_block_mask, dynamic=False)


def build_block_mask(anchors: torch.Tensor, ar_len: int, block_size: int) -> BlockMask:
    """Dual-pass mask for training: diffusion queries over cat([AR keys, diffusion keys]).

    A query in block b sees AR keys strictly before its anchor anchors[:, b] and all keys of its own
    block (bidirectional), never other blocks (paper, Sec. 3.2; official src/models/*qwen3_5.py).
    """
    num_blocks = anchors.shape[1]
    q_len = num_blocks * block_size

    def mask_mod(batch, head, q_idx, kv_idx):
        block = (q_idx // block_size).clamp(max=num_blocks - 1)  # q_idx may be padded past q_len
        ar_visible = (kv_idx < ar_len) & (kv_idx < anchors[batch, block])
        same_block = (kv_idx >= ar_len) & ((kv_idx - ar_len) // block_size == q_idx // block_size)
        return ar_visible | same_block

    builder = _compiled_create_block_mask if anchors.is_cuda else create_block_mask
    return builder(mask_mod, anchors.shape[0], None, q_len, ar_len + q_len, device=anchors.device)


def diffusion_attention(q, k, v, block_mask: BlockMask) -> torch.Tensor:
    """Masked attention of the diffusion view: compiled FlexAttention on GPU, dense SDPA on CPU.

    FlexAttention has no CPU backward, so CPU runs (tests) use the same mask materialised densely.
    """
    if q.is_cuda:
        return _compiled_flex_attention(q, k, v, block_mask=block_mask, enable_gqa=True)
    dense = create_mask(block_mask.mask_mod, q.shape[0], None, q.shape[2], k.shape[2], q.device)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=dense, enable_gqa=True)


@torch.no_grad()
def copy_diff_from_ar(model: nn.Module) -> None:
    """Warm-start every diffusion twin from its frozen AR counterpart (paper, Sec. 3.1)."""
    for module in model.modules():
        if isinstance(module, OrthrusAttention):
            for name in DIFF_MODULES:
                getattr(module, f"{name}_diff").load_state_dict(getattr(module, name).state_dict())


def freeze_to_diffusion(model: nn.Module) -> list[nn.Parameter]:
    """Freeze the AR backbone; return the trainable diffusion parameters (q/k/v/o + q/k norms)."""
    trainable = []
    for name, param in model.named_parameters():
        param.requires_grad_(any(f".{m}_diff." in name for m in DIFF_MODULES))
        if param.requires_grad:
            trainable.append(param)
    return trainable


def truncate_cache(cache: DynamicCache, length: int) -> None:
    """Drop cached positions >= length (negative crop: positive values are deprecated in 5.18)."""
    excess = cache.get_seq_length() - length
    if excess > 0:
        cache.crop(-excess)


class OrthrusAttention(Qwen3Attention):
    """Qwen3 attention with a frozen autoregressive view and a trainable diffusion twin."""

    def __init__(self, config: OrthrusConfig, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        q_dim = config.num_attention_heads * self.head_dim
        kv_dim = config.num_key_value_heads * self.head_dim
        bias = config.attention_bias
        self.q_proj_diff = nn.Linear(config.hidden_size, q_dim, bias=bias)
        self.k_proj_diff = nn.Linear(config.hidden_size, kv_dim, bias=bias)
        self.v_proj_diff = nn.Linear(config.hidden_size, kv_dim, bias=bias)
        self.o_proj_diff = nn.Linear(q_dim, config.hidden_size, bias=bias)
        self.q_norm_diff = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm_diff = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        is_diffusion_pass: bool = False,
        ar_seq_len: int | None = None,
        flex_block_mask: BlockMask | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        if not is_diffusion_pass:
            return super().forward(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )
        if past_key_values is None or ar_seq_len is None:
            raise ValueError(
                "the diffusion pass needs the AR cache (past_key_values) and ar_seq_len"
            )

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query = self.q_norm_diff(self.q_proj_diff(hidden_states).view(hidden_shape)).transpose(1, 2)
        key = self.k_norm_diff(self.k_proj_diff(hidden_states).view(hidden_shape)).transpose(1, 2)
        value = self.v_proj_diff(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # Both views attend over the single shared cache written by the frozen AR view.
        shared = past_key_values.layers[self.layer_idx]
        if shared.keys.shape[2] != ar_seq_len:
            raise ValueError(
                f"AR cache holds {shared.keys.shape[2]} positions, expected {ar_seq_len}"
            )
        keys = torch.cat([shared.keys, key], dim=2)
        values = torch.cat([shared.values, value], dim=2)

        if flex_block_mask is not None:  # training / evaluation over many packed blocks
            output = diffusion_attention(query, keys, values, flex_block_mask).transpose(1, 2)
        else:  # generation: one block sees the whole committed prefix and itself
            attention = ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation, eager_attention_forward
            )
            output, _ = attention(
                self,
                query,
                keys,
                values,
                None,
                dropout=0.0,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                is_causal=False,
                **kwargs,
            )
        return self.o_proj_diff(output.reshape(*input_shape, -1).contiguous()), None


class OrthrusDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: OrthrusConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = OrthrusAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        past_key_values: Cache | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states, _ = self.self_attn(
            hidden_states=self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class OrthrusModel(Qwen3PreTrainedModel):
    _no_split_modules = ["OrthrusDecoderLayer"]

    def __init__(self, config: OrthrusConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [OrthrusDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        is_diffusion_pass: bool = False,
        ar_seq_len: int | None = None,
        flex_block_mask: BlockMask | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("pass exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if cache_position is None:
            seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                seen, seen + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if is_diffusion_pass or self.config._attn_implementation not in ("eager", "sdpa"):
            causal_mask = attention_mask
        else:
            causal_mask = create_causal_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                is_diffusion_pass=is_diffusion_pass,
                ar_seq_len=ar_seq_len,
                flex_block_mask=flex_block_mask,
                **kwargs,
            )
        return BaseModelOutputWithPast(
            last_hidden_state=self.norm(hidden_states),
            past_key_values=past_key_values if use_cache else None,
        )


@dataclass
class GenerationStats:
    """Forward-pass accounting of one diffusion-mode generation (see evaluate.py for metrics)."""

    new_tokens: int = 0
    accepted: list[int] = field(default_factory=list)  # accepted draft tokens per cycle

    @property
    def cycles(self) -> int:  # one cycle = diffusion projection + AR verification
        return len(self.accepted)

    @property
    def forward_passes(self) -> int:  # prefill + 2 passes per cycle (paper, Sec. 4.2)
        return 1 + 2 * self.cycles


class OrthrusLM(Qwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: OrthrusConfig):
        super().__init__(config)
        self.model = OrthrusModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        is_diffusion_pass: bool = False,
        ar_seq_len: int | None = None,
        flex_block_mask: BlockMask | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            is_diffusion_pass=is_diffusion_pass,
            ar_seq_len=ar_seq_len,
            flex_block_mask=flex_block_mask,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        keep = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        return CausalLMOutputWithPast(
            logits=self.lm_head(hidden_states[:, keep, :]),
            past_key_values=outputs.past_key_values,
            hidden_states=(hidden_states,),
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int | None = None,
        max_length: int | None = None,
        temperature: float = 0.0,
        top_k: int = 20,
        top_p: float = 0.8,
        eos_token_id: int | list[int] | None = None,
        streamer=None,
        use_diffusion_mode: bool = True,
        **kwargs,
    ) -> torch.LongTensor:
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        if not use_diffusion_mode:
            return super().generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=eos_token_id,
                streamer=streamer,
                use_cache=True,
                **kwargs,
            )
        max_new_tokens = max_new_tokens or (max_length - input_ids.shape[1])
        output, _ = self.diffusion_generate(
            input_ids,
            max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=eos_token_id,
            streamer=streamer,
        )
        return output

    @torch.inference_mode()
    def diffusion_generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_k: int = 20,
        top_p: float = 0.8,
        eos_token_id: int | list[int] | None = None,
        streamer=None,
    ) -> tuple[torch.LongTensor, GenerationStats]:
        """Project K tokens in parallel, verify them with the AR view, keep the agreed prefix.

        Greedy decoding keeps a token iff it equals the AR argmax; sampling uses exact rejection
        sampling (paper, Sec. 3.3). Batch size 1, as in the official implementation.
        """
        if input_ids.shape[0] != 1:
            raise ValueError("diffusion_generate supports batch size 1")
        eos = {eos_token_id} if isinstance(eos_token_id, int) else set(eos_token_id or ())
        device = input_ids.device
        block_size, mask_token_id = self.config.block_size, self.config.mask_token_id
        prompt_len = input_ids.shape[1]
        max_length = prompt_len + max_new_tokens
        stats = GenerationStats()
        cache = DynamicCache(config=self.config)
        output = torch.full(
            (1, max_length + block_size), mask_token_id, dtype=torch.long, device=device
        )
        output[:, :prompt_len] = input_ids

        def pick(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
            if temperature < 1e-5:
                return logits.argmax(dim=-1), None
            logits = logits.float() / temperature
            if top_k > 0:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[..., -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            if top_p < 1.0:
                sorted_logits, order = torch.sort(logits, descending=True)
                drop = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                drop[..., 1:] = drop[..., :-1].clone()
                drop[..., 0] = False
                logits = logits.masked_fill(drop.scatter(-1, order, drop), float("-inf"))
            probs = F.softmax(logits, dim=-1)
            tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(probs.shape[:-1])
            return tokens, probs

        def finish(end: int) -> tuple[torch.LongTensor, GenerationStats]:
            stats.new_tokens = end - prompt_len
            if streamer is not None:
                streamer.end()
            return output[:, :end], stats

        if streamer is not None:
            streamer.put(input_ids)
        prefill = self(input_ids=input_ids, past_key_values=cache, use_cache=True)
        start = prompt_len
        next_token, _ = pick(prefill.logits[:, -1, :])
        output[:, start] = next_token
        if streamer is not None:
            streamer.put(next_token)
        if next_token.item() in eos:
            return finish(start + 1)

        while start < max_length - 1:
            block_len = min(block_size, max_length - start)
            block = torch.full((1, block_len), mask_token_id, dtype=torch.long, device=device)
            block[:, 0] = output[:, start]
            positions = torch.arange(start, start + block_len, device=device).unsqueeze(0)

            draft = self(
                input_ids=block,
                position_ids=positions,
                past_key_values=cache,
                use_cache=False,
                is_diffusion_pass=True,
                ar_seq_len=start,
            )
            if block_len > 1:
                draft_tokens, draft_probs = pick(draft.logits[:, :-1, :])
            else:
                draft_tokens = torch.empty((1, 0), dtype=torch.long, device=device)
                draft_probs = None
            proposed = torch.cat([output[:, start : start + 1], draft_tokens], dim=1)

            verify = self(
                input_ids=proposed, position_ids=positions, past_key_values=cache, use_cache=True
            )
            ar_tokens, ar_probs = pick(verify.logits)

            if temperature < 1e-5:
                matches = draft_tokens == ar_tokens[:, :-1]
                accepted = int(matches.cumprod(dim=1).sum().item())
                next_token = ar_tokens[:, accepted]
            else:
                accepted = 0
                for i in range(draft_tokens.shape[1]):
                    token = draft_tokens[0, i]
                    ratio = ar_probs[0, i, token] / draft_probs[0, i, token].clamp_min(1e-8)
                    if torch.rand((), device=device) < ratio.clamp(max=1.0):
                        accepted += 1
                    else:
                        break
                target = ar_probs[0, accepted]
                if accepted < draft_tokens.shape[1]:
                    residual = (target - draft_probs[0, accepted]).clamp_min(0.0)
                    target = residual / residual.sum() if residual.sum() > 1e-5 else target
                next_token = torch.multinomial(target, 1)
            stats.accepted.append(accepted)

            kept = proposed[:, : accepted + 1]
            eos_hits = [i for i, token in enumerate(kept[0].tolist()) if token in eos]
            if eos_hits:
                end = start + eos_hits[0] + 1
                output[:, start:end] = kept[:, : eos_hits[0] + 1]
                if streamer is not None:
                    streamer.put(kept[:, 1 : eos_hits[0] + 1])
                return finish(end)

            output[:, start : start + accepted + 1] = kept
            if streamer is not None and accepted > 0:
                streamer.put(kept[:, 1:])
            start += accepted + 1
            truncate_cache(cache, start)
            if start < max_length:
                output[:, start] = next_token
                if streamer is not None:
                    streamer.put(next_token)
                if next_token.item() in eos:
                    return finish(start + 1)
        return finish(max_length)
