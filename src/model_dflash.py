"""Qwen3 DFlash drafter conditioned on intermediate states of a frozen target."""

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3MLP,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    rotate_half,
)

from src.model import GenerationStats, build_block_mask, diffusion_attention, truncate_cache


def rotate(states, rope):
    cos, sin = (part[:, None].to(states.dtype) for part in rope)
    return states * cos + rotate_half(states) * sin


class DFlashLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = Qwen3Attention(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_norm = Qwen3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_norm = Qwen3RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, x, context, block_rope, context_rope, mask=None, cache=None):
        attn = self.attn
        h = self.input_norm(x)

        def heads(y):
            return y.unflatten(-1, (-1, attn.head_dim)).transpose(1, 2)

        # Project keys/values from both the target features and the current draft block.
        context_len = context.shape[1]
        kv_input = torch.cat((context, h), dim=1)
        key, value = heads(attn.k_proj(kv_input)), heads(attn.v_proj(kv_input))
        query = attn.q_norm(heads(attn.q_proj(h))).to(value.dtype)
        key = attn.k_norm(key).to(value.dtype)
        query = rotate(query, block_rope)
        ctx_key = rotate(key[:, :, :context_len], context_rope)
        ctx_value = value[:, :, :context_len]
        if cache is not None:
            # Only accepted target context lives in this cache, never draft block states.
            ctx_key, ctx_value = cache.update(ctx_key, ctx_value, attn.layer_idx)
        keys = torch.cat((ctx_key, rotate(key[:, :, context_len:], block_rope)), dim=2)
        values = torch.cat((ctx_value, value[:, :, context_len:]), dim=2)
        if mask is None:
            out = F.scaled_dot_product_attention(query, keys, values, enable_gqa=True)
        else:
            out = diffusion_attention(query, keys, values, mask)
        x = x + attn.o_proj(out.transpose(1, 2).flatten(2))
        return x + self.mlp(self.post_norm(x))


class DFlashLM(nn.Module):
    def __init__(self, target, draft_layers: int):
        super().__init__()
        self.target = target.requires_grad_(False).eval()
        self.config = target.config
        self.generation_config = target.generation_config
        cfg = self.config
        # Evenly spaced intermediate layer outputs, with a fallback for tiny test models.
        first, last = 1, cfg.num_hidden_layers - 3
        if cfg.num_hidden_layers < 4:
            first, last = 0, cfg.num_hidden_layers - 1
        count = min(5, last - first + 1)
        self.feature_layers = tuple(
            round(first + i * (last - first) / max(1, count - 1)) for i in range(count)
        )
        self.context_proj = nn.Linear(count * cfg.hidden_size, cfg.hidden_size, bias=False)
        self.context_norm = Qwen3RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.layers = nn.ModuleList(DFlashLayer(cfg, i) for i in range(draft_layers))
        self.norm = Qwen3RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(cfg)
        self.checkpoint_layers = False
        for module in self.modules():
            if isinstance(module, nn.Linear) and module.weight.requires_grad:
                nn.init.normal_(module.weight, std=cfg.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @property
    def lm_head(self):
        return self.target.lm_head

    def train(self, mode=True):
        super().train(mode)
        self.target.eval()
        return self

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.checkpoint_layers = True

    def forward(self, *args, **kwargs):
        return self.target(*args, **kwargs)

    def features(self, hidden_states):
        return torch.cat([hidden_states[i + 1] for i in self.feature_layers], dim=-1)

    def draft_states(self, block_ids, features, block_pos, context_pos, mask=None, cache=None):
        dtype = self.context_proj.weight.dtype
        context = self.context_norm(self.context_proj(features.to(dtype)))
        x = self.target.model.embed_tokens(block_ids).to(dtype)
        block_rope = self.rotary_emb(x, block_pos)
        context_rope = self.rotary_emb(x, context_pos)
        for layer in self.layers:
            args = (x, context, block_rope, context_rope, mask, cache)
            if self.checkpoint_layers and self.training and torch.is_grad_enabled():
                x = checkpoint(layer, *args, use_reentrant=False)
            else:
                x = layer(*args)
        return self.norm(x)

    def training_states(self, input_ids, anchors):
        with torch.no_grad():
            target = self.target.model(input_ids, use_cache=False, output_hidden_states=True)
            features = self.features(target.hidden_states)
            teacher = target.last_hidden_state
        del target
        block_size = self.config.block_size
        offsets = torch.arange(block_size, device=input_ids.device)
        positions = anchors[:, :, None] + offsets
        block = torch.full_like(positions, self.config.mask_token_id)
        block[:, :, 0] = input_ids.gather(1, anchors)
        seq_len = input_ids.shape[1]
        context_pos = torch.arange(seq_len, device=input_ids.device)[None]
        hidden = self.draft_states(
            block.flatten(1),
            features,
            positions.flatten(1),
            context_pos,
            mask=build_block_mask(anchors, seq_len, block_size),
        )
        return hidden, teacher

    @torch.inference_mode()
    def diffusion_generate(self, input_ids, max_new_tokens, eos_token_id=None, streamer=None):
        """Greedy speculative decoding with separate target and drafter KV caches."""
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError("DFlash generation requires one nonempty prompt")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        eos = {eos_token_id} if isinstance(eos_token_id, int) else set(eos_token_id or ())
        prompt_len = input_ids.shape[1]
        limit = prompt_len + max_new_tokens
        device = input_ids.device
        output = input_ids.new_full((1, limit), self.config.mask_token_id)
        output[:, :prompt_len] = input_ids
        stats = GenerationStats()
        target_cache = DynamicCache(config=self.config)
        draft_cache = DynamicCache()

        if streamer is not None:
            streamer.put(input_ids.cpu())
        prefill = self.target(
            input_ids,
            past_key_values=target_cache,
            use_cache=True,
            output_hidden_states=True,
            logits_to_keep=1,
        )
        features = self.features(prefill.hidden_states)
        next_token = prefill.logits[:, -1:].argmax(-1)
        del prefill
        start = prompt_len
        output[:, start:start + 1] = next_token
        if streamer is not None:
            streamer.put(next_token.cpu())
        end = start + 1

        while end < limit and next_token.item() not in eos:
            width = min(self.config.block_size, limit - start)
            positions = torch.arange(start, start + width, device=device)[None]
            context_pos = torch.arange(start - features.shape[1], start, device=device)[None]
            proposal = output[:, start:start + width].clone()
            hidden = self.draft_states(proposal, features, positions, context_pos, cache=draft_cache)
            proposal[:, 1:] = self.lm_head(hidden[:, 1:].to(self.lm_head.weight.dtype)).argmax(-1)
            verified = self.target(
                proposal,
                past_key_values=target_cache,
                use_cache=True,
                output_hidden_states=True,
                logits_to_keep=width,
            )
            greedy = verified.logits.argmax(-1)
            accepted = int((proposal[:, 1:] == greedy[:, :-1]).cumprod(dim=-1).sum().item())
            next_token = greedy[:, accepted:accepted + 1]
            committed = start + accepted + 1
            emitted = proposal[:, 1:accepted + 1]
            if committed < limit:
                emitted = torch.cat((emitted, next_token), dim=1)
            stop = next((i + 1 for i, token in enumerate(emitted[0].tolist()) if token in eos), None)
            emitted = emitted[:, :stop]
            end = start + 1 + emitted.shape[1]
            output[:, start + 1:end] = emitted
            stats.accepted.append(min(accepted, emitted.shape[1]))
            if streamer is not None and emitted.numel():
                streamer.put(emitted.cpu())
            if stop is not None or end == limit:
                break
            truncate_cache(target_cache, committed)
            features = self.features([h[:, :accepted + 1] for h in verified.hidden_states])
            start = committed
            del verified

        if streamer is not None:
            streamer.end()
        stats.new_tokens = end - prompt_len
        return output[:, :end], stats
