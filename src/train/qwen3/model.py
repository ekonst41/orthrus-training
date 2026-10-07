import inspect
import json
from pathlib import Path
import shutil

import torch
import torch.distributed as dist
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from torch.utils.checkpoint import checkpoint

import src.model as qwen3_implementation
from src.configuration import OrthrusConfig
from src.model import OrthrusLM
from src.train.qwen3.loss import chunked_forward_kl

DIFF_MODULE_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")


def copy_diff_from_ar(model):
    with torch.no_grad():
        for layer in model.model.layers:
            for name in DIFF_MODULE_NAMES:
                attention = layer.self_attn
                getattr(attention, name + "_diff").load_state_dict(
                    getattr(attention, name).state_dict()
                )


def freeze_to_diffusion(model):
    n_train, n_total = 0, 0
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(
            any(part.endswith("_diff") for part in name.split("."))
        )
        n_total += 1
        n_train += int(parameter.requires_grad)
    return n_train, n_total


def configure_flex_backend(backend="triton"):
    compiled = torch.compile(flex_attention, dynamic=False)

    def attention(q, k, v, mask=None):
        options = {"BACKEND": "FLASH"} if backend == "flash" else {}
        return compiled(
            q, k, v, block_mask=mask, enable_gqa=True, kernel_options=options
        )

    qwen3_implementation.fused_flex_attention = attention


class OrthrusQwen3ForTraining(OrthrusLM):
    config_class = OrthrusConfig
    _auto_class = None

    def __init__(
        self,
        config,
        *,
        activation_checkpointing=False,
        kl_chunk_size=32,
        temperature=1.0,
    ):
        super().__init__(config)
        self.activation_checkpointing = activation_checkpointing
        self.kl_chunk_size = kl_chunk_size
        self.temperature = temperature

    def forward(self, input_ids, anchors, anchor_valid=None, supervise_mask=None):
        batch_size, ar_len = input_ids.shape
        block_size = self.config.block_size
        if anchor_valid is None:
            anchor_valid = torch.ones_like(anchors, dtype=torch.bool)
        positions, keep = block_positions_and_keep(
            input_ids, anchors, block_size, supervise_mask, anchor_valid
        )

        self.eval()
        with torch.no_grad():
            teacher = self.model(
                input_ids=input_ids, use_cache=True, is_diffusion_pass=False
            )
        self.train()
        cache = teacher.past_key_values
        if cache is None or cache.get_seq_length() != ar_len:
            raise RuntimeError("Qwen3 AR forward did not produce a full KV cache.")

        masked_tokens = torch.full_like(positions, self.config.mask_token_id)
        masked_tokens[:, :, 0] = input_ids.gather(1, anchors)
        position_ids = positions.flatten(1)
        hidden = self.model.embed_tokens(masked_tokens.flatten(1))
        rope = self.model.rotary_emb(hidden, position_ids)
        mask = make_block_mask(anchors, ar_len, block_size)
        for layer in self.model.layers:

            def apply_layer(states, layer=layer):
                return layer(
                    states,
                    position_embeddings=rope,
                    position_ids=position_ids,
                    past_key_values=cache,
                    use_cache=False,
                    is_diffusion_pass=True,
                    causal_limit=None,
                    ar_seq_len=ar_len,
                    flex_block_mask=mask,
                )

            hidden = (
                checkpoint(apply_layer, hidden, use_reentrant=False)
                if self.activation_checkpointing
                else apply_layer(hidden)
            )
        hidden = self.model.norm(hidden)
        hidden = hidden.reshape(batch_size, anchors.shape[1], block_size, -1)[:, :, :-1]
        teacher_positions = positions[:, :, :-1]
        batch_indices = torch.arange(batch_size, device=input_ids.device)[
            :, None, None
        ].expand_as(teacher_positions)
        teacher_selected = teacher.last_hidden_state[
            batch_indices[keep], teacher_positions[keep]
        ].detach()
        student_selected = hidden[keep]

        count = keep.sum().to(torch.float32)
        if dist.is_initialized():
            dist.all_reduce(count)
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        loss = chunked_forward_kl(
            student_selected,
            teacher_selected,
            self.lm_head,
            chunk_size=self.kl_chunk_size,
            temperature=self.temperature,
        )
        loss = loss * world_size / count.clamp_min(1)
        return loss, hidden.detach(), keep

    def save_pretrained(self, save_directory, *args, **kwargs):
        super().save_pretrained(save_directory, *args, **kwargs)
        path = Path(save_directory)
        shutil.copyfile(inspect.getfile(OrthrusLM), path / "model.py")
        shutil.copyfile(inspect.getfile(OrthrusConfig), path / "configuration.py")
        with (path / "model.py").open("a", encoding="utf-8") as file:
            file.write(
                "\n\nfrom .configuration import OrthrusConfig as _OrthrusExportConfig\n"
                "OrthrusLM.config_class = _OrthrusExportConfig\n"
                "OrthrusModel.config_class = _OrthrusExportConfig\n"
            )
        cfg_path = path / "config.json"
        payload = json.loads(cfg_path.read_text())
        payload["architectures"] = ["OrthrusLM"]
        payload["auto_map"] = {
            "AutoConfig": "configuration.OrthrusConfig",
            "AutoModelForCausalLM": "model.OrthrusLM",
        }
        cfg_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def block_positions_and_keep(
    input_ids, anchors, block_size, supervise_mask, anchor_valid
):
    positions = anchors[:, :, None] + torch.arange(block_size, device=input_ids.device)
    keep = anchor_valid[:, :, None].expand(-1, -1, block_size - 1).clone()
    if supervise_mask is not None:
        mask = supervise_mask.bool()
        starts = mask & ~torch.cat([torch.zeros_like(mask[:, :1]), mask[:, :-1]], dim=1)
        runs = starts.long().cumsum(dim=1)
        targets = positions[:, :, 1:]
        target_mask = mask.gather(1, targets.flatten(1)).reshape_as(targets)
        target_runs = runs.gather(1, targets.flatten(1)).reshape_as(targets)
        anchor_runs = runs.gather(1, anchors)[:, :, None]
        keep &= target_mask & (target_runs == anchor_runs)
    return positions, keep


def make_block_mask(anchors, ar_len, block_size):
    diffusion_length = anchors.shape[1] * block_size

    def allowed(batch, head, query, key):
        block = (query // block_size).clamp(max=anchors.shape[1] - 1)
        ar_visible = (key < ar_len) & (key < anchors[batch, block])
        diffusion_visible = (key >= ar_len) & (
            (key - ar_len) // block_size == query // block_size
        )
        return (
            (query < diffusion_length)
            & (key < ar_len + diffusion_length)
            & (ar_visible | diffusion_visible)
        )

    return create_block_mask(
        allowed,
        B=anchors.shape[0],
        H=None,
        Q_LEN=diffusion_length,
        KV_LEN=ar_len + diffusion_length,
        device=str(anchors.device),
    )
