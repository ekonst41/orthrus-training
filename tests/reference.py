import torch

from src.model import OrthrusLM, build_block_mask


def reference_states(
    model: OrthrusLM, input_ids: torch.Tensor, anchors: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """model_states through the modules of model.py (as at inference): the reference."""
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
