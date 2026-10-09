import pytest
import torch
from transformers import Qwen3Config

from orthrus.modeling_orthrus import OrthrusLM, copy_diff_from_ar, freeze_to_diffusion

VOCAB = 512
MASK_TOKEN = VOCAB - 1


def make_tiny_model(block_size: int = 8, seed: int = 0) -> OrthrusLM:
    """A 2-layer random Qwen3 Orthrus model that runs in milliseconds on CPU (fp32)."""
    config = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=1024,
        tie_word_embeddings=True,
        rms_norm_eps=1e-6,
        eos_token_id=[VOCAB - 2, VOCAB - 3],
    )
    config.block_size = block_size
    config.mask_token_id = MASK_TOKEN
    config._attn_implementation = "sdpa"
    torch.manual_seed(seed)
    model = OrthrusLM(config).float()
    copy_diff_from_ar(model)
    freeze_to_diffusion(model)
    return model


@pytest.fixture
def tiny_model() -> OrthrusLM:
    return make_tiny_model()
