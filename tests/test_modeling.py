import torch
from conftest import MASK_TOKEN, make_tiny_model
from torch.nn.attention.flex_attention import flex_attention

from orthrus import modeling_orthrus as mo


def test_block_mask_matches_definition():
    """AR keys strictly before the block's anchor, diffusion keys only inside the same block."""
    seq_len, block = 40, 4
    anchors = torch.tensor([[3, 10, 30], [1, 2, 36]])
    mask = mo.build_block_mask(anchors, seq_len, block)
    q = torch.arange(anchors.shape[1] * block)[:, None]
    k = torch.arange(seq_len + anchors.shape[1] * block)[None, :]
    for b in range(anchors.shape[0]):
        blk = q // block
        expected = ((k < seq_len) & (k < anchors[b][blk])) | (
            (k >= seq_len) & ((k - seq_len) // block == blk)
        )
        got = mask.mask_mod(
            torch.tensor(b), torch.tensor(0), q.expand(-1, k.shape[1]), k.expand(q.shape[0], -1)
        )
        assert torch.equal(got, expected)


def test_dense_fallback_matches_flex():
    torch.manual_seed(0)
    anchors = torch.tensor([[2, 9, 17]])
    mask = mo.build_block_mask(anchors, 24, 4)
    q = torch.randn(1, 4, 12, 16)
    k, v = torch.randn(1, 2, 36, 16), torch.randn(1, 2, 36, 16)
    dense = mo.diffusion_attention(q, k, v, mask)  # CPU: dense SDPA path
    flex = flex_attention(q, k, v, block_mask=mask, enable_gqa=True)  # eager reference
    assert torch.allclose(dense, flex, atol=1e-5)


def test_freeze_keeps_only_diffusion_twins(tiny_model):
    trainable = [n for n, p in tiny_model.named_parameters() if p.requires_grad]
    assert trainable and all("_diff." in n for n in trainable)
    assert len(trainable) == 6 * tiny_model.config.num_hidden_layers


def _ar_greedy(model, prompt, steps):
    seq = prompt.clone()
    for _ in range(steps):
        next_token = model(input_ids=seq).logits[:, -1].argmax(-1, keepdim=True)
        seq = torch.cat([seq, next_token], dim=1)
    return seq


def test_greedy_diffusion_generation_is_lossless():
    model = make_tiny_model(block_size=8).eval()
    prompt = torch.randint(0, 500, (1, 11))
    with torch.inference_mode():
        out, stats = model.diffusion_generate(prompt, max_new_tokens=40, eos_token_id=[])
        reference = _ar_greedy(model, prompt, 40)
    assert torch.equal(out, reference)
    assert stats.new_tokens == 40
    assert stats.forward_passes == 1 + 2 * stats.cycles
    assert all(0 <= a <= model.config.block_size - 1 for a in stats.accepted)


def test_generation_stops_at_any_eos():
    model = make_tiny_model().eval()
    prompt = torch.randint(0, 500, (1, 7))
    with torch.inference_mode():
        free, _ = model.diffusion_generate(prompt, max_new_tokens=30, eos_token_id=[])
        stop_token = int(free[0, prompt.shape[1] + 5])  # force a stop at the 6th new token
        out, stats = model.diffusion_generate(prompt, max_new_tokens=30, eos_token_id=[stop_token])
    first = (free[0, prompt.shape[1] :] == stop_token).nonzero()[0].item()
    assert out.shape[1] == prompt.shape[1] + first + 1
    assert stats.new_tokens == first + 1


def test_truncate_cache_drops_tail(tiny_model):
    cache = mo.DynamicCache(config=tiny_model.config)
    tiny_model(input_ids=torch.randint(0, 500, (1, 9)), past_key_values=cache, use_cache=True)
    mo.truncate_cache(cache, 5)
    assert cache.get_seq_length() == 5
    assert MASK_TOKEN == tiny_model.config.mask_token_id
