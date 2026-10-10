import pytest
import torch
from conftest import make_tiny_model
from torch import nn

from orthrus.decoding import (
    Hypothesis,
    ModelBackend,
    OracleBackend,
    Strategy,
    ar_decode,
    decode,
    draft_table,
    expected_tokens,
    parse_strategies,
    tail_hypotheses,
)
from orthrus.evaluate import ar_generate

STRATEGIES = parse_strategies(
    [
        {"name": "base"},
        {"name": "tail-s0", "kind": "tail", "shifts": [0]},
        {"name": "tail-tree", "kind": "tail", "shifts": [1, 0, -1, -2]},
        {"name": "tail-x2", "kind": "tail", "shifts": [1, 0, -1], "max_tail_passes": 2},
        {"name": "tail-gated", "kind": "tail", "shifts": [0], "min_tail": 3, "min_expected": 2.5},
        {"name": "reuse-tree", "kind": "reuse", "shifts": [1, 0, -1, -2]},
    ]
)


@pytest.fixture(scope="module")
def model():
    """Tiny model whose greedy output does not collapse into one repeated token (the default
    tiny model's small weights and tied embeddings just repeat the last token)."""
    model = make_tiny_model(block_size=8).eval()
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        model.lm_head.weight = nn.Parameter(
            torch.randn(model.lm_head.weight.shape, generator=generator)
        )  # untied
        for name, param in model.model.layers.named_parameters():
            if "norm" not in name:
                param.mul_(10)
    return model


def _prompt(seed: int, length: int = 9) -> list[int]:
    return torch.randint(0, 500, (length,), generator=torch.Generator().manual_seed(seed)).tolist()


def _same_trace(a, b) -> bool:
    return (a.tokens, a.passes, a.accepted, a.branch, a.tails) == (
        b.tokens,
        b.passes,
        b.accepted,
        b.branch,
        b.tails,
    )


def test_strategy_validation():
    with pytest.raises(ValueError):
        Strategy("x", kind="tail", shifts=(2,))
    with pytest.raises(ValueError):
        Strategy("x", kind="other")
    with pytest.raises(ValueError):
        parse_strategies([{"name": "a"}, {"name": "a"}])
    assert Strategy("g", kind="tail", min_expected=2).needs_confidence


def test_tail_hypotheses_shifts_and_room():
    source = Hypothesis(None, [10, 11, 12, 13, 14, 15], [0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    hyps = tail_hypotheses(source, accepted=2, shifts=(1, 0, -1, -2), room=3)
    # rejected source[2] = 12: insertion keeps it, substitution skips it, deletions skip more
    assert [(h.shift, h.tokens) for h in hyps] == [
        (1, [12, 13, 14]),
        (0, [13, 14, 15]),
        (-1, [14, 15]),
        (-2, [15]),
    ]
    assert hyps[1].confidence == [0.6, 0.5, 0.4]
    assert tail_hypotheses(source, accepted=5, shifts=(0, -1), room=4) == []
    assert expected_tokens(Hypothesis(0, [1, 2], [0.5, 0.5])) == pytest.approx(1.75)


def test_base_strategy_equals_diffusion_generate(model):
    for seed in range(4):
        prompt = _prompt(seed)
        out, stats = model.diffusion_generate(torch.tensor([prompt]), 40, eos_token_id=[])
        gen = decode(ModelBackend(model, prompt), STRATEGIES[0], len(prompt), 40, set())
        assert gen.tokens == out[0, len(prompt) :].tolist()
        assert gen.accepted == stats.accepted
        assert gen.forward_passes == stats.forward_passes


def test_ar_decode_equals_reference_ar(model):
    prompt = _prompt(7)
    gen = ar_decode(ModelBackend(model, prompt), len(prompt), 30, set())
    reference, _ = ar_generate(model, torch.tensor([prompt]), 30, set())
    assert gen.tokens == reference and gen.forward_passes == 30


def test_tree_verification_matches_separate_chains(model):
    prompt = _prompt(3)
    hyps = [
        Hypothesis(s, torch.randint(0, 500, (n,)).tolist()) for s, n in ((1, 5), (0, 3), (-1, 6))
    ]
    tree = ModelBackend(model, prompt)
    root = tree.prefill()
    tree_preds = tree.verify("tail", len(prompt), root, hyps)
    for index, hyp in enumerate(hyps):
        chain = ModelBackend(model, prompt)
        chain.prefill()
        assert chain.verify("tail", len(prompt), root, [hyp])[0] == tree_preds[index]
    tree.commit(2, 4)  # keep the root and 4 tokens of the third hypothesis
    chain = ModelBackend(model, prompt)
    chain.prefill()
    chain.verify("tail", len(prompt), root, [hyps[2]])
    chain.commit(0, 4)
    for a, b in zip(tree.cache.layers, chain.cache.layers, strict=True):
        assert a.keys.shape == b.keys.shape
        assert torch.allclose(a.keys, b.keys, atol=1e-5)
        assert torch.allclose(a.values, b.values, atol=1e-5)


def _scripted_table(reference: list[int], block_size: int, edit: int | None = None) -> list:
    """Drafts equal to the true continuation with one edit per block at a varying offset: a
    substitution (0), a skipped token (1, shift +1 repairs it) or a spurious token (2, shift -1
    repairs it); edit None cycles through the three."""
    table = []
    for i in range(len(reference)):
        truth = (reference[i + 1 : i + block_size + 1] + [7] * block_size)[:block_size]
        j, kind = 1 + i % 5, i % 3 if edit is None else edit
        if kind == 0:
            drafts = truth[:j] + [(truth[j] + 1) % 500] + truth[j + 1 :]
        elif kind == 1:
            drafts = truth[:j] + truth[j + 1 :]
        else:
            drafts = truth[:j] + [(truth[j] + 3) % 500] + truth[j:]
        table.append(drafts[: block_size - 1])
    return table


class ScriptedBackend(ModelBackend):
    """Real AR verification, drafts from a table (stands in for a well-trained diffusion view)."""

    def __init__(self, model, prompt_ids, table):
        super().__init__(model, prompt_ids)
        self.table, self.prompt_len = table, len(prompt_ids)

    def draft(self, start, anchor, block_len, confidence):
        self._timed("draft", block_len, lambda: None)
        tokens = self.table[start - self.prompt_len][: block_len - 1]
        return Hypothesis(None, list(tokens), [0.9] * len(tokens) if confidence else None)


def test_tail_strategies_are_lossless_and_replay_exactly(model):
    prompt, new = _prompt(11), 60
    reference = ar_decode(ModelBackend(model, prompt), len(prompt), new, set()).tokens
    table = _scripted_table(reference, model.config.block_size)
    confidence = [[0.9] * len(row) for row in table]
    results = {}
    for strategy in STRATEGIES:
        real = decode(ScriptedBackend(model, prompt, table), strategy, len(prompt), new, set())
        assert real.tokens == reference, strategy.name  # lossless
        oracle = OracleBackend(len(prompt), reference, table, confidence, model.config.block_size)
        replay = decode(oracle, strategy, len(prompt), new, set())
        assert _same_trace(real, replay), strategy.name
        assert real.forward_passes == 1 + sum(
            real.passes.get(k, 0) for k in ("draft", "verify", "tail")
        )
        results[strategy.name] = real
    tree_kept = sum(t[2] for t in results["tail-tree"].tails)
    assert tree_kept > sum(t[2] for t in results["tail-s0"].tails)
    assert results["tail-tree"].tpf > results["base"].tpf
    assert any(b is not None for b in results["reuse-tree"].branch)
    assert results["reuse-tree"].passes.get("tail", 0) == 0


@pytest.mark.parametrize("edit,shift", [(0, 0), (1, 1), (2, -1)])
def test_tail_pass_repairs_each_edit_with_its_shift(model, edit, shift):
    prompt, new = _prompt(12), 60
    reference = ar_decode(ModelBackend(model, prompt), len(prompt), new, set()).tokens
    table = _scripted_table(reference, model.config.block_size, edit)
    gen = decode(ScriptedBackend(model, prompt, table), STRATEGIES[2], len(prompt), new, set())
    assert gen.tokens == reference
    kept = {s: sum(t[2] for t in gen.tails if t[1] == s) for s in (1, 0, -1, -2)}
    # the matching shift keeps the rest of each block (others only on repeated tokens)
    assert kept[shift] >= 0.8 * sum(kept.values()) > 0


def test_eos_inside_a_tail_stops_like_the_reference(model):
    prompt = _prompt(5)
    free = ar_decode(ModelBackend(model, prompt), len(prompt), 50, set()).tokens
    stop = free[23]
    reference = free[: free.index(stop) + 1]
    table = _scripted_table(free, model.config.block_size)
    for strategy in STRATEGIES:
        real = decode(ScriptedBackend(model, prompt, table), strategy, len(prompt), 50, {stop})
        assert real.tokens == reference and real.eos, strategy.name
        oracle = OracleBackend(len(prompt), reference, table, None, model.config.block_size)
        if not strategy.needs_confidence:
            assert _same_trace(real, decode(oracle, strategy, len(prompt), 50, {stop}))


def test_draft_table_matches_decoding_drafts(model):
    prompt, new = _prompt(2), 30  # no EOS: the last blocks are cut short by max_new_tokens
    reference = ar_decode(ModelBackend(model, prompt), len(prompt), new, set()).tokens
    drafts, confidence = draft_table(model, prompt, reference, new, chunk_blocks=4)
    block_size = model.config.block_size
    max_length = len(prompt) + new
    for i in range(len(reference) - 1):
        start = len(prompt) + i
        backend = ModelBackend(model, prompt + reference[:i])
        backend.prefill()  # AR cache of the committed prefix
        block_len = min(block_size, max_length - start)
        real = backend.draft(start, reference[i], block_len, confidence=True)
        assert drafts[i, : block_len - 1].tolist() == real.tokens
        assert drafts[i, block_len - 1 :].tolist() == [-1] * (block_size - block_len)
        assert confidence[i, : block_len - 1] == pytest.approx(real.confidence, abs=1e-5)
    assert (drafts[-1] == -1).all()  # no cycle starts at the last position


def test_replay_from_draft_table_matches_real_decoding(model):
    for seed in range(3):
        prompt, new = _prompt(20 + seed), 40
        eos = {_prompt(100 + seed, 1)[0]}
        reals = [decode(ModelBackend(model, prompt), s, len(prompt), new, eos) for s in STRATEGIES]
        reference = reals[0].tokens
        drafts, confidence = draft_table(model, prompt, reference, new)
        oracle_rows = [row[row >= 0].tolist() for row in drafts], confidence.tolist()
        for strategy, real in zip(STRATEGIES, reals, strict=True):
            assert real.tokens == reference
            oracle = OracleBackend(len(prompt), reference, *oracle_rows, model.config.block_size)
            assert _same_trace(real, decode(oracle, strategy, len(prompt), new, eos))
