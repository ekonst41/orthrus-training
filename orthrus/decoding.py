"""Decoding cycles of a trained Orthrus model, real and replayed, for TPF experiments.

The paper's cycle has two forward passes: the diffusion view drafts a block, the AR view verifies
it and keeps the agreed prefix plus its own next token, the bonus (paper, Sec. 3.3). Strategies:

  base   draft -> verify (the paper; same tokens and passes as OrthrusLM.diffusion_generate)
  tail   draft -> verify -> verify the rest of the rejected block after the bonus token: a third
         pass that keeps the accepted part of the tail plus one more AR token
  reuse  draft -> verify, where the verification of the next block also tests the rejected tail
         of the previous one (no extra pass, a longer verification input)

After a rejection at draft index a (drafts[a] != the AR token there, which becomes the bonus), a
tail hypothesis with shift s continues the bonus with drafts[a + 1 - s:]:
  s = 0   substitution: the bonus replaces drafts[a]
  s = 1   insertion: the draft missed the bonus, drafts[a] comes next
  s = -j  deletion: drafts[a : a + j] were spurious, drafts[a + j] should equal the bonus
Several hypotheses are verified in one pass with a tree attention mask (each sees the prefix, the
bonus and its own earlier tokens); the longest accepted one is kept, ties go to the first listed.
Only tokens the AR view predicts are ever kept, so greedy outputs equal plain AR decoding.

Greedy decoding only. Then every verification outcome is fixed by the reference output, and a
strategy can be replayed without the model from the drafts at every anchor of that output
(draft_table): OracleBackend replays any strategy with exact pass counts, which the experiment
checks against real decoding (orthrus.decode_exp).
"""

import dataclasses
import time
from collections import Counter
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from orthrus.modeling_orthrus import truncate_cache

KINDS = ("base", "tail", "reuse")


@dataclass(frozen=True)
class Strategy:
    name: str
    kind: str = "base"  # base | tail | reuse
    shifts: tuple[int, ...] = (0,)  # tail hypotheses, verified together (order breaks ties)
    min_tail: int = 1  # tail pass only if a hypothesis has at least this many tokens
    min_expected: float = 0.0  # ... and 1 + its expected accepted tokens (from the draft
    # confidences, taken as independent acceptance probabilities) reaches this (0: no gate)
    max_tail_passes: int = 1  # tail passes per cycle: a rejected tail can be retried
    oracle_min_tokens: int = 0  # replay only: a tail pass only if it would keep at least this
    # many tokens (accepted + the AR token) -- a perfect gate, the ceiling of any real gate

    def __post_init__(self):
        shifts = tuple(int(s) for s in self.shifts)
        object.__setattr__(self, "shifts", shifts)
        if self.kind not in KINDS:
            raise ValueError(f"strategy {self.name}: kind must be one of {KINDS}")
        if not shifts or len(set(shifts)) != len(shifts) or max(shifts) > 1:
            raise ValueError(f"strategy {self.name}: shifts must be distinct and <= 1")
        if self.min_tail < 1 or self.max_tail_passes < 1:
            raise ValueError(f"strategy {self.name}: min_tail and max_tail_passes must be >= 1")
        if self.oracle_min_tokens and self.kind != "tail":
            raise ValueError(f"strategy {self.name}: oracle_min_tokens needs kind tail")

    @property
    def needs_confidence(self) -> bool:
        return self.kind == "tail" and self.min_expected > 0

    def to_dict(self) -> dict:
        return {**dataclasses.asdict(self), "shifts": list(self.shifts)}


def parse_strategies(specs: list[dict]) -> list[Strategy]:
    strategies = [Strategy(**spec) for spec in specs]
    names = [s.name for s in strategies]
    if len(set(names)) != len(names):
        raise ValueError(f"strategy names must be unique: {names}")
    return strategies


@dataclass
class Hypothesis:
    shift: int | None  # None: the fresh drafts of the current block
    tokens: list[int]
    confidence: list[float] | None = None  # diffusion probability of each token


def tail_hypotheses(
    source: Hypothesis, accepted: int, shifts: tuple[int, ...], room: int
) -> list[Hypothesis]:
    """Continuations of the bonus token after source.tokens[accepted] was rejected (module doc),
    at most `room` tokens each; empty and duplicate continuations are dropped."""
    hypotheses, seen = [], set()
    for shift in shifts:
        begin = accepted + 1 - shift
        tokens = source.tokens[begin : begin + room]
        if not tokens or tuple(tokens) in seen:
            continue
        seen.add(tuple(tokens))
        confidence = source.confidence
        hypotheses.append(
            Hypothesis(
                shift, tokens, None if confidence is None else confidence[begin : begin + room]
            )
        )
    return hypotheses


def expected_tokens(hypothesis: Hypothesis) -> float:
    """Tokens a verification of the hypothesis is expected to keep: 1 (the AR token after the
    accepted part) + sum_j prod_{i<=j} confidence_i."""
    total = run = 1.0
    for p in hypothesis.confidence or ():
        run *= p
        total += run
    return total


@dataclass
class Generation:
    """One decoded output with its pass accounting (per pass kind: prefill, draft, verify, tail,
    ar)."""

    strategy: str
    tokens: list[int] = field(default_factory=list)  # new tokens
    eos: bool = False
    accepted: list[int] = field(default_factory=list)  # accepted drafts per cycle (kept hypothesis)
    branch: list[int | None] = field(default_factory=list)  # kept hypothesis per cycle: None =
    # fresh drafts, else the shift of a reused tail (strategy reuse)
    tails: list[tuple[int, int, int, int, int]] = field(default_factory=list)  # per tail pass:
    # (cycle, shift of the kept hypothesis, accepted tokens, hypotheses, longest hypothesis)
    passes: dict = field(default_factory=dict)
    input_tokens: dict = field(default_factory=dict)  # tokens fed to the model per pass kind
    seconds: dict = field(default_factory=dict)  # time inside passes per kind (host-synchronized)

    @property
    def forward_passes(self) -> int:
        return sum(self.passes.values())

    @property
    def tpf(self) -> float:
        return len(self.tokens) / max(self.forward_passes, 1)


class ModelBackend:
    """Forward passes of an OrthrusLM, batch 1, greedy, with per-kind pass/token/time counts.

    The AR cache holds the committed prefix; the newest token (the anchor) is not in it yet: the
    next verification feeds it first, like OrthrusLM.diffusion_generate."""

    def __init__(self, model, prompt_ids: list[int], record_drafts: bool = False):
        self.model = model
        self.device = next(model.parameters()).device
        self.block_size = model.config.block_size
        self.mask_token_id = model.config.mask_token_id
        self.prompt = torch.tensor([prompt_ids], device=self.device)
        self.cache = DynamicCache(config=model.config)
        self.passes, self.input_tokens, self.seconds = Counter(), Counter(), Counter()
        self.record_drafts = record_drafts
        self.drafts: dict[int, list[int]] = {}  # anchor position -> drafts (record_drafts)
        self._layout: tuple[int, list[int]] | None = None

    def _timed(self, kind: str, tokens: int, run):
        begin = time.perf_counter()
        result = run()  # ends with a host transfer (.tolist()), which synchronizes the device
        self.seconds[kind] += time.perf_counter() - begin
        self.passes[kind] += 1
        self.input_tokens[kind] += tokens
        return result

    def prefill(self) -> int:
        def run():
            logits = self.model(input_ids=self.prompt, past_key_values=self.cache, use_cache=True)
            return int(logits.logits[0, -1].argmax())

        return self._timed("prefill", self.prompt.shape[1], run)

    def draft(self, start: int, anchor: int, block_len: int, confidence: bool) -> Hypothesis:
        """Diffusion pass over [anchor, <mask> x (block_len - 1)] at positions start..: the
        drafts for positions start+1 .. start+block_len-1."""

        def run():
            block = torch.full(
                (1, block_len), self.mask_token_id, dtype=torch.long, device=self.device
            )
            block[0, 0] = anchor
            positions = torch.arange(start, start + block_len, device=self.device).unsqueeze(0)
            logits = self.model(
                input_ids=block,
                position_ids=positions,
                past_key_values=self.cache,
                use_cache=False,
                is_diffusion_pass=True,
                ar_seq_len=start,
            ).logits[0, :-1]
            tokens = logits.argmax(-1)
            probs = None
            if confidence:
                probs = F.softmax(logits.float(), dim=-1).gather(-1, tokens[:, None])[:, 0]
                probs = probs.tolist()
            return Hypothesis(None, tokens.tolist(), probs)

        drafts = self._timed("draft", block_len, run)
        if self.record_drafts:
            self.drafts[start] = drafts.tokens
        return drafts

    def verify(self, kind: str, start: int, root: int, hypotheses: list[Hypothesis]) -> list:
        """AR pass over the root (the token at `start`) and every hypothesis after it; one
        hypothesis is a plain causal chain, several form a tree. Returns, per hypothesis, the AR
        argmax after the root and after each of its tokens. Writes all of it to the cache;
        commit() keeps the root and the accepted part of one hypothesis."""
        branches = [h.tokens for h in hypotheses]
        tokens = [root] + [t for b in branches for t in b]
        depth = [0] + [i + 1 for b in branches for i in range(len(b))]
        offsets, offset = [], 1
        for b in branches:
            offsets.append(offset)
            offset += len(b)

        def run():
            ids = torch.tensor([tokens], device=self.device)
            positions = torch.tensor([[start + d for d in depth]], device=self.device)
            mask = None
            if len(branches) > 1:
                group = torch.tensor(
                    [-1] + [g for g, b in enumerate(branches) for _ in b], device=self.device
                )
                level = torch.tensor(depth, device=self.device)
                tree = (group[None, :] == -1) | (
                    (group[:, None] == group[None, :]) & (level[None, :] <= level[:, None])
                )
                prefix = torch.ones((len(tokens), start), dtype=torch.bool, device=self.device)
                mask = torch.cat([prefix, tree], dim=1)[None, None]
            logits = self.model(
                input_ids=ids,
                position_ids=positions,
                attention_mask=mask,
                past_key_values=self.cache,
                use_cache=True,
            ).logits
            return logits[0].argmax(-1).tolist()

        preds = self._timed(kind, len(tokens), run)
        self._layout = (start, offsets)
        return [[preds[0]] + preds[o : o + len(b)] for o, b in zip(offsets, branches, strict=True)]

    def commit(self, best: int, accepted: int) -> None:
        """Keep the cache of the prefix, the root and the first `accepted` tokens of hypothesis
        `best` of the last verification."""
        start, offsets = self._layout
        if offsets[best] == 1:  # the first hypothesis directly follows the root
            truncate_cache(self.cache, start + 1 + accepted)
            return
        begin = start + offsets[best]
        index = torch.cat(
            [
                torch.arange(start + 1, device=self.device),
                torch.arange(begin, begin + accepted, device=self.device),
            ]
        )
        for layer in self.cache.layers:
            layer.keys = layer.keys.index_select(-2, index)
            layer.values = layer.values.index_select(-2, index)


class OracleBackend:
    """Replays greedy decoding without the model: AR predictions come from the reference output,
    drafts from a draft table (row i: drafts of the block anchored at reference[i]). Exact when
    real decoding produced the reference and drew these drafts (the experiment checks both)."""

    def __init__(
        self,
        prompt_len: int,
        reference: list[int],
        drafts: list[list[int]],
        confidence: list[list[float]] | None,
        block_size: int,
    ):
        self.prompt_len, self.reference, self.block_size = prompt_len, reference, block_size
        self.drafts, self.confidence = drafts, confidence
        self.passes, self.input_tokens, self.seconds = Counter(), Counter(), Counter()

    def _count(self, kind: str, tokens: int) -> None:
        self.passes[kind] += 1
        self.input_tokens[kind] += tokens

    def _check(self, start: int, token: int) -> int:
        index = start - self.prompt_len
        if not 0 <= index < len(self.reference) or self.reference[index] != token:
            raise RuntimeError(f"replay left the reference output at position {start}")
        return index

    def prefill(self) -> int:
        self._count("prefill", self.prompt_len)
        return self.reference[0]

    def draft(self, start: int, anchor: int, block_len: int, confidence: bool) -> Hypothesis:
        index = self._check(start, anchor)
        tokens = self.drafts[index][: block_len - 1]
        if len(tokens) != block_len - 1 or min(tokens, default=0) < 0:
            raise RuntimeError(f"draft table has no block of {block_len} at position {start}")
        probs = self.confidence[index][: block_len - 1] if confidence else None
        self._count("draft", block_len)
        return Hypothesis(None, list(tokens), None if probs is None else list(probs))

    def verify(self, kind: str, start: int, root: int, hypotheses: list[Hypothesis]) -> list:
        index = self._check(start, root)
        self._count(kind, 1 + sum(len(h.tokens) for h in hypotheses))
        ref = self.reference

        def at(i: int) -> int:  # past the end of the output: never accepted
            return ref[i] if i < len(ref) else -1

        return [[at(index + 1 + d) for d in range(len(h.tokens) + 1)] for h in hypotheses]

    def commit(self, best: int, accepted: int) -> None:
        pass

    def peek(self, start: int, hypotheses: list[Hypothesis]) -> int:
        """Tokens a verification at `start` would keep (most accepted + 1), without a pass."""
        index = start - self.prompt_len
        best = 0
        for h in hypotheses:
            accepted = 0
            while (
                accepted < len(h.tokens)
                and index + 1 + accepted < len(self.reference)
                and h.tokens[accepted] == self.reference[index + 1 + accepted]
            ):
                accepted += 1
            best = max(best, accepted)
        return best + 1


def _verify(backend, kind: str, start: int, root: int, hypotheses: list[Hypothesis]):
    """Verify, keep the hypothesis with the most accepted tokens; (index, accepted, bonus)."""
    preds = backend.verify(kind, start, root, hypotheses)
    best, best_accepted = 0, -1
    for index, (hypothesis, pred) in enumerate(zip(hypotheses, preds, strict=True)):
        accepted = 0
        while accepted < len(hypothesis.tokens) and hypothesis.tokens[accepted] == pred[accepted]:
            accepted += 1
        if accepted > best_accepted:
            best, best_accepted = index, accepted
    backend.commit(best, best_accepted)
    return best, best_accepted, preds[best][best_accepted]


def _advance(gen: Generation, start: int, kept: list[int], bonus: int, eos, max_length: int):
    """Append the accepted tokens and the AR's next token, as diffusion_generate does; returns
    (finished, position of the new anchor)."""
    for i, token in enumerate(kept):
        if token in eos:
            gen.tokens.extend(kept[: i + 1])
            gen.eos = True
            return True, start + i + 1
    gen.tokens.extend(kept)
    start += len(kept) + 1
    if start < max_length:
        gen.tokens.append(bonus)
        if bonus in eos:
            gen.eos = True
            return True, start
    return start >= max_length - 1, start


def _worth_a_pass(strategy: Strategy, hypotheses: list[Hypothesis], backend, start: int) -> bool:
    if not hypotheses or max(len(h.tokens) for h in hypotheses) < strategy.min_tail:
        return False
    if strategy.min_expected > 0 and max(map(expected_tokens, hypotheses)) < strategy.min_expected:
        return False
    if strategy.oracle_min_tokens:
        if not hasattr(backend, "peek"):
            raise ValueError(f"strategy {strategy.name} (oracle_min_tokens) is replay-only")
        return backend.peek(start, hypotheses) >= strategy.oracle_min_tokens
    return True


def _finish(gen: Generation, backend) -> Generation:
    gen.passes = dict(backend.passes)
    gen.input_tokens = dict(backend.input_tokens)
    gen.seconds = dict(backend.seconds)
    return gen


def decode(
    backend, strategy: Strategy, prompt_len: int, max_new_tokens: int, eos: set[int]
) -> Generation:
    """Greedy Orthrus decoding with `strategy` (module doc) on a ModelBackend or OracleBackend."""
    gen = Generation(strategy.name)
    max_length = prompt_len + max_new_tokens
    gen.tokens.append(backend.prefill())
    gen.eos = done = gen.tokens[-1] in eos
    start = prompt_len  # position of the anchor gen.tokens[-1], not yet in the AR cache
    stale: list[Hypothesis] = []  # reuse: rejected tail of the previous block
    while not done and start < max_length - 1:
        block_len = min(backend.block_size, max_length - start)
        root = gen.tokens[-1]
        hypotheses = [backend.draft(start, root, block_len, strategy.needs_confidence)]
        hypotheses += [h for h in stale if h.tokens != hypotheses[0].tokens]
        best, accepted, bonus = _verify(backend, "verify", start, root, hypotheses)
        kept = hypotheses[best]
        gen.accepted.append(accepted)
        gen.branch.append(kept.shift)
        done, start = _advance(gen, start, kept.tokens[:accepted], bonus, eos, max_length)
        stale, tail_passes = [], 0
        while strategy.kind != "base" and not done and accepted < len(kept.tokens):
            hypotheses = tail_hypotheses(kept, accepted, strategy.shifts, max_length - 1 - start)
            if strategy.kind == "reuse":
                stale = hypotheses
                break
            if tail_passes == strategy.max_tail_passes or not _worth_a_pass(
                strategy, hypotheses, backend, start
            ):
                break
            tail_passes += 1
            best, accepted, bonus = _verify(backend, "tail", start, gen.tokens[-1], hypotheses)
            kept = hypotheses[best]
            longest = max(len(h.tokens) for h in hypotheses)
            gen.tails.append(
                (len(gen.accepted) - 1, kept.shift, accepted, len(hypotheses), longest)
            )
            done, start = _advance(gen, start, kept.tokens[:accepted], bonus, eos, max_length)
    return _finish(gen, backend)


def ar_decode(backend, prompt_len: int, max_new_tokens: int, eos: set[int]) -> Generation:
    """Plain greedy AR decoding through the same backend (one pass per token): the reference for
    losslessness and wall-clock speedup."""
    gen = Generation("ar")
    gen.tokens.append(backend.prefill())
    start = prompt_len
    while gen.tokens[-1] not in eos and len(gen.tokens) < max_new_tokens:
        (pred,) = backend.verify("ar", start, gen.tokens[-1], [Hypothesis(None, [])])
        backend.commit(0, 0)
        gen.tokens.append(pred[0])
        start += 1
    gen.eos = gen.tokens[-1] in eos
    return _finish(gen, backend)


@torch.inference_mode()
def draft_table(
    model, prompt_ids: list[int], reference: list[int], max_new_tokens: int, chunk_blocks: int = 64
):
    """Drafts of the diffusion view at every anchor of a reference output, as decoding computes
    them after committing reference[: i + 1]: row i holds the drafts of the block anchored at
    reference[i] (block_size - 1 of them, fewer where max_new_tokens cuts the block short).

    One AR pass caches the whole output; full blocks then go chunk_blocks at a time through one
    diffusion pass with the training mask (AR keys before the anchor, the block's own keys);
    blocks cut short by max_new_tokens run one by one. Rows without a cycle (the last position)
    stay empty. Returns numpy arrays (drafts int32, confidence float32), both
    [len(reference), block_size - 1], padded with -1 and 0."""
    device = next(model.parameters()).device
    block_size, mask_token_id = model.config.block_size, model.config.mask_token_id
    prompt_len, max_length = len(prompt_ids), len(prompt_ids) + max_new_tokens
    full = torch.tensor([prompt_ids + reference], device=device)
    total = full.shape[1]
    cache = DynamicCache(config=model.config)
    model(input_ids=full, past_key_values=cache, use_cache=True, logits_to_keep=1)
    drafts = torch.full((len(reference), block_size - 1), -1, dtype=torch.int32)
    confidence = torch.zeros((len(reference), block_size - 1), dtype=torch.float32)

    def run(anchors: list[int], block_len: int) -> None:
        count = len(anchors)
        anchor = torch.tensor(anchors, device=device)
        ids = torch.full((count, block_len), mask_token_id, dtype=torch.long, device=device)
        ids[:, 0] = full[0, anchor]
        positions = anchor[:, None] + torch.arange(block_len, device=device)
        block = torch.arange(count, device=device).repeat_interleave(block_len)
        ar_keys = torch.arange(total, device=device)[None, :] < anchor[block][:, None]
        own_block = block[:, None] == block[None, :]
        mask = torch.cat([ar_keys, own_block], dim=1)[None, None]
        logits = model(
            input_ids=ids.view(1, -1),
            position_ids=positions.view(1, -1),
            attention_mask=mask,
            past_key_values=cache,
            use_cache=False,
            is_diffusion_pass=True,
            ar_seq_len=total,
        ).logits[0]
        logits = logits.view(count, block_len, -1)[:, :-1]
        tokens = logits.argmax(-1)
        probs = F.softmax(logits.float(), dim=-1).gather(-1, tokens[..., None])[..., 0]
        rows = (anchor - prompt_len).cpu()
        drafts[rows, : block_len - 1] = tokens.int().cpu()
        confidence[rows, : block_len - 1] = probs.cpu()

    starts = range(prompt_len, min(prompt_len + len(reference), max_length - 1))
    full_blocks = [s for s in starts if s + block_size <= max_length]
    for i in range(0, len(full_blocks), chunk_blocks):
        run(full_blocks[i : i + chunk_blocks], block_size)
    for s in starts:
        if s + block_size > max_length:
            run([s], max_length - s)
    return drafts.numpy(), confidence.numpy()
