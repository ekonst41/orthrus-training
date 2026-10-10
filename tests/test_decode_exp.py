import hashlib
import json

import pytest
from test_decoding import model as lively_model  # noqa: F401  (fixture)
from test_evaluate import tiny_tokenizer

from orthrus import checkpoint, decode_exp, decode_report
from orthrus.config import load_config

STRATEGIES = [
    {"name": "base"},
    {"name": "tail-s0", "kind": "tail", "shifts": [0]},
    {"name": "tail-tree", "kind": "tail", "shifts": [1, 0, -1, -2]},
    {"name": "tail-gated", "kind": "tail", "shifts": [1, 0], "min_expected": 1.5},
    {"name": "reuse-tree", "kind": "reuse", "shifts": [1, 0, -1, -2]},
]


def _prompt_set(path, tokenizer) -> str:
    lines = []
    for offset, suite in enumerate(("math", "code")):
        for rank in range(4):
            words = " ".join(f"t{(rank * 7 + offset * 50 + i) % 400 + 10}" for i in range(5))
            messages = [{"role": "system", "content": ""}, {"role": "user", "content": words}]
            lines.append(
                {
                    "suite": suite,
                    "rank": rank,
                    "split": "dev" if rank % 2 == 0 else "test",
                    "source": "test",
                    "source_id": str(rank),
                    "messages": messages,
                    "prompt_ids": decode_exp.chat_ids(tokenizer, messages),
                }
            )
    data = "".join(json.dumps(line) + "\n" for line in lines).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _config(tmp_path, digest: str, **extra):
    overrides = [
        "run_name=exp",
        "storage.bucket=",
        f"storage.local_dir={tmp_path}",
        "model.block_size=8",
        "decode.prompt_set=unit",
        f"decode.prompt_set_sha256={digest}",
        "decode.suites=[math, code]",
        "decode.prompts_per_suite=3",
        "decode.max_new_tokens=24",
        "decode.ar_reference_prompts=2",
        "decode.chunk_blocks=4",
        f"decode.strategies={json.dumps(STRATEGIES)}",
        *[f"{k}={v}" for k, v in extra.items()],
    ]
    return load_config(overrides=overrides)


def test_experiment_runs_checks_resumes_and_reports(tmp_path, monkeypatch, lively_model):  # noqa: F811
    tokenizer = tiny_tokenizer()
    digest = _prompt_set(tmp_path / "experiments" / "prompts" / "unit.jsonl", tokenizer)
    calls = []

    def fake_load(cfg, device, dtype):
        calls.append(cfg.base)
        return lively_model, tokenizer, []

    monkeypatch.setattr(checkpoint, "load_model", fake_load)
    cfg = _config(tmp_path, digest)
    decode_exp.run(cfg)
    root = tmp_path / "experiments" / "exp"
    records = [
        json.loads(line)
        for suite in ("math", "code")
        for line in (root / "suites" / suite / "records.jsonl").read_text().splitlines()
    ]
    real = [r for r in records if r["strategy"] != "ar"]
    assert len(real) == 2 * 3 * len(STRATEGIES)
    assert sum(r["strategy"] == "ar" for r in records) == 2 * 2
    assert all(r["match_base"] and r["sim_match"] and r["draft_mismatch"] == 0 for r in real)
    assert all(r["match_ar"] for r in records if "match_ar" in r)
    manifest = json.loads((root / "run.json").read_text())
    assert manifest["prompt_set_sha256"] == digest and len(manifest["segments"]) == 1

    # a rerun skips complete suites but still records a job segment and the summary
    decode_exp.run(cfg)
    manifest = json.loads((root / "run.json").read_text())
    assert len(manifest["segments"]) == 2 and manifest["segments"][-1]["suites_completed"] == []

    text, analysis = decode_report.analyze(cfg, "exp", "small", workers=1)
    assert "## Policy search" in text and "## Anatomy" in text
    grid = {e["policy"]: e for e in analysis["grid"]}
    assert grid["base"]["dev/avg"] == pytest.approx(
        decode_report.statistics.mean(
            sum(
                r["new_tokens"]
                for r in real
                if r["strategy"] == "base" and r["suite"] == s and r["split"] == "dev"
            )
            / sum(
                r["passes"]
                for r in real
                if r["strategy"] == "base" and r["suite"] == s and r["split"] == "dev"
            )
            for s in ("math", "code")
        )  # fmt: skip
    )
    assert (root / "analysis" / "policy_grid.csv").exists()


def test_run_refuses_an_unpinned_change(tmp_path, monkeypatch, lively_model):  # noqa: F811
    tokenizer = tiny_tokenizer()
    _prompt_set(tmp_path / "experiments" / "prompts" / "unit.jsonl", tokenizer)
    monkeypatch.setattr(checkpoint, "load_model", lambda *a: (lively_model, tokenizer, []))
    with pytest.raises(ValueError, match="sha256"):
        decode_exp.run(_config(tmp_path, "0" * 64))


def test_bootstrap_paired_difference_is_exact_for_identical_rows():
    import numpy as np

    tokens = np.array([[10, 20, 30], [10, 20, 30]], float)
    passes = np.array([[5, 5, 10], [4, 5, 10]], float)
    base, other = decode_report.bootstrap_tpf(tokens, passes)
    assert base["delta"] == 0 and base["delta_ci"] == [0, 0]
    assert other["tpf"] == pytest.approx(60 / 19) and other["fewer_passes"] == pytest.approx(1 / 3)
    assert other["delta_ci"][0] >= 0
