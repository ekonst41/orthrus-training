import hashlib
import json

import numpy as np
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
MODELS = [{"tag": "small", "base": "fake/small"}, {"tag": "large", "base": "fake/large"}]


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


def _config(tmp_path, digest: str, *extra: str):
    overrides = [
        "run_name=exp",
        "storage.bucket=",
        f"storage.local_dir={tmp_path}",
        "model.block_size=8",
        f"decode.models={json.dumps(MODELS)}",
        "decode.prompt_set=unit",
        f"decode.prompt_set_sha256={digest}",
        "decode.suites=[math, code]",
        "decode.prompts_per_suite=3",
        "decode.max_new_tokens=24",
        "decode.chunk_blocks=4",
        f"decode.strategies={json.dumps(STRATEGIES)}",
        "decode.check_strategies=[tail-tree, reuse-tree, tail-gated]",
        *extra,
    ]
    return load_config(overrides=overrides)


@pytest.fixture
def setup(tmp_path, monkeypatch, lively_model):  # noqa: F811
    tokenizer = tiny_tokenizer()
    digest = _prompt_set(tmp_path / "experiments" / "prompts" / "unit.jsonl", tokenizer)
    loaded = []

    def fake_load(spec, device, dtype):
        loaded.append(spec.base)
        return lively_model, tokenizer, []

    monkeypatch.setattr(checkpoint, "load_model", fake_load)
    hub = tmp_path / "hub"

    def fake_download(repo_id, revision=None, **kwargs):  # files to be released after loading
        snapshot = hub / f"models--{repo_id.replace('/', '--')}" / "snapshots" / "rev"
        snapshot.mkdir(parents=True, exist_ok=True)
        (snapshot / "model.safetensors").write_text("weights")
        return str(snapshot)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)
    return tmp_path, digest, loaded


def _records(root, tag):
    return [
        json.loads(line)
        for path in sorted((root / tag).glob("round-*.jsonl"))
        for line in path.read_text().splitlines()
    ]


def test_rounds_check_resume_and_report(setup):
    tmp_path, digest, loaded = setup
    root = tmp_path / "experiments" / "exp"
    # a job past its time limit (no round starts), a job with one round per model, and a job
    # that continues with the remaining rounds
    decode_exp.run(_config(tmp_path, digest, "decode.max_minutes=0.0001"))
    assert not list(root.glob("*/round-*.jsonl"))
    decode_exp.run(_config(tmp_path, digest, "decode.prompts_per_suite=1"))
    assert len(list(root.glob("*/round-*.jsonl"))) == 2
    decode_exp.run(_config(tmp_path, digest))
    assert loaded.count("fake/small") >= 2 and loaded.count("fake/large") >= 2
    assert not list((tmp_path / "hub").glob("models--*"))  # released after loading
    for tag in ("small", "large"):
        records = _records(root, tag)
        assert sorted({r["round"] for r in records}) == [0, 1, 2]
        check = {r["strategy"] for r in records if r["round"] == 0}
        assert check == {"base", "tail-tree", "reuse-tree", "tail-gated", "ar"}
        assert {r["strategy"] for r in records if r["round"] > 0} == {"base"}
        assert sum(r["strategy"] == "ar" for r in records) == 1
        real = [r for r in records if r["strategy"] != "ar"]
        assert all(r["match_base"] and r["sim_match"] and r["draft_mismatch"] == 0 for r in real)
        assert all(r["match_ar"] for r in records if "match_ar" in r)
        data = np.load(root / tag / "round-0001.npz")
        assert list(data["suite"]) == ["math", "code"] and list(data["rank"]) == [1, 1]
    manifest = json.loads((root / "run.json").read_text())
    assert manifest["prompt_set_sha256"] == digest and len(manifest["segments"]) == 3
    assert [len(s["rounds_completed"]) for s in manifest["segments"]] == [0, 2, 4]

    text, results = decode_report.analyze(_config(tmp_path, digest), "exp", "small", workers=1)
    assert "## Headline" in text and "## Checks" in text and "Anatomy" in text
    for tag in ("small", "large"):
        r = results[tag]
        assert r["prompts"] == 6
        base = [x for x in _records(root, tag) if x["strategy"] == "base"]
        for suite in ("math", "code"):
            rows = [x for x in base if x["suite"] == suite]
            expected = sum(x["new_tokens"] for x in rows) / sum(x["passes"] for x in rows)
            assert r["per_suite"][suite]["base"]["tpf"] == pytest.approx(expected)
        assert r["checks"]["tail-tree"]["sim_match"] == 1.0
        assert r["ceiling"]["tpf"] >= r["avg"]["base"]["tpf"]
    assert (root / "analysis" / "policy_grid.csv").exists()


def test_run_refuses_an_unpinned_change(setup):
    tmp_path, _, _ = setup
    with pytest.raises(ValueError, match="sha256"):
        decode_exp.run(_config(tmp_path, "0" * 64))


def test_bootstrap_paired_difference_is_exact_for_identical_rows():
    tokens = np.array([[10, 20, 30], [10, 20, 30]], float)
    passes = np.array([[5, 5, 10], [4, 5, 10]], float)
    base, other = decode_report.bootstrap_tpf(tokens, passes)
    assert base["delta"] == 0 and base["delta_ci"] == [0, 0]
    assert other["tpf"] == pytest.approx(60 / 19) and other["fewer_passes"] == pytest.approx(1 / 3)
    assert other["delta_ci"][0] >= 0
    avg = decode_report.bootstrap_avg([(tokens, passes), (tokens, passes)])
    assert avg[0]["change_pct"] == 0
    assert avg[1]["change_pct"] == pytest.approx(100 * (60 / 19 / 3 - 1))
