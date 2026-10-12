import json
import re

from orthrus import compare


def write_run(root, name, gamma, tpf):
    run = root / "runs" / name
    run.mkdir(parents=True)
    train = {"loss_gamma": gamma} if gamma else {}  # runs from before the option have no key
    manifest = {"config": {"train": train}, "total_steps": 20}
    (run / "run.json").write_text(json.dumps(manifest))
    records = [
        {"step": 10, "train/loss": 2.0},
        {"step": 20, "eval/kl": 2.5, "eval/agree_k1": 0.9, "generate/tpf": tpf},
    ]
    (run / "train_metrics.jsonl").write_text("\n".join(map(json.dumps, records)))
    final = {s: {"tpf": tpf, "ar_match": 1.0} for s in compare.SUITES}
    (run / compare.FINAL_EVAL).write_text(json.dumps(final))


def test_compare_prints_runs_side_by_side(tmp_path, capsys):
    write_run(tmp_path, "base", 0, 2.0)
    write_run(tmp_path, "gamma8", 8, 2.2)
    compare.main(
        ["--runs", "base,gamma8,missing", "--config", str(_config(tmp_path))]
        + [f"storage.local_dir={tmp_path}", "storage.bucket="]
    )
    out = capsys.readouterr().out
    assert "0 (equal)" in out and "10/20" in out  # last trained step / total
    assert "+10.0%" in out  # mean final TPF of gamma8 relative to the first run
    assert re.search(r"^missing +- +0/?", out, re.M)  # a run that has not started: no crash


def _config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("run_name: base\n")
    return path
