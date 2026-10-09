from orthrus import datagen
from orthrus.config import load_config
from orthrus.storage import Bucket

TRAIN = [{"uuid": f"t{i}", "domain": "math", "prompt_ids": [1, 2, i]} for i in range(3)]
EVAL = [{"uuid": "e0", "domain": "chat", "prompt_ids": [5]}]


def test_prompt_selection_is_cached_per_settings(tmp_path, monkeypatch):
    calls = []

    def fake_sample(d, tokenizer):
        calls.append(d.samples_per_domain)
        return TRAIN, EVAL, "rev1"

    monkeypatch.setattr(datagen, "sample_prompts", fake_sample)
    bucket = Bucket("")
    cfg = load_config(overrides=["storage.bucket=", "datagen.samples_per_domain=3"])
    assert datagen.selected_prompts(cfg, None, bucket, tmp_path) == (TRAIN, EVAL, "rev1")
    assert datagen.selected_prompts(cfg, None, bucket, tmp_path) == (TRAIN, EVAL, "rev1")
    assert calls == [3]  # the second call read the cache

    other = load_config(overrides=["storage.bucket=", "datagen.samples_per_domain=4"])
    datagen.selected_prompts(other, None, bucket, tmp_path)
    assert calls == [3, 4]  # other settings: sampled again
