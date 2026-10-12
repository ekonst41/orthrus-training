#!/usr/bin/env bash
# Шаг 4 инструкции: базовая модель и данные (один раз на машину).
#
# Скачивает Qwen3-0.6B (1.2 ГБ) и сгенерированные ею данные из HF-бакета (0.9 ГБ), упаковывает их
# в строки по 2048 токенов (~251 тыс. строк, 2.6 ГБ) — ровно как в базовом прогоне. Обучающие
# скрипты потом только читают готовое, поэтому их можно запускать параллельно. ~5-10 минут.
# Запуск:  bash scripts/ablation/prepare.sh
set -euo pipefail
source "$(dirname "$0")/common.sh"
load_env
activate_venv

say "базовая модель $BASE_MODEL"
hf download "$BASE_MODEL" --revision "$BASE_REVISION" --quiet >/dev/null
echo "ok"

say "данные $DATASET из бакета $ORTHRUS_BUCKET и упаковка"
python - "$DATASET" <<'EOF'
import os
import sys
from pathlib import Path

from orthrus import data
from orthrus.evaluate import load_eval_prompts
from orthrus.storage import Bucket

dataset, root = sys.argv[1], Path("runs")
bucket = Bucket(os.environ["ORTHRUS_BUCKET"])
if not data.load_manifest(bucket, dataset, root):
    raise SystemExit(f"в бакете нет data/{dataset}: проверьте ORTHRUS_BUCKET и права HF_TOKEN")
prompts = load_eval_prompts(bucket, dataset, root)
raw = data.fetch_raw(bucket, dataset, root)
ids, _, key = data.load_packed(raw, raw.parent / "packed", 2048, 42)
print(f"{ids.shape[0]} строк по {ids.shape[1]} токенов ({key}), {len(prompts)} отложенных промптов")
EOF

say "данные готовы. Следующий шаг: bash scripts/ablation/selftest.sh (проверка, ~15 минут)"
