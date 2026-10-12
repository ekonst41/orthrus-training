#!/usr/bin/env bash
# Шаг 5 инструкции: самопроверка этой машины перед долгими прогонами (~10 минут, одна GPU).
#
# Проходит тот же путь, что gamma*.sh, но коротко: 3 шага обучения с gamma=8 на маленьком батче,
# экспорт модели, итоговая оценка на 2 промптах каждого набора. Затем проверяет результат:
# взвешенный лосс записан и конечен, gamma=8 сохранена в чекпойнте, а вывод Orthrus в fp32
# совпадает с обычной AR-генерацией на всех наборах (ar_match = 1). В конце удаляет свой прогон
# selftest-gamma8 локально и в бакете; настоящие прогоны не трогает.
# Запуск:  bash scripts/ablation/selftest.sh [GPU]
set -euo pipefail
source "$(dirname "$0")/common.sh"
load_env
activate_venv
require_prepared

GPU=${1:-0}
export RUN=selftest-gamma8
trap 'die "самопроверка не прошла: причина — в сообщениях выше"' ERR

remove_run() {
    python - "$RUN" <<'EOF'
import os
import sys

from orthrus.storage import Bucket

Bucket(os.environ["ORTHRUS_BUCKET"]).delete(f"runs/{sys.argv[1]}")
EOF
    rm -rf "runs/runs/$RUN"
}

say "самопроверка: прогон $RUN на GPU $GPU"
remove_run # остатки прошлой самопроверки
bash scripts/ablation/gamma8.sh "$GPU" \
    train.max_steps=3 train.global_batch_size=8 train.log_every=1 train.eval_every=0 \
    eval.prompts=2 eval.max_new_tokens=32

say "проверка результатов"
python - "runs/runs/$RUN" <<'EOF'
import json
import math
import sys
from pathlib import Path

run = Path(sys.argv[1])
records = [json.loads(line) for line in (run / "train_metrics.jsonl").read_text().splitlines()]
losses = [r["train/weighted_loss"] for r in records if "train/weighted_loss" in r]
assert len(losses) == 3 and all(map(math.isfinite, losses)), f"weighted loss: {losses}"
state = json.loads((run / "checkpoints/step-0000003/trainer_state.json").read_text())
assert state["recipe"]["train"]["loss_gamma"] == 8, state["recipe"]["train"]
results = json.loads((run / "eval-final-32-float32.json").read_text())
suites = {name: m for name, m in results.items() if name != "_meta"}
assert len(suites) == 5, f"evaluated suites: {sorted(suites)}"
for name, m in suites.items():
    assert m["ar_match"] == 1.0, f"{name}: ar_match {m['ar_match']} (fp32 must be 1)"
    print(f"  {name:10s} tpf {m['tpf']:.2f}  ar_match {m['ar_match']:.0f}")
print(f"  weighted loss over 3 steps: {', '.join(f'{x:.2f}' for x in losses)}")
EOF

say "удаление прогона $RUN"
remove_run
say "САМОПРОВЕРКА ПРОЙДЕНА. Следующий шаг: bash scripts/ablation/start_all.sh"
