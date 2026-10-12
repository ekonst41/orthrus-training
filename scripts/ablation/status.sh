#!/usr/bin/env bash
# Состояние прогонов абляции и сравнение с базовым прогоном (равные веса).
# Запуск:  bash scripts/ablation/status.sh
set -euo pipefail
source "$(dirname "$0")/common.sh"
load_env
activate_venv

say "прогоны на этой машине"
for gamma in "${GAMMAS[@]}"; do
    run=$(run_name "$gamma")
    if pgrep -f "orthrus.train .*run_name=$run " >/dev/null; then
        state="обучается"
    elif pgrep -f "orthrus.evaluate .*run_name=$run " >/dev/null; then
        state="итоговая оценка"
    elif [ -f "runs/runs/$run/final/config.json" ]; then
        state="обучен"
    else
        state="не запущен"
    fi
    # shellcheck disable=SC2012
    log=$(ls -t "runs/runs/$run"/logs/train-*.log 2>/dev/null | head -1 || true)
    last=""
    if [ -n "$log" ]; then
        last=$(grep -h "orthrus.train: step " "$log" | tail -1 | sed 's/.*orthrus.train: //' || true)
    fi
    printf 'gamma=%-3s %-16s %s\n' "$gamma" "$state" "${last:-}"
done

say "сравнение с базовым прогоном $BASELINE_RUN"
runs=$BASELINE_RUN
for gamma in "${GAMMAS[@]}"; do runs+=",$(run_name "$gamma")"; done
python -m orthrus.compare --config "$CONFIG" --runs "$runs"
