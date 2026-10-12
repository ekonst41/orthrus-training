#!/usr/bin/env bash
# Аккуратно остановить прогон: обучение доделает текущий шаг, сохранит чекпойнт и выйдет
# (1-3 минуты). Продолжить потом: тот же gamma*.sh или start_all.sh.
# Запуск:  bash scripts/ablation/stop.sh 8          один прогон (gamma = 8)
#          bash scripts/ablation/stop.sh all        все прогоны абляции
set -euo pipefail
source "$(dirname "$0")/common.sh"

[ $# -eq 1 ] || die "укажите gamma (${GAMMAS[*]}) или all, например: stop.sh 8"
if [ "$1" = all ]; then targets=("${GAMMAS[@]}"); else targets=("$1"); fi

for gamma in "${targets[@]}"; do
    run=$(run_name "$gamma")
    if pkill -TERM -f "orthrus.train .*run_name=$run "; then
        echo "$run: сигнал отправлен, обучение сохранит чекпойнт и завершится"
    elif pkill -TERM -f "orthrus.evaluate .*run_name=$run "; then
        echo "$run: итоговая оценка прервана (обучение завершено; оценку можно повторить, gamma$gamma.sh)"
    else
        echo "$run: не запущен"
    fi
done
