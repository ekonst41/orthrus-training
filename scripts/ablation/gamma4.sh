#!/usr/bin/env bash
# Абляция: обучение Orthrus с весами позиций блока, gamma = 4, затем итоговая оценка.
#
#   Вес токена на позиции i блока = exp(-i/4): первый токен 1.0, 8-й 0.17, 16-й 0.02.
#   На первые 8 позиций приходится 87% веса лосса (при равных весах — 26%).
#   Сильный акцент на начале блока: примерно как 0.8^k в Medusa/EAGLE-3 и gamma=4 в DFlash.
#
# Запуск:   bash scripts/ablation/gamma4.sh [GPU]
#           GPU — номер видеокарты (0, 1, ...; см. nvidia-smi), по умолчанию 0.
# Фоном:    nohup bash scripts/ablation/gamma4.sh 0 > logs/gamma4.log 2>&1 &
# Длится:   ~29 ч на A100 80GB (2 эпохи, 3920 шагов); H100 быстрее.
# Прервали: запустите ту же команду ещё раз — обучение продолжится с последнего чекпойнта.
set -euo pipefail
source "$(dirname "$0")/common.sh"

GAMMA=4
GPU=${1:-0}
shift || true # остальные аргументы (section.key=value) — дополнительные настройки

train_and_evaluate "$GAMMA" "$GPU" "$@"
