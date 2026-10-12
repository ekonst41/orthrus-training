#!/usr/bin/env bash
# Шаг 6 инструкции: все три прогона (gamma 4, 8, 16) сразу, каждый на своей GPU, в фоне.
#
# Прогоны продолжают работать после закрытия терминала или ssh. Уже идущий прогон не
# запускается второй раз, поэтому скрипт можно повторить после перезагрузки сервера: остановленные
# прогоны продолжатся с последнего чекпойнта.
# Запуск:  bash scripts/ablation/start_all.sh            gamma 4, 8, 16 на GPU 0, 1, 2
#          bash scripts/ablation/start_all.sh 3 5 6      те же прогоны на GPU 3, 5, 6
# Логи:    logs/gamma4.log, logs/gamma8.log, logs/gamma16.log
set -euo pipefail
source "$(dirname "$0")/common.sh"
load_env
activate_venv
require_prepared

if [ $# -eq 0 ]; then GPUS=(0 1 2); else GPUS=("$@"); fi
[ ${#GPUS[@]} -eq ${#GAMMAS[@]} ] ||
    die "нужно ${#GAMMAS[@]} номера GPU (по одному на gamma ${GAMMAS[*]}), например: start_all.sh 0 1 2"
available=$(nvidia-smi -L | wc -l)

mkdir -p logs
for i in "${!GAMMAS[@]}"; do
    gamma=${GAMMAS[$i]} gpu=${GPUS[$i]}
    run=$(run_name "$gamma")
    [ "$gpu" -lt "$available" ] ||
        die "GPU $gpu нет: на машине $available GPU (номера 0..$((available - 1)))"
    if pgrep -f "run_name=$run " >/dev/null; then
        echo "gamma=$gamma: прогон $run уже идёт, пропускаю"
        continue
    fi
    used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
    if [ "$used" -gt 2000 ]; then
        echo "ВНИМАНИЕ: на GPU $gpu уже занято $used MiB, проверьте nvidia-smi"
    fi
    nohup bash scripts/ablation/gamma"$gamma".sh "$gpu" >>"logs/gamma$gamma.log" 2>&1 &
    echo "gamma=$gamma: запущен на GPU $gpu, лог logs/gamma$gamma.log"
done

say "следить: bash scripts/ablation/status.sh   остановить: bash scripts/ablation/stop.sh all"
