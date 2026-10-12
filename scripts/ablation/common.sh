# Общая часть скриптов абляции. Подключается из них через `source`, отдельно не запускается.
# Переходит в корень репозитория, так что скрипты можно запускать из любой папки.
# shellcheck disable=SC2034  # переменные ниже используют подключающие скрипты

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT" || exit 1

# CONFIG и DATASET можно заменить переменными окружения (проверка скриптов на маленькой модели).
CONFIG=${CONFIG:-configs/qwen3-0.6b.yaml} # рецепт статьи: всё, кроме весов позиций, как в базе
DATASET=${DATASET:-nemotron-v2-qwen3-0.6b} # данные базового прогона (в HF-бакете)
BASE_MODEL=Qwen/Qwen3-0.6B
BASE_REVISION=c1899de289a04d12100db370d81485cdf75e47ca
BASELINE_RUN=qwen3-0.6b-paper        # готовый базовый прогон (равные веса), с ним и сравниваем
GAMMAS=(4 8 16)

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
die() {
    printf '\n\033[31mОШИБКА:\033[0m %s\n' "$*" >&2
    exit 1
}

# HF_TOKEN и ORTHRUS_BUCKET: из файла .env в корне репозитория или из окружения.
load_env() {
    if [ -f .env ]; then
        set -a
        # shellcheck disable=SC1090
        . <(tr -d '\r' <.env) # \r — если файл правили в Windows
        set +a
    fi
    [ -n "${HF_TOKEN:-}" ] || die "не задан HF_TOKEN: впишите его в файл .env (scripts/ablation/README.md, шаг 3)"
    [ -n "${ORTHRUS_BUCKET:-}" ] || die "не задан ORTHRUS_BUCKET: впишите его в файл .env (scripts/ablation/README.md, шаг 3)"
    export HF_TOKEN ORTHRUS_BUCKET
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # как в официальном коде
    export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1
}

activate_venv() {
    [ -x .venv/bin/python ] || die "нет окружения .venv: сначала запустите bash scripts/ablation/setup.sh"
    # shellcheck disable=SC1091
    source .venv/bin/activate
}

require_prepared() {
    compgen -G "runs/data/$DATASET/packed/packed-*/assistant_mask.npy" >/dev/null ||
        die "данные не подготовлены: сначала запустите bash scripts/ablation/prepare.sh"
}

run_name() { echo "qwen3-0.6b-gamma$1"; }

# Обучение одного gamma на одной GPU, затем итоговая оценка (её делает только завершённый прогон).
# Аргументы: gamma, номер GPU, дополнительные настройки section.key=value (для обучения и оценки).
# Имя прогона можно заменить переменной RUN (так делает selftest.sh).
train_and_evaluate() {
    local gamma=$1 gpu=$2
    shift 2
    local run=${RUN:-$(run_name "$gamma")}
    load_env
    activate_venv
    require_prepared
    export CUDA_VISIBLE_DEVICES=$gpu

    say "обучение $run: gamma=$gamma, GPU $gpu (повторный запуск продолжает с последнего чекпойнта)"
    python -m orthrus.train --config "$CONFIG" \
        run_name="$run" train.loss_gamma="$gamma" train.max_hours=0 "$@"

    if [ ! -f "runs/runs/$run/final/config.json" ]; then
        say "обучение $run остановлено до конца; чтобы продолжить, запустите этот же скрипт ещё раз"
        return 0
    fi

    say "итоговая оценка $run: fp32, 512 новых токенов, 32 промпта на набор"
    python -m orthrus.evaluate --config "$CONFIG" \
        run_name="$run" eval.max_new_tokens=512 eval.dtype=float32 "$@"
    say "готово: $run обучен и оценён; сравнение — bash scripts/ablation/status.sh"
}
