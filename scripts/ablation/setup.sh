#!/usr/bin/env bash
# Шаг 2 инструкции: окружение Python в папке .venv (один раз на машину).
#
# Ставит uv (если его нет), Python 3.12 и зависимости обучения из requirements/train.txt, затем
# проверяет, что torch видит GPU. sudo не нужен. Занимает ~5 минут и ~12 ГБ на диске.
# Запуск:  bash scripts/ablation/setup.sh
set -euo pipefail
source "$(dirname "$0")/common.sh"

UV_VERSION=0.11.3 # проверенная версия

say "проверка машины"
command -v nvidia-smi >/dev/null ||
    die "нет nvidia-smi: нужен драйвер NVIDIA (версия 535 или новее)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
# torch.compile собирает ядра Triton, а Triton компилирует свой загрузчик компилятором C.
command -v gcc >/dev/null || command -v cc >/dev/null ||
    die "нет компилятора C: установите его (Ubuntu: sudo apt install build-essential)"
command -v curl >/dev/null || command -v uv >/dev/null ||
    die "нет curl: установите его (Ubuntu: sudo apt install curl)"
df -h . | tail -1 | awk '{print "свободно на диске: " $4 " (нужно ~40 ГБ: окружение, данные, чекпойнты)"}'

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
    say "установка uv $UV_VERSION в ~/.local/bin"
    curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | sh
fi
uv --version

export UV_HTTP_TIMEOUT=600 UV_LINK_MODE=copy
if [ ! -x .venv/bin/python ]; then
    say "создание .venv с Python 3.12"
    # Python от uv: в нём есть заголовки Python, нужные Triton (в системном их может не быть).
    uv venv --python 3.12 --managed-python .venv
fi

say "установка зависимостей (torch 2.13 + CUDA 12.9 и др., ~3 ГБ загрузки)"
# unsafe-best-match: как pip и как при сборке lock-файла (scripts/lock.py).
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match -r requirements/train.txt

say "проверка torch и GPU"
.venv/bin/python - <<'EOF'
import torch

if not torch.cuda.is_available():
    raise SystemExit(
        "torch не видит GPU: проверьте nvidia-smi и драйвер (нужна версия 535 или новее)"
    )
name = torch.cuda.get_device_name(0)
memory = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.device_count()} GPU, "
      f"GPU 0: {name}, {memory:.0f} GB")
if memory < 40:  # пик обучения ~39 GB при train.micro_batch_size=4
    print("ВНИМАНИЕ: обучению нужно ~39 GB памяти GPU; добавьте к запуску train.micro_batch_size=2")
EOF

say "окружение готово. Следующий шаг: bash scripts/ablation/prepare.sh"
