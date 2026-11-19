#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ -f .env ]]; then set -a; source .env; set +a; fi
config="${1:-configs/qwen3-0.6b.yaml}"
if [[ $# -gt 0 ]]; then shift; fi
export TOKENIZERS_PARALLELISM=false
if [[ "${NPROC:-1}" -gt 1 ]]; then
    exec torchrun --standalone --nproc-per-node="$NPROC" --module src.train.train --config "$config" "$@"
fi
exec python -m src.train.train --config "$config" "$@"
