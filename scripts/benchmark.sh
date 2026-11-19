#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ -f .env ]]; then set -a; source .env; set +a; fi
config="${1:-configs/qwen3-0.6b.yaml}"
if [[ $# -gt 0 ]]; then shift; fi
exec python -m src.test.benchmark --config "$config" "$@"
