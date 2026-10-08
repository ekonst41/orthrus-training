#!/usr/bin/env bash
# Runs the storage probe in a minimal venv (only huggingface_hub + hf-xet, pinned in train.txt),
# so the job skips the ~5 minute install of the full training stack.
set -euo pipefail
python3.12 -m venv /tmp/hub-venv
/tmp/hub-venv/bin/pip install -q $(grep -E '^(huggingface-hub|hf-xet)==' requirements/train.txt)
exec /tmp/hub-venv/bin/python scripts/probe_storage.py "$@"
