#!/usr/bin/env bash
# Host probe for a DataSphere job (or any GPU server): needs no Python packages.
# Usage: bash scripts/probe_system.sh [report_path]
set -u
report="${1:-outputs/probe_system.txt}"
mkdir -p "$(dirname "$report")"

section() { printf '\n===== %s =====\n' "$1"; }
have() { command -v "$1" >/dev/null 2>&1; }

main() {
  section "host"
  date -u
  uname -a
  grep -E '^(PRETTY_NAME|VERSION_ID)=' /etc/os-release
  echo "cpus: $(nproc)"
  lscpu | grep -E '^(Model name|Socket|Thread|Core)'
  free -g
  df -h / "$PWD" /tmp 2>/dev/null

  section "gpu"
  if have nvidia-smi; then
    nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap,persistence_mode --format=csv
    nvidia-smi | head -4 | tail -1  # banner line with the highest CUDA version the driver supports
  else
    echo "nvidia-smi: missing"
  fi

  section "toolchain"
  for tool in python3 python3.10 python3.11 python3.12 conda pip3 gcc cc g++ nvcc git curl; do
    printf '%-10s ' "$tool"
    if have "$tool"; then "$tool" --version 2>&1 | head -1; else echo "missing"; fi
  done

  section "job environment"
  echo "pwd: $PWD"
  env | grep -E '^(DS_|JOB_|CUDA|NVIDIA|LD_LIBRARY_PATH|HOME|USER)' \
    | sed -E 's/((TOKEN|KEY|SECRET|PASSWORD)[^=]*=).*/\1***/' | sort
  ls -la "$PWD" | head -20

  section "network (status, seconds)"
  for url in https://pypi.org/simple/torch/ https://download.pytorch.org/whl/cu128/torch/ \
             https://huggingface.co/api/models/Qwen/Qwen3-0.6B https://github.com https://wheels.vllm.ai/; do
    printf '%-55s ' "$url"
    curl -sS -o /dev/null -w '%{http_code} %{time_total}\n' --max-time 20 "$url" || echo "FAILED"
  done

  section "download speed (first 300 MB)"
  printf 'huggingface: '
  curl -sS -L -r 0-314572799 -o /dev/null --max-time 120 \
    -w '%{size_download} B in %{time_total} s = %{speed_download} B/s\n' \
    "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/model.safetensors" || echo "FAILED"

  section "disk write speed (2 GB, fsync)"
  dd if=/dev/zero of="$PWD/.dd_probe" bs=1M count=2048 conv=fsync 2>&1 | tail -1
  rm -f "$PWD/.dd_probe"

  section "done"
}

main 2>&1 | tee "$report"
