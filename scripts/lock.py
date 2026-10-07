"""Regenerate requirements/*.txt from requirements/*.in with uv.

Usage: python scripts/lock.py [dev|train|datagen ...]   (all by default)

train/datagen are resolved for Linux x86_64 (DataSphere, GPU servers). The DataSphere CLI rejects
comments, `--index-url` and URL requirements in a requirements file, so these locks are written
without them; URL requirements (`pkg @ https://...`) go to <name>-urls.txt and are installed
after the lock with `pip install --no-deps -r requirements/<name>-urls.txt`.
dev is a universal lock for local development on any OS.
"""

import subprocess
import sys
from pathlib import Path

REQUIREMENTS = Path(__file__).resolve().parents[1] / "requirements"
COMMON = "--python-version 3.12 -q".split()
# pip (used by DataSphere) picks the best version across all indexes; mirror that in uv.
LINUX = "--python-platform x86_64-manylinux_2_28 --index-strategy unsafe-best-match".split()
LINUX += "--no-header --no-annotate --emit-index-url".split()
# vllm's multimodal extras exist only as CUDA 13 builds (torchcodec) or for another torch
# (torchaudio); text generation does not need them, and vllm skips them when they are absent.
NO_MULTIMODAL = "--no-emit-package torchcodec --no-emit-package torchaudio".split()
TARGETS = {"dev": ["--universal"], "train": LINUX, "datagen": LINUX + NO_MULTIMODAL}
FOR_DATASPHERE = {"train", "datagen"}


def write(filename: str, lines: list[str]) -> None:
    (REQUIREMENTS / filename).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def is_plain_requirement(line: str) -> bool:
    text = line.strip()
    return bool(text) and not text.startswith(("#", "--index-url")) and " @ " not in text


def lock(name: str) -> None:
    command = ["uv", "pip", "compile", f"{name}.in", *COMMON, *TARGETS[name]]
    result = subprocess.run(command, cwd=REQUIREMENTS, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"{name}: {result.stderr}")
    lines = result.stdout.splitlines()
    urls = []
    if name in FOR_DATASPHERE:
        urls = [line.strip() for line in lines if " @ " in line]
        lines = [line for line in lines if is_plain_requirement(line)]
    write(f"{name}.txt", lines)
    if urls:
        write(f"{name}-urls.txt", urls)
    pins = sum("==" in line and not line.lstrip().startswith("#") for line in lines)
    print(f"requirements/{name}.txt: {pins} pinned packages, {len(urls)} URL requirements")


if __name__ == "__main__":
    for target in sys.argv[1:] or TARGETS:
        lock(target)
