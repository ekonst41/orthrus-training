"""Regenerate requirements/*.txt from requirements/*.in with uv.

Usage: python scripts/lock.py [dev|train|datagen ...]   (all by default)

train/datagen are resolved for Linux x86_64 (DataSphere, GPU servers) and written without comments
and without `--index-url`, because the DataSphere CLI rejects both in a requirements file.
URL requirements (`pkg @ https://...`) go to <name>-urls.txt for the same reason; install them
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
CLEAN = "--no-header --no-annotate --emit-index-url".split()
TARGETS = {"dev": ["--universal"], "train": LINUX + CLEAN, "datagen": LINUX + CLEAN}


def write(filename: str, lines: list[str]) -> None:
    (REQUIREMENTS / filename).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def lock(name: str) -> None:
    command = ["uv", "pip", "compile", f"{name}.in", *COMMON, *TARGETS[name]]
    result = subprocess.run(command, cwd=REQUIREMENTS, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"{name}: {result.stderr}")
    lines = [line for line in result.stdout.splitlines() if not line.startswith("--index-url")]
    urls = [line for line in lines if " @ " in line]
    write(f"{name}.txt", [line for line in lines if " @ " not in line])
    if urls:
        write(f"{name}-urls.txt", urls)
    pins = sum("==" in line and not line.lstrip().startswith("#") for line in lines)
    print(f"requirements/{name}.txt: {pins} pinned packages, {len(urls)} URL requirements")


if __name__ == "__main__":
    for target in sys.argv[1:] or TARGETS:
        lock(target)
