"""Regenerate requirements/*.txt from requirements/*.in with uv.

Usage: python scripts/lock.py [dev|train|datagen ...]   (all by default)

train/datagen are resolved for Linux x86_64 (DataSphere, GPU servers) and written without comments
and without `--index-url`, because the DataSphere CLI rejects both in a requirements file.
dev is a universal lock for local development on any OS.
"""

import subprocess
import sys
from pathlib import Path

REQUIREMENTS = Path(__file__).resolve().parents[1] / "requirements"
# pip (used by DataSphere) picks the best version across all indexes; mirror that in uv.
LINUX = [
    "--python-platform",
    "x86_64-manylinux_2_28",
    "--no-header",
    "--no-annotate",
    "--index-strategy",
    "unsafe-best-match",
]
TARGETS = {
    "dev": ["--universal"],
    "train": [*LINUX, "--emit-index-url"],
    "datagen": [*LINUX, "--emit-index-url"],
}


def lock(name: str) -> None:
    command = [
        "uv",
        "pip",
        "compile",
        f"{name}.in",
        "--python-version",
        "3.12",
        "-q",
        *TARGETS[name],
    ]
    result = subprocess.run(command, cwd=REQUIREMENTS, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"{name}: {result.stderr}")
    lines = [line for line in result.stdout.splitlines() if not line.startswith("--index-url")]
    (REQUIREMENTS / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pins = sum("==" in line and not line.lstrip().startswith("#") for line in lines)
    print(f"requirements/{name}.txt: {pins} pinned packages")


if __name__ == "__main__":
    for target in sys.argv[1:] or TARGETS:
        lock(target)
