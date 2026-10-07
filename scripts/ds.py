"""Thin DataSphere CLI wrapper: project id from .env, logs in .ds_logs/, runs from the repo root.

Usage:
  python scripts/ds.py run jobs/probe-system.yaml   # submit a job and stream its logs
  python scripts/ds.py attach <job_id>              # reattach to a running job
  python scripts/ds.py list | get <id> | cancel <id> | download <id> | ttl <id> <days>
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def main() -> int:
    load_dotenv(ROOT / ".env")
    os.environ.setdefault("YC_CLI_INITIALIZATION_SILENCE", "true")
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    command, args = sys.argv[1], sys.argv[2:]
    project = os.environ.get("DS_PROJECT_ID")
    venv_bin = str(Path(sys.executable).parent)
    cli = shutil.which("datasphere") or shutil.which("datasphere", path=venv_bin)
    if cli is None:
        sys.exit("datasphere CLI not found: pip install datasphere")

    job = [cli, "--log-dir", str(ROOT / ".ds_logs"), "project", "job"]
    needs_project = {"run", "list"}
    if command in needs_project and not project:
        sys.exit("Set DS_PROJECT_ID in .env (DataSphere project page -> project id)")
    commands = {
        "run": [*job, "execute", "-p", project, "-c", *args],
        "list": [*job, "list", "-p", project],
        "attach": [*job, "attach", "--id", *args],
        "get": [*job, "get", "--id", *args],
        "cancel": [*job, "cancel", "--id", *args],
        "download": [*job, "download-files", "--id", *args],
        "ttl": [*job, "set-data-ttl", "--id", *args[:1], "--days", *args[1:]],
    }
    if command not in commands:
        sys.exit(__doc__)
    return subprocess.call(commands[command], cwd=ROOT)


if __name__ == "__main__":
    sys.exit(main())
