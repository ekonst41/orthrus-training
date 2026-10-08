"""Thin DataSphere CLI wrapper: project id from .env, per-run logs in .ds_logs/, repo-root cwd.

Usage:
  python scripts/ds.py run jobs/x.yaml [--max-minutes N] [--retry-minutes M]
      submit, stream logs, cancel after N minutes; if no VM of the requested type is free,
      resubmit every 5 minutes for up to M minutes
  python scripts/ds.py attach <job_id>        # reattach to a running job
  python scripts/ds.py list | get <id> | cancel <id> | download <id> | ttl <id> <days>

DataSphere has no server-side time limit for jobs, so `--max-minutes` cancels the job from here.
Jobs receive ORTHRUS_BUCKET / ORTHRUS_TRACKIO_SPACE from .env and ORTHRUS_GIT_COMMIT (the code
version, recorded in run manifests). The id of the last submitted job is in .ds_logs/last_job_id.
"""

import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / ".ds_logs"
CREATED_JOB = re.compile(r"created job `(\w+)`")
NO_FREE_VM = "Unable to find available VM"


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def git_commit() -> str:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
        return commit + ("-dirty" if dirty.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def find_cli() -> str:
    venv_bin = str(Path(sys.executable).parent)
    cli = shutil.which("datasphere") or shutil.which("datasphere", path=venv_bin)
    if cli is None:
        sys.exit("datasphere CLI not found: pip install datasphere")
    return cli


def take_option(args: list[str], name: str) -> tuple[float | None, list[str]]:
    if name not in args:
        return None, args
    at = args.index(name)
    return float(args[at + 1]), args[:at] + args[at + 2 :]


def run_with_deadline(cli: str, command: list[str], max_minutes: float | None) -> int:
    """Run `job execute`, remember the job id, cancel the job once the deadline passes."""
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    job_id = None
    deadline = time.monotonic() + max_minutes * 60 if max_minutes else None

    def watchdog() -> None:
        while proc.poll() is None:
            if deadline is not None and time.monotonic() > deadline:
                print(f"[ds] {max_minutes} min deadline reached, cancel {job_id}", flush=True)
                if job_id:
                    subprocess.call([cli, "project", "job", "cancel", "--id", job_id], env=env)
                proc.terminate()
                return
            time.sleep(5)

    threading.Thread(target=watchdog, daemon=True).start()
    for line in proc.stdout:
        print(line, end="", flush=True)
        match = CREATED_JOB.search(line)
        if match and job_id is None:
            job_id = match.group(1)
            (LOG_DIR / "last_job_id").write_text(job_id)
            print(f"[ds] job id {job_id} (deadline: {max_minutes or 'none'} min)", flush=True)
    return proc.wait()


def main() -> int:
    # Job logs carry arbitrary Unicode (progress bars); never let printing kill the watchdog.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv(ROOT / ".env")
    os.environ.setdefault("YC_CLI_INITIALIZATION_SILENCE", "true")
    for name in ("ORTHRUS_BUCKET", "ORTHRUS_TRACKIO_SPACE"):  # jobs list them in env.vars
        os.environ.setdefault(name, "")
    os.environ["ORTHRUS_GIT_COMMIT"] = git_commit()
    commands = ("run", "attach", "list", "get", "cancel", "download", "ttl")
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        sys.exit(__doc__)
    command, args = sys.argv[1], sys.argv[2:]
    project = os.environ.get("DS_PROJECT_ID")
    if command in ("run", "list") and not project:
        sys.exit("Set DS_PROJECT_ID in .env")
    max_minutes, args = take_option(args, "--max-minutes")
    retry_minutes, args = take_option(args, "--retry-minutes")
    cli = find_cli()
    give_up = time.monotonic() + (retry_minutes or 0) * 60

    while True:
        # One log directory per invocation: the CLI overwrites stdout.txt/system.log otherwise.
        label = Path(args[0]).stem if command == "run" and args else command
        run_logs = LOG_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{label}"
        run_logs.mkdir(parents=True, exist_ok=True)
        job = [cli, "--log-dir", str(run_logs), "project", "job"]
        argv = {
            "run": [*job, "execute", "-p", str(project), "-c", *args],
            "list": [*job, "list", "-p", str(project)],
            "attach": [*job, "attach", "--id", *args],
            "get": [*job, "get", "--id", *args],
            "cancel": [*job, "cancel", "--id", *args],
            "download": [*job, "download-files", "--id", *args],
            "ttl": [*job, "set-data-ttl", "--id", *args[:1], "--days", *args[1:]],
        }[command]
        if command != "run":
            return subprocess.call(argv, cwd=ROOT)
        code = run_with_deadline(cli, argv, max_minutes)
        system_log = run_logs / "system.log"
        no_vm = system_log.exists() and NO_FREE_VM in system_log.read_text(errors="replace")
        if code == 0 or not no_vm or time.monotonic() + 300 > give_up:
            return code
        print("[ds] no free VM of the requested type; retrying in 5 minutes", flush=True)
        time.sleep(300)


if __name__ == "__main__":
    sys.exit(main())
