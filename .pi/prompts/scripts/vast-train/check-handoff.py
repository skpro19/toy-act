#!/usr/bin/env python3
"""Read-only, bounded handoff gate. Exit: 0 ready, 1 not ready, 2 invalid, 3 terminal.

Run with: uv run --frozen python .pi/prompts/scripts/vast-train/check-handoff.py RUN_DIR
Add --wait-seconds 90 to retry pending/unknown conditions within an overall deadline.
setup.env is trusted shell configuration, just as it is for local-watcher.sh.
Only readiness fields are read; credentials and command output are never reported.
"""

import argparse
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import time


SETUP_KEYS = (
    "INSTANCE_ID", "SSH_HOST", "SSH_PORT", "SSH_KNOWN_HOSTS", "TB_URL",
    "TB_SESSION", "WATCHER_PID", "REMOTE_PROJECT", "REMOTE_STATE",
    "TRAIN_SESSION", "REPO",
)
REMOTE_KEYS = (
    "run_status", "run_name", "completed", "failed", "train_session", "backup_session",
    "backup_running", "backup_artifact_ready", "backup_last_succeeded", "backup_failed",
)


class Deadline:
    def __init__(self, *, seconds: float) -> None:
        self.end = time.monotonic() + seconds

    def run(self, *, args: list[str], maximum: float) -> subprocess.CompletedProcess[str]:
        remaining = self.end - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(args, 0)
        return subprocess.run(
            args, capture_output=True, text=True, check=False,
            timeout=min(maximum, remaining))


def load_setup(*, run_dir: Path, deadline: Deadline) -> dict[str, str]:
    setup = run_dir / "setup.env"
    if not setup.is_file():
        raise ValueError("Missing setup.env")
    # Source in a separate process; return only an explicit non-sensitive allowlist.
    command = 'source "$1" >/dev/null || exit 1; shift; for key; do printf "%s\\0" "${!key:-}"; done'
    result = deadline.run(
        args=["bash", "-c", command, "handoff-setup", str(setup), *SETUP_KEYS], maximum=5)
    values = result.stdout.split("\0")
    if result.returncode or len(values) != len(SETUP_KEYS) + 1:
        raise ValueError("Could not read trusted setup.env")
    config = dict(zip(SETUP_KEYS, values[:-1]))
    for key in ("INSTANCE_ID", "SSH_HOST", "SSH_PORT", "TB_URL", "TB_SESSION"):
        if not config[key]:
            raise ValueError(f"Missing {key} in setup.env")
    if not config["INSTANCE_ID"].isdigit() or not config["SSH_PORT"].isdigit():
        raise ValueError("Instance ID and SSH port must be numeric")
    if not 1 <= int(config["SSH_PORT"]) <= 65535:
        raise ValueError("Invalid SSH port")
    if not re.fullmatch(r"http://(?:localhost|127\.0\.0\.1):[0-9]+/", config["TB_URL"]):
        raise ValueError("TB_URL must be the recorded local HTTP forwarding URL")
    if not re.fullmatch(r"[a-zA-Z0-9.-]+", config["SSH_HOST"]):
        raise ValueError("Invalid SSH host")
    return config


def read_record(*, path: Path) -> dict[str, str]:
    lines = path.read_text().splitlines()
    record = dict(line.split("=", 1) for line in lines)
    if len(record) != len(lines):
        raise ValueError("Duplicate acknowledgement fields")
    return record


def watcher_condition(*, run_dir: Path, config: dict[str, str]) -> dict[str, str]:
    try:
        record = read_record(path=run_dir / "watcher.ready")
        pid_path = Path(config["WATCHER_PID"] or str(run_dir / "watcher.pid"))
        if not pid_path.is_absolute():
            pid_path = Path(config["REPO"] or run_dir.parent.parent) / pid_path
        pid_text = pid_path.read_text().strip()
        if not pid_text.isdigit() or int(pid_text) <= 1:
            raise ValueError("Invalid watcher PID")
        if record["instance_id"] != config["INSTANCE_ID"] or record["pid"] != pid_text:
            return condition(status="failed", detail="Stale acknowledgement: instance or PID mismatch")
        if int(record["observed_at"]) <= 0:
            raise ValueError("Invalid acknowledgement timestamp")
        pid = int(pid_text)
        os.kill(pid, 0)
        # /proc stat includes a parenthesized command name, which may contain spaces.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] in ("Z", "X") or fields[19] != record["process_start_ticks"]:
            return condition(status="failed", detail="Watcher exited or PID was reused")
        return condition(status="passed", detail="Live watcher acknowledged running and armed cleanup gate")
    except FileNotFoundError:
        return condition(status="pending", detail="Watcher acknowledgement or live process missing")
    except ProcessLookupError:
        return condition(status="failed", detail="Acknowledged watcher is dead")
    except (PermissionError, OSError):
        return condition(status="unknown", detail="Cannot verify watcher process")
    except (ValueError, KeyError, IndexError):
        return condition(status="failed", detail="Malformed watcher acknowledgement or process record")


def condition(*, status: str, detail: str) -> dict[str, str]:
    return {"status": status, "detail": detail}


def command_condition(*, deadline: Deadline, args: list[str], detail: str) -> dict[str, str]:
    try:
        result = deadline.run(args=args, maximum=5)
        return condition(status="passed" if result.returncode == 0 else "pending", detail=detail)
    except (OSError, subprocess.TimeoutExpired):
        return condition(status="unknown", detail=f"Probe unavailable or timed out: {detail}")


def check_once(*, run_dir: Path, config: dict[str, str], deadline: Deadline) -> dict:
    checks = {"watcher": watcher_condition(run_dir=run_dir, config=config)}
    checks["tb_forward"] = command_condition(
        deadline=deadline, args=["tmux", "has-session", "-t", config["TB_SESSION"]],
        detail="Recorded TensorBoard forwarding session exists")
    checks["tensorboard"] = command_condition(
        deadline=deadline, args=["curl", "--silent", "--fail", "--output", os.devnull,
                                 "--max-time", "5", "--noproxy", "*", config["TB_URL"]],
        detail="Recorded local TensorBoard URL responds")
    repo = Path(config["REPO"] or run_dir.parent.parent)
    known_hosts = Path(config["SSH_KNOWN_HOSTS"] or str(run_dir / "known_hosts"))
    if not known_hosts.is_absolute():
        known_hosts = repo / known_hosts
    state = config["REMOTE_STATE"] or f"{config['REMOTE_PROJECT'] or '/workspace/toy-act'}/.vast-train/state"
    train = config["TRAIN_SESSION"] or "train"
    remote_command = f"cd {shlex.quote(state)} || exit 3\n"
    remote_command += "printf 'run_status=%s\\n' \"$(cat run-status 2>/dev/null)\"\n"
    remote_command += "printf 'run_name=%s\\n' \"$(cat run-name 2>/dev/null)\"\n"
    for key, marker in (
        ("completed", "completed"), ("failed", "failed"),
        ("backup_running", "backup-running"), ("backup_artifact_ready", "backup-artifact-ready"),
        ("backup_last_succeeded", "backup-last-succeeded"), ("backup_failed", "backup-failed"),
    ):
        remote_command += f"if test -e {marker}; then echo {key}=yes; else echo {key}=no; fi\n"
    for key, session in (("train_session", train), ("backup_session", "ckpt-bkp")):
        remote_command += f"if tmux has-session -t {shlex.quote(session)} 2>/dev/null; then echo {key}=yes; else echo {key}=no; fi\n"
    try:
        result = deadline.run(args=[
            "ssh", "-o", f"UserKnownHostsFile={known_hosts}", "-o", "StrictHostKeyChecking=yes",
            "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-p", config["SSH_PORT"],
            f"root@{config['SSH_HOST']}", remote_command], maximum=20)
        lines = result.stdout.splitlines()
        remote = dict(line.split("=", 1) for line in lines)
        if result.returncode or len(lines) != len(REMOTE_KEYS) or set(remote) != set(REMOTE_KEYS):
            raise ValueError("Incomplete remote probe")
        if any(remote[key] not in ("yes", "no") for key in REMOTE_KEYS if key not in ("run_status", "run_name")):
            raise ValueError("Malformed remote probe")
    except (OSError, ValueError, subprocess.TimeoutExpired):
        checks["remote"] = condition(status="unknown", detail="SSH failed, timed out, or returned incomplete state")
        return {"status": "not_ready", "checks": checks}

    terminal = remote["completed"] == "yes" or remote["failed"] == "yes"
    checks["nonterminal"] = condition(
        status="failed" if terminal else "passed", detail="Remote terminal marker present" if terminal else "No terminal markers")
    checks["training"] = condition(
        status="passed" if remote["run_status"] == "running" and remote["train_session"] == "yes" else "pending",
        detail="Remote status must be running and train session must exist")
    backup_failed = remote["backup_failed"] == "yes"
    checks["backup_health"] = condition(
        status="failed" if backup_failed else "passed", detail="Backup failure marker present" if backup_failed else "No backup failure marker")
    for key in ("backup_session", "backup_running", "backup_artifact_ready", "backup_last_succeeded"):
        checks[key] = condition(status="passed" if remote[key] == "yes" else "pending", detail=f"Require {key}")
    ready = all(check["status"] == "passed" for check in checks.values())
    return {"status": "terminal" if terminal else "ready" if ready else "not_ready",
            "run_name": remote["run_name"], "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--wait-seconds", type=float, default=0)
    parser.add_argument("--poll-seconds", type=float, default=5)
    args = parser.parse_args()
    if (not math.isfinite(args.wait_seconds) or not math.isfinite(args.poll_seconds)
            or args.wait_seconds < 0 or args.poll_seconds <= 0):
        parser.error("Wait must be finite and nonnegative; poll interval must be finite and positive")
    # One-shot mode still has a finite overall budget for setup and every probe.
    deadline = Deadline(seconds=args.wait_seconds or 35)
    run_dir = args.run_dir.resolve()
    try:
        config = load_setup(run_dir=run_dir, deadline=deadline)
    except (ValueError, OSError, subprocess.TimeoutExpired):
        print(json.dumps({"status": "invalid", "error": "Missing, invalid, or unreadable setup.env; verify required readiness fields"}))
        return 2
    while True:
        report = check_once(run_dir=run_dir, config=config, deadline=deadline)
        if report["status"] == "ready":
            code = 0
            break
        if report["status"] == "terminal":
            code = 3
            break
        remaining = deadline.end - time.monotonic()
        hard_failure = any(check["status"] == "failed" for check in report["checks"].values())
        if not args.wait_seconds or remaining <= 0 or hard_failure:
            code = 1
            break
        time.sleep(min(args.poll_seconds, remaining))
    print(json.dumps(report, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
