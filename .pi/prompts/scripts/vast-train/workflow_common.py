"""Shared, credential-safe I/O for local Vast workflows (standard library only)."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import codecs
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time

HELPERS = Path(__file__).resolve().parent
REPO = HELPERS.parents[3]
PROFILE = "toy-pickplace-backup"
REGION = "ap-south-1"
IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime"
MAX_PRICE = 0.80
HEARTBEAT_SECONDS = 30


class Blocked(RuntimeError):
    """An ambiguous external state must not authorize a retry or cleanup."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def redact(text: str) -> str:
    text = re.sub(r"https?://[^\s\"'<>]*\?[^\s\"'<>]*", "[redacted URL]", text)
    text = re.sub(
        r"(?i)((?:AWS_[A-Z_]*(?:KEY|TOKEN)[A-Z_]*|VAST_API_KEY|jupyter_token)[\"']?\s*[=:]\s*[\"']?)[^\s,\"']+",
        r"\1[redacted]", text)
    for key, value in os.environ.items():
        if value and len(value) >= 8 and (key.startswith("AWS_") and ("KEY" in key or "TOKEN" in key)
                                         or key == "VAST_API_KEY"):
            text = text.replace(value, "[redacted]")
    return text


def safe_value(value: object) -> object:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, list):
        return [safe_value(item) for item in value]
    if isinstance(value, dict):
        return {key: "[redacted]" if re.fullmatch(
            r"(?i)AWS_[A-Z_]*(?:KEY|TOKEN)[A-Z_]*|VAST_API_KEY|jupyter_token", key)
                else safe_value(item) for key, item in value.items()}
    return value


def atomic_write_bytes(*, path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_write(*, path: Path, text: str) -> None:
    atomic_write_bytes(path=path, content=text.encode("utf-8"))


def write_json(*, path: Path, value: dict) -> None:
    atomic_write(path=path, text=json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_env(*, path: Path, values: dict) -> None:
    atomic_write(path=path, text="".join(f"{key}={shlex.quote(str(value))}\n" for key, value in values.items()))


@contextmanager
def lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("a") as file:
        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise Blocked(f"Another driver owns {path.parent}") from error
        yield


def capture_command(*, args: list[str], timeout: float, input_text: str | None,
                    env: dict | None, on_output: Callable[[str], None] | None,
                    on_heartbeat: Callable[[], None] | None = None) -> subprocess.CompletedProcess:
    """Stream complete lines safely, while retaining separate parseable outputs.

    Anonymous mode-600 temporary files avoid pipe deadlocks and keep sensitive
    stdin/output out of process arguments and persistent raw capture files.
    pread does not change the child's shared file offset.
    """
    with tempfile.TemporaryFile() as stdin, tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        stdin.write((input_text or "").encode())
        stdin.seek(0)
        process = subprocess.Popen(args, cwd=REPO, stdin=stdin, stdout=stdout, stderr=stderr,
                                   env=env, start_new_session=True)
        streams = [{"file": file, "offset": 0, "pending": "",
                    "decoder": codecs.getincrementaldecoder("utf-8")(errors="replace")}
                   for file in (stdout, stderr)]
        started = time.monotonic()
        next_heartbeat = started + HEARTBEAT_SECONDS

        def drain(*, final: bool = False) -> None:
            if on_output is None:
                return
            for stream in streams:
                while block := os.pread(stream["file"].fileno(), 65536, stream["offset"]):
                    stream["offset"] += len(block)
                    stream["pending"] += stream["decoder"].decode(block)
                    while "\n" in stream["pending"]:
                        line, stream["pending"] = stream["pending"].split("\n", 1)
                        on_output(line + "\n")
                if final:
                    stream["pending"] += stream["decoder"].decode(b"", final=True)
                    if stream["pending"]:
                        on_output(stream["pending"] + "\n")
                        stream["pending"] = ""

        try:
            while process.poll() is None:
                drain()
                if on_heartbeat is not None and time.monotonic() >= next_heartbeat:
                    on_heartbeat()
                    next_heartbeat = time.monotonic() + HEARTBEAT_SECONDS
                if time.monotonic() - started >= timeout:
                    raise subprocess.TimeoutExpired(args, timeout)
                time.sleep(0.1)
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            drain(final=True)
        output = os.pread(stdout.fileno(), os.fstat(stdout.fileno()).st_size, 0).decode(errors="replace")
        errors = os.pread(stderr.fileno(), os.fstat(stderr.fileno()).st_size, 0).decode(errors="replace")
        return subprocess.CompletedProcess(args, process.returncode, output, errors)


class Journal:
    def __init__(self, *, directory: Path) -> None:
        self.directory = directory
        self.started = time.monotonic()
        self.stage = "initializing"

    def progress(self, *, text: str) -> None:
        elapsed = int(time.monotonic() - self.started)
        print(f"[{elapsed // 60:02d}:{elapsed % 60:02d}] {redact(text)}", file=sys.stderr, flush=True)

    def set_stage(self, *, text: str) -> None:
        self.stage = text
        self.progress(text=text)

    def event(self, *, kind: str, **fields) -> None:
        record = {"time": now(), "kind": kind, **fields}
        # Sanitize values before serialization so redaction cannot corrupt JSONL.
        text = json.dumps(safe_value(record), sort_keys=True)
        with (self.directory / "events.jsonl").open("a") as file:
            file.write(text + "\n")
            file.flush()
            os.fsync(file.fileno())
        self.log(text=text)
        # Only display the event name and safe identity fields, not raw API records.
        if kind != "state_saved":
            identity = " ".join(f"{key}={fields[key]}" for key in ("combo", "instance_id") if key in fields)
            self.progress(text=f"{kind.replace('_', ' ')} {identity}".rstrip())

    def log(self, *, text: str) -> None:
        with (self.directory / "driver.log").open("a") as file:
            file.write(f"[{now()}] {redact(text)}\n")

    def run(self, *, args: list[str], timeout: float = 60, input_text: str | None = None,
            log_path: Path | None = None, check: bool = True, sensitive: bool = False,
            env: dict | None = None) -> subprocess.CompletedProcess:
        # Never log arguments: remote scripts may contain credentials or signed URLs.
        started = time.monotonic()
        command = Path(args[0]).name
        self.log(text=f"command started: {command}; timeout={timeout}s")
        self.progress(text=f"{self.stage}: {command} started (timeout {timeout:g}s)")
        output_file = None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            output_file = log_path.open("a")
            output_file.write(f"[{now()}] {Path(args[0]).name}: started\n")
            output_file.flush()

        def record_output(text: str) -> None:
            output_file.write(f"[{now()}] {redact(text)}")
            output_file.flush()
            print(redact(text), end="", file=sys.stderr, flush=True)

        try:
            result = capture_command(args=args, timeout=timeout, input_text=input_text, env=env,
                                     on_output=record_output if output_file is not None and not sensitive else None,
                                     on_heartbeat=lambda: self.progress(
                                         text=f"{self.stage}: waiting for {command} ({time.monotonic() - started:.0f}s elapsed)"))
            if output_file is not None:
                if sensitive:
                    output_file.write("[sensitive output omitted]\n")
                output_file.write(f"[{now()}] {Path(args[0]).name}: rc={result.returncode}\n")
        except (OSError, subprocess.TimeoutExpired) as error:
            self.log(text=f"command unavailable/timed out: {Path(args[0]).name}")
            if output_file is not None:
                output_file.write(f"[{now()}] {Path(args[0]).name}: unavailable/timed out\n")
            raise Blocked(f"{Path(args[0]).name} unavailable/timed out; external state is unknown") from error
        finally:
            if output_file is not None:
                output_file.close()
        self.log(text=f"command finished: {Path(args[0]).name}; rc={result.returncode}; elapsed={time.monotonic() - started:.1f}s")
        self.progress(text=f"{self.stage}: {command} finished (exit {result.returncode}, {time.monotonic() - started:.1f}s)")
        if check and result.returncode:
            raise Blocked(f"{Path(args[0]).name} failed (exit {result.returncode}); see local logs")
        return result


def load_vast_key(journal: Journal) -> None:
    if os.environ.get("VAST_API_KEY"):
        return
    stored = Path.home() / ".config/vastai/vast_api_key"
    key = stored.read_text().strip() if stored.is_file() else ""
    if not key and (REPO / ".env").is_file():
        result = journal.run(
            args=["bash", "-c", 'source "$1" >/dev/null 2>&1 || exit 1; printf "%s" "${VAST_API_KEY:-}"',
                  "vast-key", str(REPO / ".env")], sensitive=True, timeout=5)
        key = result.stdout.strip()
    if not key or any(character.isspace() for character in key):
        raise Blocked("VAST_API_KEY is missing/invalid in environment, CLI config, or .env")
    os.environ["VAST_API_KEY"] = key


def instances(journal: Journal) -> list[dict]:
    load_vast_key(journal)
    result = journal.run(args=["vastai", "show", "instances", "--raw"], sensitive=True)
    try:
        records = json.loads(result.stdout)
        if not isinstance(records, list) or any(
                not isinstance(record, dict) or not isinstance(record.get("id"), int)
                or isinstance(record["id"], bool) for record in records):
            raise ValueError("Invalid instance list")
        if len({record["id"] for record in records}) != len(records):
            raise ValueError("Duplicate instance IDs")
        return records
    except (ValueError, TypeError) as error:
        raise Blocked("Vast returned malformed instance data; state is unknown") from error


def ssh_args(*, run: dict, command: str) -> list[str]:
    return ["ssh", "-o", f"UserKnownHostsFile={run['known_hosts']}", "-o", "StrictHostKeyChecking=yes",
            "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3", "-p", str(run["port"]), f"root@{run['host']}", command]


def scp_args(*, run: dict, source: Path, destination: str) -> list[str]:
    return ["scp", "-o", f"UserKnownHostsFile={run['known_hosts']}", "-o", "StrictHostKeyChecking=yes",
            "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-P", str(run["port"]),
            str(source), f"root@{run['host']}:{destination}"]


def git_preflight(*, journal: Journal, expected: str | None = None) -> str:
    args = ["bash", str(HELPERS / "git-preflight.sh")]
    if expected is not None:
        args += ["--expected-commit", expected]
    result = journal.run(args=args, timeout=120, log_path=journal.directory / "preflight.log")
    sha = result.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40,64}", sha):
        raise Blocked("Git preflight returned an invalid commit")
    return sha
