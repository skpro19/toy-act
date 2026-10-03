"""Bounded, resumable setup and watcher handoff for one accepted rental."""

from collections.abc import Callable
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import time
import tomllib

from workflow_common import (
    Blocked, HELPERS, PROFILE, REGION, REPO, Journal, atomic_write, digest,
    instances, scp_args, ssh_args, write_env, write_json,
)


def local_preflight(journal: Journal) -> None:
    for command in ("vastai", "aws", "jq", "ssh", "ssh-keyscan", "scp", "git", "tmux", "flock", "ss", "curl", "uv", "timeout"):
        if shutil.which(command) is None:
            raise Blocked(f"Missing required command: {command}")
    instances(journal)
    journal.run(args=["aws", "sts", "get-caller-identity", "--profile", PROFILE], sensitive=True)
    journal.run(args=["aws", "s3", "ls", "s3://toy-act/checkpoints/act_v2/", "--profile", PROFILE], sensitive=True)
    journal.run(args=["uv", "lock", "--check"], log_path=journal.directory / "preflight.log")


def dataset_path(config: dict) -> str:
    path = Path(config["dataset"])
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "datasets":
        raise Blocked("Dataset must be a project-relative path under datasets/")
    return path.as_posix()


def dataset_head(*, journal: Journal, key: str) -> dict | None:
    result = journal.run(args=["aws", "s3api", "head-object", "--bucket", "toy-act", "--key", key,
                               "--profile", PROFILE, "--region", REGION], check=False, sensitive=True)
    if result.returncode:
        if "(404)" in result.stderr or "(NoSuchKey)" in result.stderr:
            return None
        raise Blocked("Dataset S3 HEAD failed; do not interpret access/network errors as absence")
    try:
        head = json.loads(result.stdout)
        if not isinstance(head["ContentLength"], int) or head["ContentLength"] <= 0:
            raise ValueError("Invalid dataset size")
        return head
    except (ValueError, KeyError, TypeError) as error:
        raise Blocked("Invalid dataset S3 metadata") from error


def validate_dataset_source(*, journal: Journal, config: dict, config_path: Path) -> None:
    key = dataset_path(config)
    if not (REPO / key).is_file() and dataset_head(journal=journal, key=key) is None:
        raise Blocked(f"Dataset absent locally and on S3: {key}")
    # Config validation itself is delegated to the pinned project module.
    code = """from pathlib import Path
import sys
from scripts.train_v2 import load_config
from scripts.rollout import camera_names_from_image_keys
c = load_config(path=Path(sys.argv[1]))
camera_names_from_image_keys(image_keys=tuple(c['image_keys']))
"""
    journal.run(args=["uv", "run", "--frozen", "python", "-c", code, str(config_path)],
                timeout=120, log_path=journal.directory / "preflight.log")


def transfer_credentials(*, journal: Journal, run: dict) -> None:
    credentials = journal.run(args=["aws", "configure", "export-credentials", "--profile", PROFILE,
                                    "--format", "env-no-export"], sensitive=True).stdout
    allowed = {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"}
    values = {}
    for line in credentials.splitlines():
        key, separator, value = line.partition("=")
        if separator and key in allowed:
            try:
                parts = shlex.split(value)
            except ValueError as error:
                raise Blocked("AWS credential export has malformed quoting") from error
            if len(parts) != 1 or key in values:
                raise Blocked("AWS credential export has duplicate/invalid fields")
            values[key] = parts[0]
    if not values.get("AWS_ACCESS_KEY_ID") or not values.get("AWS_SECRET_ACCESS_KEY"):
        raise Blocked("AWS credential export is incomplete")
    values.update(AWS_REGION=REGION, AWS_DEFAULT_REGION=REGION, S3_BUCKET="toy-act")
    fd, name = tempfile.mkstemp(prefix="toy-act-aws-")
    os.close(fd)
    try:
        write_env(path=Path(name), values=values)
        journal.run(args=scp_args(run=run, source=Path(name), destination="/workspace/toy-act/.vast-train/s3-env.env"),
                    timeout=120, sensitive=True)
        journal.run(args=ssh_args(run=run, command="chmod 600 /workspace/toy-act/.vast-train/s3-env.env"))
    finally:
        Path(name).unlink(missing_ok=True)


def prepare_dataset(*, journal: Journal, run: dict, config: dict, log_path: Path) -> None:
    key = dataset_path(config)
    local = REPO / key
    head = dataset_head(journal=journal, key=key)
    sha = ""
    if local.is_file():
        size = local.stat().st_size
        sha = digest(local)
        if head is None or head["ContentLength"] != size:
            env = dict(os.environ, AWS_MAX_ATTEMPTS="10", AWS_RETRY_MODE="adaptive")
            for attempt in range(3):
                result = journal.run(args=["aws", "s3", "cp", str(local), f"s3://toy-act/{key}",
                                           "--profile", PROFILE, "--region", REGION, "--only-show-errors"],
                                     timeout=7200, env=env, check=False, log_path=log_path)
                if result.returncode == 0:
                    break
                time.sleep(5 * (attempt + 1))
            else:
                raise Blocked("Dataset upload failed after three attempts")
            head = dataset_head(journal=journal, key=key)
    if head is None:
        raise Blocked("Dataset S3 object is absent")
    size = head["ContentLength"]
    write_json(path=Path(run["run_dir"]) / "dataset.json", value={
        "path": key, "s3_uri": f"s3://toy-act/{key}", "bytes": size,
        "expected_local_sha256": sha or None, "s3_etag": head.get("ETag"),
        "s3_version_id": head.get("VersionId"),
    })
    journal.event(kind="dataset_download_requested", instance_id=run["instance_id"], path=key, bytes=size)
    script = f"""set -euo pipefail
cd /workspace/toy-act
set -a
source .vast-train/s3-env.env
set +a
mkdir -p {shlex.quote(str(Path(key).parent))}
AWS_MAX_ATTEMPTS=10 AWS_RETRY_MODE=adaptive aws s3 cp {shlex.quote('s3://toy-act/' + key)} {shlex.quote(key)} --only-show-errors
export EXPECTED_BYTES={size}
export EXPECTED_SHA={shlex.quote(sha)}
""" + (HELPERS / "verify-dataset.sh").read_text()
    journal.run(args=ssh_args(run=run, command="bash -s"), input_text=script, timeout=7200, log_path=log_path)


def allocate_lease(*, journal: Journal, run: dict) -> dict:
    owner = f"toy-act-{run['instance_id']}"
    # Recover a lease allocated just before an interrupted metadata write.
    existing = []
    for path in Path('/tmp').glob('toy-act-local-wrapper-*.owner'):
        try:
            if path.read_text().strip() == owner:
                existing.append(path)
        except FileNotFoundError:
            continue  # Another run's watcher released its lease during the scan.
    if len(existing) > 1:
        raise Blocked("Multiple local leases for the same instance")
    if existing:
        index = int(existing[0].name.removeprefix('toy-act-local-wrapper-').removesuffix('.owner'))
        return {"INDEX": index, "SSH_SESSION": f"act-ssh-{index}", "TB_SESSION": f"act-tb-{index}",
                "TB_PORT": 6006 + index, "LOCAL_OWNER_FILE": str(existing[0])}
    result = journal.run(args=["bash", str(HELPERS / "local-wrapper-lease.sh"), "allocate", owner], timeout=30)
    fields = result.stdout.splitlines()
    if len(fields) != 4 or not fields[0].isdigit() or not fields[3].isdigit():
        raise Blocked("Invalid local wrapper allocation")
    index, ssh, tb, port = fields
    return {"INDEX": int(index), "SSH_SESSION": ssh, "TB_SESSION": tb, "TB_PORT": int(port),
            "LOCAL_OWNER_FILE": f"/tmp/toy-act-local-wrapper-{index}.owner"}


def setup(*, journal: Journal, combo: dict, config_path: Path,
          commit: str, save: Callable[[], None]) -> None:
    run = combo["run"]
    run_dir = Path(run["run_dir"])
    config = tomllib.loads(config_path.read_text())
    log_path = run_dir / "setup.log"
    script = f"export GIT_COMMIT={shlex.quote(commit)}\n" + (HELPERS / "remote-setup.sh").read_text()
    journal.run(args=ssh_args(run=run, command="bash -s"), input_text=script, timeout=2400, log_path=log_path)
    journal.run(args=scp_args(run=run, source=config_path, destination="/workspace/toy-act/.vast-train/train-config.toml"),
                timeout=120, log_path=log_path)
    transfer_credentials(journal=journal, run=run)
    prepare_dataset(journal=journal, run=run, config=config, log_path=log_path)
    combo["lease"] = allocate_lease(journal=journal, run=run)
    save()
    values = {
        "REPO": str(REPO), "INSTANCE_ID": run["instance_id"], "INSTANCE_LABEL": run["label"],
        "SSH_HOST": run["host"], "SSH_PORT": run["port"], "SSH_KNOWN_HOSTS": run["known_hosts"],
        "GIT_COMMIT": commit, "OFFER_ID": run["offer"]["id"], "MACHINE_ID": run["offer"]["machine_id"],
        "OFFER_DPH_TOTAL": run["offer"]["dph_total"], "ACTUAL_DPH_TOTAL": run["actual_price"],
        "CONFIG_SLUG": config["name"], "RUN_NAME": "", "RESOLVED_CONFIG": str(config_path), "DATASET_PATH": config["dataset"],
        "CHECKPOINT_EVERY": config["checkpoint_every"], "STEPS": config["steps"],
        "S3_CHECKPOINT_BASE": "checkpoints/act_v2", "S3_RUNS_BASE": "runs/act_v2",
        "AWS_PROFILE": PROFILE, "AWS_REGION": REGION, "WATCHER_ASSUME_STARTED": "yes",
        "WATCHER_PID": str(run_dir / "watcher.pid"), "WATCHER_LOG": str(run_dir / "watcher.log"),
        "REPORT": str(run_dir / "report.txt"), **combo["lease"],
    }
    values["TB_URL"] = f"http://localhost:{values['TB_PORT']}/"
    write_env(path=run_dir / "setup.env", values=values)
    combo["setup_sha256"] = digest(run_dir / "setup.env")
    watcher = run_dir / "watcher.sh"
    atomic_write(path=watcher, text=(HELPERS / "local-watcher.sh").read_text())
    combo["watcher_sha256"] = digest(watcher)
    combo["status"] = "ready_to_launch"
    save()


def ensure_forwarding(*, journal: Journal, combo: dict) -> None:
    run, lease = combo["run"], combo["lease"]
    restored = journal.run(args=["bash", str(HELPERS / "local-wrapper-lease.sh"), "restore",
                                 f"toy-act-{run['instance_id']}", str(lease["INDEX"])], timeout=30)
    if restored.stdout.splitlines() != [str(lease["INDEX"]), lease["SSH_SESSION"],
                                       lease["TB_SESSION"], str(lease["TB_PORT"])]:
        raise Blocked("Recorded forwarding lease metadata does not match restored ownership")
    base = ssh_args(run=run, command="")[:-1]
    for session, command in (
        (lease["SSH_SESSION"], base),
        (lease["TB_SESSION"], base[:-1] + ["-o", "ExitOnForwardFailure=yes", "-N", "-L",
                                         f"{lease['TB_PORT']}:127.0.0.1:6006", base[-1]]),
    ):
        result = journal.run(args=["tmux", "has-session", "-t", session], check=False)
        if result.returncode:
            journal.run(args=["tmux", "new-session", "-d", "-s", session, shlex.join(command)])


def ensure_watcher(*, journal: Journal, combo: dict) -> None:
    run_dir = Path(combo["run_dir"])
    if not (run_dir / "setup.env").is_file() or digest(run_dir / "setup.env") != combo.get("setup_sha256"):
        raise Blocked("Saved watcher setup is missing/modified; refusing restart")
    # The watcher holds this lock for its entire lifetime. No PID-only liveness check.
    with (run_dir / "watcher.lock").open("a") as file:
        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        if (run_dir / "report.txt").exists():
            raise Blocked("Watcher already produced a report; reconcile before restarting")
        script = run_dir / "watcher.sh"
        # Never reconstruct a missing/modified historical watcher from new code.
        if not script.is_file() or digest(script) != combo.get("watcher_sha256"):
            raise Blocked("Saved watcher snapshot is missing/modified; refusing replacement")
    # Release our probe lock before spawning: the child acquires it nonblocking.
    # The driver holds the iteration lock throughout this operation.
    with (run_dir / "watcher.out").open("a") as output:
        process = subprocess.Popen(["bash", str(script), str(run_dir)], cwd=REPO,
                                   stdin=subprocess.DEVNULL, stdout=output, stderr=output,
                                   start_new_session=True, close_fds=True)
    journal.event(kind="watcher_spawned", combo=combo["id"], pid=process.pid)
    time.sleep(0.2)


def launch(*, journal: Journal, combo: dict, save: Callable[[], None]) -> None:
    ensure_forwarding(journal=journal, combo=combo)
    combo["status"] = "awaiting_handoff"
    combo["training_launch_requested"] = True
    save()  # From now on no provisioning/helper cleanup can destroy this rental.
    ensure_watcher(journal=journal, combo=combo)
    journal.run(args=ssh_args(run=combo["run"], command="bash -s"),
                input_text=(HELPERS / "start-services.sh").read_text(), timeout=120,
                log_path=Path(combo["run_dir"]) / "setup.log")
    combo["services_started"] = True
    save()


def handoff(*, journal: Journal, combo: dict, save: Callable[[], None], wait_seconds: int = 690) -> int:
    run_dir = Path(combo["run_dir"])
    result = journal.run(args=["uv", "run", "--frozen", "python", str(HELPERS / "check-handoff.py"),
                               str(run_dir), "--wait-seconds", str(wait_seconds)],
                         timeout=max(wait_seconds, 35) + 15, check=False)
    try:
        report = json.loads(result.stdout)
    except ValueError as error:
        raise Blocked("Handoff checker returned invalid output") from error
    write_json(path=run_dir / "handoff.json", value=report)
    if result.returncode == 0 and report.get("status") == "ready":
        combo["status"] = "running"
        if report.get("run_name"):
            combo["remote_run_name"] = report["run_name"]
        save()
        journal.event(kind="handoff_passed", combo=combo["id"], instance_id=combo["instance_id"])
    else:
        combo["status"] = "awaiting_handoff"
        save()
        if result.returncode != 3:
            raise Blocked(f"Handoff not ready (exit {result.returncode}); watcher owns cleanup; see {run_dir / 'handoff.json'}")
    return result.returncode
