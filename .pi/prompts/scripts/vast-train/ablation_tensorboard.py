"""One local, S3-backed TensorBoard dashboard per sweep iteration.

These services never launch training or destroy rentals. The synchronizer stays
alive after handoff to collect final uploads; cached runs remain available.
"""

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from workflow_common import (
    Blocked, HELPERS, PROFILE, REGION, REPO, Journal, lock, now, write_json,
)


def session_exists(*, name: str) -> bool:
    return subprocess.run(["tmux", "has-session", "-t", f"={name}"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def ensure_session(*, journal: Journal, name: str, directory: Path, command: list[str]) -> None:
    """Only reuse a session explicitly owned by this iteration."""
    import shlex

    if session_exists(name=name):
        result = journal.run(args=["tmux", "show-options", "-v", "-t", name, "@iteration"], check=False)
        if result.stdout.strip() != str(directory):
            raise Blocked(f"Unrelated tmux session occupies {name}")
        return
    # Set ownership in the same tmux command sequence as creation.
    journal.run(args=["tmux", "new-session", "-d", "-s", name, "-c", str(REPO),
                      shlex.join(command), ";", "set-option", "-t", name,
                      "@iteration", str(directory)], timeout=30)


def port_available(*, port: int) -> bool:
    with socket.socket() as connection:
        try:
            connection.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def ensure_dashboard(iteration) -> None:
    if iteration.manifest["kind"] != "sweep":
        return
    directory = iteration.directory.resolve()
    root = directory / "tensorboard"
    root.mkdir(mode=0o700, exist_ok=True)
    (root / "logs").mkdir(mode=0o700, exist_ok=True)
    journal = iteration.journal
    # Serialize allocation across iterations as well as recovery within one.
    with lock(REPO / ".vast-train-local" / "tensorboard-allocation.lock"):
        metadata_path = root / "service.json"
        if metadata_path.exists():
            service = json.loads(metadata_path.read_text())
            if service["directory"] != str(directory):
                raise Blocked("TensorBoard service belongs to another iteration")
        else:
            port = next((port for port in range(16006, 17006) if port_available(port=port)), None)
            if port is None:
                raise Blocked("No free ablation TensorBoard port")
            service = {"directory": str(directory), "port": port,
                       "url": f"http://localhost:{port}/",
                       "server_session": f"ablate-tb-{directory.name}",
                       "sync_session": f"ablate-sync-{directory.name}"}
            write_json(path=metadata_path, value=service)
        server = service["server_session"]
        if not session_exists(name=server) and not port_available(port=service["port"]):
            raise Blocked(f"Saved TensorBoard port {service['port']} is occupied; see {metadata_path}")
        ensure_session(journal=journal, name=server, directory=directory,
                       command=["uv", "run", "--frozen", "--only-group", "train", "python",
                                str(HELPERS / "ablation_tensorboard.py"), "serve", str(directory)])
        # Do not report a healthy dashboard merely because tmux exists.
        for _ in range(30):
            result = journal.run(args=["curl", "--silent", "--fail", "--output", os.devnull,
                                       "--max-time", "2", "--noproxy", "*", service["url"]],
                                 timeout=5, check=False)
            if result.returncode == 0:
                break
            if not session_exists(name=server):
                raise Blocked(f"TensorBoard exited; see {root / 'server.log'}")
            time.sleep(1)
        else:
            raise Blocked(f"TensorBoard did not become ready; see {root / 'server.log'}")
        ensure_session(journal=journal, name=service["sync_session"], directory=directory,
                       command=["uv", "run", "--frozen", "--only-group", "train", "python",
                                str(HELPERS / "ablation_tensorboard.py"), "sync", str(directory)])
    iteration.state["tensorboard"] = service
    iteration.save()
    journal.progress(text=f"Shared ablation TensorBoard: {service['url']} (S3-backed; updates are delayed)")


def sync_once(*, directory: Path, journal: Journal) -> dict:
    """Use atomic state snapshots and only explicitly recorded run prefixes."""
    state = json.loads((directory / "state.json").read_text())
    results = {}
    for combo in state["combos"]:
        run_name = combo.get("remote_run_name")
        if not run_name:
            continue
        # Restrict S3 prefixes and local paths, even for corrupt runtime state.
        if Path(run_name).name != run_name or run_name in {".", ".."} or "/" in run_name:
            raise ValueError("Invalid registered TensorBoard run name")
        combo_id = combo["id"]
        if not combo_id.isdigit():
            raise ValueError("Invalid TensorBoard combo ID")
        target = directory / "tensorboard" / "logs" / combo_id / run_name
        target.mkdir(parents=True, exist_ok=True, mode=0o700)
        result = journal.run(args=["aws", "s3", "sync", f"s3://toy-act/runs/act_v2/{run_name}/",
                                   str(target), "--exclude", "*", "--include", "*tfevents*",
                                   "--only-show-errors", "--profile", PROFILE, "--region", REGION],
                             timeout=120, check=False, sensitive=True)
        results[combo_id] = {"run": run_name, "ok": result.returncode == 0, "checked_at": now()}
    write_json(path=directory / "tensorboard" / "sync-status.json", value=results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["serve", "sync"])
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    directory = args.directory.resolve()
    root = directory / "tensorboard"
    if args.mode == "serve":
        service = json.loads((root / "service.json").read_text())
        with (root / "server.log").open("a") as output:
            os.dup2(output.fileno(), 1)
            os.dup2(output.fileno(), 2)
            os.execv(sys.executable, [sys.executable, "-m", "tensorboard.main", "--logdir",
                                     str(root / "logs"), "--host", "127.0.0.1", "--port",
                                     str(service["port"])])
    with lock(root / "sync.lock"):
        journal = Journal(directory=root)
        while True:
            try:
                results = sync_once(directory=directory, journal=journal)
                journal.event(kind="tensorboard_sync", results=results)
            except (OSError, ValueError, KeyError, TypeError, Blocked) as error:
                journal.event(kind="tensorboard_sync_failed", reason=str(error))
            time.sleep(30)


if __name__ == "__main__":
    main()
