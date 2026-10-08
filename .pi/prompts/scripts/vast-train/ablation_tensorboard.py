"""One local, S3-backed TensorBoard dashboard per sweep iteration.

These services never launch training or destroy rentals. The synchronizer stays
alive after handoff to collect final uploads; cached runs remain available.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
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
        if service.get("disabled"):
            iteration.state["tensorboard"] = service
            iteration.save()
            journal.progress(text="Shared ablation TensorBoard was explicitly stopped; not restarting")
            return
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


def publish_snapshot(*, source: Path, destination: Path) -> None:
    """Preserve the watched inode and append only a matching snapshot's growth."""
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not destination.exists():
        temporary = source.with_name(source.name + ".publish")
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
        return
    previous_size = destination.stat().st_size
    if source.stat().st_size < previous_size:
        raise Blocked(f"Event snapshot shrank: {destination}")
    # Fail closed if a snapshot rewrites history. Never silently replace an
    # existing watched file, because TensorBoard holds its original inode open.
    with source.open("rb") as incoming, destination.open("rb") as cached:
        while chunk := cached.read(1024 * 1024):
            if incoming.read(len(chunk)) != chunk:
                raise Blocked(f"Event snapshot changed existing records: {destination}")
        with destination.open("ab") as output:
            shutil.copyfileobj(incoming, output)
            output.flush()
            os.fsync(output.fileno())
    shutil.copystat(source, destination)


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
        # AWS downloads via temporary files and rename. Keep those partial files
        # outside the watched logdir; publish only successful downloads.
        staging = directory / "tensorboard" / "staging" / combo_id / run_name
        staging.mkdir(parents=True, exist_ok=True, mode=0o700)
        result = journal.run(args=["aws", "s3", "sync", f"s3://toy-act/runs/act_v2/{run_name}/",
                                   str(staging), "--exclude", "*", "--include", "*tfevents*",
                                   "--only-show-errors", "--profile", PROFILE, "--region", REGION],
                             timeout=120, check=False, sensitive=True)
        if result.returncode == 0:
            for source in staging.rglob("*tfevents*"):
                if not source.is_file() or source.name.endswith(".publish"):
                    continue
                destination = target / source.relative_to(staging)
                if (destination.exists()
                        and destination.stat().st_size == source.stat().st_size
                        and destination.stat().st_mtime_ns == source.stat().st_mtime_ns):
                    continue
                publish_snapshot(source=source, destination=destination)
        results[combo_id] = {"run": run_name, "ok": result.returncode == 0, "checked_at": now()}
    write_json(path=directory / "tensorboard" / "sync-status.json", value=results)
    return results


def stop_owned_sessions(*, directory: Path, service: dict, journal: Journal) -> None:
    expected = {"server_session": f"ablate-tb-{directory.name}",
                "sync_session": f"ablate-sync-{directory.name}"}
    if service["directory"] != str(directory):
        raise Blocked("Dashboard belongs to another iteration")
    # Validate both sessions before stopping either; exact names prevent prefix matches.
    for field, name in expected.items():
        if service[field] != name:
            raise Blocked("Unexpected dashboard session identity")
        if session_exists(name=name):
            owner = journal.run(args=["tmux", "show-options", "-v", "-t", name, "@iteration"])
            if owner.stdout.strip() != str(directory):
                raise Blocked(f"Unrelated tmux session occupies {name}")
    for name in expected.values():
        if session_exists(name=name):
            journal.run(args=["tmux", "kill-session", "-t", f"={name}"])
            if session_exists(name=name):
                raise Blocked(f"Dashboard session still exists after stop: {name}")


def stop_dashboard(*, directory: Path) -> None:
    """Stop only iteration-owned services; retain logs/cache and disable auto-recovery."""
    from iteration import Iteration

    with lock(directory / "driver.lock"):
        iteration = Iteration(directory=directory)
        root = directory / "tensorboard"
        service = json.loads((root / "service.json").read_text())
        if iteration.manifest["kind"] != "sweep":
            raise Blocked("Dashboard does not belong to a sweep")
        journal = Journal(directory=root)
        stop_owned_sessions(directory=directory, service=service, journal=journal)
        service.update(disabled=True, stopped_at=now())
        write_json(path=root / "service.json", value=service)
        iteration.state["tensorboard"] = service
        iteration.save()
        journal.event(kind="tensorboard_stopped", port=service["port"])
        print("Stopped this iteration's TensorBoard server and synchronizer; cache/logs retained")


def refresh_dashboard(*, directory: Path) -> None:
    """Recover only a saved dashboard; never provision or execute training."""
    from iteration import Iteration

    with lock(directory / "driver.lock"):
        iteration = Iteration(directory=directory)
        root = directory / "tensorboard"
        service = json.loads((root / "service.json").read_text())
        if service["directory"] != str(directory) or iteration.manifest["kind"] != "sweep":
            raise Blocked("Dashboard does not belong to this sweep")
        journal = Journal(directory=root)
        stop_owned_sessions(directory=directory, service=service, journal=journal)
        with lock(root / "sync.lock"):
            results = sync_once(directory=directory, journal=journal)
            journal.event(kind="tensorboard_refresh", results=results)
            if not all(result["ok"] for result in results.values()):
                raise Blocked(f"Dashboard download failed; see {root / 'driver.log'}")
        # Only an explicit refresh opts a deliberately stopped dashboard back in.
        service.pop("disabled", None)
        service.pop("stopped_at", None)
        write_json(path=root / "service.json", value=service)
        ensure_dashboard(iteration)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["serve", "sync", "refresh", "stop"])
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    directory = args.directory.resolve()
    root = directory / "tensorboard"
    if args.mode in {"refresh", "stop"}:
        # Report blockers concisely instead of a traceback; callers surface the
        # diagnostic path and never fall back to stopping an unrelated service.
        try:
            if args.mode == "stop":
                stop_dashboard(directory=directory)
            else:
                refresh_dashboard(directory=directory)
        except (Blocked, ValueError, KeyError, TypeError, IndexError, OSError) as error:
            print(f"Blocked: {error}", file=sys.stderr)
            raise SystemExit(1)
        return
    if args.mode == "serve":
        service = json.loads((root / "service.json").read_text())
        with (root / "server.log").open("a") as output:
            os.dup2(output.fileno(), 1)
            os.dup2(output.fileno(), 2)
            os.execv(sys.executable, [sys.executable, "-m", "tensorboard.main", "--logdir",
                                     str(root / "logs"), "--host", "127.0.0.1", "--port",
                                     str(service["port"]), "--reload_interval=5"])
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
