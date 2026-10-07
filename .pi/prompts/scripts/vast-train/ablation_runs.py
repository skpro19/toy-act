#!/usr/bin/env python3
"""Read-only reports for one saved ablation iteration.

The ``runs`` mode lists run names, bucket links, and the sweep's fixed and
ablated params. The ``report`` mode prints a full tabular report (identity,
configs, execution, verification, records, S3, cache, progress). Neither mode
writes files, downloads objects, or calls Vast, and neither prints sensitive
instance records.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tomllib

from workflow_common import PROFILE, REGION

FINAL = {"done", "failed", "verification_failed"}
S3_BUCKET = "s3://toy-act"
CHECKPOINT_PREFIX = f"{S3_BUCKET}/checkpoints/act_v2"
RUNS_PREFIX = f"{S3_BUCKET}/runs/act_v2"
PROGRESS_RE = re.compile(
    r"progress: epoch (?P<epoch>\d+) step (?P<step>\d+)/(?P<total>\d+).*?loss=(?P<loss>[\d.]+)")


def read_json(*, path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def read_report(*, run_dir: Path) -> dict:
    """Parse the watcher's `key: value` report without trusting its ordering."""
    report = {}
    path = run_dir / "report.txt"
    if not path.is_file():
        return report
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return report
    for line in lines:
        key, separator, value = line.partition(": ")
        if separator and key not in report:
            report[key] = value
    return report


def session_exists(*, name: str) -> bool:
    return subprocess.run(["tmux", "has-session", "-t", f"={name}"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def port_state(*, iteration_id: str, port: int | None) -> str:
    if session_exists(name=f"ablate-tb-{iteration_id}"):
        return "own"
    if port is None:
        return "unknown"
    with socket.socket() as connection:
        try:
            connection.bind(("127.0.0.1", port))
        except OSError:
            return "held"
    return "free"


def watcher_alive(*, run_dir: Path) -> bool:
    try:
        pid = int((run_dir / "watcher.pid").read_text().strip())
    except (OSError, ValueError):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def human_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"


def modified(path: Path) -> str:
    try:
        stamp = path.stat().st_mtime
    except OSError:
        return "-"
    return datetime.fromtimestamp(stamp, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def render_table(*, headers: list[str], rows: list[list[object]]) -> str:
    text_rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in text_rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def line(cells: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(cells)) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([line(headers), separator, *[line(row) for row in text_rows]])


def directory_inventory(*, directory: Path) -> dict:
    try:
        files = [path for path in directory.iterdir() if path.is_file()]
    except OSError:
        return {"files": 0, "bytes": 0, "newest": "-"}
    if not files:
        return {"files": 0, "bytes": 0, "newest": "-"}
    total = sum(path.stat().st_size for path in files)
    newest = max(files, key=lambda path: path.stat().st_mtime)
    return {"files": len(files), "bytes": total, "newest": newest.name}


def config_row(*, config_path: Path) -> dict:
    try:
        config = tomllib.loads(config_path.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    rollout = config.get("rollout", {})
    steps = rollout.get("n_action_steps", [])
    return {
        "dataset": config.get("dataset", "-"),
        "lr": config.get("lr", "-"),
        "batch_size": config.get("batch_size", "-"),
        "action_chunk_size": config.get("action_chunk_size", "-"),
        "use_z": config.get("use_z", "-"),
        "n_action_steps": ",".join(str(value) for value in steps) or "-",
        "steps": config.get("steps", "-"),
    }


def format_param(*, value: object) -> str:
    """Render a TOML value compactly for a single table cell."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ", ".join(format_param(value=item) for item in value)
    if isinstance(value, dict):
        return ", ".join(f"{key}={format_param(value=item)}" for key, item in value.items())
    return str(value)


def spec_params(*, directory: Path, manifest: dict) -> tuple[dict, dict]:
    """Read the sweep spec snapshot named by the manifest.

    Returns the ``[fixed]`` and ``[grid]`` tables: the parameters held constant
    and the parameters being ablated. Missing or unreadable specs yield empties.
    """
    source = manifest.get("source")
    for entry in manifest.get("inputs", []):
        if entry.get("source") != source or not entry.get("path"):
            continue
        try:
            spec = tomllib.loads((directory / entry["path"]).read_text())
        except (OSError, tomllib.TOMLDecodeError):
            break
        fixed = spec.get("fixed")
        grid = spec.get("grid")
        return (fixed if isinstance(fixed, dict) else {},
                grid if isinstance(grid, dict) else {})
    return {}, {}


def run_name_from_record(*, record: dict) -> str | None:
    """Recover a started run's name from state or its local run records.

    ``remote_run_name`` is recorded only once handoff passes, so a run that
    started and then failed can still be named by `handoff.json` or `report.txt`
    in its run directory.
    """
    if record.get("remote_run_name"):
        return record["remote_run_name"]
    run_dir = Path(record["run_dir"]) if record.get("run_dir") else None
    if run_dir is None:
        return None
    handoff = read_json(path=run_dir / "handoff.json") or {}
    if handoff.get("run_name"):
        return handoff["run_name"]
    return read_report(run_dir=run_dir).get("run_name")


def s3_folders(*, prefix: str, timeout: float = 120) -> set[str] | None:
    """List immediate child folder names under a prefix; None if untrusted."""
    try:
        result = subprocess.run(
            ["aws", "s3", "ls", f"{prefix}/", "--profile", PROFILE, "--region", REGION],
            capture_output=True, text=True, timeout=timeout, env=dict(os.environ), check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    folders = set()
    for line in result.stdout.splitlines():
        if line.lstrip().startswith("PRE "):
            folders.add(line.split("PRE ", 1)[1].strip().rstrip("/"))
    return folders


def started_run_names(*, directory: Path, manifest: dict, state: dict) -> list[str]:
    """Every run started by this iteration, newest first.

    A run is identified by its recorded name and by the S3 folders it wrote.
    Iteration run names all end with `-i<iteration-hash>`, so the bucket listing
    is filtered by that marker.
    """
    names = {name for name in (run_name_from_record(record=record)
                               for record in state.get("combos", [])) if name}
    iteration_id = manifest.get("id", directory.name)
    marker = f"-i{iteration_id.rsplit('-', 1)[-1]}"
    for prefix in (CHECKPOINT_PREFIX, RUNS_PREFIX):
        folders = s3_folders(prefix=prefix)
        if folders is not None:
            names |= {name for name in folders if name.endswith(marker)}
    return sorted(names, reverse=True)


def runs_report(*, directory: Path) -> str:
    """Compact report: fixed params, ablated params, and per-run bucket folders."""
    manifest = read_json(path=directory / "manifest.json")
    state = read_json(path=directory / "state.json")
    if manifest is None or state is None:
        raise ValueError(f"Not a saved iteration: {directory}")
    fixed, grid = spec_params(directory=directory, manifest=manifest)
    names = started_run_names(directory=directory, manifest=manifest, state=state)

    fixed_rows = [[key, format_param(value=value)] for key, value in fixed.items()]
    grid_rows = [[key, format_param(value=value)] for key, value in grid.items()]
    run_rows = [[name, f"{CHECKPOINT_PREFIX}/{name}/", f"{RUNS_PREFIX}/{name}/"]
                for name in names]

    sections = [
        "## Runs\n\n" + render_table(
            headers=["Run name", "Checkpoints folder", "Runs folder"], rows=run_rows),
        "## Fixed params\n\n" + render_table(
            headers=["Param", "Value"], rows=fixed_rows),
        "## Ablated params\n\n" + render_table(
            headers=["Param", "Values"], rows=grid_rows),
    ]
    return "\n\n".join(sections) + "\n"


def s3_objects(*, prefix: str, timeout: float = 120) -> list[dict] | None:
    """List a prefix read-only; None means the listing could not be trusted."""
    try:
        result = subprocess.run(
            ["aws", "s3", "ls", prefix, "--recursive", "--profile", PROFILE, "--region", REGION],
            capture_output=True, text=True, timeout=timeout, env=dict(os.environ), check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    objects = []
    for line in result.stdout.splitlines():
        fields = line.split(None, 3)
        if len(fields) != 4 or not fields[2].isdigit():
            continue
        objects.append({"modified": f"{fields[0]}T{fields[1]}Z", "size": int(fields[2]),
                        "key": fields[3]})
    return objects


def s3_object_state(*, objects: list[dict] | None, match: str) -> str:
    if objects is None:
        return "unknown"
    return "yes" if any(match in item["key"] for item in objects) else "no"


def live_progress(*, run_dir: Path) -> dict | None:
    path = run_dir / "watcher.log"
    if not path.is_file():
        return None
    latest = None
    try:
        for line in path.read_text(errors="replace").splitlines():
            match = PROGRESS_RE.search(line)
            if match:
                latest = match
    except OSError:
        return None
    return latest.groupdict() if latest else None


def report(*, directory: Path, combo: str | None, files: bool, no_s3: bool) -> str:
    manifest = read_json(path=directory / "manifest.json")
    state = read_json(path=directory / "state.json")
    service = read_json(path=directory / "tensorboard/service.json") or {}
    sync = read_json(path=directory / "tensorboard/sync-status.json") or {}
    if manifest is None or state is None:
        raise ValueError(f"Not a saved iteration: {directory}")

    manifest_combos = {entry["id"]: entry for entry in manifest.get("combos", [])}
    combos = [record for record in state.get("combos", [])
              if combo is None or record["id"] == combo]
    if combo is not None and not combos:
        raise ValueError(f"Unknown combo {combo!r}")
    # Read each watcher report once; it covers execution and verification rows.
    reports = {}
    for record in combos:
        values = dict(record.get("report", {}))
        if record.get("run_dir"):
            values.update(read_report(run_dir=Path(record["run_dir"])))
        reports[record["id"]] = values

    sections = []

    # 1. Iteration identity.
    iteration_id = manifest.get("id", directory.name)
    completed = [record for record in combos if record["status"] in FINAL]
    concluded = [record for record in combos
                 if reports[record["id"]].get("outcome") in {"success", "failure"}]
    sections.append(f"## Iteration\n\n" + render_table(headers=["Field", "Value"], rows=[
        ["Iteration", iteration_id],
        ["Spec", manifest.get("source", "-")],
        ["Created (UTC)", manifest.get("created_at", "-")],
        ["Kind", manifest.get("kind", "-")],
        ["Pinned commit", manifest.get("git_commit", "-")],
        ["Driver status", state.get("driver_status", "-")],
        ["Combos selected", len(combos)],
        ["Terminal (state)", len(completed)],
        ["Reports with outcome", len(concluded)],
        ["Recorded port", service.get("port", "-")],
        ["Port state", port_state(iteration_id=iteration_id, port=service.get("port"))],
        ["Shared TensorBoard", service.get("url", "-")],
        ["Directory", str(directory)],
        ["Diagnostics", str(directory / "tensorboard")],
    ]))

    # 2. Configs.
    config_rows = []
    for record in combos:
        entry = manifest_combos.get(record["id"], {})
        values = config_row(config_path=directory / entry.get("config", ""))
        config_rows.append([record["id"], entry.get("config", "-"), values.get("dataset", "-"),
                            values.get("lr", "-"), values.get("batch_size", "-"),
                            values.get("action_chunk_size", "-"), values.get("use_z", "-"),
                            values.get("n_action_steps", "-"), values.get("steps", "-")])
    sections.append("## Configs\n\n" + render_table(
        headers=["Combo", "Config", "Dataset", "lr", "Batch", "Action chunk", "use_z",
                 "n_action_steps", "Steps"], rows=config_rows))

    # 3. Execution.
    execution_rows = []
    for record in combos:
        run = record.get("run", {})
        elapsed = reports[record["id"]].get("elapsed_seconds", "")
        costs = "-"
        try:
            elapsed_hours = float(elapsed) / 3600.0
            costs = f"${float(run.get('actual_price', 0)) * elapsed_hours:.2f}"
        except (TypeError, ValueError):
            elapsed_hours = 0.0
        execution_rows.append([
            record["id"], record["status"], record.get("instance_id", "-"),
            run.get("offer", {}).get("dph_total", "-"), run.get("actual_price", "-"),
            f"{elapsed_hours:.2f}" if elapsed_hours else "-", costs,
            record.get("remote_run_name", "-")])
    sections.append("## Execution\n\n" + render_table(
        headers=["Combo", "Status", "Instance", "Offer $/h", "Actual $/h", "Elapsed h",
                 "Est. cost", "Run name"], rows=execution_rows))

    # 4. Verification.
    verification_rows = []
    for record in combos:
        values = reports[record["id"]]
        verification_rows.append([
            record["id"], values.get("outcome", "-"), values.get("failure_reason", "-"),
            values.get("s3_verification", "-"), values.get("tensorboard_verification", "-"),
            values.get("final_cleanup_status", "-"), values.get("generated", "-")])
    sections.append("## Verification\n\n" + render_table(
        headers=["Combo", "Outcome", "Failure reason", "S3 verification",
                 "TB verification", "Cleanup", "Report generated"], rows=verification_rows))

    # 5. Local records.
    local_rows = []
    for record in combos:
        run_dir = Path(record["run_dir"]) if record.get("run_dir") else None
        inventory = directory_inventory(directory=run_dir) if run_dir else {"files": 0, "bytes": 0, "newest": "-"}
        handoff = read_json(path=run_dir / "handoff.json") if run_dir else None
        local_rows.append([
            record["id"], str(run_dir) if run_dir else "-", inventory["files"],
            human_bytes(inventory["bytes"]), inventory["newest"],
            "alive" if run_dir and watcher_alive(run_dir=run_dir) else "stopped",
            (handoff or {}).get("status", "-"),
            "present" if run_dir and (run_dir / "report.txt").is_file() else "missing"])
    sections.append("## Local records\n\n" + render_table(
        headers=["Combo", "Run dir", "Files", "Total size", "Newest file", "Watcher",
                 "Handoff", "Report"], rows=local_rows))

    # 6. S3 artifacts.
    artifact_rows = []
    prefix_rows = []
    for record in combos:
        run_name = record.get("remote_run_name")
        if not run_name:
            artifact_rows.append([record["id"], "-", "-", "-", "-", "-", "-", "-", "-"])
            prefix_rows.append([record["id"], "-", "-"])
            continue
        checkpoints = f"{S3_BUCKET}/checkpoints/act_v2/{run_name}/"
        runs = f"{S3_BUCKET}/runs/act_v2/{run_name}/"
        prefix_rows.append([record["id"], checkpoints, runs])
        if no_s3:
            artifact_rows.append([record["id"], "skipped", "skipped", "skipped", "skipped",
                                  "skipped", "skipped", "skipped", "skipped"])
            continue
        checkpoint_objects = s3_objects(prefix=checkpoints)
        run_objects = s3_objects(prefix=runs)
        latest = max(checkpoint_objects, key=lambda item: item["key"]) if checkpoint_objects else None
        artifact_rows.append([
            record["id"],
            len(checkpoint_objects) if checkpoint_objects is not None else "unknown",
            human_bytes(sum(item["size"] for item in checkpoint_objects)) if checkpoint_objects is not None else "unknown",
            latest["key"].rsplit("/", 1)[-1] if latest else "-",
            latest["modified"] if latest else "-",
            len(run_objects) if run_objects is not None else "unknown",
            human_bytes(sum(item["size"] for item in run_objects if "tfevents" in item["key"]))
            if run_objects is not None else "unknown",
            s3_object_state(objects=run_objects, match="config.json"),
            s3_object_state(objects=run_objects, match="training-log-tail.txt")])
    sections.append("## S3 artifacts\n\n" + render_table(
        headers=["Combo", "Ckpt objects", "Ckpt size", "Latest checkpoint", "Latest (UTC)",
                 "Run objects", "Event size", "config.json", "log tail"], rows=artifact_rows))
    sections.append("## S3 prefixes\n\n" + render_table(
        headers=["Combo", "Checkpoints", "Runs"], rows=prefix_rows))

    # 7. TensorBoard cache.
    cache_rows = []
    for record in combos:
        run_name = record.get("remote_run_name")
        cached = directory / "tensorboard/logs" / record["id"] / run_name if run_name else None
        inventory = directory_inventory(directory=cached) if cached and cached.is_dir() else {"files": 0, "bytes": 0}
        status = sync.get(record["id"], {})
        cache_rows.append([record["id"], inventory["files"], human_bytes(inventory["bytes"]),
                           status.get("checked_at", "-"), "yes" if status.get("ok") else "-"])
    sections.append("## TensorBoard cache\n\n" + render_table(
        headers=["Combo", "Cached files", "Cached size", "Last sync (UTC)", "Sync ok"],
        rows=cache_rows))

    # 8. Live progress, nonterminal combos only.
    progress_rows = []
    for record in combos:
        # A written report already proves the run left the training phase.
        if record["status"] in FINAL or reports[record["id"]].get("outcome") or not record.get("run_dir"):
            continue
        run_dir = Path(record["run_dir"])
        values = live_progress(run_dir=run_dir)
        progress_rows.append([
            record["id"], values.get("step", "-") if values else "-",
            values.get("total", "-") if values else "-", values.get("epoch", "-") if values else "-",
            values.get("loss", "-") if values else "-", modified(run_dir / "watcher.log")])
    if progress_rows:
        sections.append("## Live progress\n\n" + render_table(
            headers=["Combo", "Step", "Total", "Epoch", "Loss", "Last log (UTC)"],
            rows=progress_rows))

    # 9. Full file inventory, opt-in only.
    if files:
        inventory_rows = []
        for record in combos:
            run_dir = Path(record["run_dir"]) if record.get("run_dir") else None
            if run_dir and run_dir.is_dir():
                for path in sorted(run_dir.iterdir()):
                    if path.is_file():
                        inventory_rows.append([record["id"], "local", path.name,
                                               human_bytes(path.stat().st_size), modified(path)])
            run_name = record.get("remote_run_name")
            if run_name and not no_s3:
                for prefix in (f"{S3_BUCKET}/checkpoints/act_v2/{run_name}/",
                               f"{S3_BUCKET}/runs/act_v2/{run_name}/"):
                    for item in s3_objects(prefix=prefix) or []:
                        inventory_rows.append([record["id"], "s3", item["key"],
                                               human_bytes(item["size"]), item["modified"]])
        sections.append("## File inventory\n\n" + render_table(
            headers=["Combo", "Where", "File", "Size", "Modified (UTC)"], rows=inventory_rows))

    return "\n\n".join(sections) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["report", "runs"])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--combo")
    parser.add_argument("--files", action="store_true")
    parser.add_argument("--no-s3", action="store_true")
    args = parser.parse_args()
    directory = args.directory.resolve()
    if not directory.is_dir():
        print(f"Not an iteration directory: {directory}", file=sys.stderr)
        return 2
    try:
        if args.mode == "runs":
            print(runs_report(directory=directory))
        else:
            print(report(directory=directory, combo=args.combo, files=args.files, no_s3=args.no_s3))
    except ValueError as error:
        print(f"Blocked: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
