"""Immutable iteration manifests and atomic, locked workflow state."""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import tomllib
import uuid

from workflow_common import (
    Blocked, REPO, Journal, atomic_write, atomic_write_bytes, digest, git_preflight, now, write_json,
)

ROOT = REPO / ".vast-train-local/ablations"
FINAL = {"done", "failed", "verification_failed"}


def input_paths(*, source: Path, kind: str) -> list[Path]:
    paths = [source]
    if kind == "sweep":
        spec = tomllib.loads(source.read_text())
        current = (REPO / spec["group"] / "BASE.toml").resolve()
    else:
        current = source
    seen = set()
    while current not in seen:
        seen.add(current)
        if current not in paths:
            paths.append(current)
        raw = tomllib.loads(current.read_text())
        base = raw.get("base_config")
        if base is None:
            return paths
        current = (current.parent / base).resolve()
    raise Blocked("Config inheritance cycle")


def new_directory(*, source: Path, root: Path = ROOT) -> Path:
    stem = re.sub(r"[^a-zA-Z0-9._-]", "-", source.stem)[:80] or "experiment"
    identity = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:12]
    directory = root / stem / identity
    directory.mkdir(parents=True, mode=0o700)
    return directory


def locate(*, identity: str, root: Path = ROOT) -> Path:
    # Accept an explicit iteration path or a globally unique iteration ID.
    supplied = Path(identity).expanduser()
    if supplied.is_dir():
        directory = supplied.resolve()
        if not directory.is_relative_to(root.resolve()):
            raise Blocked("Resume path must be inside the local iteration root")
        return directory
    if not re.fullmatch(r"[0-9]{8}T[0-9]{6}Z-[0-9a-f]{12}", identity):
        raise Blocked("Invalid iteration ID/path")
    matches = list(root.glob(f"*/{identity}"))
    if len(matches) != 1:
        raise Blocked("Iteration ID is missing or ambiguous; use its explicit directory")
    return matches[0].resolve()


def create(*, directory: Path, source: Path, kind: str, journal: Journal,
           expected_commit: str | None = None) -> dict:
    commit = git_preflight(journal=journal, expected=expected_commit)
    paths = input_paths(source=source, kind=kind)
    snapshots = []
    for index, path in enumerate(paths):
        destination = directory / "inputs" / f"{index:03d}-{path.name}"
        atomic_write_bytes(path=destination, content=path.read_bytes())
        snapshots.append({"source": str(path), "path": str(destination.relative_to(directory)), "sha256": digest(path)})
    configs = directory / "configs"
    configs.mkdir(mode=0o700)
    resolved_manifest = directory / "resolved-paths.json"
    if kind == "sweep":
        journal.run(args=["uv", "run", "--frozen", "python", "-m", "scripts.ablate", "--spec", str(source),
                          "--resolve-dir", str(configs), "--manifest", str(resolved_manifest)],
                    timeout=120, log_path=directory / "resolution.log")
        config_paths = [Path(path) for path in json.loads(resolved_manifest.read_text())]
    else:
        target = configs / f"{source.stem}.toml"
        journal.run(args=["uv", "run", "--frozen", "python", "-m", "scripts.resolve_config", "--config", str(source),
                          "--out", str(target)], timeout=120, log_path=directory / "resolution.log")
        config_paths = [target]
    # Source edits during resolution must not silently change the experiment.
    if any(digest(Path(record["source"])) != record["sha256"]
           or digest(directory / record["path"]) != record["sha256"] for record in snapshots):
        raise Blocked("Input changed during resolution; no rentals were made")
    if not config_paths or len(set(config_paths)) != len(config_paths):
        raise Blocked("Empty or duplicate resolved config manifest")
    # Import only after Git preflight and project resolution succeeded.
    import tomli_w
    combos = []
    for index, path in enumerate(config_paths):
        if not path.resolve().is_relative_to(configs.resolve()) or not path.is_file():
            raise Blocked("Resolved manifest contains an invalid config path")
        config = tomllib.loads(path.read_text())
        # Training adds a seconds-resolution timestamp. This suffix prevents S3
        # collisions when concurrent iterations launch identical configs.
        name = f"{config['name']}-i{directory.name.rsplit('-', 1)[1]}"
        if any(ord(character) < 32 or ord(character) == 127 or character == '/' for character in name):
            raise Blocked("Run name contains unsupported path/control characters")
        if len(name.encode("utf-8")) + 20 > 255:
            raise Blocked("Run name plus training timestamp exceeds filesystem limits; shorten the config name")
        config["name"] = name
        atomic_write(path=path, text=tomli_w.dumps(config))
        combos.append({"id": f"{index:04d}", "slug": path.stem, "name": name,
                       "config": str(path.relative_to(directory)), "sha256": digest(path)})
    manifest = {"version": 1, "id": directory.name, "kind": kind, "source": str(source),
                "created_at": now(), "git_commit": commit, "inputs": snapshots, "combos": combos}
    write_json(path=directory / "manifest.json", value=manifest)
    state = {"version": 1, "manifest_sha256": digest(directory / "manifest.json"), "driver_status": "created",
             "combos": [{"id": combo["id"], "status": "pending", "attempts": []} for combo in combos]}
    write_json(path=directory / "state.json", value=state)
    journal.event(kind="iteration_created", iteration=directory.name, combos=len(combos), git_commit=commit)
    resolved_manifest.unlink(missing_ok=True)
    return manifest


class Iteration:
    def __init__(self, *, directory: Path) -> None:
        self.directory = directory
        self.journal = Journal(directory=directory)
        self.manifest = json.loads((directory / "manifest.json").read_text())
        self.state = json.loads((directory / "state.json").read_text())
        self.validate()

    def validate(self) -> None:
        if self.manifest.get("version") != 1 or self.state.get("version") != 1:
            raise Blocked("Unsupported iteration state version")
        if digest(self.directory / "manifest.json") != self.state["manifest_sha256"]:
            raise Blocked("Iteration manifest was modified")
        if not re.fullmatch(r"[0-9a-f]{40,64}", self.manifest["git_commit"]):
            raise Blocked("Invalid saved revision")
        ids = [combo["id"] for combo in self.manifest["combos"]]
        if len(set(ids)) != len(ids) or ids != [combo["id"] for combo in self.state["combos"]]:
            raise Blocked("State and manifest combo lists differ")
        for record in [*self.manifest["inputs"], *self.manifest["combos"]]:
            path = self.directory / record.get("path", record.get("config", ""))
            if not path.resolve().is_relative_to(self.directory.resolve()) or digest(path) != record["sha256"]:
                raise Blocked(f"Saved snapshot/config was modified: {path}")
        valid = {"pending", "provisioning", "setup", "ready_to_launch", "awaiting_handoff", "running", *FINAL}
        if any(combo["status"] not in valid for combo in self.state["combos"]):
            raise Blocked("Invalid combo status")
        for combo in self.state["combos"]:
            attempts = combo.get("attempts")
            if not isinstance(attempts, list) or len(attempts) > 3:
                raise Blocked("Invalid provisioning attempt history")
            if any(not isinstance(attempt, dict) or attempt.get("status") not in {
                    "create_requested", "created", "accepted", "removed"} for attempt in attempts):
                raise Blocked("Invalid provisioning attempt status")
            if combo["status"] == "pending" and attempts:
                raise Blocked("An attempted combo cannot be reset to pending")
            if combo.get("training_launch_requested") and combo["status"] not in FINAL | {"awaiting_handoff", "running"}:
                raise Blocked("Training launch intent cannot return to provisioning/setup")
            for key, filename in (("setup_sha256", "setup.env"), ("watcher_sha256", "watcher.sh")):
                if key in combo and digest(Path(combo["run_dir"]) / filename) != combo[key]:
                    raise Blocked(f"Saved run input was modified: {filename}")

    def save(self) -> None:
        write_json(path=self.directory / "state.json", value=self.state)
        self.journal.event(kind="state_saved", driver_status=self.state["driver_status"],
                           combos=[{"id": combo["id"], "status": combo["status"],
                                    "instance_id": combo.get("instance_id")} for combo in self.state["combos"]])
        self.summary()

    def summary(self) -> str:
        rows = [f"Iteration: {self.directory.name}", f"Directory: {self.directory}",
                f"Commit: {self.manifest['git_commit']}", f"Driver: {self.state['driver_status']}",
                "combo  status                instance  tensorboard / run directory"]
        for combo, entry in zip(self.state["combos"], self.manifest["combos"]):
            lease = combo.get("lease", {})
            tb = f"http://localhost:{lease['TB_PORT']}/" if lease else "-"
            rows.append(f"{combo['id']}   {combo['status']:<21} {str(combo.get('instance_id', '-')):<9} {tb}  {entry['slug']}")
            if combo.get("run_dir"):
                rows.append(f"       {combo['run_dir']}")
            run = combo.get("run", {})
            if run:
                offer_price = run.get("offer", {}).get("dph_total", "unknown")
                rows.append(f"       offer=${offer_price}/h actual=${run.get('actual_price', 'unknown')}/h")
            if combo.get("remote_run_name"):
                run_name = combo["remote_run_name"]
                rows.append(f"       run={run_name}")
                rows.append(f"       s3://toy-act/checkpoints/act_v2/{run_name}/")
                rows.append(f"       s3://toy-act/runs/act_v2/{run_name}/")
            report = combo.get("report", {})
            if report:
                rows.append(f"       outcome={report.get('outcome')} cleanup={report.get('final_cleanup_status')} s3={report.get('s3_uri')}")
        rows.append(f"Logs: {self.directory / 'driver.log'}; {self.directory / 'events.jsonl'}")
        text = "\n".join(rows) + "\n"
        atomic_write(path=self.directory / "summary.txt", text=text)
        return text
