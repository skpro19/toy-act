#!/usr/bin/env python3
"""Run a fresh sweep/single config, or explicitly resume one saved iteration.

No historical iteration is consulted for a fresh invocation. Exit 0 means all
combos were handed off or reconciled; 1 blocked; 2 invalid input; 130 interrupted.
--plan resolves and snapshots configs without accessing Vast/AWS or renting.
"""

import argparse
import fcntl
import os
from pathlib import Path
import re
import signal
import sys
import time
import tomllib
from types import FrameType

from iteration import FINAL, Iteration, create, locate, new_directory
from provision import provision
from setup_run import (
    ensure_forwarding, ensure_watcher, handoff, launch, local_preflight, setup, validate_dataset_source,
)
from workflow_common import Blocked, Journal, atomic_write, git_preflight, instances, lock


def watcher_alive(run_dir: Path) -> bool:
    with (run_dir / "watcher.lock").open("a") as file:
        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return False
        except BlockingIOError:
            return True


def read_report(*, run_dir: Path) -> dict:
    report = {}
    for line in (run_dir / "report.txt").read_text().splitlines():
        key, separator, value = line.partition(": ")
        if separator:
            if key in report:
                raise Blocked("Duplicate watcher report fields")
            report[key] = value
    return report


def recover_watcher_report(*, iteration: Iteration, combo: dict) -> None:
    run_dir = Path(combo["run_dir"])
    report_path = run_dir / "report.txt"
    archived = run_dir / "report-history" / f"{time.time_ns()}.txt"
    atomic_write(path=archived, text=report_path.read_text())
    report_path.unlink()
    combo["status"] = "awaiting_handoff"
    iteration.save()
    iteration.journal.event(kind="watcher_report_archived", combo=combo["id"], path=str(archived))
    ensure_watcher(journal=iteration.journal, combo=combo)


def reconcile(*, iteration: Iteration, combo: dict) -> bool:
    """No cleanup or launch decisions from absence, dead PIDs, or API failures."""
    if combo["status"] in FINAL:
        return True
    if not combo.get("training_launch_requested"):
        return False
    run_dir = Path(combo["run_dir"])
    if (run_dir / "report.txt").exists():
        # Reports are written at exit. Avoid reading a partial report from a live writer.
        if watcher_alive(run_dir):
            return False
        report = read_report(run_dir=run_dir)
        run = combo["run"]
        if report.get("instance_id") != str(run["instance_id"]) or report.get("instance_label") != run["label"]:
            raise Blocked("Watcher report identity mismatch")
        cleanup = report.get("final_cleanup_status")
        if cleanup == "refused_gate_closed":
            # A killed/crashed watcher can write a nonterminal report. Archive it
            # and recover conservatively; it never authorizes driver destruction.
            recover_watcher_report(iteration=iteration, combo=combo)
            return False
        if report.get("outcome") not in {"success", "failure"}:
            raise Blocked("Watcher report has no confirmed outcome or recoverable cleanup status")
        records = instances(iteration.journal)
        if any(record["id"] == run["instance_id"] for record in records):
            if cleanup == "destroy_unverified":
                recover_watcher_report(iteration=iteration, combo=combo)
                return False
            raise Blocked("Watcher report exists but instance remains; no automatic destruction/replacement")
        if cleanup not in {"destroyed_and_verified", "destroy_unverified"}:
            raise Blocked("Watcher cleanup status cannot be reconciled automatically")
        combo["cleanup_verified_on_resume"] = True
        combo["report"] = report
        if report["outcome"] == "failure":
            combo["status"] = "failed"
        elif (report.get("s3_verification", "").startswith("verified ")
              and report.get("tensorboard_verification", "").startswith("verified:")):
            combo["status"] = "done"
        else:
            combo["status"] = "verification_failed"
        iteration.save()
        iteration.journal.event(kind="terminal_reconciled", combo=combo["id"], status=combo["status"])
        return True
    if not watcher_alive(run_dir):
        # Explicit resume is allowed to restart the saved conservative watcher.
        ensure_watcher(journal=iteration.journal, combo=combo)
        combo["status"] = "awaiting_handoff"
        iteration.save()
    return False


def execute(iteration: Iteration) -> None:
    journal = iteration.journal
    # Reconcile only this explicit iteration. No global old-run launch gate.
    for combo in iteration.state["combos"]:
        reconcile(iteration=iteration, combo=combo)
    actionable = [combo for combo in iteration.state["combos"] if combo["status"] not in FINAL | {"running"}]
    if not actionable:
        return
    # Existing monitoring/reconciliation is possible even after a branch advances.
    # Every additional provisioning/setup/launch still requires the pinned revision.
    git_preflight(journal=journal, expected=iteration.manifest["git_commit"])
    local_preflight(journal)
    total = len(iteration.state["combos"])
    for index, (combo, entry) in enumerate(zip(iteration.state["combos"], iteration.manifest["combos"]), start=1):
        if reconcile(iteration=iteration, combo=combo) or combo["status"] == "running":
            continue
        label = f"combo {index}/{total} ({combo['id']})"
        journal.set_stage(text=f"{label}: validating inputs and dataset source")
        iteration.validate()
        config_path = iteration.directory / entry["config"]
        if combo["status"] in {"pending", "provisioning", "setup"}:
            git_preflight(journal=journal, expected=iteration.manifest["git_commit"])
            validate_dataset_source(journal=journal, config=tomllib.loads(config_path.read_text()), config_path=config_path)
        if combo["status"] in {"pending", "provisioning"}:
            journal.set_stage(text=f"{label}: provisioning and hardware/network checks")
            provision(journal=journal, combo=combo, directory=iteration.directory / "combos" / combo["id"],
                      commit=iteration.manifest["git_commit"], save=iteration.save)
        if combo["status"] == "setup":
            journal.set_stage(text=f"{label}: remote setup and dataset validation")
            git_preflight(journal=journal, expected=iteration.manifest["git_commit"])
            setup(journal=journal, combo=combo, config_path=config_path,
                  commit=iteration.manifest["git_commit"], save=iteration.save)
        if combo["status"] == "ready_to_launch":
            journal.set_stage(text=f"{label}: launching training and monitoring")
            git_preflight(journal=journal, expected=iteration.manifest["git_commit"])
            launch(journal=journal, combo=combo, save=iteration.save)
        elif combo["status"] == "awaiting_handoff":
            journal.set_stage(text=f"{label}: restoring monitoring")
            ensure_forwarding(journal=journal, combo=combo)
            ensure_watcher(journal=journal, combo=combo)
            # Reissue only an unacknowledged request, using an idempotent helper.
            # Known launches need monitoring recovery, not another service launch.
            if not combo.get("services_started"):
                launch(journal=journal, combo=combo, save=iteration.save)
        journal.set_stage(text=f"{label}: waiting for handoff")
        code = handoff(journal=journal, combo=combo, save=iteration.save)
        if code == 0:
            # Independent, fresh check before advancing the sweep, not saved JSON.
            code = handoff(journal=journal, combo=combo, save=iteration.save, wait_seconds=0)
        if code == 0:
            journal.progress(text=f"{label}: handoff verified (training continues independently)")
            print(iteration.summary(), flush=True)
        if code == 3 and not reconcile(iteration=iteration, combo=combo):
            raise Blocked("Training became terminal before handoff; watcher is finishing backup/cleanup. Resume this iteration later")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--spec", type=Path)
    source.add_argument("--config", type=Path)
    parser.add_argument("--resume", help="explicit iteration ID or local iteration directory")
    parser.add_argument("--expected-commit", help="optional caller-pinned Git SHA")
    parser.add_argument("--plan", action="store_true", help="resolve/snapshot only; no Vast/AWS operations")
    args = parser.parse_args()
    if not args.resume and args.spec is None and args.config is None:
        parser.error("Provide --spec, --config, or --resume; no implicit defaults")
    if args.plan and args.resume:
        parser.error("--plan creates a new iteration; use --resume without --plan to launch it")
    if args.expected_commit and not re.fullmatch(r"[0-9a-f]{40,64}", args.expected_commit):
        parser.error("--expected-commit must be a full hexadecimal Git SHA")
    return args


def main() -> int:
    os.umask(0o077)
    args = parse_args()
    source = args.spec or args.config
    if source is not None:
        source = source.expanduser().resolve()
        if not args.resume and not source.is_file():
            print(f"Input does not exist: {source}", file=sys.stderr)
            return 2
    iteration = None
    directory = None
    # TERM from an outer timeout should persist driver interruption, never clean up.
    def interrupted(signum: int, frame: FrameType | None, /) -> None:
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        directory = locate(identity=args.resume) if args.resume else new_directory(source=source)
        journal = Journal(directory=directory)
        with lock(directory / "driver.lock"):
            # Error/interruption state is persisted while still holding the lock.
            try:
                journal.progress(text=f"Iteration: {directory.name}\nLogs: {directory}\nResume: uv run --frozen python .pi/prompts/scripts/vast-train/workflow.py --resume {directory.name}")
                if not args.resume:
                    journal.set_stage(text="Git preflight and immutable config resolution")
                    create(directory=directory, source=source, kind="sweep" if args.spec else "single",
                           journal=journal, expected_commit=args.expected_commit)
                iteration = Iteration(directory=directory)
                journal.progress(text=f"Combos: {len(iteration.state['combos'])}; mode: {'plan' if args.plan else 'resume' if args.resume else 'fresh'}")
                if args.resume and source is not None and str(source) != iteration.manifest["source"]:
                    raise Blocked("Resume source path differs; saved configs must not be substituted")
                if args.expected_commit and args.expected_commit != iteration.manifest["git_commit"]:
                    raise Blocked("Caller revision differs from the saved iteration")
                iteration.state["driver_status"] = "planned" if args.plan else "running"
                iteration.save()
                journal.event(kind="driver_started", resumed=bool(args.resume), pid=os.getpid(), plan=args.plan)
                if not args.plan:
                    execute(iteration)
                    iteration.state["driver_status"] = "handed_off"
                    iteration.save()
                journal.event(kind="driver_finished", status=iteration.state["driver_status"])
                print(iteration.summary())
                return 0
            except (Blocked, ValueError, KeyError, TypeError, IndexError, OSError) as error:
                journal.event(kind="driver_blocked", reason=str(error))
                if iteration is not None:
                    iteration.state["driver_status"] = "blocked"
                    iteration.save()
                    print(iteration.summary())
                print(f"Blocked: {error}\nLogs: {directory}", file=sys.stderr)
                return 1
            except KeyboardInterrupt:
                journal.event(kind="driver_interrupted")
                if iteration is not None:
                    iteration.state["driver_status"] = "interrupted"
                    iteration.save()
                    print(iteration.summary())
                print("Driver interrupted; existing instances/watchers were left untouched", file=sys.stderr)
                return 130
    except (Blocked, OSError) as error:
        # In particular, a second driver cannot write state/logs owned by the first.
        print(f"Blocked: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
