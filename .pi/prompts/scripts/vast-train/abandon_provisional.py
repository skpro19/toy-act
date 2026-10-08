#!/usr/bin/env python3
"""Explicitly retire one known, unlaunched provisioning rental and its combo.

No Git gate is needed for cleanup, and the pinned revision is never changed.
This helper never provisions or launches. It cannot cancel launched training;
that requires cancel_run.py and watcher-owned cleanup instead.
"""

import argparse
import signal
from types import FrameType

from iteration import Iteration, locate
from provision import destroy_provisional
from workflow_common import Blocked, instances, lock, now, redact


def abandon(*, iteration: Iteration, instance_id: int) -> None:
    matches = [combo for combo in iteration.state["combos"] if combo.get("instance_id") == instance_id]
    if type(instance_id) is not int or instance_id <= 0 or len(matches) != 1:
        raise Blocked("Instance must belong to exactly one combo in this iteration")
    combo = matches[0]
    if (combo.get("training_launch_requested") or combo.get("run") or combo.get("services_started")
            or combo["status"] not in {"provisioning", "abandoned"}):
        raise Blocked("Only an unlaunched provisioning rental can be abandoned")
    attempts = [attempt for attempt in combo["attempts"] if attempt.get("instance_id") == instance_id]
    if len(attempts) != 1:
        raise Blocked("Instance must match exactly one saved provisioning attempt")
    attempt = attempts[0]
    if attempt["status"] not in {"created", "removed"} or not attempt.get("label"):
        raise Blocked("Attempt is not a known provisional rental")
    if combo["status"] == "abandoned" and not combo.get("provisional_abandon_requested"):
        raise Blocked("Abandoned combo has no saved abandonment intent")
    if any(other["status"] != "removed" for other in combo["attempts"] if other is not attempt):
        raise Blocked("Another unresolved attempt exists; refusing abandonment")
    records = instances(iteration.journal)
    by_id = [record for record in records if record["id"] == instance_id]
    by_label = [record for record in records if record.get("label") == attempt["label"]]
    if by_id != by_label or len(by_label) > 1:
        raise Blocked("Exact instance ID/label mismatch; refusing removal")
    if combo["status"] == "abandoned" and by_id:
        raise Blocked("Previously abandoned instance is visible again; refusing automatic removal")
    if attempt["status"] == "removed" and by_id:
        raise Blocked("Removed attempt is visible again; refusing automatic removal")
    combo.setdefault("provisional_abandon_requested", now())
    iteration.save()  # Durable intent before any destructive command.
    iteration.journal.event(kind="provisional_abandon_requested", combo=combo["id"], instance_id=instance_id)
    destroy_provisional(journal=iteration.journal, attempt=attempt, save=iteration.save)
    combo["status"] = "abandoned"
    combo["provisional_abandon_verified_at"] = now()
    iteration.save()
    iteration.journal.event(kind="provisional_abandoned", combo=combo["id"], instance_id=instance_id)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", required=True, help="explicit iteration ID or directory")
    parser.add_argument("--instance-id", type=int, required=True)
    parser.add_argument("--confirm-abandon", action="store_true", help="remove this provisional rental and retire its combo")
    args = parser.parse_args()
    if not args.confirm_abandon or args.instance_id <= 0:
        parser.error("A positive instance ID and explicit --confirm-abandon are required")
    def interrupted(signum: int, frame: FrameType | None, /) -> None:
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        directory = locate(identity=args.resume)
        with lock(directory / "driver.lock"):
            iteration = Iteration(directory=directory)
            abandon(iteration=iteration, instance_id=args.instance_id)
            print(f"Provisional instance {args.instance_id} removal verified; selected combo abandoned.")
            print("No replacement rented; no training started. Other combos are unchanged.")
            print(f"Pinned SHA remains {iteration.manifest['git_commit']}.")
        return 0
    except Blocked as error:
        print(f"Abandonment blocked: {redact(str(error))}")
        return 1
    except (ValueError, KeyError, TypeError, OSError):
        print("Abandonment blocked: invalid or unreadable saved state; inspect local diagnostics")
        return 1
    except KeyboardInterrupt:
        print("Abandonment interrupted; repeat this exact command to verify removal. Do not resume provisioning.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
