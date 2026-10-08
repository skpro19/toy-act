#!/usr/bin/env python3
"""Record an operator-reviewed provider rejection for one uncertain create.

This is not an absence-based override. The operator must obtain confirmation
from Vast that the exact request created no rental, then explicitly attest that
confirmation. This tool records the evidence reference and validates current
absence. It never provisions, launches, destroys, or changes the pinned SHA.
"""

import argparse
import json
from pathlib import Path

from iteration import Iteration, locate
from workflow_common import Blocked, digest, instances, lock, now, redact, write_json


def resolve_rejection(*, iteration: Iteration, combo_id: str, evidence: dict) -> None:
    combo = next((combo for combo in iteration.state["combos"] if combo["id"] == combo_id), None)
    if combo is None or combo["status"] != "provisioning" or combo.get("training_launch_requested"):
        raise Blocked("Resolution requires an unlaunched provisioning combo")
    attempt = next((attempt for attempt in reversed(combo["attempts"]) if attempt["status"] != "removed"), None)
    if (attempt is None or attempt["status"] != "create_requested"
            or attempt.get("preexisting_label") or attempt.get("create_rejected")
            or attempt.get("instance_id") or attempt.get("returned_instance_id")
            or combo.get("instance_id") or combo.get("run_dir")):
        raise Blocked("Only an uncertain create without a known instance identity can be resolved")
    expected = {"iteration_id": iteration.directory.name, "combo_id": combo_id,
                "label": attempt["label"], "offer_id": attempt["offer"]["id"],
                "created_at": attempt["created_at"], "conclusion": "provider_confirmed_no_rental"}
    if set(evidence) != set(expected) | {"provider_reference", "reviewed_by"}:
        raise Blocked("Evidence must contain exact request identity, conclusion, provider_reference and reviewed_by")
    if any(type(evidence[key]) is not type(value) or evidence[key] != value for key, value in expected.items()):
        raise Blocked("Provider evidence does not match this exact create request")
    for key in ("provider_reference", "reviewed_by"):
        value = evidence[key]
        if (not isinstance(value, str) or not value.strip() or len(value) > 200
                or any(ord(character) < 32 for character in value)
                or redact(value) != value or "://" in value):
            raise Blocked("Use a short non-secret support/request reference and reviewer name, not raw records or URLs")
    receipt_path = iteration.directory / "combos" / combo_id / f"create-{attempt['label']}.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("version") != 1 or receipt.get("label") != attempt["label"]
                or receipt.get("offer_id") != attempt["offer"]["id"]
                or receipt.get("outcome") != "unknown"):
            raise Blocked("Create receipt has a known result or invalid identity; use normal reconciliation")
    records = instances(iteration.journal)  # Failed/malformed API queries never prove absence.
    if any(record.get("label") == attempt["label"] for record in records):
        raise Blocked("Instance with the exact label exists; resume for reconciliation instead")
    snapshot = iteration.directory / "combos" / combo_id / f"rejection-{attempt['label']}.json"
    if snapshot.exists():
        if json.loads(snapshot.read_text()) != evidence:
            raise Blocked("Existing provider-evidence snapshot differs; refusing overwrite")
    else:
        write_json(path=snapshot, value=evidence)
    attempt["provider_resolution"] = {"evidence": str(snapshot.relative_to(iteration.directory)),
                                      "sha256": digest(snapshot), "recorded_at": now()}
    # Keep the complete attempt history and provisioning status. Resume performs
    # a fresh search within the original three-attempt cap and Git revision gate.
    attempt["status"] = "removed"
    iteration.save()
    iteration.journal.event(kind="create_provider_rejection_resolved", combo=combo_id,
                            label=attempt["label"], diagnostic=str(snapshot))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", required=True, help="explicit iteration ID or directory")
    parser.add_argument("--combo", required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--confirm-provider-rejection", action="store_true",
                        help="attest that Vast confirmed no rental was created; absence alone is insufficient")
    args = parser.parse_args()
    if not args.confirm_provider_rejection:
        parser.error("Provider confirmation and explicit --confirm-provider-rejection are required")
    try:
        directory = locate(identity=args.resume)
        with lock(directory / "driver.lock"):
            iteration = Iteration(directory=directory)
            evidence = json.loads(args.evidence.read_text())
            if not isinstance(evidence, dict):
                raise Blocked("Evidence must be a JSON object")
            resolve_rejection(iteration=iteration, combo_id=args.combo, evidence=evidence)
            print("Provider rejection recorded. No instance was created or removed by this command.")
            print(f"Pinned SHA remains {iteration.manifest['git_commit']}; normal Git gates still apply on resume.")
        return 0
    except Blocked as error:
        print(f"Resolution blocked: {redact(str(error))}")
        return 1
    except (ValueError, KeyError, TypeError, OSError):
        # Parsing/I/O exception text could expose raw evidence or secrets.
        print("Resolution blocked: invalid or unreadable evidence/state; verify local snapshots")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
