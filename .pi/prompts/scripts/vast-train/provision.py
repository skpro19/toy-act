"""Offer selection and durable provisional rentals. Never owns a training run."""

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shlex
import time
from collections.abc import Callable
import uuid

from hardware_gate import cpu_rank, evaluate
from workflow_common import (
    Blocked, HELPERS, IMAGE, MAX_PRICE, PROFILE, REGION, REPO, Journal,
    atomic_write, git_preflight, instances, lock, now, ssh_args, write_json,
)

OFFER_QUERY = (
    "gpu_name=RTX_4090 num_gpus=1 gpu_ram>=24 gpu_max_power>=400 compute_cap>=890 "
    "cpu_cores_effective>=24 cpu_ram>=64 pci_gen>=4 pcie_bw>=20 inet_down>=500 "
    "inet_up>=200 reliability>=0.99 rentable=true verification=verified gpu_display_active=false disk_space>=100"
)
# Vast search syntax uses GB; raw RAM fields are reported in MB by the CLI.
OFFER_MINIMUMS = {
    "gpu_ram": 24000, "gpu_max_power": 400, "compute_cap": 890,
    "cpu_cores_effective": 24, "cpu_ram": 64000, "pci_gen": 4,
    "pcie_bw": 20, "inet_down": 500, "inet_up": 200,
    "reliability": 0.99, "disk_space": 100,
}


def shortlist(*, offers: list, quarantine: dict[str, float], epoch: float) -> list[dict]:
    ranked = []
    for offer in offers:
        try:
            rank = cpu_rank(offer["cpu_name"])
            if (offer["gpu_name"] != "RTX 4090" or offer["num_gpus"] != 1
                    or offer["rentable"] is not True or offer["verification"] != "verified"
                    or offer["gpu_display_active"] is not False):
                continue
            if any(not math.isfinite(float(offer[key])) or float(offer[key]) < minimum
                   for key, minimum in OFFER_MINIMUMS.items()):
                continue
            numeric = {key: float(offer[key]) for key in (
                "dph_total", "disk_bw", "reliability", "cpu_cores_effective", "cpu_ram", "pcie_bw")}
            if rank is None or any(not math.isfinite(value) for value in numeric.values()):
                continue
            if not 0 <= numeric["dph_total"] <= MAX_PRICE:
                continue
            if quarantine.get(str(offer["machine_id"]), 0) > epoch - 86400:
                continue
            if not isinstance(offer["id"], int) or not isinstance(offer["machine_id"], int):
                continue
            ranked.append((rank, numeric["dph_total"], -numeric["disk_bw"], -numeric["reliability"], offer))
        except (KeyError, TypeError, ValueError):
            continue
    ranked.sort(key=lambda entry: entry[:4])
    return [entry[-1] for entry in ranked[:3]]


def search_offers(*, journal: Journal, directory: Path, excluded: set[int]) -> list[dict]:
    result = journal.run(args=["vastai", "search", "offers", OFFER_QUERY, "--order", "dph_total+", "--storage", "100", "--raw"],
                         log_path=directory / "offers.log")
    try:
        offers = json.loads(result.stdout)
        if not isinstance(offers, list):
            raise ValueError("Not a list")
    except ValueError as error:
        raise Blocked("Invalid offer search output") from error
    quarantine = {}
    path = REPO / ".vast-train-local/failed-machines.tsv"
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                timestamp, machine, reason = line.split("\t")
                if reason == "gpu-start-error":
                    quarantine[machine] = max(float(timestamp), quarantine.get(machine, 0))
            except ValueError:
                continue
    chosen = shortlist(offers=[offer for offer in offers if offer.get("machine_id") not in excluded],
                       quarantine=quarantine, epoch=time.time())
    write_json(path=directory / "offers.json", value={"captured_at": now(), "offers": chosen})
    if not chosen:
        raise Blocked("No supported, non-quarantined offers under the price cap")
    return chosen


def quarantine_machine(machine: int) -> None:
    path = REPO / ".vast-train-local/failed-machines.tsv"
    with lock(REPO / ".vast-train-local/quarantine.lock"):
        with path.open("a") as file:
            file.write(f"{int(time.time())}\t{machine}\tgpu-start-error\n")
            file.flush()
            os.fsync(file.fileno())


def destroy_provisional(*, journal: Journal, attempt: dict, save: Callable[[], None]) -> None:
    # This helper is called only before any training launch intent. Never call it
    # from remote setup/service failure handling or from terminal reconciliation.
    identity = attempt["instance_id"]
    records = instances(journal)
    record = next((record for record in records if record["id"] == identity), None)
    if record is not None and record.get("label") != attempt["label"]:
        raise Blocked("Provisional instance identity mismatch; refusing destruction")
    if record is not None:
        journal.run(args=["vastai", "destroy", "instance", str(identity), "-y"], check=False,
                    log_path=journal.directory / "provision.log")
    for _ in range(24):
        # A failed or malformed query raises; it never proves absence.
        if not any(record["id"] == identity for record in instances(journal)):
            attempt["status"] = "removed"
            save()
            journal.event(kind="provisional_removed", instance_id=identity)
            return
        time.sleep(5)
    raise Blocked(f"Removal of provisional instance {identity} unverified; no replacement rented")


def reconcile_create(*, journal: Journal, attempt: dict, save: Callable[[], None]) -> dict:
    matches = [record for record in instances(journal) if record.get("label") == attempt["label"]]
    if len(matches) != 1:
        raise Blocked(f"Create outcome unknown for {attempt['label']}; do not issue another create")
    record = matches[0]
    expected = attempt.get("instance_id", attempt.get("returned_instance_id"))
    if expected not in (None, record["id"]):
        raise Blocked("Create identity changed; refusing to continue")
    attempt.update(instance_id=record["id"], status="created")
    save()
    return record


def network_gate(*, journal: Journal, run: dict, directory: Path) -> bool:
    # Presigned URLs travel through SSH stdin, never through process arguments or logs.
    code = (
        "import boto3, json, uuid\n"
        f"s3 = boto3.Session(profile_name={PROFILE!r}, region_name={REGION!r}).client('s3')\n"
        "p = {'Bucket': 'toy-act', 'Key': 'runs/act_v2/.netgate/' + uuid.uuid4().hex}\n"
        "print(json.dumps({k: s3.generate_presigned_url(k, Params=p, ExpiresIn=900)\n"
        "                  for k in ('put_object', 'delete_object')}))\n"
    )
    result = journal.run(args=["uv", "run", "--frozen", "python", "-c", code], sensitive=True)
    try:
        urls = json.loads(result.stdout)
        script = (f"export S3_PRESIGNED_PUT={shlex.quote(urls['put_object'])}\n"
                  f"export S3_PRESIGNED_DELETE={shlex.quote(urls['delete_object'])}\n"
                  + (HELPERS / "network-gate.sh").read_text())
    except (ValueError, KeyError) as error:
        raise Blocked("Could not generate network-gate credentials") from error
    result = journal.run(args=ssh_args(run=run, command="bash -s"), input_text=script,
                         timeout=700, check=False, log_path=directory / "network.log")
    return result.returncode == 0 and any(line.startswith("PASSED") for line in result.stdout.splitlines())


def provision(*, journal: Journal, combo: dict, directory: Path,
              commit: str, save: Callable[[], None]) -> dict:
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    attempts = combo.setdefault("attempts", [])
    while True:
        active = next((attempt for attempt in reversed(attempts) if attempt["status"] != "removed"), None)
        if active is None:
            if len(attempts) >= 3:
                raise Blocked("All three provisioning attempts failed; no filters were weakened")
            excluded = {attempt["offer"]["machine_id"] for attempt in attempts}
            combo.pop("run_dir", None)
            combo.pop("instance_id", None)
            offer = search_offers(journal=journal, directory=directory, excluded=excluded)[0]
            git_preflight(journal=journal, expected=commit)
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
            active = {"label": f"toy-act-train-actv2-{timestamp}-{uuid.uuid4().hex[:12]}", "offer": offer,
                      "status": "create_requested", "created_at": now()}
            attempts.append(active)
            combo["status"] = "provisioning"
            save()  # Durable label/intent before any create request.
            journal.event(kind="create_requested", combo=combo["id"], label=active["label"], offer_id=offer["id"])
            if any(record.get("label") == active["label"] for record in instances(journal)):
                active["preexisting_label"] = True
                save()
                raise Blocked("Instance label existed before create; refusing to rent or adopt it")
            result = journal.run(args=["vastai", "create", "instance", str(offer["id"]),
                                      "--image", IMAGE, "--disk", "100", "--ssh", "--direct",
                                      "--label", active["label"], "--cancel-unavail", "--raw"],
                                 check=False, sensitive=True, timeout=120)
            journal.event(kind="create_returned", combo=combo["id"], exit_code=result.returncode)
            try:
                response = json.loads(result.stdout)
                if isinstance(response, dict):
                    identity = response.get("new_contract")
                    if response.get("success") is True and type(identity) is int and identity > 0:
                        active["returned_instance_id"] = identity
                        save()
                    elif response.get("success") is False and not identity:
                        active["create_rejected"] = True
                        save()
            except ValueError:
                pass  # Response loss/malformed output is not proof of rejection.
            time.sleep(2)
        if active.get("preexisting_label"):
            raise Blocked("Recorded label predates this request; refusing automatic adoption")
        if active.get("create_rejected"):
            matches = [record for record in instances(journal) if record.get("label") == active["label"]]
            if not matches:
                active["status"] = "removed"
                save()
                journal.event(kind="create_explicitly_rejected", combo=combo["id"], label=active["label"])
                continue
        if active.get("rejected"):
            # Destruction may have succeeded just before the state write was
            # interrupted. A valid API absence can finish this provisional step.
            destroy_provisional(journal=journal, attempt=active, save=save)
            continue
        record = reconcile_create(journal=journal, attempt=active, save=save)
        identity = active["instance_id"]
        run_dir = REPO / f".vast-train-local/toy-act-{identity}"
        run_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        existing_record = run_dir / "instance.json"
        if existing_record.exists() and json.loads(existing_record.read_text()).get("label") != active["label"]:
            raise Blocked("Local run directory belongs to a different instance label")
        combo.update(run_dir=str(run_dir), instance_id=identity)
        active["run_dir"] = str(run_dir)
        write_json(path=run_dir / "instance.json", value=record)
        save()
        rejected = False
        for _ in range(60):
            message = str(record.get("status_msg", ""))
            if re.search(r"gpu.*(?:error|fail)|(?:error|fail).*gpu|unable to start", message, re.IGNORECASE):
                quarantine_machine(active["offer"]["machine_id"])
                rejected = True
                break
            if record.get("actual_status") == "running":
                break
            time.sleep(10)
            record = reconcile_create(journal=journal, attempt=active, save=save)
        else:
            rejected = True
        if rejected:
            active["rejected"] = "instance did not start"
            save()
            destroy_provisional(journal=journal, attempt=active, save=save)
            continue
        write_json(path=run_dir / "instance.json", value=record)  # Sensitive, mode 600.
        url = journal.run(args=["vastai", "ssh-url", str(identity)]).stdout.strip()
        match = re.fullmatch(r"ssh://root@([a-zA-Z0-9.-]+):([0-9]+)", url)
        if not match or not 1 <= int(match[2]) <= 65535:
            raise Blocked("Invalid SSH URL; provisional instance left recoverable")
        run = {"host": match[1], "port": int(match[2]), "known_hosts": str(run_dir / "known_hosts"),
               "instance_id": identity, "label": active["label"], "run_dir": str(run_dir),
               "offer": active["offer"], "actual_price": record.get("dph_total", "unknown")}
        keys = ""
        for _ in range(12):
            keyscan = journal.run(args=["ssh-keyscan", "-T", "10", "-p", str(run["port"]), run["host"]],
                                  timeout=35, check=False)
            keys = "\n".join(line for line in keyscan.stdout.splitlines() if not line.startswith("#"))
            if not keyscan.returncode and keys:
                break
            time.sleep(5)
        else:
            raise Blocked("SSH host-key scan failed; retry this iteration, not another rental")
        # Never silently replace a recorded host key on resume.
        known_hosts = Path(run["known_hosts"])
        if known_hosts.exists() and set(known_hosts.read_text().splitlines()) != set(keys.splitlines()):
            raise Blocked("SSH host key changed; refusing automatic replacement")
        atomic_write(path=known_hosts, text=keys + "\n")
        probe_result = journal.run(args=ssh_args(run=run, command="bash -s"),
                                   input_text=(HELPERS / "hardware-probe.sh").read_text(),
                                   timeout=120, check=False, log_path=directory / f"hardware-{identity}.log")
        try:
            probe = json.loads(probe_result.stdout) if not probe_result.returncode else {}
        except ValueError:
            probe = {}
        report = evaluate(probe=probe, offer=active["offer"])
        write_json(path=directory / f"hardware-{identity}.json", value=report)
        if not report["passed"] or not network_gate(journal=journal, run=run, directory=directory):
            active["rejected"] = "hardware/network gate rejected"
            save()
            destroy_provisional(journal=journal, attempt=active, save=save)
            continue
        combo["run"] = run
        combo["status"] = "setup"
        active["status"] = "accepted"
        save()
        journal.event(kind="provisioned", combo=combo["id"], instance_id=identity)
        return run
