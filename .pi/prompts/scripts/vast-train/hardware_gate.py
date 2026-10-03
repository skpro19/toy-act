"""Fail-closed validation of the structured remote hardware probe."""

import argparse
import json
import math
from pathlib import Path
import re


def cpu_rank(model: str) -> int | None:
    model = model.upper()
    families = (
        r"EPYC\s+9[0-9A-Z]{2}5", r"EPYC\s+9[0-9A-Z]{2}4",
        r"THREADRIPPER(?:\s+PRO)?\s+7[0-9]{3}", r"EPYC\s+7[0-9A-Z]{2}3",
        r"RYZEN\s+[3579]\s+(?:PRO\s+)?[79][0-9]{3}",
    )
    return next((rank for rank, pattern in enumerate(families) if re.search(pattern, model)), None)


def number(*, values: dict, key: str) -> float:
    value = values[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Missing/invalid numeric measurement: {key}")
    return float(value)


def evaluate(*, probe: dict, offer: dict) -> dict:
    failures = []
    try:
        model = probe["cpu_model"]
        if not isinstance(model, str) or cpu_rank(model) is None:
            failures.append("CPU generation/family is unsupported")
        if number(values=probe, key="physical_cores") < 24:
            failures.append("fewer than 24 allowed online physical cores")
        vcpus = number(values=offer, key="cpu_cores_effective")
        if vcpus < 24 or number(values=probe, key="logical_cpus") < vcpus * 0.9:
            failures.append("logical CPU allocation below offer/24-core requirement")
        quota = probe["cpu_quota"]
        if quota is not None and number(values=probe, key="cpu_quota") < vcpus * 0.9:
            failures.append("CPU quota below 90% of advertised effective vCPUs")
        ram = number(values=probe, key="memory_bytes")
        advertised_ram = number(values=offer, key="cpu_ram") * 1_000_000
        if advertised_ram < 64_000_000_000 or ram < 64_000_000_000 or ram < advertised_ram * 0.9:
            failures.append("RAM below 64 GB or inconsistent with offer")
        if number(values=probe, key="workspace_available_bytes") < 100_000_000_000:
            failures.append("less than 100 GB available at /workspace")
        if number(values=offer, key="pcie_bw") < 20:
            failures.append("advertised PCIe bandwidth below 20 GB/s")
        gpus = probe["gpus"]
        if not isinstance(gpus, list) or len(gpus) != 1:
            raise ValueError("Exactly one GPU is required")
        gpu = gpus[0]
        if "RTX 4090" not in gpu["name"]:
            failures.append("GPU is not an RTX 4090")
        if not 22 * 1024 <= number(values=gpu, key="memory_mib") <= 26 * 1024:
            failures.append("GPU memory is not approximately 24 GiB")
        for key, minimum in (("power_watts", 400), ("pcie_gen", 4), ("pcie_width", 16)):
            if number(values=gpu, key=key) < minimum:
                failures.append(f"GPU {key} below {minimum}")
        for key in ("thermal_slowdown", "power_brake_slowdown"):
            if gpu[key] != "Not Active":
                failures.append(f"GPU {key} active or unknown")
    except (KeyError, TypeError, ValueError) as error:
        failures.append(f"Incomplete or malformed hardware probe: {error}")
    return {"passed": not failures, "failures": failures, "probe": probe}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--offer", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(probe=json.loads(args.probe.read_text()), offer=json.loads(args.offer.read_text()))
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
