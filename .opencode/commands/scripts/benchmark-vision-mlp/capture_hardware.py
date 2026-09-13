import argparse
import json
import os
import platform
import subprocess
from pathlib import Path

from common import write_json


GIB = 1024**3


def read_optional(*, path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def cpu_quota_cores() -> float | None:
    cpu_max_path = Path("/sys/fs/cgroup/cpu.max")
    cpu_max = read_optional(path=cpu_max_path)
    if cpu_max_path.exists():
        if cpu_max is None:
            raise ValueError("cgroup v2 CPU quota is unreadable")
        quota, period = cpu_max.split()
        return None if quota == "max" else int(quota) / int(period)

    quota_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    quota = read_optional(path=quota_path)
    period = read_optional(path=period_path)
    if not quota_path.exists() or not period_path.exists() or quota is None or period is None:
        raise ValueError("no readable cgroup CPU quota files")
    if int(quota) < 0:
        return None
    return int(quota) / int(period)


def memory_limit_bytes() -> int | None:
    memory_max_path = Path("/sys/fs/cgroup/memory.max")
    memory_max = read_optional(path=memory_max_path)
    if memory_max_path.exists():
        if memory_max is None:
            raise ValueError("cgroup v2 memory limit is unreadable")
        return None if memory_max == "max" else int(memory_max)

    memory_limit_path = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    value = read_optional(path=memory_limit_path)
    if not memory_limit_path.exists() or value is None:
        raise ValueError("no readable cgroup memory limit file")
    parsed = int(value)
    return None if parsed >= 2**60 else parsed


def cpu_identity() -> dict[str, object]:
    fields: dict[str, str] = {}
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if not line.strip():
            break
        key, value = line.split(":", maxsplit=1)
        fields[key.strip()] = value.strip()
    return {
        "model_name": fields["model name"],
        "vendor_id": fields["vendor_id"],
        "family": int(fields["cpu family"]),
        "model": int(fields["model"]),
    }


def allowed_physical_cores() -> tuple[list[int], int]:
    allowed = sorted(os.sched_getaffinity(0))
    physical = set()
    for cpu in allowed:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        core = int((topology / "core_id").read_text())
        socket = int((topology / "physical_package_id").read_text())
        physical.add((socket, core))
    return allowed, len(physical)


def gpu_rows() -> list[dict[str, str]]:
    fields = [
        "name",
        "memory.total",
        "power.limit",
        "power.default_limit",
        "pcie.link.gen.max",
        "pcie.link.width.max",
        "clocks_throttle_reasons.hw_thermal_slowdown",
        "clocks_throttle_reasons.hw_power_brake_slowdown",
    ]
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return [
        {key: value.strip() for key, value in zip(fields, line.split(","), strict=True)}
        for line in output.splitlines()
        if line.strip()
    ]


def require_offer_thresholds(*, offer: dict) -> list[str]:
    failures = []

    def number(key: str) -> float:
        if key not in offer:
            failures.append(f"offer is missing {key}")
            return float("-inf")
        return float(offer[key])

    if str(offer.get("gpu_name", "")).replace("_", " ") != "RTX 4090":
        failures.append("offer GPU is not RTX_4090")
    thresholds = {
        "gpu_frac": 1,
        "num_gpus": 1,
        "gpu_ram": 24,
        "gpu_max_power": 400,
        "compute_cap": 890,
        "total_flops": 80,
        "cpu_cores_effective": 24,
        "cpu_ram": 64,
        "disk_bw": 1000,
        "pci_gen": 4,
        "pcie_bw": 20,
        "inet_down": 500,
        "inet_up": 200,
        "reliability": 0.99,
    }
    for key, threshold in thresholds.items():
        value = number(key)
        if key in {"gpu_ram", "cpu_ram"} and value >= threshold * 100:
            value /= 1024
        if value < threshold:
            failures.append(f"offer {key}={value} is below {threshold}")
    if number("gpu_frac") != 1 or number("num_gpus") != 1:
        failures.append("offer must provide exactly one full GPU")
    if offer.get("rentable") not in {True, 1, "true", "True"}:
        failures.append("offer is not rentable")
    if str(offer.get("verification", "")).lower() != "verified":
        failures.append("offer is not verified")
    if offer.get("gpu_display_active") not in {False, 0, "false", "False"}:
        failures.append("offer GPU has an active display")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    offer = json.loads(args.offer.read_text())
    identity = cpu_identity()
    allowed_cpus, physical_cores = allowed_physical_cores()
    quota_cores = cpu_quota_cores()
    finite_memory_limit = memory_limit_bytes()
    visible_memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    usable_memory = finite_memory_limit or visible_memory
    gpus = gpu_rows()

    failures = require_offer_thresholds(offer=offer)
    if identity["vendor_id"] != "AuthenticAMD" or int(identity["family"]) < 25:
        failures.append("CPU must be AMD Zen 3 or newer")
    if "EPYC" in str(identity["model_name"]) and any(
        marker in str(identity["model_name"]) for marker in (" 7", "7001", "7002")
    ) and int(identity["family"]) < 25:
        failures.append("EPYC 7001/7002 is not accepted")
    if physical_cores < 24:
        failures.append(f"only {physical_cores} allowed physical cores")
    advertised_vcpus = float(offer["cpu_cores_effective"])
    if quota_cores is not None and quota_cores < advertised_vcpus * 0.9:
        failures.append(
            f"CPU quota {quota_cores:.2f} is below 90% of {advertised_vcpus:.2f}"
        )
    if usable_memory < 64 * GIB:
        failures.append(f"usable memory is only {usable_memory / GIB:.1f} GiB")
    if len(gpus) != 1 or "RTX 4090" not in gpus[0]["name"]:
        failures.append("exactly one RTX 4090 is required")
    elif (
        float(gpus[0]["memory.total"]) < 23 * 1024
        or float(gpus[0]["power.default_limit"]) < 400
        or float(gpus[0]["power.limit"]) < 400
        or int(gpus[0]["pcie.link.gen.max"]) < 4
        or int(gpus[0]["pcie.link.width.max"]) < 16
    ):
        failures.append("GPU power or PCIe capability is below requirements")
    elif any(
        gpus[0][field] != "Not Active"
        for field in (
            "clocks_throttle_reasons.hw_thermal_slowdown",
            "clocks_throttle_reasons.hw_power_brake_slowdown",
        )
    ):
        failures.append("GPU thermal or power-brake throttling is active")
    if failures:
        raise ValueError("; ".join(failures))

    write_json(
        path=args.output,
        value={
            "success": True,
            "cpu": identity,
            "allowed_cpus": allowed_cpus,
            "allowed_logical_cpus": len(allowed_cpus),
            "allowed_physical_cores": physical_cores,
            "cpu_quota_cores": quota_cores,
            "visible_memory_bytes": visible_memory,
            "memory_limit_bytes": finite_memory_limit,
            "usable_memory_bytes": usable_memory,
            "gpus": gpus,
            "offer": offer,
            "kernel": platform.release(),
        },
    )


if __name__ == "__main__":
    main()
