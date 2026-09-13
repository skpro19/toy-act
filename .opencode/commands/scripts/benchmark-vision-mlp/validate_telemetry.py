import argparse
import csv
import math
import re
from pathlib import Path


def require_text(*, path: Path, patterns: tuple[str, ...]) -> None:
    text = path.read_text()
    if not text.strip() or not all(re.search(pattern, text) for pattern in patterns):
        raise ValueError(f"invalid telemetry file: {path}")


def numeric_value(*, value: str) -> float:
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", value)
    if match is None:
        raise ValueError(f"non-numeric GPU telemetry value: {value!r}")
    parsed = float(match.group())
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite GPU telemetry value: {value!r}")
    return parsed


def validate_telemetry(*, gpu: Path, pidstat: Path, vmstat: Path, time_file: Path) -> None:
    with gpu.open(newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    required_fields = {
        "timestamp",
        "utilization.gpu",
        "utilization.memory",
        "power.draw",
        "temperature.gpu",
        "clocks.current.graphics",
        "clocks.current.memory",
        "memory.used",
        "memory.total",
    }
    if reader.fieldnames is None or not required_fields.issubset(reader.fieldnames):
        raise ValueError("GPU telemetry header is incomplete")
    if not rows:
        raise ValueError("GPU telemetry has no samples")
    for row in rows:
        for field in required_fields - {"timestamp"}:
            numeric_value(value=row[field])

    require_text(path=pidstat, patterns=(r"Linux", r"(?:UID|CPU)"))
    require_text(path=vmstat, patterns=(r"procs", r"swpd", r"free"))
    require_text(
        path=time_file,
        patterns=(
            r"User time \(seconds\)",
            r"System time \(seconds\)",
            r"Maximum resident set size \(kbytes\)",
            r"File system inputs",
            r"File system outputs",
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=Path, required=True)
    parser.add_argument("--pidstat", type=Path, required=True)
    parser.add_argument("--vmstat", type=Path, required=True)
    parser.add_argument("--time", type=Path, required=True)
    args = parser.parse_args()
    validate_telemetry(
        gpu=args.gpu,
        pidstat=args.pidstat,
        vmstat=args.vmstat,
        time_file=args.time,
    )


if __name__ == "__main__":
    main()
