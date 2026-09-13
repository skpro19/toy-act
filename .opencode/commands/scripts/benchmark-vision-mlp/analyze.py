import argparse
import csv
import hashlib
import io
import json
import math
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from common import write_json, write_text
from validate_result import validate_result
from validate_telemetry import validate_telemetry


TRAINING_ID = re.compile(r"^(dw(0|2|4|8)-p([01]))-r([1-5])$")
WORKER_ID = re.compile(r"^(evaluation|dagger)-w(6|12)-r([1-5])$")


def load_json(*, path: Path) -> dict:
    return json.loads(path.read_text())


def read_plan(*, path: Path, fields: list[str]) -> list[dict[str, str]]:
    lines = path.read_text().splitlines()
    if not lines or lines[0].split("|") != fields:
        raise ValueError(f"invalid plan header: {path}")
    rows = [
        dict(zip(fields, line.split("|"), strict=True))
        for line in lines[1:]
        if line
    ]
    if len({row["trial_id"] for row in rows}) != len(rows):
        raise ValueError(f"duplicate trial ID in {path}")
    return rows


def expected_training_rows(*, batch_size: int) -> list[dict[str, str]]:
    orders = [
        ["dw0-p0", "dw2-p0", "dw2-p1", "dw4-p0", "dw4-p1", "dw8-p0", "dw8-p1"],
        ["dw8-p1", "dw8-p0", "dw4-p1", "dw4-p0", "dw2-p1", "dw2-p0", "dw0-p0"],
        ["dw4-p0", "dw4-p1", "dw8-p0", "dw8-p1", "dw0-p0", "dw2-p0", "dw2-p1"],
    ]
    rows = []
    for repetition, configurations in enumerate(orders, start=1):
        for configuration in configurations:
            workers, persistence = configuration.split("-")
            rows.append(
                {
                    "order": str(len(rows) + 1),
                    "repetition": str(repetition),
                    "batch_size": str(batch_size),
                    "dataloader_workers": workers.removeprefix("dw"),
                    "persistent_workers": str(persistence == "p1").lower(),
                    "trial_id": f"{configuration}-r{repetition}",
                }
            )
    return rows


def expected_worker_rows() -> list[dict[str, str]]:
    orders = [[6, 12], [12, 6], [6, 12]]
    rows = []
    for repetition, worker_order in enumerate(orders, start=1):
        for workers in worker_order:
            for workload in ("evaluation", "dagger"):
                rows.append(
                    {
                        "order": str(len(rows) + 1),
                        "repetition": str(repetition),
                        "workload": workload,
                        "workers": str(workers),
                        "trial_id": f"{workload}-w{workers}-r{repetition}",
                    }
                )
    return rows


def status_identity(*, control_dir: Path, trial_id: str) -> tuple[str, int]:
    values = (control_dir / "status" / f"{trial_id}.status").read_text().split()
    if len(values) != 4 or values[:2] != ["succeeded", "0"]:
        raise ValueError(f"trial is not successful: {trial_id}")
    return values[2], int(values[3])


def parse_number(*, value: str) -> float:
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", value)
    if match is None:
        raise ValueError(f"no number in telemetry value: {value!r}")
    return float(match.group())


def percentile(*, values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def parse_gpu_timestamp(*, value: str, timezone) -> datetime:
    parsed = datetime.strptime(value.strip(), "%Y/%m/%d %H:%M:%S.%f")
    return parsed.replace(tzinfo=timezone)


def telemetry_summary(
    *, control_dir: Path, trial_id: str, attempt: int, result: dict
) -> dict:
    prefix = f"{trial_id}-a{attempt}"
    paths = {
        "gpu": control_dir / "telemetry" / f"{prefix}-gpu.csv",
        "pidstat": control_dir / "telemetry" / f"{prefix}-pidstat.txt",
        "vmstat": control_dir / "telemetry" / f"{prefix}-vmstat.txt",
        "time": control_dir / "time" / f"{prefix}.txt",
        "log": control_dir / "logs" / f"{prefix}.log",
        "omissions": control_dir / "telemetry" / f"{prefix}-omissions.json",
    }
    empty = [name for name, path in paths.items() if not path.is_file() or path.stat().st_size == 0]
    if empty:
        raise ValueError(f"missing telemetry for {trial_id}: {empty}")
    validate_telemetry(
        gpu=paths["gpu"],
        pidstat=paths["pidstat"],
        vmstat=paths["vmstat"],
        time_file=paths["time"],
    )

    with paths["gpu"].open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"GPU telemetry is empty: {trial_id}")
    utilization = [parse_number(value=row["utilization.gpu"]) for row in rows]
    power = [parse_number(value=row["power.draw"]) for row in rows]
    memory = [parse_number(value=row["memory.used"]) for row in rows]

    time_values = {}
    for line in paths["time"].read_text().splitlines():
        if ":" in line:
            key, value = line.strip().split(":", maxsplit=1)
            time_values[key] = value.strip()
    required_time = [
        "User time (seconds)",
        "System time (seconds)",
        "Maximum resident set size (kbytes)",
        "File system inputs",
        "File system outputs",
    ]
    if any(key not in time_values for key in required_time):
        raise ValueError(f"incomplete /usr/bin/time telemetry: {trial_id}")
    measured_rows = rows
    if result["kind"] == "training" and trial_id != "preflight":
        measured_start = datetime.fromisoformat(result["measured_started_utc"])
        measured_end = datetime.fromisoformat(result["measured_finished_utc"])
        measured_rows = [
            row
            for row in rows
            if measured_start
            <= parse_gpu_timestamp(
                value=row["timestamp"], timezone=measured_start.tzinfo
            )
            <= measured_end
        ]
        if not measured_rows:
            raise ValueError(f"no GPU samples in measured window: {trial_id}")
    measured_utilization = [
        parse_number(value=row["utilization.gpu"]) for row in measured_rows
    ]
    omissions = load_json(path=paths["omissions"])["omitted_gpu_fields"]
    return {
        "gpu_samples": len(rows),
        "measured_gpu_samples": len(measured_rows),
        "gpu_utilization_median": statistics.median(utilization),
        "measured_gpu_utilization_median": statistics.median(measured_utilization),
        "measured_gpu_idle_fraction": (
            sum(value < 5 for value in measured_utilization)
            / len(measured_utilization)
        ),
        "gpu_power_median_watts": statistics.median(power),
        "gpu_power_p95_watts": percentile(values=power, fraction=0.95),
        "gpu_memory_peak_mb": max(memory),
        "user_cpu_seconds": float(time_values["User time (seconds)"]),
        "system_cpu_seconds": float(time_values["System time (seconds)"]),
        "maximum_rss_kb": int(time_values["Maximum resident set size (kbytes)"]),
        "filesystem_inputs": int(time_values["File system inputs"]),
        "filesystem_outputs": int(time_values["File system outputs"]),
        "omitted_gpu_fields": omissions,
        "files": {
            name: {
                "path": str(path.relative_to(control_dir)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in paths.items()
        },
    }


def validate_trial(*, control_dir: Path, kind: str, trial_id: str, path: Path) -> tuple[dict, dict]:
    generation, attempt = status_identity(control_dir=control_dir, trial_id=trial_id)
    validate_result(
        control_dir=control_dir,
        kind=kind,
        result_path=path,
        trial_id=trial_id,
        generation=generation,
        attempt=attempt,
    )
    result = load_json(path=path)
    return result, telemetry_summary(
        control_dir=control_dir,
        trial_id=trial_id,
        attempt=attempt,
        result=result,
    )


def aggregate(*, values: list[float]) -> dict[str, float]:
    median = statistics.median(values)
    return {
        "median": median,
        "minimum": min(values),
        "maximum": max(values),
        "relative_spread": (max(values) - min(values)) / median,
    }


def training_summary(*, control_dir: Path, rows: list[dict[str, str]]) -> tuple[list[dict], dict[str, dict]]:
    trials = []
    grouped = defaultdict(list)
    for plan in rows:
        trial_id = plan["trial_id"]
        match = TRAINING_ID.fullmatch(trial_id)
        if match is None:
            raise ValueError(f"invalid training trial ID: {trial_id}")
        result, telemetry = validate_trial(
            control_dir=control_dir,
            kind="training",
            trial_id=trial_id,
            path=control_dir / "results/training" / f"{trial_id}.json",
        )
        row = {
            "trial_id": trial_id,
            "configuration": match.group(1),
            "repetition": int(match.group(4)),
            "workers": result["inputs"]["dataloader_workers"],
            "persistent_workers": result["inputs"]["persistent_workers"],
            "samples_per_second": result["samples_per_second"],
            "measured_seconds": result["measured_seconds"],
            "peak_allocated_mb": result["peak_allocated_bytes"] / 1024**2,
            "telemetry": telemetry,
        }
        trials.append(row)
        grouped[row["configuration"]].append(row)
    configurations = {
        key: {
            **aggregate(values=[row["samples_per_second"] for row in rows]),
            "median_measured_seconds": statistics.median(
                row["measured_seconds"] for row in rows
            ),
            "median_gpu_idle_fraction": statistics.median(
                row["telemetry"]["measured_gpu_idle_fraction"] for row in rows
            ),
            "median_maximum_rss_kb": statistics.median(
                row["telemetry"]["maximum_rss_kb"] for row in rows
            ),
            "repetitions": len(rows),
        }
        for key, rows in grouped.items()
    }
    return trials, configurations


def worker_summary(*, control_dir: Path, rows: list[dict[str, str]]) -> tuple[list[dict], dict[str, dict]]:
    trials = []
    paired = defaultdict(dict)
    for plan in rows:
        trial_id = plan["trial_id"]
        match = WORKER_ID.fullmatch(trial_id)
        if match is None:
            raise ValueError(f"invalid worker trial ID: {trial_id}")
        workload = match.group(1)
        result, telemetry = validate_trial(
            control_dir=control_dir,
            kind=workload,
            trial_id=trial_id,
            path=control_dir / "results/workers" / f"{trial_id}.json",
        )
        row = {
            "trial_id": trial_id,
            "workload": workload,
            "workers": int(match.group(2)),
            "repetition": int(match.group(3)),
            "wall_seconds": result["wall_seconds"],
            "episodes": result["episodes"],
            "telemetry": telemetry,
        }
        trials.append(row)
        paired[(row["workers"], row["repetition"])][workload] = row["wall_seconds"]
    paired_values = defaultdict(list)
    for (workers, repetition), workloads in paired.items():
        if set(workloads) != {"evaluation", "dagger"}:
            raise ValueError(f"incomplete worker pair: workers={workers}, rep={repetition}")
        paired_values[workers].append(workloads["evaluation"] + workloads["dagger"])
    worker_resources = defaultdict(list)
    for row in trials:
        worker_resources[row["workers"]].append(row["telemetry"])
    configurations = {
        str(workers): {
            **aggregate(values=values),
            "median_maximum_rss_kb": statistics.median(
                item["maximum_rss_kb"] for item in worker_resources[workers]
            ),
            "median_cpu_seconds": statistics.median(
                item["user_cpu_seconds"] + item["system_cpu_seconds"]
                for item in worker_resources[workers]
            ),
            "repetitions": len(values),
        }
        for workers, values in paired_values.items()
    }
    return trials, configurations


def stability_requirements(*, training: dict[str, dict], workers: dict[str, dict]) -> dict[str, list[str]]:
    production = {key: value for key, value in training.items() if key.endswith("-p0")}
    persistent = {key: value for key, value in training.items() if key.endswith("-p1")}
    candidates = [
        *sorted(production, key=lambda key: production[key]["median"], reverse=True)[:3],
        *sorted(persistent, key=lambda key: persistent[key]["median"], reverse=True)[:3],
    ]
    return {
        "training": [key for key in candidates if training[key]["relative_spread"] > 0.10],
        "workers": [key for key, value in workers.items() if value["relative_spread"] > 0.10],
    }


def expected_stability_training(*, needed: list[str], batch_size: int) -> list[dict[str, str]]:
    rows = []
    for configuration in needed:
        workers, persistence = configuration.split("-")
        for repetition in (4, 5):
            rows.append(
                {
                    "order": str(len(rows) + 1),
                    "repetition": str(repetition),
                    "batch_size": str(batch_size),
                    "dataloader_workers": workers.removeprefix("dw"),
                    "persistent_workers": str(persistence == "p1").lower(),
                    "trial_id": f"{configuration}-r{repetition}",
                }
            )
    return rows


def expected_stability_workers(*, needed: list[str]) -> list[dict[str, str]]:
    rows = []
    for workers in needed:
        for repetition in (4, 5):
            for workload in ("evaluation", "dagger"):
                rows.append(
                    {
                        "order": str(len(rows) + 1),
                        "repetition": str(repetition),
                        "workload": workload,
                        "workers": workers,
                        "trial_id": f"{workload}-w{workers}-r{repetition}",
                    }
                )
    return rows


def csv_text(*, rows: list[dict], fields: list[str]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def select_training_winner(*, configurations: dict[str, dict]) -> str:
    fastest = max(value["median"] for value in configurations.values())
    near_fastest = [key for key, value in configurations.items() if value["median"] >= fastest * 0.98]
    return min(
        near_fastest,
        key=lambda key: (
            configurations[key]["median_measured_seconds"],
            configurations[key]["median_gpu_idle_fraction"],
            configurations[key]["median_maximum_rss_kb"],
            int(key.removeprefix("dw").split("-")[0]),
        ),
    )


def select_worker_winner(*, configurations: dict[str, dict]) -> str:
    fastest = min(value["median"] for value in configurations.values())
    near_fastest = [key for key, value in configurations.items() if value["median"] <= fastest * 1.02]
    return min(
        near_fastest,
        key=lambda key: (
            configurations[key]["median_maximum_rss_kb"],
            configurations[key]["median_cpu_seconds"],
            int(key),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    metadata = load_json(path=args.control_dir / "metadata.json")
    manifest = load_json(path=args.control_dir / "data/manifest.json")
    if metadata["dataset_digest"] != manifest["aggregate_sha256"]:
        raise ValueError("metadata and manifest dataset digests differ")

    training_fields = ["order", "repetition", "batch_size", "dataloader_workers", "persistent_workers", "trial_id"]
    worker_fields = ["order", "repetition", "workload", "workers", "trial_id"]
    training_rows = read_plan(path=args.control_dir / "plans/training.plan", fields=training_fields)
    worker_rows = read_plan(path=args.control_dir / "plans/workers.plan", fields=worker_fields)
    if training_rows != expected_training_rows(batch_size=metadata["batch_size"]):
        raise ValueError("training plan differs from the benchmark contract")
    if worker_rows != expected_worker_rows():
        raise ValueError("worker plan differs from the benchmark contract")

    validate_trial(
        control_dir=args.control_dir,
        kind="training",
        trial_id="preflight",
        path=args.control_dir / "results/training/preflight.json",
    )
    validate_trial(
        control_dir=args.control_dir,
        kind="bootstrap",
        trial_id="bootstrap",
        path=args.control_dir / "results/workers/bootstrap.json",
    )
    initial_training_trials, initial_training = training_summary(control_dir=args.control_dir, rows=training_rows)
    initial_worker_trials, initial_workers = worker_summary(control_dir=args.control_dir, rows=worker_rows)
    stability_needed = stability_requirements(training=initial_training, workers=initial_workers)
    write_json(path=args.output_dir / "stability-needed.json", value=stability_needed)

    if args.require_complete:
        stability_training = read_plan(
            path=args.control_dir / "plans/stability-training.plan", fields=training_fields
        )
        stability_workers = read_plan(
            path=args.control_dir / "plans/stability-workers.plan", fields=worker_fields
        )
        expected_training = expected_stability_training(
            needed=stability_needed["training"], batch_size=metadata["batch_size"]
        )
        expected_workers = expected_stability_workers(needed=stability_needed["workers"])
        if stability_training != expected_training or stability_workers != expected_workers:
            raise ValueError("stability plans differ from computed requirements")
        training_trials, training = training_summary(
            control_dir=args.control_dir, rows=[*training_rows, *stability_training]
        )
        worker_trials, workers = worker_summary(
            control_dir=args.control_dir, rows=[*worker_rows, *stability_workers]
        )
    else:
        training_trials, training = initial_training_trials, initial_training
        worker_trials, workers = initial_worker_trials, initial_workers

    production = {key: value for key, value in training.items() if key.endswith("-p0")}
    persistent = {key: value for key, value in training.items() if key.endswith("-p1")}
    production_winner = select_training_winner(configurations=production)
    persistent_winner = select_training_winner(configurations=persistent)
    worker_winner = select_worker_winner(configurations=workers)
    persistent_speedups = {
        key: value["median"] / production[key.replace("-p1", "-p0")]["median"] - 1
        for key, value in persistent.items()
    }
    baseline_training = (
        f"dw{metadata['baseline']['dataloader_workers']}-"
        f"p{int(metadata['baseline']['persistent_workers'])}"
    )
    if baseline_training not in training:
        raise ValueError("configured DataLoader baseline is not benchmarked")
    baseline_worker = str(metadata["baseline"]["workers"])
    if baseline_worker not in workers:
        raise ValueError("configured worker baseline is not benchmarked")
    baseline_speedups = {
        "production_winner": (
            production[production_winner]["median"]
            / training[baseline_training]["median"]
            - 1
        ),
        "persistent_winner": (
            persistent[persistent_winner]["median"]
            / training[baseline_training]["median"]
            - 1
        ),
        "worker_winner": (
            workers[baseline_worker]["median"] / workers[worker_winner]["median"] - 1
        ),
    }
    retries = sorted(path.name for path in (args.control_dir / "status/history").glob("*.status"))
    finished_path = args.control_dir / "state/workload-finished.json"
    elapsed_seconds = None
    estimated_cost = None
    if finished_path.is_file():
        started = datetime.fromisoformat(
            metadata["instance"].get("created_utc", metadata["started_utc"])
        )
        finished = datetime.fromisoformat(load_json(path=finished_path)["finished_utc"].replace("Z", "+00:00"))
        elapsed_seconds = (finished - started).total_seconds()
        price = metadata["instance"].get("price_per_hour", metadata["instance"].get("dph_total"))
        if price is not None:
            estimated_cost = elapsed_seconds / 3600 * float(price)
    telemetry_omissions = sorted(
        {
            field
            for row in [*training_trials, *worker_trials]
            for field in row["telemetry"]["omitted_gpu_fields"]
        }
    )

    summary = {
        "benchmark": metadata["benchmark"],
        "branch": metadata["branch"],
        "commit": metadata["commit"],
        "config": metadata["config"],
        "config_sha256": metadata["config_sha256"],
        "config_contents": metadata["config_contents"],
        "batch_size": metadata["batch_size"],
        "global_seed": metadata["global_seed"],
        "seeds": metadata["seeds"],
        "fixed": metadata["fixed"],
        "baseline": metadata["baseline"],
        "baseline_speedups": baseline_speedups,
        "dataset_digest": manifest["aggregate_sha256"],
        "dataset_frames": manifest["total_frames"],
        "instance": metadata["instance"],
        "hardware": metadata["hardware"],
        "software": metadata["software"],
        "plans": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((args.control_dir / "plans").glob("*.plan"))
        },
        "training_trials": training_trials,
        "training_configurations": training,
        "production_winner": production_winner,
        "persistent_winner": persistent_winner,
        "persistent_speedups": persistent_speedups,
        "worker_trials": worker_trials,
        "worker_configurations": workers,
        "worker_winner": int(worker_winner),
        "stability_needed": stability_needed,
        "retry_statuses": retries,
        "telemetry_omissions": telemetry_omissions,
        "elapsed_seconds": elapsed_seconds,
        "estimated_cost": estimated_cost,
        "no_full_flywheel": True,
        "no_batch_size_sweep": True,
        "no_policy_quality_comparison": True,
    }
    write_json(path=args.output_dir / "benchmark-summary.json", value=summary)
    write_text(
        path=args.output_dir / "training-throughput.csv",
        text=csv_text(
            rows=training_trials,
            fields=["trial_id", "configuration", "repetition", "workers", "persistent_workers", "samples_per_second", "measured_seconds", "peak_allocated_mb"],
        ),
    )
    write_text(
        path=args.output_dir / "worker-throughput.csv",
        text=csv_text(
            rows=worker_trials,
            fields=["trial_id", "workload", "workers", "repetition", "wall_seconds", "episodes"],
        ),
    )

    lines = [
        "# Vision MLP Infrastructure Benchmark",
        "",
        f"- Branch/commit: `{metadata['branch']}` / `{metadata['commit']}`",
        f"- Config/batch size: `{metadata['config']}` / {metadata['batch_size']}",
        f"- Seeds: `{json.dumps(metadata['seeds'], sort_keys=True)}`",
        f"- Dataset: {manifest['episode_count']} episodes, {manifest['total_frames']} frames, `{manifest['aggregate_sha256']}`",
        f"- Instance: `{metadata['instance'].get('id', 'unknown')}`",
        f"- Accepted hardware: {metadata['hardware']['allowed_physical_cores']} physical cores, {metadata['hardware']['usable_memory_bytes'] / 1024**3:.1f} GiB RAM, {metadata['hardware']['gpus'][0]['name']}",
        f"- Retries: {len(retries)}; telemetry omissions: {telemetry_omissions}",
        f"- Elapsed/cost: {elapsed_seconds if elapsed_seconds is not None else 'in progress'} seconds / {estimated_cost if estimated_cost is not None else 'unavailable'}",
        "",
        "## Production DataLoader Ranking",
        "",
        "| Configuration | Median samples/s | Spread | Repetitions |",
        "|---|---:|---:|---:|",
    ]
    for key, value in sorted(production.items(), key=lambda item: item[1]["median"], reverse=True):
        lines.append(f"| {key} | {value['median']:.0f} | {value['relative_spread']:.1%} | {value['repetitions']} |")
    lines.extend(["", f"Recommendation: `{production_winner}`.", "", "## Persistent Workers", "", "| Configuration | Median samples/s | Speedup | Spread |", "|---|---:|---:|---:|"])
    for key, value in sorted(persistent.items(), key=lambda item: item[1]["median"], reverse=True):
        lines.append(f"| {key} | {value['median']:.0f} | {persistent_speedups[key]:+.1%} | {value['relative_spread']:.1%} |")
    lines.extend(["", f"Persistent candidate: `{persistent_winner}` (not a policy-quality result).", "", "## Simulator Workers", "", "| Workers | Median paired seconds | Spread | Repetitions |", "|---:|---:|---:|---:|"])
    for key, value in sorted(workers.items(), key=lambda item: int(item[0])):
        lines.append(f"| {key} | {value['median']:.2f} | {value['relative_spread']:.1%} | {value['repetitions']} |")
    lines.extend(
        [
            "",
            f"Recommendation: `{worker_winner}` workers.",
            "",
            "## Baseline Speedups",
            "",
            f"- Production winner vs `{baseline_training}`: {baseline_speedups['production_winner']:+.1%}",
            f"- Persistent winner vs `{baseline_training}`: {baseline_speedups['persistent_winner']:+.1%}",
            f"- Worker winner vs `{baseline_worker}` workers: {baseline_speedups['worker_winner']:+.1%}",
            "",
            "## Training Trials",
            "",
            "| Trial | samples/s | measured s | GPU util | GPU idle | max RSS KiB |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in training_trials:
        telemetry = row["telemetry"]
        lines.append(
            f"| {row['trial_id']} | {row['samples_per_second']:.0f} | "
            f"{row['measured_seconds']:.2f} | "
            f"{telemetry['measured_gpu_utilization_median']:.1f}% | "
            f"{telemetry['measured_gpu_idle_fraction']:.1%} | "
            f"{telemetry['maximum_rss_kb']} |"
        )
    lines.extend(
        [
            "",
            "## Worker Trials",
            "",
            "| Trial | wall s | GPU power p95 W | GPU memory peak MiB | CPU s | max RSS KiB |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in worker_trials:
        telemetry = row["telemetry"]
        lines.append(
            f"| {row['trial_id']} | {row['wall_seconds']:.2f} | "
            f"{telemetry['gpu_power_p95_watts']:.1f} | "
            f"{telemetry['gpu_memory_peak_mb']:.1f} | "
            f"{telemetry['user_cpu_seconds'] + telemetry['system_cpu_seconds']:.1f} | "
            f"{telemetry['maximum_rss_kb']} |"
        )
    lines.extend(
        [
            "",
            "## Controls And Limitations",
            "",
            f"- Fixed controls: `{json.dumps(metadata['fixed'], sort_keys=True)}`",
            f"- Failed attempts preserved in history: `{retries}`",
            "- Skipped trials: none; completion requires every planned trial.",
            "- Telemetry details and file paths are embedded per trial in `benchmark-summary.json`.",
            "- Exact config contents are embedded in `benchmark-summary.json` metadata.",
            "",
            "No full flywheel, batch-size sweep, or policy-quality comparison was run.",
            "Run `/ablate-flywheel-params batch_size` for a later batch-size quality ablation.",
        ]
    )
    write_text(path=args.output_dir / "benchmark-summary.md", text="\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
