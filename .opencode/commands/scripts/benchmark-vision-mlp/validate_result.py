import argparse
import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np

from validate_checkpoint import inspect_checkpoint


def load_json(*, path: Path) -> dict:
    return json.loads(path.read_text())


def resolve_control_path(*, path_value: str, control_dir: Path, anchor: str) -> Path:
    path = Path(path_value)
    try:
        relative = Path(*path.parts[path.parts.index(anchor) :])
    except ValueError as error:
        raise ValueError(f"artifact path has no {anchor!r} component: {path}") from error
    local_path = control_dir / relative
    return local_path.resolve(strict=True)


def project_root_for(*, control_dir: Path) -> Path:
    configured = os.environ.get("BENCHMARK_PROJECT_ROOT")
    if configured is not None:
        project_root = Path(configured).resolve(strict=True)
        if not (project_root / "scripts/policy_runtimes").is_dir():
            raise ValueError("BENCHMARK_PROJECT_ROOT is not a project checkout")
        return project_root
    for candidate in control_dir.parents:
        if (candidate / "scripts/policy_runtimes").is_dir():
            return candidate
    raise ValueError("cannot locate the project root for checkpoint validation")


def read_plan_rows(*, path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    lines = path.read_text().splitlines()
    header = lines[0].split("|")
    return [dict(zip(header, line.split("|"), strict=True)) for line in lines[1:] if line]


def require_finite_positive(*, name: str, values: list[float]) -> None:
    if not all(math.isfinite(float(value)) and float(value) > 0 for value in values):
        raise ValueError(f"{name} contains non-finite or non-positive values")


def find_plan_row(*, control_dir: Path, trial_id: str) -> dict[str, str]:
    rows = [
        *read_plan_rows(path=control_dir / "plans/training.plan"),
        *read_plan_rows(path=control_dir / "plans/workers.plan"),
        *read_plan_rows(path=control_dir / "plans/stability-training.plan"),
        *read_plan_rows(path=control_dir / "plans/stability-workers.plan"),
    ]
    matches = [row for row in rows if row["trial_id"] == trial_id]
    if len(matches) != 1:
        raise ValueError(f"expected one plan row for {trial_id}, got {len(matches)}")
    return matches[0]


def validate_common(
    *,
    result: dict,
    metadata: dict,
    manifest: dict,
    kind: str,
    trial_id: str,
    generation: str,
    attempt: int,
) -> None:
    if result.get("success") is not True:
        raise ValueError("result success must be JSON true")
    if type(result.get("schema_version")) is not int or result["schema_version"] != 1:
        raise ValueError("result schema_version must be integer 1")
    if type(result.get("attempt")) is not int:
        raise ValueError("result attempt must be an integer")
    expected = {
        "kind": kind,
        "trial_id": trial_id,
        "generation": generation,
        "attempt": attempt,
        "git_commit": metadata["commit"],
        "dataset_digest": manifest["aggregate_sha256"],
    }
    for key, value in expected.items():
        if type(result.get(key)) is not type(value) or result.get(key) != value:
            raise ValueError(f"result {key} differs: {result.get(key)!r} != {value!r}")


def validate_training(
    *, result: dict, metadata: dict, manifest: dict, plan: dict[str, str] | None
) -> None:
    inputs = result["inputs"]
    if plan is None:
        expected = {
            "batch_size": metadata["batch_size"],
            "dataloader_workers": 0,
            "persistent_workers": False,
            "sample_seed": metadata["seeds"]["train"],
            "model_seed": metadata["seeds"]["train"],
            "warmup_epochs": 1,
            "measured_epochs": 1,
        }
    else:
        expected = {
            "batch_size": int(plan["batch_size"]),
            "dataloader_workers": int(plan["dataloader_workers"]),
            "persistent_workers": plan["persistent_workers"] == "true",
            "sample_seed": metadata["seeds"]["train"],
            "model_seed": metadata["seeds"]["train"],
            "warmup_epochs": 1,
            "measured_epochs": 5,
        }
    if inputs != expected or any(
        type(inputs.get(key)) is not type(value) for key, value in expected.items()
    ):
        raise ValueError(f"training inputs differ: {inputs!r} != {expected!r}")
    if type(inputs["persistent_workers"]) is not bool:
        raise ValueError("persistent_workers must be a JSON boolean")
    if "RTX 4090" not in result.get("device_name", ""):
        raise ValueError("training result has the wrong CUDA device")
    if result["samples_per_epoch"] != manifest["total_frames"]:
        raise ValueError("training sample count differs from the manifest")
    expected_batches = math.ceil(result["samples_per_epoch"] / inputs["batch_size"])
    if result["batches_per_epoch"] != expected_batches:
        raise ValueError("training batch count is incorrect")
    epoch_seconds = result["epoch_seconds"]
    if len(epoch_seconds) != inputs["measured_epochs"]:
        raise ValueError("training epoch count is incorrect")
    require_finite_positive(name="epoch timing", values=epoch_seconds)
    require_finite_positive(
        name="training timing",
        values=[result["measured_seconds"], result["samples_per_second"]],
    )
    expected_sps = (
        result["samples_per_epoch"] * inputs["measured_epochs"]
    ) / result["measured_seconds"]
    if not math.isclose(result["samples_per_second"], expected_sps, rel_tol=1e-9):
        raise ValueError("samples_per_second does not recompute")
    if (
        type(result["peak_allocated_bytes"]) is not int
        or type(result["peak_reserved_bytes"]) is not int
        or result["peak_allocated_bytes"] <= 0
        or result["peak_reserved_bytes"] <= 0
    ):
        raise ValueError("training recorded no CUDA memory")
    metrics = result["epoch_metrics"]
    if len(metrics) != inputs["measured_epochs"]:
        raise ValueError("training metrics count is incorrect")
    if not all(
        set(epoch) == {"loss", "joints_loss", "gripper_loss"}
        and all(math.isfinite(float(value)) for value in epoch.values())
        for epoch in metrics
    ):
        raise ValueError("training metrics contain non-finite values")
    started = datetime.fromisoformat(result["measured_started_utc"])
    finished = datetime.fromisoformat(result["measured_finished_utc"])
    timestamp_seconds = (finished - started).total_seconds()
    if timestamp_seconds <= 0 or not math.isclose(
        timestamp_seconds, result["measured_seconds"], rel_tol=0.05, abs_tol=1.0
    ):
        raise ValueError("training measured timestamps differ from measured_seconds")


def validate_checkpoint(*, result: dict, metadata: dict, control_dir: Path) -> None:
    expected_inputs = {
        "batch_size": metadata["batch_size"],
        "sample_seed": metadata["seeds"]["train"],
        "model_seed": metadata["seeds"]["train"],
        "eval_seed": metadata["seeds"]["eval"],
        "eval_max_steps": metadata["fixed"]["eval_max_steps"],
        "eval_capture_hz": metadata["fixed"]["eval_capture_hz"],
        "epochs": 120,
        "dataloader_workers": 0,
        "persistent_workers": False,
    }
    if result.get("inputs") != expected_inputs:
        raise ValueError("bootstrap inputs differ from the benchmark contract")
    checkpoint_metadata = load_json(path=control_dir / "bootstrap/checkpoint.json")
    if (
        checkpoint_metadata.get("success") is not True
        or checkpoint_metadata.get("arch") != "vision_mlp"
        or checkpoint_metadata.get("epoch") != 120
    ):
        raise ValueError("bootstrap checkpoint metadata is invalid")
    checkpoint_path = resolve_control_path(
        path_value=checkpoint_metadata["checkpoint_path"],
        control_dir=control_dir,
        anchor="bootstrap",
    )
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if digest != checkpoint_metadata["checkpoint_sha256"]:
        raise ValueError("bootstrap checkpoint digest changed")
    if result["checkpoint_path"] != checkpoint_metadata["checkpoint_path"]:
        raise ValueError("bootstrap result points to a different checkpoint")
    if result["checkpoint_sha256"] != digest:
        raise ValueError("bootstrap result digest differs")
    inspected = inspect_checkpoint(
        project_root=project_root_for(control_dir=control_dir),
        checkpoint_path=checkpoint_path,
    )
    if inspected["checkpoint_sha256"] != digest:
        raise ValueError("independent bootstrap checkpoint validation differs")
    require_finite_positive(name="bootstrap timing", values=[result["wall_seconds"]])


def validate_worker(
    *, result: dict, metadata: dict, control_dir: Path, plan: dict[str, str]
) -> None:
    fixed = metadata["fixed"]
    kind = result["kind"]
    workers = int(plan["workers"])
    expected = {
        "seed": metadata["seeds"]["eval" if kind == "evaluation" else "dagger"],
        "workers": workers,
        "episodes": fixed["eval_episodes" if kind == "evaluation" else "dagger_episodes"],
        "max_steps": fixed["eval_max_steps" if kind == "evaluation" else "dagger_max_steps"],
        "capture_hz": fixed["eval_capture_hz" if kind == "evaluation" else "train_capture_hz"],
    }
    for key, value in expected.items():
        if type(result.get(key)) is not type(value) or result.get(key) != value:
            raise ValueError(f"worker {key} differs: {result.get(key)!r} != {value!r}")
    checkpoint_metadata = load_json(path=control_dir / "bootstrap/checkpoint.json")
    if result["checkpoint_path"] != checkpoint_metadata["checkpoint_path"]:
        raise ValueError("worker checkpoint path differs")
    if result["checkpoint_sha256"] != checkpoint_metadata["checkpoint_sha256"]:
        raise ValueError("worker checkpoint digest differs")
    require_finite_positive(name="worker timing", values=[result["wall_seconds"]])
    if kind == "evaluation":
        scores = result["metrics"]["scores"]
        if len(scores) != expected["episodes"]:
            raise ValueError("evaluation episode count differs")
        if not all(math.isfinite(float(value)) for value in scores):
            raise ValueError("evaluation scores contain non-finite values")
    else:
        if result["output_episode_count"] != expected["episodes"]:
            raise ValueError("DAgger episode count differs")
        if result["intervention_threshold"] != fixed["intervention_threshold"]:
            raise ValueError("DAgger threshold differs")
        if result["intervention_steps"] != fixed["intervention_steps"]:
            raise ValueError("DAgger intervention steps differ")
        output_dir = resolve_control_path(
            path_value=result["output_dir"],
            control_dir=control_dir,
            anchor="results",
        )
        expected_names = {
            f"pick_place_{index:06d}.npz" for index in range(expected["episodes"])
        }
        if {path.name for path in output_dir.glob("*.npz")} != expected_names:
            raise ValueError("DAgger output files differ from the episode contract")
        for path in output_dir.glob("*.npz"):
            with np.load(path, allow_pickle=False) as episode:
                if not {"obs", "actions", "img_obs"}.issubset(episode.files):
                    raise ValueError(f"DAgger episode schema is incomplete: {path}")
                lengths = [len(episode[key]) for key in ("obs", "actions", "img_obs")]
                if not lengths[0] or len(set(lengths)) != 1:
                    raise ValueError(f"DAgger episode lengths differ: {path}")
                images = episode["img_obs"]
                if images.dtype != np.uint8 or images.shape[1:] != (64, 64, 3):
                    raise ValueError(f"DAgger image schema differs: {path}")


def validate_result(
    *,
    control_dir: Path,
    kind: str,
    result_path: Path,
    trial_id: str,
    generation: str,
    attempt: int,
) -> None:
    metadata = load_json(path=control_dir / "metadata.json")
    manifest = load_json(path=control_dir / "data/manifest.json")
    result = load_json(path=result_path)
    validate_common(
        result=result,
        metadata=metadata,
        manifest=manifest,
        kind=kind,
        trial_id=trial_id,
        generation=generation,
        attempt=attempt,
    )
    if kind == "training":
        plan = None if trial_id == "preflight" else find_plan_row(
            control_dir=control_dir, trial_id=trial_id
        )
        validate_training(result=result, metadata=metadata, manifest=manifest, plan=plan)
    elif kind == "bootstrap":
        validate_checkpoint(result=result, metadata=metadata, control_dir=control_dir)
    elif kind in {"evaluation", "dagger"}:
        plan = find_plan_row(control_dir=control_dir, trial_id=trial_id)
        if plan["workload"] != kind:
            raise ValueError("worker workload differs from its plan")
        validate_worker(result=result, metadata=metadata, control_dir=control_dir, plan=plan)
    else:
        raise ValueError(f"unsupported result kind: {kind}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--kind", required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--attempt", type=int, required=True)
    args = parser.parse_args()
    validate_result(
        control_dir=args.control_dir,
        kind=args.kind,
        result_path=args.result,
        trial_id=args.trial_id,
        generation=args.generation,
        attempt=args.attempt,
    )


if __name__ == "__main__":
    main()
