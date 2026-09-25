"""Remap ACT v2 TensorBoard scalar tags to the post-refactor namespace layout."""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader
from tensorboard.compat.proto import event_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter

BACKUP_SUFFIX = ".pre_namespace_remap.bak"

EPOCH_TRAIN_TAG_MAP = {
    "train/loss": "epoch_metrics/loss",
    "train/l1_loss": "epoch_metrics/action_loss",
    "train/kl_loss": "epoch_metrics/kl_loss",
    "train/weighted_kl_loss": "epoch_metrics/weighted_kl_loss",
    "train/kl_fraction": "epoch_metrics/kl_fraction",
    "train/l1_fraction": "epoch_metrics/l1_fraction",
}

DEBUG_SCALAR_TAG_MAP = {
    "debug/batch_loss": "batch_metrics/loss",
    "debug/l1_loss": "batch_metrics/action_loss",
    "debug/kl_loss": "batch_metrics/kl_loss",
    "debug/weighted_kl_loss": "batch_metrics/weighted_kl_loss",
    "debug/kl_fraction": "batch_metrics/kl_fraction",
    "debug/grad_norm_global": "optimizer/grad_norm_global",
    "debug/param_norm_global": "optimizer/param_norm_global",
    "debug/update_norm_global": "optimizer/update_norm_global",
    "debug/lr": "optimizer/lr",
    "debug/adam_exp_avg_norm": "optimizer/adam_exp_avg_norm",
    "debug/adam_exp_avg_sq_norm": "optimizer/adam_exp_avg_sq_norm",
    "debug/l1_joint": "denorm_l1/joint",
    "debug/l1_gripper": "denorm_l1/gripper",
    "debug/pred_min_joint": "ranges/pred_min_joint",
    "debug/pred_max_joint": "ranges/pred_max_joint",
    "debug/pred_min_gripper": "ranges/pred_min_gripper",
    "debug/pred_max_gripper": "ranges/pred_max_gripper",
    "debug/target_min_joint": "ranges/target_min_joint",
    "debug/target_max_joint": "ranges/target_max_joint",
    "debug/target_min_gripper": "ranges/target_min_gripper",
    "debug/target_max_gripper": "ranges/target_max_gripper",
    "debug/mu_norm": "latent/mu_norm",
    "debug/log_sigma_x2_mean": "latent/log_sigma_x2_mean",
    "debug/sigma_mean": "latent/sigma_mean",
}

FORBIDDEN_TAG_PREFIXES = ("debug/",)


def remap_scalar_tag(*, tag: str) -> str:
    if tag in DEBUG_SCALAR_TAG_MAP:
        return DEBUG_SCALAR_TAG_MAP[tag]
    if tag.startswith("debug/activations/"):
        return f"activations/{tag.removeprefix('debug/activations/')}"
    if tag in EPOCH_TRAIN_TAG_MAP:
        return EPOCH_TRAIN_TAG_MAP[tag]
    if tag.startswith("train/batch/"):
        return f"batch_metrics/{tag.removeprefix('train/batch/')}"
    if tag.startswith("train/epoch/"):
        return f"epoch_metrics/{tag.removeprefix('train/epoch/')}"
    return tag


def remap_summary_event(*, event: event_pb2.Event) -> tuple[event_pb2.Event, int]:
    if not event.summary or not event.summary.value:
        return event, 0

    remapped = event_pb2.Event()
    remapped.CopyFrom(event)
    changes = 0
    for value in remapped.summary.value:
        new_tag = remap_scalar_tag(tag=value.tag)
        if new_tag != value.tag:
            value.tag = new_tag
            changes += 1
    return remapped, changes


def event_file_paths(*, run_dir: Path) -> list[Path]:
    paths = sorted(run_dir.glob("events.out.tfevents.*"))
    return [
        path
        for path in paths
        if not path.name.endswith(BACKUP_SUFFIX)
        and ".bak" not in path.name
    ]


def backup_path(*, event_path: Path) -> Path:
    return event_path.with_name(event_path.name + BACKUP_SUFFIX)


def ensure_backup(*, event_path: Path) -> Path:
    path = backup_path(event_path=event_path)
    if not path.is_file():
        shutil.copy2(event_path, path)
    return path


def inspect_event_file(*, event_path: Path) -> dict:
    event_count = 0
    tag_changes = 0
    tensor_values = 0
    unknown_tags: set[str] = set()

    for event in LegacyEventFileLoader(str(event_path)).Load():
        event_count += 1
        if event.summary:
            for value in event.summary.value:
                if value.HasField("tensor"):
                    tensor_values += 1
                new_tag = remap_scalar_tag(tag=value.tag)
                if new_tag != value.tag:
                    tag_changes += 1
                if any(new_tag.startswith(prefix) for prefix in FORBIDDEN_TAG_PREFIXES):
                    unknown_tags.add(value.tag)

    return {
        "event_count": event_count,
        "tag_changes": tag_changes,
        "tensor_values": tensor_values,
        "unknown_tags": unknown_tags,
    }


def rewrite_event_file(*, event_path: Path, dry_run: bool) -> dict:
    source_path = event_path
    inspection = inspect_event_file(event_path=source_path)
    existing_backup = backup_path(event_path=event_path)
    if inspection["tensor_values"] > 0 and existing_backup.is_file():
        source_path = existing_backup
        inspection = inspect_event_file(event_path=source_path)

    event_count = inspection["event_count"]
    tag_changes = inspection["tag_changes"]
    unknown_tags = inspection["unknown_tags"]

    if dry_run or tag_changes == 0:
        return {
            "event_path": str(event_path),
            "event_count": event_count,
            "tag_changes": tag_changes,
            "unknown_tags": sorted(unknown_tags),
            "dry_run": dry_run,
            "skipped": tag_changes == 0,
        }

    source_path = ensure_backup(event_path=event_path)
    with tempfile.TemporaryDirectory() as temp_dir:
        writer = EventFileWriter(logdir=temp_dir, flush_secs=1)
        for event in LegacyEventFileLoader(str(source_path)).Load():
            remapped_event, _ = remap_summary_event(event=event)
            writer.add_event(remapped_event)
        writer.close()

        temp_events = sorted(Path(temp_dir).glob("events.out.tfevents.*"))
        if len(temp_events) != 1:
            raise RuntimeError(
                f"expected one remapped event file in {temp_dir}, found {len(temp_events)}",
            )
        event_path.unlink(missing_ok=True)
        shutil.move(temp_events[0], event_path)

    return {
        "event_path": str(event_path),
        "event_count": event_count,
        "tag_changes": tag_changes,
        "unknown_tags": sorted(unknown_tags),
        "dry_run": False,
        "skipped": False,
    }


def remap_extracted_metrics(*, metrics_path: Path, dry_run: bool) -> dict:
    with metrics_path.open() as file:
        payload = json.load(file)

    step_data = payload.get("step_data")
    if not isinstance(step_data, dict):
        return {"metrics_path": str(metrics_path), "key_changes": 0, "skipped": True}

    remapped_step_data: dict = {}
    key_changes = 0
    for key, value in step_data.items():
        new_key = remap_scalar_tag(tag=key)
        if new_key != key:
            key_changes += 1
        if new_key in remapped_step_data:
            raise RuntimeError(
                f"duplicate key after remap in {metrics_path}: {key!r} and existing {new_key!r}",
            )
        remapped_step_data[new_key] = value

    if not dry_run and key_changes > 0:
        backup = metrics_path.with_name(metrics_path.name + BACKUP_SUFFIX)
        if not backup.is_file():
            shutil.copy2(metrics_path, backup)
        payload["step_data"] = remapped_step_data
        with metrics_path.open("w") as file:
            json.dump(payload, file, indent=2, sort_keys=True)
            file.write("\n")

    return {
        "metrics_path": str(metrics_path),
        "key_changes": key_changes,
        "skipped": False,
        "dry_run": dry_run,
    }


def verify_with_accumulator(*, event_path: Path) -> dict:
    accumulator = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    accumulator.Reload()
    scalar_tags = sorted(accumulator.Tags().get("scalars", []))
    tensor_tags = sorted(accumulator.Tags().get("tensors", []))
    tags = sorted(set(scalar_tags) | set(tensor_tags))
    stale = [tag for tag in tags if tag.startswith("debug/")]
    return {
        "event_path": event_path.name,
        "scalar_tags": scalar_tags,
        "tensor_tags": tensor_tags,
        "tags": tags,
        "stale_debug_tags": stale,
        "ok": not stale and not tensor_tags,
    }


def verify_run_dir(*, run_dir: Path) -> dict:
    issues: list[str] = []
    tag_union: set[str] = set()
    event_files = event_file_paths(run_dir=run_dir)

    for event_path in event_files:
        accumulator = verify_with_accumulator(event_path=event_path)
        tags = set(accumulator["tags"])
        tag_union |= tags
        if accumulator["stale_debug_tags"]:
            issues.append(
                f"{event_path.name}: stale debug tags "
                f"{accumulator['stale_debug_tags']}"
            )
        if accumulator["tensor_tags"]:
            issues.append(
                f"{event_path.name}: metrics classified as tensors "
                f"{accumulator['tensor_tags']}"
            )
        non_idempotent = sorted(
            tag for tag in tags if remap_scalar_tag(tag=tag) != tag
        )
        if non_idempotent:
            issues.append(f"{event_path.name}: tags still need remap {non_idempotent}")

    metrics_path = run_dir / "extracted_metrics.json"
    if metrics_path.is_file():
        with metrics_path.open() as file:
            payload = json.load(file)
        step_data = payload.get("step_data", {})
        if isinstance(step_data, dict):
            stale_keys = sorted(key for key in step_data if key.startswith("debug/"))
            if stale_keys:
                issues.append(
                    f"extracted_metrics.json: stale debug keys ({len(stale_keys)} total)",
                )

    return {
        "run_dir": run_dir.name,
        "event_files": len(event_files),
        "scalar_tags": len(tag_union),
        "tags": sorted(tag_union),
        "ok": not issues,
        "issues": issues,
    }


def remap_run_dir(*, run_dir: Path, dry_run: bool) -> list[dict]:
    results: list[dict] = []
    for event_path in event_file_paths(run_dir=run_dir):
        results.append(rewrite_event_file(event_path=event_path, dry_run=dry_run))

    metrics_path = run_dir / "extracted_metrics.json"
    if metrics_path.is_file():
        results.append(remap_extracted_metrics(metrics_path=metrics_path, dry_run=dry_run))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remap ACT v2 TensorBoard scalar tags to the current namespace layout.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs/act_v2"),
        help="directory containing ACT v2 run folders",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report changes without rewriting files",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify tag layout under runs-root without rewriting",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_root: Path = args.runs_root
    if not runs_root.is_dir():
        raise FileNotFoundError(f"runs root not found: {runs_root}")

    run_dirs = sorted(path for path in runs_root.iterdir() if path.is_dir())
    if not run_dirs:
        print(f"no run directories under {runs_root.resolve()}")
        return

    if args.verify_only:
        all_ok = True
        for run_dir in run_dirs:
            summary = verify_run_dir(run_dir=run_dir)
            status = "OK" if summary["ok"] else "FAIL"
            print(f"[{status}] {summary['run_dir']} tags={summary['scalar_tags']}")
            if summary["issues"]:
                all_ok = False
                for issue in summary["issues"]:
                    print(f"  - {issue}")
        if not all_ok:
            sys.exit(1)
        return

    for run_dir in run_dirs:
        print(f"remapping {run_dir.name} ...")
        for result in remap_run_dir(run_dir=run_dir, dry_run=args.dry_run):
            if "event_path" in result:
                print(
                    f"  events: {Path(result['event_path']).name} "
                    f"count={result['event_count']} tag_changes={result['tag_changes']} "
                    f"skipped={result['skipped']}",
                )
                if result["unknown_tags"]:
                    print(f"  WARNING unknown tags: {result['unknown_tags']}")
            elif result.get("metrics_path"):
                print(
                    f"  metrics: key_changes={result['key_changes']} "
                    f"skipped={result.get('skipped', False)}",
                )

    if args.dry_run:
        return

    print("verifying ...")
    all_ok = True
    for run_dir in run_dirs:
        summary = verify_run_dir(run_dir=run_dir)
        status = "OK" if summary["ok"] else "FAIL"
        print(f"[{status}] {summary['run_dir']} tags={summary['scalar_tags']}")
        if not summary["ok"]:
            all_ok = False
            for issue in summary["issues"]:
                print(f"  - {issue}")
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
