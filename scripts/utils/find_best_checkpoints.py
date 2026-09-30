"""Find the best checkpoint of each training run by training-time eval score.

Each training run logs ``eval/success_rate`` to TensorBoard every
``checkpoint_every`` steps (see ``scripts/train_v2.py``). This script reads those
event files, picks the global step with the highest score for every run (ties
resolved toward the later step), and writes a manifest that pairs the run with
its best ``step_*.pt`` checkpoint and the run metadata needed for labelling
(batch size, image keys).

Examples:
    uv run python scripts/utils/find_best_checkpoints.py

    uv run python scripts/utils/find_best_checkpoints.py \\
        --runs-dir runs/act_v2 --output assets/rollout-two-camera/best_checkpoints.json

    uv run python scripts/utils/find_best_checkpoints.py \\
        --run-name 20260929-195938_bs64_lr1e-04_...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RUNS_DIR = REPO_ROOT / "runs" / "act_v2"
DEFAULT_METRIC = "eval/success_rate"
EVENT_GLOB = "events.out.tfevents.*"
CONFIG_FILENAME = "config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="directory searched recursively for run directories with event files",
    )
    parser.add_argument(
        "--run-name",
        action="append",
        default=None,
        help="restrict to these run directory names (repeatable)",
    )
    parser.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        help=f"scalar tag used to rank checkpoints (default: {DEFAULT_METRIC})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="write the manifest JSON here (default: print to stdout)",
    )
    return parser.parse_args()


def discover_run_dirs(*, runs_dir: Path) -> list[Path]:
    run_dirs = {path.parent for path in runs_dir.rglob(EVENT_GLOB)}
    return sorted(run_dirs)


def read_scalar_series(*, run_dir: Path, tag: str) -> list[tuple[int, float]]:
    series: dict[int, float] = {}
    for event_path in sorted(run_dir.glob(EVENT_GLOB)):
        accumulator = EventAccumulator(str(event_path))
        accumulator.Reload()
        if tag not in accumulator.Tags().get("scalars", []):
            continue
        for event in accumulator.Scalars(tag):
            series[event.step] = float(event.value)
    return sorted(series.items())


def read_run_config(*, run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / CONFIG_FILENAME
    if not config_path.is_file():
        return {}
    payload = json.loads(config_path.read_text())
    return payload.get("config", {})


def best_step(*, series: list[tuple[int, float]]) -> tuple[int, float]:
    step, value = max(series, key=lambda item: (item[1], item[0]))
    return step, value


def checkpoint_name_for_step(*, global_step: int) -> str:
    return f"step_{global_step:09d}.pt"


def build_manifest(*, args: argparse.Namespace) -> dict[str, Any]:
    runs_dir = args.runs_dir.resolve()
    run_dirs = discover_run_dirs(runs_dir=runs_dir)
    if args.run_name is not None:
        wanted = set(args.run_name)
        run_dirs = [run_dir for run_dir in run_dirs if run_dir.name in wanted]
    if not run_dirs:
        raise SystemExit(f"no run directories with event files under {runs_dir}")

    runs: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        series = read_scalar_series(run_dir=run_dir, tag=args.metric)
        if not series:
            print(f"skip {run_dir.name}: no {args.metric} scalars")
            continue
        step, value = best_step(series=series)
        config = read_run_config(run_dir=run_dir)
        image_keys = config.get("image_keys", [])
        runs.append(
            {
                "run_name": run_dir.name,
                "run_dir": str(run_dir.relative_to(REPO_ROOT)),
                "checkpoint": checkpoint_name_for_step(global_step=step),
                "global_step": step,
                "eval_success_rate": value,
                "batch_size": config.get("batch_size"),
                "image_keys": image_keys,
                "num_cameras": len(image_keys),
                "points": [{"step": s, "value": v} for s, v in series],
            }
        )

    runs.sort(key=lambda run: run["eval_success_rate"], reverse=True)
    return {"metric": args.metric, "runs_dir": str(runs_dir), "runs": runs}


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args=args)
    text = json.dumps(manifest, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(f"wrote {args.output}")
    else:
        print(text)

    for run in manifest["runs"]:
        print(
            f"{run['run_name']}: best step {run['global_step']} "
            f"({run['checkpoint']}) {manifest['metric']}={run['eval_success_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
