"""Plot training-time eval curves for a set of runs side by side.

Reads the ``eval/success_rate`` and ``eval/horizon_mean`` scalars that
``scripts/train_v2.py`` writes to TensorBoard and renders them as a row of two
panels so an ablation (for example batch size) can be compared at a glance. The
canvas and panel sizes match ``plot_training_run.py`` so every curve in the
README lines up. Runs are ordered by their ``batch_size`` from ``config.json``
(falling back to the ``bs<N>`` token in the run directory name).

Examples:
    uv run python scripts/utils/plot_eval_curves.py \\
        --runs-dir runs/act_v2/v4 \\
        --run-name 20260929-113816_bs8_lr1e-04_... \\
        --run-name 20260929-162138_bs16_lr1e-04_... \\
        --output assets/eval-curves/batch_size_uniform_panels.png
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from plot_layout import FIGURE_DPI, FIGURE_HEIGHT, FIGURE_WIDTH, place_panel_row

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RUNS_DIR = REPO_ROOT / "runs" / "act_v2"
SUCCESS_RATE_TAG = "eval/success_rate"
HORIZON_TAG = "eval/horizon_mean"
EVENT_GLOB = "events.out.tfevents.*"
CONFIG_FILENAME = "config.json"
BATCH_SIZE_PATTERN = re.compile(r"_bs(\d+)_")


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
        "--output",
        type=Path,
        required=True,
        help="path of the PNG figure to write",
    )
    parser.add_argument(
        "--title",
        default="",
        help="optional figure suptitle",
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


def run_label(*, run_dir: Path, config: dict[str, Any]) -> str:
    batch_size = config.get("batch_size")
    if batch_size is None:
        match = BATCH_SIZE_PATTERN.search(run_dir.name)
        batch_size = int(match.group(1)) if match else None
    num_cameras = len(config.get("image_keys", []))
    if batch_size is None:
        return run_dir.name
    if num_cameras:
        camera_word = "camera" if num_cameras == 1 else "cameras"
        return f"bs={batch_size} · {num_cameras} {camera_word}"
    return f"bs={batch_size}"


def build_runs(*, runs_dir: Path, run_names: list[str] | None) -> list[dict[str, Any]]:
    run_dirs = discover_run_dirs(runs_dir=runs_dir)
    if run_names is not None:
        wanted = set(run_names)
        run_dirs = [run_dir for run_dir in run_dirs if run_dir.name in wanted]
    if not run_dirs:
        raise SystemExit(f"no run directories with event files under {runs_dir}")

    runs: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        success = read_scalar_series(run_dir=run_dir, tag=SUCCESS_RATE_TAG)
        horizon = read_scalar_series(run_dir=run_dir, tag=HORIZON_TAG)
        if not success and not horizon:
            print(f"skip {run_dir.name}: no eval scalars")
            continue
        runs.append(
            {
                "run_dir": run_dir,
                "config": read_run_config(run_dir=run_dir),
                "success_rate": success,
                "horizon": horizon,
            }
        )
    runs.sort(key=lambda run: run["config"].get("batch_size") or 0)
    return runs


def plot_runs(*, runs: list[dict[str, Any]], output: Path, title: str) -> None:
    figure, axes = plt.subplots(
        1, 2, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), sharex=True, squeeze=False
    )
    place_panel_row(axes=list(axes[0, :]))
    success_ax, horizon_ax = axes[0, :]

    for run in runs:
        label = run_label(run_dir=run["run_dir"], config=run["config"])
        if run["success_rate"]:
            steps, values = zip(*run["success_rate"])
            success_ax.plot(steps, values, marker="o", markersize=3, linewidth=1.5, label=label)
        if run["horizon"]:
            steps, values = zip(*run["horizon"])
            horizon_ax.plot(steps, values, marker="o", markersize=3, linewidth=1.5, label=label)

    success_ax.set_ylabel("success rate")
    success_ax.set_ylim(0.0, 1.0)
    horizon_ax.set_ylabel("mean horizon")

    for ax in (success_ax, horizon_ax):
        ax.set_xlabel("training step")
        ax.grid(alpha=0.3)

    handles, labels = success_ax.get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="center left", bbox_to_anchor=(0.005, 0.5))

    if title:
        figure.suptitle(title)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=FIGURE_DPI)
    plt.close(figure)
    print(f"wrote {output}")


def main() -> None:
    args = parse_args()
    runs = build_runs(runs_dir=args.runs_dir.resolve(), run_names=args.run_name)
    plot_runs(runs=runs, output=args.output, title=args.title)


if __name__ == "__main__":
    main()
