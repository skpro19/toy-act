"""Render the main TensorBoard metrics from one ACT training run as PNGs.

Produces one figure per metric group, with a single panel per metric, so the
training dynamics of a run can be inspected at a glance:

- ``batch_metrics_horizontal.png``: action, weighted KL, and total losses (smoothed, log scale)
- ``denorm_l1_horizontal.png``: denormalized joint and gripper L1 errors (smoothed, log scale)
- ``eval_horizontal.png``: episode horizon and success rate
- ``latent_log_scale.png``: latent ``mu`` norm and ``sigma`` mean (log scale)

Example:
    uv run python scripts/utils/plot_training_run.py \\
        --events runs/act_v2/v4/20260929-195938_bs64_lr1e-04_/events.out.tfevents.1790711978... \\
        --output-dir assets/training-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

SMOOTHING_WINDOW = 500
SMOOTHED_STRIDE = 50
PANEL_Y_LIMITS = {"eval/success_rate": (0.0, 1.0)}

BATCH_METRICS = (
    ("batch_metrics/action_loss", "Action L1 loss"),
    ("batch_metrics/weighted_kl_loss", "Weighted KL loss"),
    ("batch_metrics/loss", "Total loss"),
)
DENORM_METRICS = (
    ("denorm_l1/joint", "Joint L1 error"),
    ("denorm_l1/gripper", "Gripper L1 error"),
)
EVAL_METRICS = (
    ("eval/horizon_mean", "Mean episode horizon"),
    ("eval/success_rate", "Success rate"),
)
LATENT_METRICS = (
    ("latent/mu_norm", "mu norm"),
    ("latent/sigma_mean", "sigma mean"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--events", type=Path, required=True, help="TensorBoard event file")
    parser.add_argument("--output-dir", type=Path, required=True, help="where PNGs are written")
    return parser.parse_args()


def read_series(*, accumulator: EventAccumulator, tag: str) -> tuple[np.ndarray, np.ndarray]:
    if tag not in accumulator.Tags()["scalars"]:
        raise ValueError(f"missing TensorBoard scalar: {tag}")
    events = accumulator.Scalars(tag)
    return (
        np.array([event.step for event in events]),
        np.array([event.value for event in events]),
    )


def smooth_series(*, values: np.ndarray, window: int = SMOOTHING_WINDOW) -> np.ndarray:
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    indices = np.arange(len(values))
    starts = np.maximum(indices - window + 1, 0)
    return (cumulative[indices + 1] - cumulative[starts]) / (indices - starts + 1)


def style_axis(*, axis: Axes, label: str, tag: str, log_scale: bool) -> None:
    axis.set_ylabel(f"{label} (log scale)" if log_scale else label)
    axis.grid(alpha=0.25)
    if log_scale:
        axis.set_yscale("log")
    if tag in PANEL_Y_LIMITS:
        axis.set_ylim(*PANEL_Y_LIMITS[tag])


def render_group(
    *,
    accumulator: EventAccumulator,
    output_path: Path,
    title: str,
    metrics: tuple[tuple[str, str], ...],
    smooth: bool,
    log_scale: bool,
) -> None:
    figure, axes = plt.subplots(1, len(metrics), figsize=(4.0 * len(metrics), 3.2), sharex=True, squeeze=False)
    for axis, (tag, label) in zip(axes[0, :], metrics):
        steps, values = read_series(accumulator=accumulator, tag=tag)
        if smooth:
            steps, values = steps[::SMOOTHED_STRIDE], smooth_series(values=values)[::SMOOTHED_STRIDE]
        axis.plot(steps, values, linewidth=1.5, color="tab:blue")
        style_axis(axis=axis, label=label, tag=tag, log_scale=log_scale)
        axis.set_xlabel("Training step")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"wrote {output_path}")


def main() -> None:
    args = parse_args()
    accumulator = EventAccumulator(str(args.events), size_guidance={"scalars": 0})
    accumulator.Reload()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    render_group(
        accumulator=accumulator,
        output_path=args.output_dir / "batch_metrics_horizontal.png",
        title="Batch metrics (500-step moving average)",
        metrics=BATCH_METRICS,
        smooth=True,
        log_scale=True,
    )
    render_group(
        accumulator=accumulator,
        output_path=args.output_dir / "denorm_l1_horizontal.png",
        title="Denormalized action error (500-step moving average)",
        metrics=DENORM_METRICS,
        smooth=True,
        log_scale=True,
    )
    render_group(
        accumulator=accumulator,
        output_path=args.output_dir / "eval_horizontal.png",
        title="Evaluation during training",
        metrics=EVAL_METRICS,
        smooth=False,
        log_scale=False,
    )
    render_group(
        accumulator=accumulator,
        output_path=args.output_dir / "latent_log_scale.png",
        title="Latent statistics",
        metrics=LATENT_METRICS,
        smooth=False,
        log_scale=True,
    )


if __name__ == "__main__":
    main()
