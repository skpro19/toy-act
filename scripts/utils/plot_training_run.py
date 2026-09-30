"""Render the main TensorBoard metrics from one ACT training run as PNGs.

Example:
    uv run python scripts/utils/plot_training_run.py \
        --events /path/to/events.out.tfevents... \
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
from matplotlib.figure import Figure
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def read_series(*, accumulator: EventAccumulator, tag: str) -> tuple[np.ndarray, np.ndarray]:
    if tag not in accumulator.Tags()["scalars"]:
        raise ValueError(f"missing TensorBoard scalar: {tag}")
    events = accumulator.Scalars(tag)
    return (
        np.array([event.step for event in events]),
        np.array([event.value for event in events]),
    )


def style_axis(*, axis: Axes, ylabel: str) -> None:
    axis.set_ylabel(ylabel)
    axis.set_xlabel("Training step")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)


def save_figure(*, figure: Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"wrote {path}")


def plot_evaluation(*, accumulator: EventAccumulator, output_dir: Path) -> None:
    figure, (success_axis, horizon_axis) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for axis, tag, label in (
        (success_axis, "eval/success_rate", "Success rate"),
        (horizon_axis, "eval/horizon_mean", "Mean episode horizon"),
    ):
        steps, values = read_series(accumulator=accumulator, tag=tag)
        axis.plot(steps, values, marker="o", markersize=3, linewidth=1.5, label=label)
        style_axis(axis=axis, ylabel=label)
    success_axis.set_ylim(0, 1)
    success_axis.set_title("Evaluation during training")
    save_figure(figure=figure, path=output_dir / "evaluation.png")


def smooth_series(*, values: np.ndarray, window: int = 500) -> np.ndarray:
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    indices = np.arange(len(values))
    starts = np.maximum(indices - window + 1, 0)
    return (cumulative[indices + 1] - cumulative[starts]) / (indices - starts + 1)


def plot_training_loss(*, accumulator: EventAccumulator, output_dir: Path) -> None:
    figure, (loss_axis, kl_axis) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for axis, tag, label in (
        (loss_axis, "batch_metrics/loss", "Total loss"),
        (loss_axis, "batch_metrics/action_loss", "Action L1 loss"),
        (kl_axis, "batch_metrics/weighted_kl_loss", "Weighted KL loss"),
    ):
        steps, values = read_series(accumulator=accumulator, tag=tag)
        axis.plot(steps[::50], smooth_series(values=values)[::50], linewidth=1.5, label=label)
    style_axis(axis=loss_axis, ylabel="Loss")
    style_axis(axis=kl_axis, ylabel="Weighted KL loss")
    loss_axis.set_yscale("log")
    kl_axis.set_yscale("log")
    loss_axis.set_title("Training loss (500-step moving average)")
    save_figure(figure=figure, path=output_dir / "training_loss.png")


def plot_action_error(*, accumulator: EventAccumulator, output_dir: Path) -> None:
    figure, (joint_axis, gripper_axis) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for axis, tag, label in (
        (joint_axis, "denorm_l1/joint", "Joint L1 error"),
        (gripper_axis, "denorm_l1/gripper", "Gripper L1 error"),
    ):
        steps, values = read_series(accumulator=accumulator, tag=tag)
        axis.plot(steps[::50], smooth_series(values=values)[::50], linewidth=1.5, label=label)
        style_axis(axis=axis, ylabel=label)
        axis.set_yscale("log")
    joint_axis.set_title("Denormalized action error (500-step moving average)")
    save_figure(figure=figure, path=output_dir / "action_error.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    accumulator = EventAccumulator(str(args.events), size_guidance={"scalars": 0})
    accumulator.Reload()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_evaluation(accumulator=accumulator, output_dir=args.output_dir)
    plot_training_loss(accumulator=accumulator, output_dir=args.output_dir)
    plot_action_error(accumulator=accumulator, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
