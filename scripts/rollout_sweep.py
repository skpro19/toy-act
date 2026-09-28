"""Run rollouts over every checkpoint in a training run and save the results.

Evaluates each ``epoch_*.pt`` snapshot in a checkpoint directory with the same
rollout procedure as ``scripts/rollout.py`` and writes the run's artifacts to
``rollouts/<run-name>/``: a success-rate curve (``success_rate.png``), the raw
per-checkpoint summaries (``results.json``), and the same arrays as a NumPy
``results.npz``.

Robosuite samples object placements from the global NumPy RNG, so the sweep
seeds ``numpy.random`` from ``--seed`` before every episode. This gives all
checkpoints the same initial states (common random numbers) and keeps their
success rates directly comparable.

Examples:
    PYTHONPATH=. uv run python scripts/rollout_sweep.py \\
        --model-version act_v2 \\
        --checkpoint-dir checkpoints/act_v2/<run-name> \\
        --n-rollouts 10 --horizon 250 --no-on-screen
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from scripts.models.act_v1.config import IMG_DIMS
from scripts.rollout import (
    DEFAULT_DATASET,
    DEFAULT_HORIZON,
    close_env,
    configure_renderer,
    create_rollout_env,
    load_model,
    make_rollout_env_meta,
    run_rollout,
    summarize_rollouts,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "rollouts"
EPOCH_CHECKPOINT_PATTERN = re.compile(r"^epoch_(\d+)\.pt$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="directory containing epoch_*.pt checkpoints",
    )
    parser.add_argument(
        "--model-version",
        choices=("act_v1", "act_v2"),
        default="act_v1",
        help="checkpoint model architecture",
    )
    parser.add_argument(
        "--use-z",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="act_v2: decode with latent z (use --no-use-z for no-z checkpoints)",
    )
    parser.add_argument(
        "--n-rollouts",
        type=int,
        default=30,
        help="number of evaluation episodes per checkpoint",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help="max steps per episode",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed for env resets")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="robomimic low-dim hdf5 used to recreate PickPlaceCan",
    )
    parser.add_argument(
        "--terminate-on-success",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="stop an episode early once the task succeeds",
    )
    parser.add_argument(
        "--on-screen",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="render live in the MuJoCo viewer during each rollout",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="plot output directory (default: rollouts/<run-name>)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="keep going if a checkpoint rollout fails",
    )
    return parser.parse_args()


def parse_epoch_checkpoint(*, path: Path) -> int | None:
    match = EPOCH_CHECKPOINT_PATTERN.match(path.name)
    if match is None:
        return None
    return int(match.group(1))


def select_checkpoints(*, checkpoint_dir: Path) -> list[tuple[int, Path]]:
    checkpoints: list[tuple[int, Path]] = []
    for path in checkpoint_dir.glob("epoch_*.pt"):
        epoch = parse_epoch_checkpoint(path=path)
        if epoch is not None:
            checkpoints.append((epoch, path))
    if not checkpoints:
        raise SystemExit(f"no epoch_*.pt checkpoints found in {checkpoint_dir}")
    return sorted(checkpoints)


def evaluate_checkpoint(
    *,
    checkpoint: Path,
    env,
    device: torch.device,
    model_version: str,
    use_z: bool,
    n_rollouts: int,
    horizon: int,
    terminate_on_success: bool,
    on_screen: bool,
    reset_seed_base: int,
    progress: tqdm,
) -> dict[str, float | int]:
    model, normalization = load_model(
        checkpoint_path=checkpoint,
        device=device,
        model_version=model_version,
        use_z=use_z,
    )
    rollouts: list[dict[str, float | int | bool]] = []
    try:
        for rollout_idx in range(n_rollouts):
            np.random.seed(reset_seed_base + rollout_idx)
            rollout_stats = run_rollout(
                model=model,
                env=env,
                device=device,
                normalization=normalization,
                horizon=horizon,
                terminate_on_success=terminate_on_success,
                render=on_screen,
                video_writer=None,
                video_skip=1,
            )
            rollouts.append(rollout_stats)
            progress.set_postfix(
                rollout=f"{rollout_idx + 1}/{n_rollouts}",
                success=str(rollout_stats["success"]),
            )
            progress.update(1)
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return summarize_rollouts(rollouts=rollouts)


def save_results(
    *,
    results: list[dict[str, float | int | str]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        epochs=np.array([int(record["epoch"]) for record in results], dtype=np.int64),
        checkpoints=np.array([str(record["checkpoint"]) for record in results]),
        num_rollouts=np.array(
            [int(record["num_rollouts"]) for record in results],
            dtype=np.int64,
        ),
        num_success=np.array(
            [int(record["num_success"]) for record in results],
            dtype=np.int64,
        ),
        success_rate=np.array(
            [float(record["success_rate"]) for record in results],
            dtype=np.float64,
        ),
        return_mean=np.array(
            [float(record["return_mean"]) for record in results],
            dtype=np.float64,
        ),
        horizon_mean=np.array(
            [float(record["horizon_mean"]) for record in results],
            dtype=np.float64,
        ),
        num_truncated=np.array(
            [int(record["num_truncated"]) for record in results],
            dtype=np.int64,
        ),
    )


def plot_success_rate(
    *,
    results: list[dict[str, float | int | str]],
    run_name: str,
    output_path: Path,
) -> None:
    epochs = [int(record["epoch"]) for record in results]
    rates = [float(record["success_rate"]) for record in results]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, rates, marker="o", color="tab:blue", label="success rate")
    for epoch, record in zip(epochs, results):
        ax.annotate(
            f"{int(record['num_success'])}/{int(record['num_rollouts'])}",
            xy=(epoch, float(record["success_rate"])),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("success rate")
    ax.set_ylim(-0.02, 1.08)
    ax.set_title(
        "rollout success rate\n" + "\n".join(textwrap.wrap(run_name, width=80)),
        fontsize=11,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, format="png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir.resolve()
    if not checkpoint_dir.is_dir():
        raise SystemExit(f"checkpoint directory not found: {checkpoint_dir}")

    run_name = checkpoint_dir.name
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else DEFAULT_OUTPUT_ROOT / run_name
    )
    checkpoints = select_checkpoints(checkpoint_dir=checkpoint_dir)

    configure_renderer(on_screen=args.on_screen)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_meta = make_rollout_env_meta(
        dataset_path=args.dataset,
        camera_height=IMG_DIMS[0],
        camera_width=IMG_DIMS[1],
    )
    env = create_rollout_env(
        env_meta=env_meta,
        on_screen=args.on_screen,
        write_video=False,
    )

    print(
        f"sweep: {len(checkpoints)} checkpoints x {args.n_rollouts} rollouts "
        f"= {len(checkpoints) * args.n_rollouts} episodes"
    )
    print("checkpoints: " + ", ".join(path.name for _, path in checkpoints))

    results: list[dict[str, float | int | str]] = []
    failures: list[str] = []
    progress = tqdm(
        total=len(checkpoints) * args.n_rollouts,
        unit="rollout",
        desc="sweep",
        dynamic_ncols=True,
    )
    try:
        for epoch, checkpoint in checkpoints:
            progress.set_description(f"epoch {epoch}")
            try:
                summary = evaluate_checkpoint(
                    checkpoint=checkpoint,
                    env=env,
                    device=device,
                    model_version=args.model_version,
                    use_z=args.use_z,
                    n_rollouts=args.n_rollouts,
                    horizon=args.horizon,
                    terminate_on_success=args.terminate_on_success,
                    on_screen=args.on_screen,
                    reset_seed_base=args.seed,
                    progress=progress,
                )
            except Exception as exc:
                failures.append(checkpoint.name)
                tqdm.write(f"{checkpoint.name}: failed ({exc})")
                if not args.continue_on_error:
                    raise
                continue

            record = {"epoch": epoch, "checkpoint": checkpoint.name, **summary}
            results.append(record)
            tqdm.write(
                f"epoch {epoch}: success={summary['num_success']}/{summary['num_rollouts']} "
                f"rate={summary['success_rate']:.3f} "
                f"return={summary['return_mean']:.3f} "
                f"horizon={summary['horizon_mean']:.1f}"
            )
    finally:
        progress.close()
        close_env(env)

    if results:
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = output_dir / "success_rate.png"
        plot_success_rate(results=results, run_name=run_name, output_path=plot_path)
        results_path = output_dir / "results.npz"
        save_results(results=results, output_path=results_path)
        records_path = output_dir / "results.json"
        records_path.write_text(json.dumps(results, indent=2) + "\n")
        print(f"saved plot: {plot_path}")
        print(f"saved results: {results_path}")
        print(f"saved records: {records_path}")

    if failures:
        raise SystemExit("rollout failed for: " + ", ".join(failures))


if __name__ == "__main__":
    main()
