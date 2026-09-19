"""Run rollout.py sequentially over selected checkpoints in a training run.

By default every 5th snapshot is evaluated: epochs 5, 25, 45, and so on.
Only checkpoint files that exist on disk are run.

Examples:
    uv run python -m scripts.rollout_sweep

    uv run python -m scripts.rollout_sweep \\
        --checkpoint-dir checkpoints/act_v1/20260919-143154_bs250_lr1e-04 \\
        --no-on-screen

    uv run python -m scripts.rollout_sweep -- --no-terminate-on-success
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

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
DEFAULT_CHECKPOINT_DIR = (
    REPO_ROOT / "checkpoints" / "act_v1" / "20260919-143154_bs250_lr1e-04"
)
EPOCH_CHECKPOINT_PATTERN = re.compile(r"^epoch_(\d+)\.pt$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="directory containing epoch_*.pt checkpoints",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=5,
        help="first epoch number to include in the sweep",
    )
    parser.add_argument(
        "--end-epoch",
        type=int,
        default=None,
        help="last epoch number to include in the sweep (inclusive; no upper bound if omitted)",
    )
    parser.add_argument(
        "--epoch-step",
        type=int,
        default=20,
        help="evaluate every Nth epoch starting from --start-epoch (5, 25, 45, ...)",
    )
    parser.add_argument(
        "--n-rollouts",
        type=int,
        default=3,
        help="number of evaluation episodes per checkpoint",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed for env resets",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="max steps per episode (uses rollout default if omitted)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="dataset path (uses rollout default if omitted)",
    )
    parser.add_argument(
        "--on-screen",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="render live in the MuJoCo viewer during each rollout",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="keep going if a checkpoint rollout fails",
    )
    parser.add_argument(
        "rollout_args",
        nargs=argparse.REMAINDER,
        help="extra arguments forwarded to rollout.py after `--`",
    )
    return parser.parse_args()


def parse_epoch_checkpoint(*, path: Path) -> int | None:
    match = EPOCH_CHECKPOINT_PATTERN.match(path.name)
    if match is None:
        return None
    return int(match.group(1))


def select_checkpoints(
    *,
    checkpoint_dir: Path,
    start_epoch: int,
    end_epoch: int | None,
    epoch_step: int,
) -> list[Path]:
    checkpoints: list[tuple[int, Path]] = []
    for path in sorted(checkpoint_dir.glob("epoch_*.pt")):
        epoch = parse_epoch_checkpoint(path=path)
        if epoch is None:
            continue
        checkpoints.append((epoch, path))

    if not checkpoints:
        raise SystemExit(f"no epoch_*.pt checkpoints found in {checkpoint_dir}")

    selected = [
        path
        for epoch, path in checkpoints
        if epoch >= start_epoch
        and (end_epoch is None or epoch <= end_epoch)
        and (epoch - start_epoch) % epoch_step == 0
    ]
    if not selected:
        available = ", ".join(str(epoch) for epoch, _ in checkpoints)
        end_label = "none" if end_epoch is None else str(end_epoch)
        raise SystemExit(
            f"no checkpoints matched start={start_epoch}, end={end_label}, step={epoch_step} "
            f"in {checkpoint_dir}; available epochs: {available}"
        )
    return selected


def parse_rollout_overrides(*, rollout_args: list[str]) -> tuple[bool, int]:
    extra_args = list(rollout_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    terminate_on_success = True
    video_skip = 5
    index = 0
    while index < len(extra_args):
        arg = extra_args[index]
        if arg in {"--terminate-on-success", "--no-terminate-on-success"}:
            terminate_on_success = arg == "--terminate-on-success"
            index += 1
            continue
        if arg == "--video-skip" and index + 1 < len(extra_args):
            video_skip = int(extra_args[index + 1])
            index += 2
            continue
        if arg.startswith("--video"):
            raise SystemExit(
                "rollout_sweep does not support --video; run scripts/rollout.py directly for video capture"
            )
        raise SystemExit(f"unsupported rollout argument: {arg!r}")

    return terminate_on_success, video_skip


def evaluate_checkpoint(
    *,
    checkpoint: Path,
    env,
    device: torch.device,
    n_rollouts: int,
    horizon: int,
    terminate_on_success: bool,
    on_screen: bool,
    video_skip: int,
    progress: tqdm,
) -> dict[str, float | int]:
    epoch = parse_epoch_checkpoint(path=checkpoint)
    epoch_label = str(epoch) if epoch is not None else checkpoint.stem
    model = load_model(checkpoint_path=checkpoint, device=device)
    rollouts: list[dict[str, float | int | bool]] = []
    try:
        for rollout_idx in range(1, n_rollouts + 1):
            window_title = f"epoch {epoch_label} | rollout {rollout_idx}/{n_rollouts}"
            progress.set_postfix(
                checkpoint=checkpoint.name,
                rollout=f"{rollout_idx}/{n_rollouts}",
                refresh=False,
            )
            rollout_stats = run_rollout(
                model=model,
                env=env,
                device=device,
                horizon=horizon,
                terminate_on_success=terminate_on_success,
                render=on_screen,
                video_writer=None,
                video_skip=video_skip,
                window_title=window_title if on_screen else None,
            )
            rollouts.append(rollout_stats)
            progress.set_postfix(
                checkpoint=checkpoint.name,
                rollout=f"{rollout_idx}/{n_rollouts}",
                success=str(rollout_stats["success"]),
                refresh=True,
            )
            progress.update(1)
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return summarize_rollouts(rollouts=rollouts)


def main() -> None:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir.resolve()
    if not checkpoint_dir.is_dir():
        raise SystemExit(f"checkpoint directory not found: {checkpoint_dir}")
    if args.end_epoch is not None and args.start_epoch > args.end_epoch:
        raise SystemExit(
            f"--start-epoch ({args.start_epoch}) must be <= --end-epoch ({args.end_epoch})"
        )

    checkpoints = select_checkpoints(
        checkpoint_dir=checkpoint_dir,
        start_epoch=args.start_epoch,
        end_epoch=args.end_epoch,
        epoch_step=args.epoch_step,
    )
    terminate_on_success, video_skip = parse_rollout_overrides(rollout_args=args.rollout_args)
    horizon = args.horizon if args.horizon is not None else DEFAULT_HORIZON
    dataset = args.dataset if args.dataset is not None else DEFAULT_DATASET
    on_screen = args.on_screen

    configure_renderer(on_screen=on_screen)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_meta = make_rollout_env_meta(
        dataset_path=dataset,
        camera_height=IMG_DIMS[0],
        camera_width=IMG_DIMS[1],
    )
    env = create_rollout_env(env_meta=env_meta, on_screen=on_screen, write_video=False)

    total_rollouts = len(checkpoints) * args.n_rollouts
    tqdm.write(
        f"sweep: {len(checkpoints)} checkpoints x {args.n_rollouts} rollouts "
        f"= {total_rollouts} episodes from {checkpoint_dir.name}"
    )
    tqdm.write("checkpoints: " + ", ".join(path.name for path in checkpoints))

    results: list[dict[str, float | int | str]] = []
    failures: list[str] = []
    progress = tqdm(total=total_rollouts, unit="rollout", dynamic_ncols=True)
    try:
        for checkpoint in checkpoints:
            progress.set_description(f"ckpt {checkpoint.name}")
            try:
                summary = evaluate_checkpoint(
                    checkpoint=checkpoint,
                    env=env,
                    device=device,
                    n_rollouts=args.n_rollouts,
                    horizon=horizon,
                    terminate_on_success=terminate_on_success,
                    on_screen=on_screen,
                    video_skip=video_skip,
                    progress=progress,
                )
            except Exception as exc:
                failures.append(checkpoint.name)
                tqdm.write(f"{checkpoint.name}: failed ({exc})")
                if not args.continue_on_error:
                    raise
                continue

            record = {"checkpoint": checkpoint.name, **summary}
            results.append(record)
            tqdm.write(
                f"{checkpoint.name}: "
                f"success={record['num_success']}/{record['num_rollouts']} "
                f"rate={record['success_rate']:.3f} "
                f"return={record['return_mean']:.3f} "
                f"horizon={record['horizon_mean']:.1f}"
            )
    finally:
        progress.close()
        close_env(env)

    if results:
        tqdm.write("sweep summary")
        tqdm.write(json.dumps(results, indent=2))

    if failures:
        failed = ", ".join(failures)
        raise SystemExit(f"rollout failed for: {failed}")


if __name__ == "__main__":
    main()
