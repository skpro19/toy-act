"""Record two-camera rollout episodes for a set of runs to per-run videos.

Reads the manifest written by ``scripts/utils/find_best_checkpoints.py``, loads
each run's best checkpoint from ``checkpoints/<version>/<run-name>/``, and runs
``--episodes`` rollouts of ``--horizon`` steps in ``PickPlaceCan``. Every
``--frame-skip`` env steps it renders each camera in the checkpoint's
``image_keys`` and writes the frames side by side into
``<out-dir>/<run-name>.mp4``. The per-run success score and episode lengths are
written to ``<out-dir>/recordings.json`` for ``build_rollout_gifs.py`` to label.

The off-screen (EGL) renderer is used, so this runs headless.

Examples:
    uv run python scripts/utils/record_rollout_episodes.py \\
        --manifest assets/rollout-two-camera/best_checkpoints.json \\
        --out-dir assets/rollout-two-camera/_recordings
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio
import numpy as np
import torch

from scripts.models.act_v2.config import ACTION_CHUNK_SIZE, IMG_DIMS
from scripts.rollout import (
    camera_names_from_image_keys,
    close_env,
    configure_renderer,
    create_rollout_env,
    load_checkpoint,
    load_model,
    make_rollout_env_meta,
    model_action_to_sim,
    predict_action_chunk,
    rollout_settings_from_checkpoint,
    summarize_rollouts,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATASET = (
    REPO_ROOT / "datasets" / "can" / "ph"
    / "2026-09-29_02-01-42_agentview_robot0_eye_in_hand.hdf5"
)
DEFAULT_CHECKPOINTS_ROOT = REPO_ROOT / "checkpoints" / "act_v2"
RECORDINGS_FILENAME = "recordings.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="best-checkpoint manifest from find_best_checkpoints.py",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="directory for the per-run mp4 recordings and recordings.json",
    )
    parser.add_argument(
        "--checkpoints-root",
        type=Path,
        default=DEFAULT_CHECKPOINTS_ROOT,
        help="root holding <run-name>/<checkpoint> files",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="robomimic hdf5 used to recreate the rollout environment",
    )
    parser.add_argument("--episodes", type=int, default=10, help="episodes per run")
    parser.add_argument("--horizon", type=int, default=200, help="max steps per episode")
    parser.add_argument("--seed", type=int, default=0, help="base seed for env resets")
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="record every Nth env step (1 records every step)",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=256,
        help="render height for the recorded frames (model input stays 84x84)",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=256,
        help="render width for the recorded frames (model input stays 84x84)",
    )
    parser.add_argument("--fps", type=int, default=20, help="fps metadata for the recordings")
    parser.add_argument(
        "--run-name",
        action="append",
        default=None,
        help="restrict to these run names (repeatable; default: all in the manifest)",
    )
    return parser.parse_args()


def checkpoint_path_for(
    *,
    checkpoints_root: Path,
    run_name: str,
    checkpoint: str) -> Path:
    return checkpoints_root / run_name / checkpoint


def render_camera_row(
    *,
    env,
    camera_names: tuple[str, ...],
    render_height: int,
    render_width: int) -> np.ndarray:
    frames = [
        env.render(
            mode="rgb_array",
            height=render_height,
            width=render_width,
            camera_name=camera_name,
        )
        for camera_name in camera_names
    ]
    return np.concatenate([np.asarray(frame, dtype=np.uint8) for frame in frames], axis=1)


def record_episode(
    *,
    model,
    env,
    device: torch.device,
    normalization,
    image_keys: tuple[str, ...],
    camera_names: tuple[str, ...],
    horizon: int,
    terminate_on_success: bool,
    frame_skip: int,
    render_height: int,
    render_width: int,
    video_writer) -> dict[str, float | int | bool]:
    obs = env.reset()
    total_reward = 0.0
    success = False
    action_chunk = None
    chunk_step = 0

    for step_index in range(horizon):
        if action_chunk is None or chunk_step >= ACTION_CHUNK_SIZE:
            action_chunk = predict_action_chunk(
                model=model,
                obs=obs,
                device=device,
                normalization=normalization,
                image_keys=image_keys,
            )
            chunk_step = 0

        sim_action = model_action_to_sim(model_action=action_chunk[chunk_step], obs=obs)
        obs, reward, done, info = env.step(sim_action)
        total_reward += float(reward)
        chunk_step += 1
        success = success or bool(info["is_success"]["task"])

        if video_writer is not None and step_index % frame_skip == 0:
            video_writer.append_data(
                render_camera_row(
                    env=env,
                    camera_names=camera_names,
                    render_height=render_height,
                    render_width=render_width,
                )
            )

        if done or (terminate_on_success and success):
            return {
                "success": success,
                "return": total_reward,
                "horizon": step_index + 1,
                "truncated": False,
            }

    return {
        "success": success,
        "return": total_reward,
        "horizon": horizon,
        "truncated": False,
    }


def record_run(
    *,
    run: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    video_path: Path) -> dict[str, Any]:
    run_name = run["run_name"]
    checkpoint = checkpoint_path_for(
        checkpoints_root=args.checkpoints_root,
        run_name=run_name,
        checkpoint=run["checkpoint"],
    )
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")

    print(f"[{run_name}] checkpoint {checkpoint.name}")
    checkpoint_data = load_checkpoint(checkpoint_path=checkpoint, device=device)
    use_z, image_keys = rollout_settings_from_checkpoint(checkpoint=checkpoint_data)
    model, normalization = load_model(
        checkpoint=checkpoint_data,
        device=device,
        use_z=use_z,
    )
    camera_names = camera_names_from_image_keys(image_keys=image_keys)

    env_meta = make_rollout_env_meta(
        dataset_path=args.dataset,
        camera_names=camera_names,
        camera_height=IMG_DIMS[0],
        camera_width=IMG_DIMS[1],
    )
    env = create_rollout_env(
        env_meta=env_meta,
        on_screen=False,
        write_video=True,
    )
    writer = imageio.get_writer(
        str(video_path),
        fps=args.fps,
        codec="libx264",
        output_params=["-crf", "0", "-pix_fmt", "yuv444p"],
        macro_block_size=1,
    )
    rollouts: list[dict[str, float | int | bool]] = []
    try:
        for episode_index in range(args.episodes):
            np.random.seed(args.seed + episode_index)
            stats = record_episode(
                model=model,
                env=env,
                device=device,
                normalization=normalization,
                image_keys=image_keys,
                camera_names=camera_names,
                horizon=args.horizon,
                terminate_on_success=True,
                frame_skip=args.frame_skip,
                render_height=args.camera_height,
                render_width=args.camera_width,
                video_writer=writer,
            )
            rollouts.append(stats)
            print(
                f"[{run_name}] episode {episode_index + 1}/{args.episodes}: "
                f"success={stats['success']} horizon={stats['horizon']}"
            )
    finally:
        writer.close()
        close_env(env)

    summary = summarize_rollouts(rollouts=rollouts)
    return {
        "run_name": run_name,
        "batch_size": run.get("batch_size"),
        "checkpoint": run["checkpoint"],
        "global_step": run["global_step"],
        "train_eval_success_rate": run.get("eval_success_rate"),
        "image_keys": list(image_keys),
        "camera_names": list(camera_names),
        "episodes": args.episodes,
        "horizon": args.horizon,
        "seed": args.seed,
        "frame_skip": args.frame_skip,
        "camera_height": args.camera_height,
        "camera_width": args.camera_width,
        "fps": args.fps,
        "recording": video_path.name,
        "episode_lengths": [int(stats["horizon"]) for stats in rollouts],
        "episode_success": [bool(stats["success"]) for stats in rollouts],
        **summary,
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.read_text())
    runs = manifest["runs"]
    if args.run_name is not None:
        wanted = set(args.run_name)
        runs = [run for run in runs if run["run_name"] in wanted]
    if not runs:
        raise SystemExit("no runs selected")

    configure_renderer(on_screen=False)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records: list[dict[str, Any]] = []
    for run in runs:
        video_path = args.out_dir / f"{run['run_name']}.mp4"
        records.append(
            record_run(
                run=run,
                args=args,
                device=device,
                video_path=video_path,
            )
        )

    recordings_path = args.out_dir / RECORDINGS_FILENAME
    recordings_path.write_text(json.dumps({"recordings": records}, indent=2) + "\n")
    print(f"wrote {recordings_path}")
    for record in records:
        print(
            f"{record['run_name']}: success={record['num_success']}/{record['episodes']} "
            f"rate={record['success_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
