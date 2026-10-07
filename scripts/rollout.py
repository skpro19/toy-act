"""Evaluate a trained ACT v2 checkpoint in PickPlaceCan.

Examples:
    uv run python scripts/rollout.py \\
        --checkpoint checkpoints/act_v2/.../step_000002000.pt \\
        --n-rollouts 20

    uv run python scripts/rollout.py \\
        --checkpoint checkpoints/act_v2/.../step_000002000.pt \\
        --no-on-screen --n-rollouts 20

    uv run python scripts/rollout.py \\
        --checkpoint checkpoints/act_v2/.../step_000002000.pt \\
        --video replays/rollout.mp4 --n-rollouts 5 --seed 0
"""

from __future__ import annotations

import argparse
import random
import copy
import json
import os
import time
from pathlib import Path

import imageio
import numpy as np
import torch

from scripts.dataset import NormalizationStats, build_proprio, image_to_tensor
from scripts.models.act_v2.config import (
    ACTION_CHUNK_SIZE,
    D_MODEL,
    IMG_DIMS,
    JOINT_DIMS,
    N_HEAD,
    NUM_LAYERS,
    PROPRIO_DIMS,
    Z_DIMS,
)
from scripts.models.act_v2.model import ACTV2

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = REPO_ROOT / "datasets" / "can" / "ph" / "low_dim_v15.hdf5"
DEFAULT_HORIZON = 400
DEFAULT_CAMERA = "agentview"
CHECKPOINT_CONFIG_VERSIONS = ("v3", "v4")
CHECKPOINT_CONFIG_KEYS = (
    "action_loss",
    "batch_size",
    "steps",
    "image_keys",
    "lr",
    "seed",
    "beta",
    "checkpoint_every",
    "version",
    "beta_start",
    "beta_warmup_steps",
    "use_z",
)
ROBOSUITE_CAMERAS = (
    "agentview",
    "frontview",
    "birdview",
    "robot0_robotview",
    "robot0_eye_in_hand",
)
GRIPPER_APERTURE_THRESHOLD = 0.002
# robosuite Panda GRIP convention: -1 opens the fingers, +1 closes them.
GRIPPER_OPEN_COMMAND = -1.0
GRIPPER_CLOSE_COMMAND = 1.0
OPENCV_RENDER_WINDOW = "offscreen render"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="path to ACT v2 checkpoint (.pt) saved by scripts/train_v2.py",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="robomimic low-dim hdf5 used to recreate PickPlaceCan",
    )
    parser.add_argument("--n-action-steps", type=int, nargs="+",
                        help="execution lengths to evaluate; overrides checkpoint rollout configuration")
    parser.add_argument("--n-rollouts", type=int, default=20, help="number of evaluation episodes")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON, help="max steps per episode")
    parser.add_argument("--seed", type=int, default=0, help="random seed for env resets")
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
        help="render live in the MuJoCo viewer",
    )
    parser.add_argument("--video", type=Path, help="write rollout video to this mp4 path")
    parser.add_argument("--video-skip", type=int, default=5, help="record every Nth env step to video")
    return parser.parse_args()


def configure_renderer(*, on_screen: bool) -> None:
    os.environ["MUJOCO_GL"] = "glfw" if on_screen else "egl"


def initialize_obs_modalities() -> None:
    import robomimic.utils.obs_utils as ObsUtils

    ObsUtils.initialize_obs_modality_mapping_from_dict(
        {
            "rgb": [f"{camera}_image" for camera in ROBOSUITE_CAMERAS],
            "low_dim": [
                "robot0_joint_pos",
                "robot0_joint_vel",
                "robot0_joint_pos_cos",
                "robot0_joint_pos_sin",
                "robot0_gripper_qpos",
                "robot0_gripper_qvel",
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_eef_quat_site",
                "robot0_proprio-state",
                "object-state",
                "Can_pos",
                "Can_quat",
                "Can_to_robot0_eef_pos",
                "Can_to_robot0_eef_quat",
            ],
        }
    )


def make_rollout_controller_config() -> dict:
    from robosuite.controllers import load_composite_controller_config

    controller_config = load_composite_controller_config(controller="BASIC", robot="Panda")
    controller_config = copy.deepcopy(controller_config)
    controller_config["body_parts"]["right"] = {
        "type": "JOINT_POSITION",
        "input_type": "absolute",
        "input_max": 1,
        "input_min": -1,
        "output_max": 0.05,
        "output_min": -0.05,
        "kp": 50,
        "damping_ratio": 1,
        "impedance_mode": "fixed",
        "kp_limits": [0, 300],
        "damping_ratio_limits": [0, 10],
        "qpos_limits": None,
        "interpolation": None,
        "ramp_ratio": 0.2,
        "gripper": {"type": "GRIP"},
    }
    return controller_config


def camera_names_from_image_keys(*, image_keys: tuple[str, ...]) -> tuple[str, ...]:
    """Map checkpoint image keys (e.g. ``agentview_image``) to robosuite camera names."""
    suffix = "_image"
    camera_names = []
    for key in image_keys:
        if not key.endswith(suffix):
            raise ValueError(
                f"image key {key!r} does not end with {suffix!r}; cannot map it to a camera"
            )
        camera_names.append(key[: -len(suffix)])

    unknown = [name for name in camera_names if name not in ROBOSUITE_CAMERAS]
    if unknown:
        raise ValueError(
            f"image keys map to unknown robosuite cameras {unknown}; "
            f"supported cameras are {list(ROBOSUITE_CAMERAS)}"
        )
    return tuple(camera_names)


def make_rollout_env_meta(
    *,
    dataset_path: Path,
    camera_names: tuple[str, ...],
    camera_height: int,
    camera_width: int,
) -> dict:
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=str(dataset_path))
    env_meta = copy.deepcopy(env_meta)
    env_kwargs = env_meta["env_kwargs"]
    env_kwargs["controller_configs"] = make_rollout_controller_config()
    env_kwargs["use_camera_obs"] = True
    env_kwargs["camera_names"] = list(camera_names)
    env_kwargs["camera_heights"] = camera_height
    env_kwargs["camera_widths"] = camera_width
    initialize_obs_modalities()
    EnvUtils.set_env_specific_obs_processing(env_meta=env_meta)
    return env_meta


def create_rollout_env(
    *,
    env_meta: dict,
    on_screen: bool,
    write_video: bool,
):
    import robomimic.utils.env_utils as EnvUtils

    return EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=on_screen,
        render_offscreen=write_video or (not on_screen),
        use_image_obs=True,
    )


def load_checkpoint(*, checkpoint_path: Path, device: torch.device) -> dict:
    return torch.load(checkpoint_path, map_location=device, weights_only=False)


def training_config_from_checkpoint(*, checkpoint: dict) -> dict:
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise KeyError(
            "checkpoint is missing 'config'; use a checkpoint saved by scripts/train_v2.py"
        )

    missing = [key for key in CHECKPOINT_CONFIG_KEYS if key not in config]
    if missing:
        raise KeyError(f"checkpoint config missing required keys: {missing}")

    if config["version"] not in CHECKPOINT_CONFIG_VERSIONS:
        allowed = ", ".join(repr(version) for version in CHECKPOINT_CONFIG_VERSIONS)
        raise ValueError(
            f"checkpoint config version must be one of {{{allowed}}}, "
            f"got {config['version']!r}"
        )

    image_keys = config["image_keys"]
    if not isinstance(image_keys, list) or len(image_keys) == 0:
        raise ValueError("checkpoint config 'image_keys' must be a non-empty list")
    if not all(isinstance(key, str) and key for key in image_keys):
        raise ValueError("checkpoint config 'image_keys' must contain non-empty strings")

    if not isinstance(config["use_z"], bool):
        raise ValueError(
            f"checkpoint config 'use_z' must be a bool, got {type(config['use_z']).__name__}"
        )

    return config


def rollout_settings_from_checkpoint(
    *,
    checkpoint: dict) -> tuple[bool, tuple[str, ...], int]:
    config = training_config_from_checkpoint(checkpoint=checkpoint)
    use_z = config["use_z"]
    image_keys = tuple(config["image_keys"])
    action_chunk_size = config.get("action_chunk_size", ACTION_CHUNK_SIZE)
    if (
        not isinstance(action_chunk_size, int)
        or isinstance(action_chunk_size, bool)
        or action_chunk_size <= 0
    ):
        raise ValueError(
            f"checkpoint config 'action_chunk_size' must be > 0, got {action_chunk_size!r}"
        )
    return use_z, image_keys, action_chunk_size


def normalization_from_checkpoint(*, checkpoint: dict) -> NormalizationStats:
    if "normalization" not in checkpoint:
        raise KeyError(
            "checkpoint is missing 'normalization' stats; use a checkpoint saved by "
            "scripts/train_v2.py"
        )
    stats = checkpoint["normalization"]
    return NormalizationStats(
        proprio_mean=np.asarray(stats["proprio_mean"], dtype=np.float32),
        proprio_std=np.asarray(stats["proprio_std"], dtype=np.float32),
        action_mean=np.asarray(stats["action_mean"], dtype=np.float32),
        action_std=np.asarray(stats["action_std"], dtype=np.float32),
    )


def load_model(
    *,
    checkpoint: dict,
    device: torch.device,
    use_z: bool,
    action_chunk_size: int = ACTION_CHUNK_SIZE,
) -> tuple[ACTV2, NormalizationStats]:
    model = ACTV2(
        d_model=D_MODEL,
        nhead=N_HEAD,
        num_layers=NUM_LAYERS,
        z_dims=Z_DIMS,
        action_chunk_size=action_chunk_size,
        proprio_dims=PROPRIO_DIMS,
        use_z=use_z,
    )
    if "model" not in checkpoint:
        raise KeyError("checkpoint is missing 'model' state dict")
    model.load_state_dict(checkpoint["model"])
    model.to(device=device)
    model.eval()
    normalization = normalization_from_checkpoint(checkpoint=checkpoint)
    return model, normalization


def obs_image_frame_to_tensor(*, image: np.ndarray, device: torch.device) -> torch.Tensor:
    if image.dtype == np.uint8:
        tensor = image_to_tensor(image)
    else:
        array = image
        if array.max() <= 1.0:
            array = (array * 255.0).astype(np.uint8)
        else:
            array = array.astype(np.uint8)
        tensor = image_to_tensor(array)
    return tensor.to(device=device)


def obs_images_to_tensor(
    *,
    obs: dict,
    image_keys: tuple[str, ...],
    device: torch.device) -> torch.Tensor:
    frames = []
    for key in image_keys:
        if key not in obs:
            raise KeyError(f"expected observation key {key!r}, got {sorted(obs.keys())}")
        frames.append(obs_image_frame_to_tensor(image=obs[key], device=device))
    return torch.stack(frames, dim=0)


def obs_to_proprio_tensor(
    *,
    obs: dict,
    device: torch.device,
    normalization: NormalizationStats,
) -> torch.Tensor:
    proprio = build_proprio(
        joint_pos=np.asarray(obs["robot0_joint_pos"], dtype=np.float32),
        gripper_qpos=np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
    )
    proprio = normalization.normalize_proprio(value=proprio)
    return torch.from_numpy(proprio).float().reshape(1, 1, PROPRIO_DIMS).to(device=device)


def current_gripper_aperture(*, obs: dict) -> float:
    gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)
    return float(gripper_qpos[0] - gripper_qpos[1])


def gripper_command(*, target_aperture: float, current_aperture: float) -> float:
    delta = target_aperture - current_aperture
    if delta > GRIPPER_APERTURE_THRESHOLD:
        return GRIPPER_OPEN_COMMAND
    if delta < -GRIPPER_APERTURE_THRESHOLD:
        return GRIPPER_CLOSE_COMMAND
    return 0.0


def model_action_to_sim(*, model_action: np.ndarray, obs: dict) -> np.ndarray:
    sim_action = np.zeros(JOINT_DIMS + 1, dtype=np.float32)
    sim_action[:JOINT_DIMS] = model_action[:JOINT_DIMS]
    sim_action[JOINT_DIMS] = gripper_command(
        target_aperture=float(model_action[JOINT_DIMS]),
        current_aperture=current_gripper_aperture(obs=obs),
    )
    return sim_action


def denormalize_action(
    *,
    action: np.ndarray,
    normalization: NormalizationStats,
) -> np.ndarray:
    return (action * normalization.action_std + normalization.action_mean).astype(np.float32)


def predict_action_chunk(
    *,
    model: ACTV2,
    obs: dict,
    device: torch.device,
    normalization: NormalizationStats,
    image_keys: tuple[str, ...],
) -> np.ndarray:
    images = obs_images_to_tensor(obs=obs, image_keys=image_keys, device=device)
    img_tensor = images.unsqueeze(0)
    proprio_tensor = obs_to_proprio_tensor(
        obs=obs,
        device=device,
        normalization=normalization,
    )
    with torch.no_grad():
        pred = model.infer(proprio=proprio_tensor, img=img_tensor)
    return denormalize_action(action=pred[0].detach().cpu().numpy(), normalization=normalization)


def set_render_window_title(*, env, title: str | None) -> None:
    if title is None:
        return

    import cv2

    robosuite_env = env.env if hasattr(env, "env") else env
    viewer = getattr(robosuite_env, "viewer", None)
    if viewer is None:
        return

    if viewer.__class__.__name__ == "OpenCVRenderer":
        cv2.setWindowTitle(OPENCV_RENDER_WINDOW, title)
        return

    if viewer.__class__.__name__ == "MjviewerRenderer":
        handle = getattr(viewer, "viewer", None)
        if handle is None:
            return
        simulate = handle._get_sim() if hasattr(handle, "_get_sim") else None
        if simulate is not None and hasattr(simulate, "filename"):
            simulate.filename = title


def validate_n_action_steps(*, values: list[int], action_chunk_size: int) -> list[int]:
    """Require explicit, unique execution lengths within the predicted chunk."""
    if not isinstance(values, list) or not values:
        raise ValueError("n_action_steps must be a nonempty list")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= action_chunk_size:
            raise ValueError(f"n_action_steps must contain integers in 1..{action_chunk_size}, got {value!r}")
    if len(set(values)) != len(values):
        raise ValueError("n_action_steps must not contain duplicates")
    return list(values)


def run_rollout(
    *,
    model: ACTV2,
    env,
    device: torch.device,
    normalization: NormalizationStats,
    image_keys: tuple[str, ...],
    horizon: int,
    terminate_on_success: bool,
    render: bool,
    video_writer,
    video_skip: int,
    n_action_steps: int,
    action_chunk_size: int = ACTION_CHUNK_SIZE,
    time_limit_s: float | None = None,
    window_title: str | None = None,) -> dict[str, float | int | bool]:
    validate_n_action_steps(values=[n_action_steps], action_chunk_size=action_chunk_size)
    start = time.monotonic()
    obs = env.reset()
    total_reward = 0.0
    success = False
    action_chunk = None
    chunk_step = 0
    video_count = 0

    for step_idx in range(horizon):
        if action_chunk is None or chunk_step >= n_action_steps:
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

        step_success = bool(info["is_success"]["task"])
        success = success or step_success

        if render:
            env.render(mode="human", camera_name=DEFAULT_CAMERA)
            set_render_window_title(env=env, title=window_title)
        if video_writer is not None and video_count % video_skip == 0:
            frame = env.render(
                mode="rgb_array",
                height=IMG_DIMS[0],
                width=IMG_DIMS[1],
                camera_name=DEFAULT_CAMERA,
            )
            video_writer.append_data(frame)
        video_count += 1

        if done or (terminate_on_success and success):
            return {
                "success": success,
                "return": total_reward,
                "horizon": step_idx + 1,
                "truncated": False,
            }
        if time_limit_s is not None and time.monotonic() - start >= time_limit_s:
            return {
                "success": success,
                "return": total_reward,
                "horizon": step_idx + 1,
                "truncated": True,
            }

    return {
        "success": success,
        "return": total_reward,
        "horizon": horizon,
        "truncated": False,
    }


def close_env(env) -> None:
    if hasattr(env, "close"):
        env.close()
        return
    if hasattr(env, "env") and hasattr(env.env, "close"):
        env.env.close()


def summarize_rollouts(*, rollouts: list[dict[str, float | int | bool]]) -> dict[str, float | int]:
    success_flags = [bool(rollout["success"]) for rollout in rollouts]
    returns = [float(rollout["return"]) for rollout in rollouts]
    horizons = [int(rollout["horizon"]) for rollout in rollouts]
    truncated_flags = [bool(rollout["truncated"]) for rollout in rollouts]
    return {
        "num_rollouts": len(rollouts),
        "num_success": int(sum(success_flags)),
        "success_rate": float(np.mean(success_flags)),
        "return_mean": float(np.mean(returns)),
        "horizon_mean": float(np.mean(horizons)),
        "num_truncated": int(sum(truncated_flags)),
    }


def main() -> None:
    args = parse_args()
    write_video = args.video is not None
    on_screen = args.on_screen and not write_video
    configure_renderer(on_screen=on_screen)

    if write_video:
        args.video.parent.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = load_checkpoint(checkpoint_path=args.checkpoint, device=device)
    use_z, image_keys, action_chunk_size = rollout_settings_from_checkpoint(checkpoint=checkpoint)
    execution_lengths = args.n_action_steps
    if execution_lengths is None:
        execution_lengths = checkpoint.get("config", {}).get("rollout", {}).get("n_action_steps")
    execution_lengths = validate_n_action_steps(
        values=execution_lengths, action_chunk_size=action_chunk_size)
    model, normalization = load_model(
        checkpoint=checkpoint,
        device=device,
        use_z=use_z,
        action_chunk_size=action_chunk_size,
    )
    camera_names = camera_names_from_image_keys(image_keys=image_keys)

    run_name = checkpoint.get("run_name")
    if run_name is not None:
        print(f"run_name => {run_name}")
    print(f"use_z => {use_z}")
    print(f"action_chunk_size => {action_chunk_size}")
    print(f"image_keys => {list(image_keys)}")
    print(f"camera_names => {list(camera_names)}")

    env_meta = make_rollout_env_meta(
        dataset_path=args.dataset,
        camera_names=camera_names,
        camera_height=IMG_DIMS[0],
        camera_width=IMG_DIMS[1],
    )
    env = create_rollout_env(
        env_meta=env_meta,
        on_screen=on_screen,
        write_video=write_video,
    )

    video_writer = imageio.get_writer(args.video, fps=20) if write_video else None
    print(f"n_action_steps => {execution_lengths}")
    summaries = {}
    try:
        for execution_length in execution_lengths:
            rollouts = []
            for episode_idx in range(args.n_rollouts):
                np.random.seed(args.seed + episode_idx)
                random.seed(args.seed + episode_idx)
                torch.manual_seed(args.seed + episode_idx)
                rollout_stats = run_rollout(
                    model=model, env=env, device=device, normalization=normalization,
                    image_keys=image_keys, action_chunk_size=action_chunk_size,
                    n_action_steps=execution_length, horizon=args.horizon,
                    terminate_on_success=args.terminate_on_success, render=on_screen,
                    video_writer=video_writer, video_skip=args.video_skip,
                )
                rollouts.append(rollout_stats)
                print(f"n_action_steps={execution_length} episode {episode_idx + 1}/{args.n_rollouts}: {rollout_stats}")
            summaries[str(execution_length)] = summarize_rollouts(rollouts=rollouts)
    finally:
        if video_writer is not None:
            video_writer.close()
        close_env(env)

    print("rollout summaries by n_action_steps")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
