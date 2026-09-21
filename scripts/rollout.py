"""Evaluate a trained ACTV1 checkpoint in PickPlaceCan.

Examples:
    uv run python scripts/rollout.py \\
        --checkpoint checkpoints/act_v1/.../epoch_010.pt \\
        --n-rollouts 20

    uv run python scripts/rollout.py \\
        --checkpoint checkpoints/act_v1/.../last.pt \\
        --no-on-screen --n-rollouts 20

    uv run python scripts/rollout.py \\
        --checkpoint checkpoints/act_v1/.../last.pt \\
        --video replays/rollout.mp4 --n-rollouts 5 --seed 0
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

import imageio
import numpy as np
import torch

from scripts.dataset import build_proprio, image_to_tensor
from scripts.models.act_v1 import ACTV1
from scripts.models.act_v1.config import (
    ACTION_CHUNK_SIZE,
    D_MODEL,
    IMG_DIMS,
    JOINT_DIMS,
    NUM_LAYERS,
    N_HEAD,
    PROPRIO_DIMS,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = REPO_ROOT / "datasets" / "can" / "ph" / "low_dim_v15.hdf5"
DEFAULT_HORIZON = 400
DEFAULT_CAMERA = "agentview"
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
        help="path to ACTV1 checkpoint (.pt)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="robomimic low-dim hdf5 used to recreate PickPlaceCan",
    )
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
            "rgb": ["agentview_image", "robot0_eye_in_hand_image"],
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


def make_rollout_env_meta(*, dataset_path: Path, camera_height: int, camera_width: int) -> dict:
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=str(dataset_path))
    env_meta = copy.deepcopy(env_meta)
    env_kwargs = env_meta["env_kwargs"]
    env_kwargs["controller_configs"] = make_rollout_controller_config()
    env_kwargs["use_camera_obs"] = True
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


def load_model(*, checkpoint_path: Path, device: torch.device) -> ACTV1:
    model = ACTV1(
        d_model=D_MODEL,
        nhead=N_HEAD,
        num_layers=NUM_LAYERS,
        action_chunk_size=ACTION_CHUNK_SIZE,
        proprio_dims=PROPRIO_DIMS,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device=device)
    model.eval()
    return model


def obs_image_to_tensor(*, image: np.ndarray, device: torch.device) -> torch.Tensor:
    if image.dtype == np.uint8:
        tensor = image_to_tensor(image)
    else:
        array = image
        if array.max() <= 1.0:
            array = (array * 255.0).astype(np.uint8)
        else:
            array = array.astype(np.uint8)
        tensor = image_to_tensor(array)
    return tensor.unsqueeze(0).to(device=device)


def obs_to_proprio_tensor(*, obs: dict, device: torch.device) -> torch.Tensor:
    proprio = build_proprio(
        joint_pos=np.asarray(obs["robot0_joint_pos"], dtype=np.float32),
        gripper_qpos=np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
    )
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


def predict_action_chunk(
    *,
    model: ACTV1,
    obs: dict,
    device: torch.device,
) -> np.ndarray:
    image_key = f"{DEFAULT_CAMERA}_image"
    if image_key not in obs:
        raise KeyError(f"expected observation key {image_key!r}, got {sorted(obs.keys())}")

    img_tensor = obs_image_to_tensor(image=obs[image_key], device=device)
    proprio_tensor = obs_to_proprio_tensor(obs=obs, device=device)
    with torch.no_grad():
        pred = model(img_tensor=img_tensor, proprio_tensor=proprio_tensor)
    return pred[0].detach().cpu().numpy()


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


def run_rollout(
    *,
    model: ACTV1,
    env,
    device: torch.device,
    horizon: int,
    terminate_on_success: bool,
    render: bool,
    video_writer,
    video_skip: int,
    time_limit_s: float | None = None,
    window_title: str | None = None,
) -> dict[str, float | int | bool]:
    start = time.monotonic()
    obs = env.reset()
    total_reward = 0.0
    success = False
    action_chunk = None
    chunk_step = 0
    video_count = 0

    for step_idx in range(horizon):
        if action_chunk is None or chunk_step >= ACTION_CHUNK_SIZE:
            action_chunk = predict_action_chunk(model=model, obs=obs, device=device)
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

    env_meta = make_rollout_env_meta(
        dataset_path=args.dataset,
        camera_height=IMG_DIMS[0],
        camera_width=IMG_DIMS[1],
    )
    env = create_rollout_env(
        env_meta=env_meta,
        on_screen=on_screen,
        write_video=write_video,
    )
    model = load_model(checkpoint_path=args.checkpoint, device=device)

    video_writer = imageio.get_writer(args.video, fps=20) if write_video else None
    rollouts: list[dict[str, float | int | bool]] = []
    try:
        for episode_idx in range(args.n_rollouts):
            rollout_stats = run_rollout(
                model=model,
                env=env,
                device=device,
                horizon=args.horizon,
                terminate_on_success=args.terminate_on_success,
                render=on_screen,
                video_writer=video_writer,
                video_skip=args.video_skip,
            )
            rollouts.append(rollout_stats)
            print(
                f"episode {episode_idx + 1}/{args.n_rollouts}: "
                f"success={rollout_stats['success']} "
                f"return={rollout_stats['return']:.3f} "
                f"horizon={rollout_stats['horizon']}"
            )
    finally:
        if video_writer is not None:
            video_writer.close()
        close_env(env)

    summary = summarize_rollouts(rollouts=rollouts)
    print("rollout summary")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
