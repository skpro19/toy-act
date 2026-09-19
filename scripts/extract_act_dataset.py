"""Extract a slim ACT training HDF5 from robomimic CAN PH low-dim data.

Renders one camera view from each recorded simulator state and keeps only the
keys needed for ACT training, plus ``states`` for sim replay.

Examples:
    # quick sanity check on 3 demos
    uv run python scripts/extract_act_dataset.py --n 3

    # full CAN PH extraction
    uv run python scripts/extract_act_dataset.py

    # custom paths and resolution
    uv run python scripts/extract_act_dataset.py \\
        --input datasets/can/ph/low_dim_v15.hdf5 \\
        --output datasets/can/ph/act_agentview.hdf5 \\
        --camera-height 480 --camera-width 640
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils

REPO_ROOT = Path(__file__).resolve().parent.parent
EXTRACTION_SCRIPT = (
    REPO_ROOT / "third_party" / "robomimic" / "robomimic" / "scripts" / "dataset_states_to_obs.py"
)
DEFAULT_INPUT = REPO_ROOT / "datasets" / "can" / "ph" / "low_dim_v15.hdf5"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "datasets" / "can" / "ph"
DEFAULT_OUTPUT_NAME = "act_agentview.hdf5"
DEFAULT_CAMERA = "agentview"

OBS_KEYS = (
    "agentview_image",
    "robot0_joint_pos",
    "robot0_gripper_qpos",
)


def default_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return DEFAULT_OUTPUT_DIR / f"{timestamp}_{DEFAULT_OUTPUT_NAME}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="path to robomimic low-dim CAN PH hdf5",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output_path(),
        help="path for the slim ACT training hdf5 (defaults to DATE_TIME_act_agentview.hdf5)",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default=DEFAULT_CAMERA,
        help="single camera name to render",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=84,
        help="rendered image height",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=84,
        help="rendered image width",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="process only the first n demos (useful for debugging)",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="gzip-compress image datasets in the output hdf5",
    )
    return parser.parse_args()


def configure_renderer() -> None:
    os.environ["MUJOCO_GL"] = "egl"


def load_extraction_module():
    spec = importlib.util.spec_from_file_location("robomimic_dataset_states_to_obs", EXTRACTION_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sorted_demo_names(*, dataset: h5py.File) -> list[str]:
    demos = list(dataset["data"].keys())
    demo_indices = np.argsort([int(name[5:]) for name in demos])
    return [demos[index] for index in demo_indices]


def create_dataset(
    *,
    group: h5py.Group,
    name: str,
    data: np.ndarray,
    compress: bool,
) -> None:
    if compress and name.endswith("_image"):
        group.create_dataset(name, data=data, compression="gzip")
        return
    group.create_dataset(name, data=data)


def write_obs_group(
    *,
    group: h5py.Group,
    prefix: str,
    obs_dict: dict[str, np.ndarray],
    compress: bool,
) -> None:
    obs_group = group.require_group(prefix)
    for key in OBS_KEYS:
        create_dataset(
            group=obs_group,
            name=key,
            data=np.array(obs_dict[key]),
            compress=compress,
        )


def write_demo(
    *,
    output_group: h5py.Group,
    demo_name: str,
    traj: dict,
    camera_info: dict | None,
    source_demo: h5py.Group,
    compress: bool,
) -> int:
    ep_group = output_group.create_group(demo_name)
    ep_group.create_dataset("states", data=np.array(traj["states"]))
    write_obs_group(group=ep_group, prefix="obs", obs_dict=traj["obs"], compress=compress)
    write_obs_group(group=ep_group, prefix="next_obs", obs_dict=traj["next_obs"], compress=compress)

    if "model" in traj["initial_state_dict"]:
        ep_group.attrs["model_file"] = traj["initial_state_dict"]["model"]
    if "ep_meta" in source_demo.attrs:
        ep_group.attrs["ep_meta"] = source_demo.attrs["ep_meta"]
    if camera_info is not None:
        ep_group.attrs["camera_info"] = json.dumps(camera_info, indent=4)

    num_samples = int(traj["states"].shape[0])
    ep_group.attrs["num_samples"] = num_samples
    return num_samples


def extract_act_dataset(
    *,
    input_path: Path,
    output_path: Path,
    camera: str,
    camera_height: int,
    camera_width: int,
    num_demos: int | None,
    compress: bool,
) -> None:
    extraction = load_extraction_module()

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=str(input_path))
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=[camera],
        camera_height=camera_height,
        camera_width=camera_width,
        reward_shaping=False,
    )
    is_robosuite_env = EnvUtils.is_robosuite_env(env_meta=env_meta)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    total_samples = 0
    with h5py.File(input_path, "r") as source_file, h5py.File(output_path, "w") as output_file:
        data_group = output_file.create_group("data")
        demo_names = sorted_demo_names(dataset=source_file)
        if num_demos is not None:
            demo_names = demo_names[:num_demos]

        for demo_name in tqdm(demo_names, desc="extracting demos"):
            source_demo = source_file[f"data/{demo_name}"]
            states = source_demo["states"][()]
            initial_state = {"states": states[0]}
            if is_robosuite_env:
                initial_state["model"] = source_demo.attrs["model_file"]
                initial_state["ep_meta"] = source_demo.attrs.get("ep_meta", None)

            traj, camera_info = extraction.extract_trajectory(
                env=env,
                initial_state=initial_state,
                states=states,
                actions=source_demo["actions"][()],
                actions_abs=None,
                done_mode=2,
                camera_names=[camera],
                camera_height=camera_height,
                camera_width=camera_width,
            )
            total_samples += write_demo(
                output_group=data_group,
                demo_name=demo_name,
                traj=traj,
                camera_info=camera_info,
                source_demo=source_demo,
                compress=compress,
            )

        if "mask" in source_file:
            source_file.copy("mask", output_file)

        data_group.attrs["total"] = total_samples
        data_group.attrs["env_args"] = json.dumps(env.serialize(), indent=4)

    print(f"wrote {len(demo_names)} demos ({total_samples} samples) to {output_path}")


def main() -> None:
    args = parse_args()
    configure_renderer()
    extract_act_dataset(
        input_path=args.input,
        output_path=args.output,
        camera=args.camera,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        num_demos=args.n,
        compress=args.compress,
    )


if __name__ == "__main__":
    main()
