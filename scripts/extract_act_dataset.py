"""Extract a slim ACT training HDF5 from robomimic CAN PH low-dim data.

Renders one or more camera views from each recorded simulator state and keeps
only the keys needed for ACT training, plus ``states`` for sim replay.

Examples:
    # quick sanity check on 3 demos
    uv run python scripts/extract_act_dataset.py --n 3

    # full CAN PH extraction (agentview only)
    uv run python scripts/extract_act_dataset.py

    # agentview plus the wrist camera
    uv run python scripts/extract_act_dataset.py --cameras agentview robot0_eye_in_hand

    # only the demos tagged "better" in the multi-human dataset
    uv run python scripts/extract_act_dataset.py \\
        --input datasets/can/mh/low_dim_v15.hdf5 \\
        --filter-key better \\
        --cameras agentview
    # custom paths and resolution
    uv run python scripts/extract_act_dataset.py \\
        --input datasets/can/ph/low_dim_v15.hdf5 \\
        --output datasets/can/ph/act_agentview.hdf5 \\
        --cameras agentview --camera-height 480 --camera-width 640
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
DEFAULT_CAMERAS = ("agentview",)

PROPRIO_KEYS = (
    "robot0_joint_pos",
    "robot0_gripper_qpos",
)


def obs_keys(*, cameras: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"{camera}_image" for camera in cameras) + PROPRIO_KEYS


def default_output_path(
    *, input_path: Path, cameras: list[str] | tuple[str, ...], filter_key: str | None
) -> Path:
    """Name the output after the input dataset's folder, keeping the same convention."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parts = [timestamp]
    if filter_key is not None:
        parts.append(filter_key)
    parts.append("_".join(cameras))
    return input_path.parent / f"{'_'.join(parts)}.hdf5"


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
        default=None,
        help="path for the slim ACT training hdf5 (defaults to DATE_TIME_<cam-list>.hdf5)",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="+",
        default=list(DEFAULT_CAMERAS),
        help="camera name(s) to render as image observations",
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
        "--filter-key",
        type=str,
        default=None,
        help=(
            "use only the demos listed under mask/<filter-key> in the input hdf5 "
            "(e.g. better, okay, worse, better_okay); defaults to all demos"
        ),
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


def filtered_demo_names(*, dataset: h5py.File, filter_key: str) -> list[str]:
    """Return the demos referenced by ``mask/<filter_key>``, in sorted order."""
    mask_path = f"mask/{filter_key}"
    if mask_path not in dataset:
        available = sorted(dataset["mask"].keys()) if "mask" in dataset else []
        raise KeyError(
            f"filter key {mask_path!r} not found in dataset (available: {available})"
        )
    tagged = {name.decode("utf-8") for name in dataset[mask_path][:]}
    return [name for name in sorted_demo_names(dataset=dataset) if name in tagged]


def copy_filter_keys(
    *,
    source_file: h5py.File,
    output_file: h5py.File,
    demo_names: set[str],
) -> None:
    """Copy the source ``mask`` group, keeping only filter keys whose demos all survived.

    A filter key that references a demo dropped during extraction (for example when
    ``--filter-key`` selected a subset) would point at missing groups, so it is skipped.
    """
    output_mask = output_file.create_group("mask")
    for key in source_file["mask"]:
        tagged = [name.decode("utf-8") for name in source_file[f"mask/{key}"][:]]
        if set(tagged) - demo_names:
            continue
        output_mask.create_dataset(key, data=np.array(sorted(tagged), dtype="S"))


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
    obs_keys: tuple[str, ...],
    compress: bool,
) -> None:
    obs_group = group.require_group(prefix)
    for key in obs_keys:
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
    obs_keys: tuple[str, ...],
    camera_info: dict | None,
    source_demo: h5py.Group,
    compress: bool,
) -> int:
    ep_group = output_group.create_group(demo_name)
    ep_group.create_dataset("states", data=np.array(traj["states"]))
    write_obs_group(
        group=ep_group, prefix="obs", obs_dict=traj["obs"], obs_keys=obs_keys, compress=compress
    )
    write_obs_group(
        group=ep_group, prefix="next_obs", obs_dict=traj["next_obs"], obs_keys=obs_keys, compress=compress
    )

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
    cameras: list[str],
    camera_height: int,
    camera_width: int,
    filter_key: str | None,
    num_demos: int | None,
    compress: bool,
) -> None:
    extraction = load_extraction_module()
    written_obs_keys = obs_keys(cameras=cameras)

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=str(input_path))
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=list(cameras),
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
        if filter_key is None:
            demo_names = sorted_demo_names(dataset=source_file)
        else:
            demo_names = filtered_demo_names(dataset=source_file, filter_key=filter_key)
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
                camera_names=list(cameras),
                camera_height=camera_height,
                camera_width=camera_width,
            )
            total_samples += write_demo(
                output_group=data_group,
                demo_name=demo_name,
                traj=traj,
                obs_keys=written_obs_keys,
                camera_info=camera_info,
                source_demo=source_demo,
                compress=compress,
            )

        if "mask" in source_file:
            # every demo in this file is already restricted to @filter_key, so only the
            # filter keys that stay valid are carried over
            copy_filter_keys(
                source_file=source_file,
                output_file=output_file,
                demo_names=set(demo_names),
            )

        data_group.attrs["total"] = total_samples
        data_group.attrs["env_args"] = json.dumps(env.serialize(), indent=4)

    print(f"wrote {len(demo_names)} demos ({total_samples} samples) to {output_path}")


def main() -> None:
    args = parse_args()
    configure_renderer()
    output_path = args.output or default_output_path(
        input_path=args.input, cameras=args.cameras, filter_key=args.filter_key
    )
    extract_act_dataset(
        input_path=args.input,
        output_path=output_path,
        cameras=args.cameras,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        filter_key=args.filter_key,
        num_demos=args.n,
        compress=args.compress,
    )


if __name__ == "__main__":
    main()
