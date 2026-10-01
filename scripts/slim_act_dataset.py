"""Create a training-only slim copy of a rendered ACT HDF5 dataset.

``scripts/dataset.py`` reads only a small part of a rendered ACT HDF5: the ``obs``
images and proprioception, the ``next_obs`` proprioception, and the per-demo
``num_samples`` attribute. Everything else -- ``next_obs`` images, simulator states,
and the per-demo ``model_file``/``camera_info`` metadata -- is dead weight during
training, where the ``next_obs`` images alone account for roughly half the file size.

This script writes a new file that keeps only what training reads, so the source
dataset is never modified. The ``data/env_args`` attribute is always preserved because
rollout evaluation in ``scripts/train_v2.py`` builds its environment from it. Optional
flags retain the fields needed by other tooling:

    --keep-next-obs-images   keep ``next_obs`` images (identical schema to the source)
    --keep-sim-replay        keep ``states`` and the ``model_file``/``camera_info`` attrs
                             (needed by scripts/replay.py and scripts/utils/capture_cameras.py)
    --keep-mask              keep the ``mask`` demo-split group

Examples:
    # training-only copy: <input>.hdf5 -> <input>_trimmed.hdf5
    uv run python scripts/slim_act_dataset.py \\
        --input datasets/can/ph/2026-09-29_02-01-42_agentview_robot0_eye_in_hand.hdf5

    # keep the fields needed to replay the recorded episodes in simulation
    uv run python scripts/slim_act_dataset.py \\
        --input datasets/can/ph/2026-09-29_02-01-42_agentview_robot0_eye_in_hand.hdf5 \\
        --keep-sim-replay
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
from tqdm import tqdm

IMAGE_SUFFIX = "_image"
PROPRIO_KEYS = ("robot0_joint_pos", "robot0_gripper_qpos")
DEMO_ATTRS = ("num_samples", "ep_meta")
SIM_REPLAY_DEMO_ATTRS = ("model_file", "camera_info")
DATA_ATTRS = ("env_args",)
DEFAULT_OUTPUT_SUFFIX = "_trimmed"


def sorted_demo_names(*, dataset: h5py.File) -> list[str]:
    demos = list(dataset["data"].keys())
    return sorted(demos, key=lambda name: int(name.split("_")[1]))


def image_keys(*, demo: h5py.Group) -> tuple[str, ...]:
    return tuple(sorted(key for key in demo["obs"] if key.endswith(IMAGE_SUFFIX)))


def copy_dataset(*, source: h5py.Group, output: h5py.Group, name: str) -> None:
    """Copy a dataset verbatim, preserving its dtype, chunks, and filters."""
    source.copy(name, output, name=name)


def copy_demo(
    *,
    source_demo: h5py.Group,
    output_group: h5py.Group,
    demo_name: str,
    keep_next_obs_images: bool,
    keep_sim_replay: bool) -> int:
    output_demo = output_group.create_group(demo_name)

    output_obs = output_demo.create_group("obs")
    for key in (*image_keys(demo=source_demo), *PROPRIO_KEYS):
        copy_dataset(source=source_demo["obs"], output=output_obs, name=key)

    output_next_obs = output_demo.create_group("next_obs")
    if keep_next_obs_images:
        for key in image_keys(demo=source_demo):
            copy_dataset(source=source_demo["next_obs"], output=output_next_obs, name=key)
    for key in PROPRIO_KEYS:
        copy_dataset(source=source_demo["next_obs"], output=output_next_obs, name=key)

    if keep_sim_replay:
        copy_dataset(source=source_demo, output=output_demo, name="states")

    retained_attrs = DEMO_ATTRS + (SIM_REPLAY_DEMO_ATTRS if keep_sim_replay else ())
    for attr_key in retained_attrs:
        if attr_key in source_demo.attrs:
            output_demo.attrs[attr_key] = source_demo.attrs[attr_key]

    return int(source_demo.attrs["num_samples"])


def default_output_path(*, input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}{DEFAULT_OUTPUT_SUFFIX}{input_path.suffix}")


def slim_dataset(
    *,
    input_path: Path,
    output_path: Path,
    keep_next_obs_images: bool,
    keep_sim_replay: bool,
    keep_mask: bool) -> None:
    if input_path.resolve() == output_path.resolve():
        raise ValueError(f"output path must differ from the input path: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    total_samples = 0
    with h5py.File(input_path, "r") as source_file, h5py.File(output_path, "w") as output_file:
        for attr_key, value in source_file.attrs.items():
            output_file.attrs[attr_key] = value

        data_group = output_file.create_group("data")
        demo_names = sorted_demo_names(dataset=source_file)
        for demo_name in tqdm(demo_names, desc="slimming demos"):
            total_samples += copy_demo(
                source_demo=source_file[f"data/{demo_name}"],
                output_group=data_group,
                demo_name=demo_name,
                keep_next_obs_images=keep_next_obs_images,
                keep_sim_replay=keep_sim_replay,
            )

        data_group.attrs["total"] = total_samples
        for attr_key in DATA_ATTRS:
            if attr_key in source_file["data"].attrs:
                data_group.attrs[attr_key] = source_file["data"].attrs[attr_key]

        if keep_mask and "mask" in source_file:
            source_file.copy("mask", output_file, name="mask")

    print(f"wrote {len(demo_names)} demos ({total_samples} samples) to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="path to a rendered ACT hdf5 dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"path for the slim hdf5 (defaults to <input>{DEFAULT_OUTPUT_SUFFIX}.hdf5)",
    )
    parser.add_argument(
        "--keep-next-obs-images",
        action="store_true",
        help="keep next_obs images (dropped by default; training does not read them)",
    )
    parser.add_argument(
        "--keep-sim-replay",
        action="store_true",
        help="keep states and model_file/camera_info for simulation replay",
    )
    parser.add_argument(
        "--keep-mask",
        action="store_true",
        help="keep the mask demo-split group",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or default_output_path(input_path=args.input)
    slim_dataset(
        input_path=args.input,
        output_path=output_path,
        keep_next_obs_images=args.keep_next_obs_images,
        keep_sim_replay=args.keep_sim_replay,
        keep_mask=args.keep_mask,
    )


if __name__ == "__main__":
    main()
