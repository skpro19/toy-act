"""Merge two ACT training HDF5 files into one, re-indexing demos to avoid collisions.

Both robomimic sources name their demonstrations ``demo_0``, ``demo_1``, ... so a naive
copy would collide. This script copies the first dataset's demos unchanged and offsets
the second dataset's demo indices past the end of the first, so every demo in the result
has a unique name.

Image and proprio datasets are copied verbatim, so the merged file keeps exactly the
schema of its inputs. The ``mask`` group is intentionally dropped: its keys reference
the source demo names and would be stale after re-indexing.

Examples:
    uv run python scripts/utils/merge_act_datasets.py \\
        --first datasets/can/ph/2026-09-29_02-01-42_agentview_robot0_eye_in_hand.hdf5 \\
        --second datasets/can/mh/2026-10-01_03-44-12_better_agentview_robot0_eye_in_hand.hdf5 \\
        --output datasets/can/ph_mh_better/merged.hdf5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def sorted_demo_names(*, dataset: h5py.File) -> list[str]:
    demos = list(dataset["data"].keys())
    return sorted(demos, key=lambda name: int(name.split("_")[1]))


def demo_attr_keys(*, demo: h5py.Group) -> list[str]:
    return sorted(demo.attrs.keys())


def copy_demo(
    *,
    source_demo: h5py.Group,
    output_group: h5py.Group,
    demo_name: str,
) -> int:
    """Copy one demonstration group verbatim, preserving nested obs/next_obs groups."""
    output_demo = output_group.create_group(demo_name)

    for key in source_demo:
        if key in ("obs", "next_obs"):
            continue
        source_demo.copy(key, output_demo, name=key)

    for prefix in ("obs", "next_obs"):
        if prefix not in source_demo:
            continue
        output_obs = output_demo.create_group(prefix)
        for key in source_demo[prefix]:
            source_demo.copy(f"{prefix}/{key}", output_obs, name=key)

    for attr_key in demo_attr_keys(demo=source_demo):
        output_demo.attrs[attr_key] = source_demo.attrs[attr_key]

    return int(source_demo.attrs["num_samples"])


def merge_datasets(*, first_path: Path, second_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    total_samples = 0
    with h5py.File(first_path, "r") as first_file, h5py.File(second_path, "r") as second_file:
        env_args_first = first_file["data"].attrs["env_args"]
        env_args_second = second_file["data"].attrs["env_args"]
        if env_args_first != env_args_second:
            raise ValueError(
                "datasets have different env_args and cannot be merged:\n"
                f"  first:  {env_args_first}\n"
                f"  second: {env_args_second}"
            )

        first_demos = sorted_demo_names(dataset=first_file)
        second_demos = sorted_demo_names(dataset=second_file)
        offset = len(first_demos)

        with h5py.File(output_path, "w") as output_file:
            data_group = output_file.create_group("data")

            for index, demo_name in enumerate(tqdm(first_demos, desc="copying first")):
                total_samples += copy_demo(
                    source_demo=first_file[f"data/{demo_name}"],
                    output_group=data_group,
                    demo_name=f"demo_{index}",
                )

            for index, demo_name in enumerate(tqdm(second_demos, desc="copying second")):
                total_samples += copy_demo(
                    source_demo=second_file[f"data/{demo_name}"],
                    output_group=data_group,
                    demo_name=f"demo_{offset + index}",
                )

            data_group.attrs["total"] = total_samples
            data_group.attrs["env_args"] = env_args_first

    print(f"first:  {first_path} ({len(first_demos)} demos)")
    print(f"second: {second_path} ({len(second_demos)} demos)")
    print(f"wrote {len(first_demos) + len(second_demos)} demos ({total_samples} samples) to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--first", type=Path, required=True, help="dataset whose demos keep their indices")
    parser.add_argument("--second", type=Path, required=True, help="dataset whose demos are appended after the first")
    parser.add_argument("--output", type=Path, required=True, help="path for the merged hdf5")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge_datasets(first_path=args.first, second_path=args.second, output_path=args.output)


if __name__ == "__main__":
    main()
