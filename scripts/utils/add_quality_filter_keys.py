"""Add the combined operator-quality filter keys used by the ACT quality-subset datasets.

The multi-human (MH) datasets were collected by 6 operators, 2 each of ``better``,
``okay``, and ``worse`` proficiency. Each operator contributes 50 demonstrations, so
the three quality tiers are 100 demos each out of 300 total.

Robomimic ships unions of those tiers in *ascending* quality order (``okay_better``,
``worse_okay``, ``worse_better``). This script adds the descending-order names that the
quality-subset extraction uses, so ``--filter-key better_okay`` and
``--filter-key better_okay_worse`` work alongside the shipped keys.

Both new keys are aliases of data that already exists -- no demos are reshuffled, and
running the script twice is a no-op.

Examples:
    uv run python scripts/utils/add_quality_filter_keys.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = REPO_ROOT / "datasets" / "can" / "mh" / "low_dim_v15.hdf5"

# filter key to create -> the existing key it should be an alias of
ALIASED_KEYS = {
    # ascending-quality "okay_better" is the same 200 demos, in descending order
    "better_okay": "okay_better",
    # the full 300-demo dataset; "train" + "valid" partition all of it
    "better_okay_worse": None,
}


def read_filter_key(*, dataset: h5py.File, key: str) -> list[str]:
    mask_path = f"mask/{key}"
    if mask_path not in dataset:
        available = sorted(dataset["mask"].keys()) if "mask" in dataset else []
        raise KeyError(f"filter key {mask_path!r} not found (available: {available})")
    return [name.decode("utf-8") for name in dataset[mask_path][:]]


def write_filter_key(*, dataset: h5py.File, key: str, demo_names: list[str]) -> None:
    mask_path = f"mask/{key}"
    if mask_path in dataset:
        del dataset[mask_path]
    dataset.create_dataset(mask_path, data=np.array(sorted(demo_names), dtype="S"))


def all_demo_names(*, dataset: h5py.File) -> list[str]:
    return sorted(dataset["data"].keys())


def add_quality_filter_keys(*, input_path: Path) -> None:
    with h5py.File(input_path, "a") as dataset:
        if "mask" not in dataset:
            raise KeyError(f"no mask group in {input_path}; is this a robomimic dataset?")

        for key, source_key in ALIASED_KEYS.items():
            if source_key is None:
                demo_names = all_demo_names(dataset=dataset)
                origin = "all demos"
            else:
                demo_names = read_filter_key(dataset=dataset, key=source_key)
                origin = f"mask/{source_key}"
            write_filter_key(dataset=dataset, key=key, demo_names=demo_names)
            print(f"wrote mask/{key} ({len(demo_names)} demos) from {origin}")

        print("")
        for key in ["better", "okay", "worse", "better_okay", "better_okay_worse"]:
            print(f"  {key:20s} {len(read_filter_key(dataset=dataset, key=key)):4d} demos")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="path to the robomimic multi-human low-dim hdf5",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    add_quality_filter_keys(input_path=args.input)


if __name__ == "__main__":
    main()
