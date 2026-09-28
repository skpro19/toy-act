"""Render every camera in the simulator for a handful of frames of one episode.

Replays a robomimic episode by loading its recorded simulator states, renders all
cameras at each selected frame, and writes a labelled grid image per frame.

Examples:
    # 10 evenly spaced frames of demo_0 -> debug/cameras/frame_*.png
    uv run python scripts/capture_cameras.py

    # a specific episode and frame count
    uv run python scripts/capture_cameras.py --demo demo_42 --frames 10

    # pick the episode at random (reproducible)
    uv run python scripts/capture_cameras.py --random --seed 0
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATASET = REPO_ROOT / "datasets" / "can" / "ph" / "low_dim_v15.hdf5"
DEFAULT_OUT_DIR = REPO_ROOT / "debug" / "cameras"

ALL_CAMERAS = [
    "agentview",
    "frontview",
    "birdview",
    "robot0_robotview",
    "robot0_eye_in_hand",
]
GRID_COLUMNS = 3
LABEL_HEIGHT = 28


def configure_renderer() -> None:
    # robosuite reads MUJOCO_GL at import time, so set it before importing robomimic.
    os.environ["MUJOCO_GL"] = "egl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="path to robomimic hdf5 dataset")
    parser.add_argument("--demo", type=str, default="demo_0", help="episode group name to replay")
    parser.add_argument("--random", action="store_true", help="choose a random episode instead of --demo")
    parser.add_argument("--seed", type=int, default=None, help="seed for random episode selection")
    parser.add_argument("--frames", type=int, default=10, help="number of evenly spaced frames to render")
    parser.add_argument("--camera-height", type=int, default=256, help="rendered image height")
    parser.add_argument("--camera-width", type=int, default=256, help="rendered image width")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="directory for grid images")
    return parser.parse_args()


def evenly_spaced_indices(*, total: int, count: int) -> list[int]:
    if count >= total:
        return list(range(total))
    import numpy as np

    return sorted({int(index) for index in np.linspace(0, total - 1, count)})


def render_grid(
    *,
    images: dict,
    cameras: list[str],
    columns: int,
):
    from PIL import Image, ImageDraw, ImageFont
    from matplotlib import font_manager

    font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size=18)
    tile_width, tile_height = next(iter(images.values())).size
    rows = (len(cameras) + columns - 1) // columns
    cell_height = tile_height + LABEL_HEIGHT
    grid = Image.new("RGB", (columns * tile_width, rows * cell_height), color=(20, 20, 20))
    draw = ImageDraw.Draw(grid)
    for position, camera in enumerate(cameras):
        row, column = divmod(position, columns)
        x = column * tile_width
        y = row * cell_height
        draw.text((x + 8, y + 5), camera, fill=(255, 255, 255), font=font)
        grid.paste(images[camera], (x, y + LABEL_HEIGHT))
    return grid


def capture(
    *,
    dataset_path: Path,
    demo_name: str,
    frame_count: int,
    camera_height: int,
    camera_width: int,
    out_dir: Path,
) -> None:
    import h5py
    import numpy as np
    from PIL import Image

    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=str(dataset_path))
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=list(ALL_CAMERAS),
        camera_height=camera_height,
        camera_width=camera_width,
        reward_shaping=False,
    )

    with h5py.File(dataset_path, "r") as dataset:
        demo = dataset[f"data/{demo_name}"]
        states = demo["states"][()]
        ep_meta = demo.attrs.get("ep_meta", None)
        model = demo.attrs["model_file"]

    initial_state = {"states": states[0]}
    if model is not None:
        initial_state["model"] = model
        initial_state["ep_meta"] = ep_meta
    env.reset_to(initial_state)

    indices = evenly_spaced_indices(total=states.shape[0], count=frame_count)
    out_dir.mkdir(parents=True, exist_ok=True)
    for frame_index in indices:
        obs = env.reset_to({"states": states[frame_index]})
        images = {
            camera: Image.fromarray(np.asarray(obs[f"{camera}_image"], dtype=np.uint8))
            for camera in ALL_CAMERAS
        }
        grid = render_grid(images=images, cameras=ALL_CAMERAS, columns=GRID_COLUMNS)
        output_path = out_dir / f"frame_{frame_index:04d}.png"
        grid.save(output_path)
        print(f"wrote {output_path} ({len(ALL_CAMERAS)} cameras)")


def main() -> None:
    args = parse_args()
    configure_renderer()

    demo_name = args.demo
    if args.random:
        import random

        import h5py

        if args.seed is not None:
            random.seed(args.seed)
        with h5py.File(args.dataset, "r") as dataset:
            demo_name = random.choice(list(dataset["data"].keys()))

    capture(
        dataset_path=args.dataset,
        demo_name=demo_name,
        frame_count=args.frames,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
