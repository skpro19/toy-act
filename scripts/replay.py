"""Replay robomimic datasets in the MuJoCo simulator.

Examples:
    # random 3 episodes -> side-by-side video (offscreen)
    uv run python scripts/replay.py --video replays/can_ph.mp4 --n 3

    # one random episode on screen
    uv run python scripts/replay.py --on-screen

    # reproducible random selection
    uv run python scripts/replay.py --video replays/can_ph.mp4 --n 3 --seed 0

    # open-loop action playback (checks actions reproduce the recorded states)
    uv run python scripts/replay.py --video replays/can_ph_actions.mp4 --n 2 --use-actions

    # replay every episode in order on screen
    uv run python scripts/replay.py --on-screen --all
"""

import argparse
import importlib.util
import os
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PLAYBACK_SCRIPT = REPO_ROOT / "third_party" / "robomimic" / "robomimic" / "scripts" / "playback_dataset.py"
DEFAULT_DATASET = REPO_ROOT / "datasets" / "can" / "ph" / "low_dim_v15.hdf5"
DEFAULT_VIDEO_CAMERAS = ["agentview", "robot0_eye_in_hand"]
DEFAULT_SCREEN_CAMERAS = ["agentview"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="path to robomimic hdf5 dataset")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--on-screen", action="store_true", help="render live in the MuJoCo viewer")
    mode.add_argument("--video", type=Path, help="render offscreen to this mp4 path")
    parser.add_argument("--n", type=int, default=1, help="number of random episodes to replay")
    parser.add_argument("--all", action="store_true", help="replay every episode in order (overrides --n)")
    parser.add_argument("--seed", type=int, default=None, help="seed for random episode selection")
    parser.add_argument("--cameras", type=str, nargs="+", default=None, help="camera name(s); defaults per mode")
    parser.add_argument("--use-actions", action="store_true", help="open-loop action playback instead of loading sim states")
    parser.add_argument("--first", action="store_true", help="only replay the first frame of each episode")
    parser.add_argument("--video-skip", type=int, default=5, help="render every Nth frame to video")
    return parser.parse_args()


def configure_renderer(*, on_screen: bool) -> None:
    # robosuite reads MUJOCO_GL at import time, so set it before importing robomimic.
    os.environ["MUJOCO_GL"] = "glfw" if on_screen else "egl"


def load_playback_module():
    # robomimic's scripts directory is not an installed package, so load the file by path.
    spec = importlib.util.spec_from_file_location("robomimic_playback_dataset", PLAYBACK_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    configure_renderer(on_screen=args.on_screen)
    if args.seed is not None:
        random.seed(args.seed)

    cameras = args.cameras or (DEFAULT_SCREEN_CAMERAS if args.on_screen else DEFAULT_VIDEO_CAMERAS)
    if args.on_screen and len(cameras) != 1:
        raise SystemExit("--on-screen supports exactly one camera")
    if args.video is not None:
        args.video.parent.mkdir(parents=True, exist_ok=True)

    playback = load_playback_module()
    playback.playback_dataset(
        argparse.Namespace(
            dataset=str(args.dataset),
            filter_key=None,
            n=None if args.all else args.n,
            use_obs=False,
            use_actions=args.use_actions,
            render=args.on_screen,
            video_path=str(args.video) if args.video is not None else None,
            video_skip=args.video_skip,
            render_image_names=cameras,
            render_depth_names=None,
            first=args.first,
        )
    )


if __name__ == "__main__":
    main()
