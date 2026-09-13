import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from common import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    files = sorted(args.data_dir.glob("*.npz"))
    expected_names = {
        f"pick_place_{index:06d}.npz" for index in range(args.expected_episodes)
    }
    actual_names = {path.name for path in files}
    if actual_names != expected_names:
        raise ValueError(
            f"dataset names differ: missing={sorted(expected_names - actual_names)}, "
            f"extra={sorted(actual_names - expected_names)}"
        )

    records = []
    total_frames = 0
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as episode:
            if not {"obs", "actions", "img_obs"}.issubset(episode.files):
                raise ValueError(f"{path.name} lacks required arrays")
            obs = episode["obs"]
            actions = episode["actions"]
            images = episode["img_obs"]
            frames = obs.shape[0]
            if frames <= 0 or actions.shape[0] != frames or images.shape[0] != frames:
                raise ValueError(f"{path.name} has inconsistent frame counts")
            if obs.ndim != 2 or actions.ndim != 2:
                raise ValueError(f"{path.name} has invalid observation/action rank")
            if images.shape != (frames, 64, 64, 3) or images.dtype != np.uint8:
                raise ValueError(f"{path.name} has invalid image data")
            record = {
                "filename": path.name,
                "bytes": path.stat().st_size,
                "sha256": digest,
                "frames": frames,
                "obs_shape": list(obs.shape),
                "actions_shape": list(actions.shape),
                "img_obs_shape": list(images.shape),
                "img_obs_dtype": str(images.dtype),
            }
            records.append(record)
            total_frames += frames

    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    write_json(
        path=args.output,
        value={
            "schema_version": 1,
            "success": True,
            "episode_count": len(records),
            "total_frames": total_frames,
            "aggregate_sha256": hashlib.sha256(canonical).hexdigest(),
            "files": records,
        },
    )


if __name__ == "__main__":
    main()
