import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

from common import write_json


def inspect_checkpoint(*, project_root: Path, checkpoint_path: Path) -> dict:
    checkpoint_path = checkpoint_path.resolve(strict=True)
    sys.path.insert(0, str(project_root / "scripts"))
    from policy_runtimes.registry import load_runtime

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("arch") != "vision_mlp" or checkpoint.get("epoch") != 120:
        raise ValueError("bootstrap checkpoint architecture or epoch is incorrect")
    if checkpoint.get("normalize") is not True:
        raise ValueError("bootstrap checkpoint is not normalized")
    for key in ("arm_actions_mean", "arm_actions_std", "arm_obs_mean", "arm_obs_std"):
        values = np.asarray(checkpoint[key])
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError(f"invalid normalization array: {key}")
    load_runtime(model_path=checkpoint_path, device=torch.device("cpu"))
    return {
        "success": True,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "run_directory": str(checkpoint_path.parent),
        "arch": checkpoint["arch"],
        "epoch": checkpoint["epoch"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    write_json(
        path=args.output,
        value=inspect_checkpoint(
            project_root=args.project_root,
            checkpoint_path=args.checkpoint,
        ),
    )


if __name__ == "__main__":
    main()
