import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from common import write_json


REQUIRED_CONFIG_KEYS = (
    "arch",
    "batch_size",
    "global_seed",
    "num_expert_episodes",
    "max_steps",
    "train_capture_hz",
    "eval_episodes",
    "eval_max_steps",
    "eval_capture_hz",
    "dagger_episodes",
    "rollout_max_steps",
    "intervention_threshold",
    "intervention_steps",
    "dataloader_workers",
    "persistent_workers",
    "workers",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--hardware", type=Path, required=True)
    parser.add_argument("--script-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError(f"config is missing required keys: {missing}")
    if config["arch"] != "vision_mlp" or config["num_expert_episodes"] != 100:
        raise ValueError("benchmark requires vision_mlp and 100 expert episodes")
    if config.get("persistent_workers", False) and config.get("dataloader_workers", 0) == 0:
        raise ValueError("config enables persistent workers with zero workers")

    sys.path.insert(0, str(args.project_root / "scripts"))
    from flywheel import make_flywheel_seeds

    instance = json.loads(args.instance.read_text())
    hardware = json.loads(args.hardware.read_text())
    script_manifest = json.loads(args.script_manifest.read_text())
    if hardware.get("success") is not True:
        raise ValueError("hardware acceptance did not succeed")
    if not script_manifest:
        raise ValueError("script manifest is empty")
    seeds = make_flywheel_seeds(global_seed=int(config["global_seed"]))

    write_json(
        path=args.output,
        value={
            "schema_version": 1,
            "benchmark": "vision_mlp infrastructure",
            "branch": args.branch,
            "commit": args.commit,
            "config": str(args.config.relative_to(args.project_root)),
            "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
            "config_contents": args.config.read_text(),
            "batch_size": int(config["batch_size"]),
            "global_seed": int(config["global_seed"]),
            "seeds": {
                "expert": seeds["expert_seed"],
                "train": seeds["train_seed"],
                "eval": seeds["eval_seed"],
                "dagger": seeds["dagger_seed"],
            },
            "fixed": {
                "expert_episodes": int(config["num_expert_episodes"]),
                "expert_max_steps": int(config["max_steps"]),
                "train_capture_hz": float(config["train_capture_hz"]),
                "eval_episodes": int(config["eval_episodes"]),
                "eval_max_steps": int(config["eval_max_steps"]),
                "eval_capture_hz": float(config["eval_capture_hz"]),
                "dagger_episodes": int(config["dagger_episodes"]),
                "dagger_max_steps": int(config["rollout_max_steps"]),
                "intervention_threshold": float(config["intervention_threshold"]),
                "intervention_steps": int(config["intervention_steps"]),
            },
            "baseline": {
                "dataloader_workers": int(config["dataloader_workers"]),
                "persistent_workers": bool(config["persistent_workers"]),
                "workers": int(config["workers"]),
            },
            "instance": instance,
            "hardware": hardware,
            "script_manifest": script_manifest,
            "telemetry_omissions": [],
            "started_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


if __name__ == "__main__":
    main()
