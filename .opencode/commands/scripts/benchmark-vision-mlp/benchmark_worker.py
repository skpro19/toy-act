import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from common import write_json


def common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--checkpoint-metadata", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--capture-hz", type=float, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--dataset-digest", required=True)
    parser.add_argument("--output", type=Path, required=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="kind", required=True)
    eval_parser = subparsers.add_parser("evaluation")
    common_parser(eval_parser)
    dagger_parser = subparsers.add_parser("dagger")
    common_parser(dagger_parser)
    dagger_parser.add_argument("--intervention-threshold", type=float, required=True)
    dagger_parser.add_argument("--intervention-steps", type=int, required=True)
    dagger_parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    metadata = json.loads(args.checkpoint_metadata.read_text())
    if metadata.get("success") is not True:
        raise ValueError("checkpoint metadata is not successful")
    checkpoint = Path(metadata["checkpoint_path"]).resolve(strict=True)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != metadata["checkpoint_sha256"]:
        raise ValueError("checkpoint digest changed")

    sys.path.insert(0, str(args.project_root))
    sys.path.insert(0, str(args.project_root / "scripts"))
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    payload = {}
    if args.kind == "evaluation":
        from eval import score_ckpt

        metrics = score_ckpt(
            ckpt_path=str(checkpoint),
            seed=args.seed,
            max_steps=args.max_steps,
            episodes=args.episodes,
            workers=args.workers,
            capture_hz=args.capture_hz,
        )
        scores = metrics.get("scores")
        if not isinstance(scores, list) or len(scores) != args.episodes:
            raise ValueError("evaluation did not return the requested episode count")
        numeric_values = [float(value) for value in scores]
        numeric_values.extend(
            float(value) for key, value in metrics.items() if key != "scores"
        )
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError("evaluation returned non-finite metrics")
        payload["metrics"] = metrics
    else:
        from rollout import rollout

        args.output_dir.mkdir(parents=True, exist_ok=False)
        rollout(
            model_path=str(checkpoint),
            randomize_scene=True,
            seed=args.seed,
            episodes=args.episodes,
            max_steps=args.max_steps,
            log_root=None,
            train_npz_dir=None,
            log_rollout=False,
            dagger=True,
            dagger_root=args.output_dir,
            beta=0.0,
            intervention_mode="threshold",
            intervention_threshold=args.intervention_threshold,
            intervention_steps=args.intervention_steps,
            create_dagger_subdir=False,
            headless=True,
            workers=args.workers,
            capture_hz=args.capture_hz,
        )
        files = sorted(args.output_dir.glob("*.npz"))
        expected_names = {
            f"pick_place_{index:06d}.npz" for index in range(args.episodes)
        }
        if {path.name for path in files} != expected_names:
            raise ValueError("DAgger output episode set is incorrect")
        payload.update(
            {
                "output_dir": str(args.output_dir),
                "output_episode_count": len(files),
                "intervention_threshold": args.intervention_threshold,
                "intervention_steps": args.intervention_steps,
            }
        )

    wall_seconds = time.perf_counter() - started
    write_json(
        path=args.output,
        value={
            "schema_version": 1,
            "kind": args.kind,
            "success": True,
            "trial_id": args.trial_id,
            "generation": args.generation,
            "attempt": args.attempt,
            "git_commit": args.git_commit,
            "dataset_digest": args.dataset_digest,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": digest,
            "seed": args.seed,
            "workers": args.workers,
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "capture_hz": args.capture_hz,
            "started_utc": started_utc,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": wall_seconds,
            **payload,
        },
    )


if __name__ == "__main__":
    main()
