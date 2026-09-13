import argparse
import json
import subprocess
import time
from pathlib import Path

from common import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--sample-seed", type=int, required=True)
    parser.add_argument("--eval-seed", type=int, required=True)
    parser.add_argument("--eval-max-steps", type=int, required=True)
    parser.add_argument("--eval-capture-hz", type=float, required=True)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--dataset-digest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    attempt_root = args.control_dir / f"bootstrap/attempt-{args.attempt}"
    checkpoint_root = attempt_root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=False)
    if any(checkpoint_root.iterdir()):
        raise ValueError("bootstrap checkpoint root is not empty")
    command = [
        "/root/.local/bin/uv",
        "run",
        "python",
        "-c",
        (
            "import random,runpy,sys,numpy as np,torch; "
            f"seed={args.sample_seed}; random.seed(seed); np.random.seed(seed); "
            "torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); "
            "sys.argv=['scripts/train.py',*sys.argv[1:]]; "
            "runpy.run_path('scripts/train.py',run_name='__main__')"
        ),
        "--arch",
        "vision_mlp",
        "--base",
        "vision-benchmark-bootstrap",
        "--checkpoint_root",
        str(checkpoint_root),
        "--log_root",
        str(attempt_root / "runs"),
        "--epochs",
        "120",
        "--batch-size",
        str(args.batch_size),
        "--npz",
        str(args.control_dir / "data/expert"),
        "--sample-seed",
        str(args.sample_seed),
        "--eval-interval",
        "120",
        "--eval-seed",
        str(args.eval_seed),
        "--eval-episodes",
        "1",
        "--eval-max-steps",
        str(args.eval_max_steps),
        "--eval-capture-hz",
        str(args.eval_capture_hz),
        "--eval-workers",
        "12",
        "--dataloader-workers",
        "0",
        "--early-stop-patience",
        "0",
    ]
    started = time.perf_counter()
    subprocess.run(command, cwd=args.project_root, check=True)
    wall_seconds = time.perf_counter() - started

    run_directories = [path for path in checkpoint_root.iterdir() if path.is_dir()]
    if len(run_directories) != 1:
        raise ValueError("bootstrap did not create exactly one run directory")
    checkpoint = (run_directories[0] / "last.pt").resolve(strict=True)
    if not (run_directories[0] / "best.pt").is_file():
        raise ValueError("bootstrap did not create best.pt")
    checkpoint_metadata = args.control_dir / "bootstrap/checkpoint.json"
    subprocess.run(
        [
            "/root/.local/bin/uv",
            "run",
            "python",
            str(args.control_dir / "tools/validate_checkpoint.py"),
            "--project-root",
            str(args.project_root),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(checkpoint_metadata),
        ],
        cwd=args.project_root,
        check=True,
    )
    metadata = json.loads(checkpoint_metadata.read_text())
    write_json(
        path=args.output,
        value={
            "schema_version": 1,
            "kind": "bootstrap",
            "success": True,
            "trial_id": args.trial_id,
            "generation": args.generation,
            "attempt": args.attempt,
            "git_commit": args.git_commit,
            "dataset_digest": args.dataset_digest,
            "inputs": {
                "batch_size": args.batch_size,
                "sample_seed": args.sample_seed,
                "model_seed": args.sample_seed,
                "eval_seed": args.eval_seed,
                "eval_max_steps": args.eval_max_steps,
                "eval_capture_hz": args.eval_capture_hz,
                "epochs": 120,
                "dataloader_workers": 0,
                "persistent_workers": False,
            },
            "wall_seconds": wall_seconds,
            "checkpoint_path": metadata["checkpoint_path"],
            "checkpoint_sha256": metadata["checkpoint_sha256"],
        },
    )


if __name__ == "__main__":
    main()
