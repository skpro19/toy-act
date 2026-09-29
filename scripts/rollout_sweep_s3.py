"""Run rollouts over every S3 checkpoint of one or more training runs.

For each run name, streams every ``epoch_*.pt`` checkpoint from
``s3://<bucket>/checkpoints/<version>/<run-name>/``, evaluates it with the same
rollout procedure as ``scripts/rollout.py``, and incrementally rewrites and
uploads the run's artifacts to ``s3://<bucket>/rollouts/<version>/<run-name>/``:

- ``results.json``: metadata header plus the per-checkpoint summaries,
- ``results.npz``: the same summaries as NumPy arrays,
- ``success_rate.png``: the success-rate curve across epochs.

The run's ``config.json`` is copied from ``runs/<version>/<run-name>/`` and
uploaded once; its ``config.use_z`` value decides whether the act_v2 decoder
runs with or without the latent ``z``.

Checkpoints are downloaded one at a time and deleted after evaluation, so disk
usage stays bounded regardless of run length. The results and curve are
rewritten and re-uploaded after every checkpoint, so a crash or an aborted
sweep still leaves the partial output intact and the curve current.

Examples:
    PYTHONPATH=. uv run python scripts/rollout_sweep_s3.py \\
        --run-name 20260927-174619_bs8_lr1e-04_beta0.01_beta_start0.001_wu80_ep1000_seed0_ckpt10_use_z1_l1 \\
        --version act_v2 --n-rollouts 30 --horizon 250 --no-on-screen
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import boto3
import numpy as np
import torch
from botocore.exceptions import ClientError
from tqdm import tqdm

from scripts.models.act_v1.config import IMG_DIMS
from scripts.rollout import (
    DEFAULT_IMAGE_KEYS,
    close_env,
    configure_renderer,
    create_rollout_env,
    make_rollout_env_meta,
)
from scripts.rollout_sweep import (
    evaluate_checkpoint,
    parse_epoch_checkpoint,
    plot_success_rate,
    save_results,
)

DEFAULT_VERSION = "act_v2"
DEFAULT_DATASET = Path("datasets/can/ph/2026-09-19_03-14-50_act_agentview.hdf5")
CONFIG_FILENAME = "config.json"
ARTIFACT_FILENAMES = ("results.json", "results.npz", "success_rate.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run-name",
        action="append",
        required=True,
        help="training run directory name (repeat for multiple runs)",
    )
    parser.add_argument(
        "--version",
        choices=(DEFAULT_VERSION,),
        default=DEFAULT_VERSION,
        help="checkpoint/model version (only act_v2 is supported)",
    )
    parser.add_argument(
        "--bucket",
        default=os.environ.get("S3_BUCKET"),
        help="S3 bucket (default: $S3_BUCKET)",
    )
    parser.add_argument(
        "--region",
        default=os.environ.get("AWS_REGION"),
        help="S3 region (default: $AWS_REGION)",
    )
    parser.add_argument(
        "--n-rollouts",
        type=int,
        default=30,
        help="number of evaluation episodes per checkpoint",
    )
    parser.add_argument("--horizon", type=int, default=250, help="max steps per episode")
    parser.add_argument("--seed", type=int, default=0, help="random seed for env resets")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="robomimic hdf5 used to recreate the rollout environment",
    )
    parser.add_argument(
        "--on-screen",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="render live in the MuJoCo viewer instead of off-screen",
    )
    parser.add_argument(
        "--terminate-on-success",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="stop an episode early once the task succeeds",
    )
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="record a failed checkpoint and continue with the next one",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="working directory for checkpoint downloads (default: a temp directory)",
    )
    return parser.parse_args()


def s3_client(*, region: str | None) -> Any:
    return boto3.client("s3", region_name=region)


def list_checkpoints(
    *,
    client: Any,
    bucket: str,
    prefix: str) -> list[tuple[int, str]]:
    paginator = client.get_paginator("list_objects_v2")
    checkpoints: list[tuple[int, str]] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/"):
        for item in page.get("Contents", []):
            key = item["Key"]
            epoch = parse_epoch_checkpoint(path=Path(key))
            if epoch is not None:
                checkpoints.append((epoch, key))
    if not checkpoints:
        raise SystemExit(f"no epoch_*.pt checkpoints found under s3://{bucket}/{prefix}/")
    return sorted(checkpoints)


def download_json(*, client: Any, bucket: str, key: str) -> dict:
    try:
        response = client.get_object(Bucket=bucket, Key=key)
    except ClientError as error:
        raise SystemExit(f"could not read s3://{bucket}/{key}: {error}") from error
    return json.loads(response["Body"].read())


def resolve_use_z(*, config: dict) -> bool:
    return bool(config.get("config", {}).get("use_z", True))


def resolve_image_keys(*, config: dict) -> tuple[str, ...]:
    image_keys = config.get("config", {}).get("image_keys", list(DEFAULT_IMAGE_KEYS))
    return tuple(image_keys)


def git_commit() -> str:
    return os.environ.get("GIT_COMMIT", "")


def write_text_atomic(*, path: Path, text: str) -> None:
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(text)
    os.replace(temp_path, path)


def write_npz_atomic(*, results: list[dict], path: Path) -> None:
    temp_path = path.with_name(f"{path.stem}.tmp.npz")
    save_results(results=results, output_path=temp_path)
    os.replace(temp_path, path)


def write_plot_atomic(*, results: list[dict], run_name: str, path: Path) -> None:
    temp_path = path.with_name(f"{path.name}.tmp")
    plot_success_rate(results=results, run_name=run_name, output_path=temp_path)
    os.replace(temp_path, path)


def write_run_outputs(
    *,
    results: list[dict],
    metadata: dict,
    failures: list[dict],
    run_name: str,
    output_dir: Path) -> None:
    payload = {**metadata, "failures": failures, "checkpoints": results}
    write_text_atomic(
        path=output_dir / "results.json",
        text=json.dumps(payload, indent=2) + "\n",
    )
    write_npz_atomic(results=results, path=output_dir / "results.npz")
    write_plot_atomic(
        results=results,
        run_name=run_name,
        path=output_dir / "success_rate.png",
    )


def upload_artifact(*, client: Any, bucket: str, prefix: str, path: Path) -> None:
    key = f"{prefix}/{path.name}"
    client.upload_file(str(path), bucket, key)
    remote = client.head_object(Bucket=bucket, Key=key)
    if remote["ContentLength"] != path.stat().st_size:
        raise RuntimeError(f"upload size mismatch for s3://{bucket}/{key}")


def sweep_run(
    *,
    args: argparse.Namespace,
    run_name: str,
    client: Any,
    device: torch.device,
    env) -> dict[str, int]:
    checkpoint_prefix = f"checkpoints/{args.version}/{run_name}"
    config_key = f"runs/{args.version}/{run_name}/{CONFIG_FILENAME}"
    output_prefix = f"rollouts/{args.version}/{run_name}"

    config = download_json(client=client, bucket=args.bucket, key=config_key)
    use_z = resolve_use_z(config=config)
    image_keys = resolve_image_keys(config=config)
    checkpoints = list_checkpoints(
        client=client,
        bucket=args.bucket,
        prefix=checkpoint_prefix,
    )

    run_dir = args.workspace / run_name
    output_dir = run_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path=output_dir / CONFIG_FILENAME, text=json.dumps(config, indent=2) + "\n")
    upload_artifact(
        client=client,
        bucket=args.bucket,
        prefix=output_prefix,
        path=output_dir / CONFIG_FILENAME,
    )

    metadata = {
        "run_name": run_name,
        "version": args.version,
        "use_z": use_z,
        "image_keys": list(image_keys),
        "seed": args.seed,
        "n_rollouts": args.n_rollouts,
        "horizon": args.horizon,
        "terminate_on_success": args.terminate_on_success,
        "on_screen": args.on_screen,
        "dataset": str(args.dataset),
        "git_commit": git_commit(),
        "checkpoints_total": len(checkpoints),
        "rollouts_planned": len(checkpoints) * args.n_rollouts,
    }

    print(
        f"[{run_name}] {len(checkpoints)} checkpoints x {args.n_rollouts} rollouts "
        f"(use_z={use_z})"
    )

    results: list[dict] = []
    failures: list[dict] = []
    progress = tqdm(
        total=len(checkpoints) * args.n_rollouts,
        unit="rollout",
        desc=run_name,
        dynamic_ncols=True,
    )
    try:
        for epoch, key in checkpoints:
            local_checkpoint = run_dir / PurePosixPath(key).name
            client.download_file(args.bucket, key, str(local_checkpoint))
            try:
                summary = evaluate_checkpoint(
                    checkpoint=local_checkpoint,
                    env=env,
                    device=device,
                    model_version=args.version,
                    use_z=use_z,
                    image_keys=image_keys,
                    n_rollouts=args.n_rollouts,
                    horizon=args.horizon,
                    terminate_on_success=args.terminate_on_success,
                    on_screen=args.on_screen,
                    reset_seed_base=args.seed,
                    progress=progress,
                )
            except Exception as error:
                failures.append(
                    {"epoch": epoch, "checkpoint": local_checkpoint.name, "error": str(error)}
                )
                tqdm.write(f"[{run_name}] {local_checkpoint.name}: failed ({error})")
                if not args.continue_on_error:
                    raise
            else:
                results.append(
                    {"epoch": epoch, "checkpoint": local_checkpoint.name, **summary}
                )
                tqdm.write(
                    f"[{run_name}] epoch {epoch}: success="
                    f"{summary['num_success']}/{summary['num_rollouts']} "
                    f"rate={summary['success_rate']:.3f} "
                    f"return={summary['return_mean']:.3f} "
                    f"horizon={summary['horizon_mean']:.1f}"
                )
            finally:
                local_checkpoint.unlink(missing_ok=True)

            if results:
                write_run_outputs(
                    results=results,
                    metadata=metadata,
                    failures=failures,
                    run_name=run_name,
                    output_dir=output_dir,
                )
                for filename in ARTIFACT_FILENAMES:
                    upload_artifact(
                        client=client,
                        bucket=args.bucket,
                        prefix=output_prefix,
                        path=output_dir / filename,
                    )
    finally:
        progress.close()

    return {
        "checkpoints_total": len(checkpoints),
        "completed": len(results),
        "failed": len(failures),
    }


def main() -> None:
    args = parse_args()
    if not args.bucket:
        raise SystemExit("--bucket or S3_BUCKET is required")

    configure_renderer(on_screen=args.on_screen)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    client = s3_client(region=args.region)

    if args.workspace is not None:
        args.workspace.mkdir(parents=True, exist_ok=True)
    else:
        args.workspace = Path(tempfile.mkdtemp(prefix="rollout-sweep-"))

    env_meta = make_rollout_env_meta(
        dataset_path=args.dataset,
        camera_height=IMG_DIMS[0],
        camera_width=IMG_DIMS[1],
    )
    env = create_rollout_env(
        env_meta=env_meta,
        on_screen=args.on_screen,
        write_video=False,
    )

    summaries: dict[str, dict[str, int]] = {}
    try:
        for run_name in args.run_name:
            summaries[run_name] = sweep_run(
                args=args,
                run_name=run_name,
                client=client,
                device=device,
                env=env,
            )
    finally:
        close_env(env)

    print("rollout sweep summary")
    for run_name, summary in summaries.items():
        print(
            f"  {run_name}: completed {summary['completed']}/"
            f"{summary['checkpoints_total']} checkpoints, "
            f"{summary['failed']} failed"
        )

    failed_runs = [name for name, summary in summaries.items() if summary["failed"]]
    if failed_runs:
        raise SystemExit("checkpoint failures in: " + ", ".join(failed_runs))


if __name__ == "__main__":
    main()
