import argparse
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from common import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--dataloader-workers", type=int, required=True)
    parser.add_argument(
        "--persistent-workers", action=argparse.BooleanOptionalAction, required=True
    )
    parser.add_argument("--sample-seed", type=int, required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--warmup-epochs", type=int, required=True)
    parser.add_argument("--measured-epochs", type=int, required=True)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--dataset-digest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.persistent_workers and args.dataloader_workers == 0:
        raise ValueError("persistent workers require at least one worker")
    sys.path.insert(0, str(args.project_root))
    sys.path.insert(0, str(args.project_root / "scripts"))
    from scripts.dataset import PickPlaceVisionDataset
    from train_core.engine import run_epoch
    from train_core.recipes.vision_mlp import VisionMlpRecipe

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device_name = torch.cuda.get_device_name(0)
    if "RTX 4090" not in device_name:
        raise RuntimeError(f"expected RTX 4090, got {device_name}")
    device = torch.device("cuda")
    torch.manual_seed(args.model_seed)
    torch.cuda.manual_seed_all(args.model_seed)

    dataset = PickPlaceVisionDataset(
        data_dirs=[args.data_dir],
        sample_ratios=None,
        dagger_intervention_ratio=None,
        action_space="joint_delta",
        normalize=True,
    )
    sampler = WeightedRandomSampler(
        weights=dataset.sample_weights,
        num_samples=dataset.samples_per_epoch,
        replacement=True,
        generator=torch.Generator().manual_seed(args.sample_seed),
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.dataloader_workers,
        pin_memory=args.dataloader_workers > 0,
        persistent_workers=args.persistent_workers,
    )
    recipe = VisionMlpRecipe()
    model = recipe.build_model(device=device)
    optimizer = recipe.build_optimizer(model=model)
    joint_loss_fn = nn.MSELoss()
    gripper_loss_fn = nn.BCEWithLogitsLoss()

    epoch_kwargs = {
        "model": model,
        "dataloader": dataloader,
        "optimizer": optimizer,
        "recipe": recipe,
        "device": device,
        "joint_loss_fn": joint_loss_fn,
        "gripper_loss_fn": gripper_loss_fn,
    }
    for _ in range(args.warmup_epochs):
        run_epoch(**epoch_kwargs)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    measured_started_utc = datetime.now(timezone.utc).isoformat()
    epoch_seconds = []
    epoch_metrics = []
    measured_started = time.perf_counter()
    for _ in range(args.measured_epochs):
        torch.cuda.synchronize()
        started = time.perf_counter()
        metrics = run_epoch(**epoch_kwargs)
        torch.cuda.synchronize()
        epoch_seconds.append(time.perf_counter() - started)
        epoch_metrics.append({key: float(value) for key, value in metrics.items()})
    measured_seconds = time.perf_counter() - measured_started
    measured_finished_utc = datetime.now(timezone.utc).isoformat()
    samples = dataset.samples_per_epoch * args.measured_epochs

    write_json(
        path=args.output,
        value={
            "schema_version": 1,
            "kind": "training",
            "success": True,
            "trial_id": args.trial_id,
            "generation": args.generation,
            "attempt": args.attempt,
            "git_commit": args.git_commit,
            "dataset_digest": args.dataset_digest,
            "inputs": {
                "batch_size": args.batch_size,
                "dataloader_workers": args.dataloader_workers,
                "persistent_workers": args.persistent_workers,
                "sample_seed": args.sample_seed,
                "model_seed": args.model_seed,
                "warmup_epochs": args.warmup_epochs,
                "measured_epochs": args.measured_epochs,
            },
            "device_name": device_name,
            "samples_per_epoch": dataset.samples_per_epoch,
            "batches_per_epoch": math.ceil(dataset.samples_per_epoch / args.batch_size),
            "epoch_seconds": epoch_seconds,
            "epoch_metrics": epoch_metrics,
            "measured_seconds": measured_seconds,
            "samples_per_second": samples / measured_seconds,
            "measured_started_utc": measured_started_utc,
            "measured_finished_utc": measured_finished_utc,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
    )


if __name__ == "__main__":
    main()
