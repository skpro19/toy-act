"""Measure whether the ACT v2 decoder uses the CVAE latent z.

See Phase 5 of notes/z-debug.md. For a fixed set of observations, decode with
z = 0, z = mu, shuffled mu, and several fixed prior samples (z ~ N(0, I)),
then compare the resulting action chunks.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from scripts.dataset import CanPhDataset, NormalizationStats
from scripts.models.act_v2.config import (
    ACTION_CHUNK_SIZE,
    JOINT_DIMS,
    PROPRIO_DIMS,
    Z_DIMS,
)
from scripts.models.act_v2.model import ACTV2
from scripts.utils.analyze_z import load_model, make_subset_indices

DEFAULT_DATASET = Path("datasets/can/ph/2026-09-19_03-14-50_act_agentview.hdf5")
DEFAULT_OUTPUT_DIR = Path("debug/z-debug")
BASELINE_VARIANT = "zero"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="ACT v2 .pt checkpoint")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="run identifier recorded in meta.json; defaults to the checkpoint's parent directory name",
    )
    parser.add_argument("--limit", type=int, default=500, help="number of samples to analyze (<=0 for all)")
    parser.add_argument("--n-prior", type=int, default=5, help="number of fixed z ~ N(0, I) draws")
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def make_output_dir(*, output_root: Path, run_name: str, checkpoint_path: Path) -> Path:
    output_dir = output_root / run_name / checkpoint_path.stem / "sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def variant_names(*, n_prior: int) -> list[str]:
    return [BASELINE_VARIANT, "mu", "shuffled_mu"] + [f"prior_{i}" for i in range(n_prior)]


def build_z_variants(
    *,
    mu: np.ndarray,
    n_prior: int,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    batch_size = mu.shape[0]
    shuffled_index = rng.permutation(batch_size)
    prior = rng.standard_normal((n_prior,) + mu.shape).astype(mu.dtype)

    variants = {
        BASELINE_VARIANT: np.zeros_like(mu),
        "mu": mu.copy(),
        "shuffled_mu": mu[shuffled_index],
    }
    for i in range(n_prior):
        variants[f"prior_{i}"] = prior[i]

    return variants, shuffled_index, prior


def collect_sensitivity(
    *,
    model: ACTV2,
    dataset: CanPhDataset,
    indices: np.ndarray,
    n_prior: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict:
    subset = Subset(dataset, indices.tolist())
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)

    names = variant_names(n_prior=n_prior)
    rng = np.random.default_rng(seed)

    demo_names: list[str] = []
    timesteps: list[int] = []
    mu_rows: list[np.ndarray] = []
    log_sigma_x2_rows: list[np.ndarray] = []
    shuffled_index_rows: list[np.ndarray] = []
    prior_rows: list[np.ndarray] = []
    pred_rows: dict[str, list[np.ndarray]] = {name: [] for name in names}

    with torch.no_grad():
        for batch in tqdm(loader, desc="sensitivity"):
            proprio = batch["proprio"].to(device=device)
            actions = batch["target_actions"].to(device=device)
            img = batch["image"].to(device=device)

            img_tokens = model.image_encoder(img)
            proprio_tokens = model.proprio_encoder(proprio)

            mu, log_sigma_x2 = model.posterior(proprio=proprio, actions=actions)
            mu_np = mu.detach().cpu().numpy()

            variants, shuffled_index, prior = build_z_variants(
                mu=mu_np,
                n_prior=n_prior,
                rng=rng,
            )

            for name in names:
                z = torch.from_numpy(variants[name]).to(device=device)
                pred = model.decode_from_tokens(
                    img_tokens=img_tokens,
                    proprio_tokens=proprio_tokens,
                    z=z,
                    use_z=True,
                )
                pred_rows[name].append(pred.detach().cpu().numpy())

            mu_rows.append(mu_np[:, 0, :])
            log_sigma_x2_rows.append(log_sigma_x2.detach().cpu().numpy()[:, 0, :])
            shuffled_index_rows.append(shuffled_index)
            prior_rows.append(prior[:, :, 0, :])
            demo_names.extend(batch["demo_name"])
            timesteps.extend(int(t) for t in batch["timestep"])

    return {
        "demo_name": np.asarray(demo_names, dtype=object),
        "timestep": np.asarray(timesteps, dtype=np.int64),
        "mu": np.concatenate(mu_rows, axis=0),
        "log_sigma_x2": np.concatenate(log_sigma_x2_rows, axis=0),
        "shuffled_index": np.concatenate(shuffled_index_rows, axis=0),
        "z_prior": np.concatenate(prior_rows, axis=1),
        "preds": {name: np.concatenate(rows, axis=0) for name, rows in pred_rows.items()},
    }


def denormalize_action(*, action: np.ndarray, normalization: NormalizationStats) -> np.ndarray:
    return action * normalization.action_std + normalization.action_mean


def l1_metrics(
    *,
    pred: np.ndarray,
    baseline: np.ndarray,
    normalization: NormalizationStats,
) -> dict[str, float]:
    normalized = np.abs(pred - baseline)
    physical = np.abs(
        denormalize_action(action=pred, normalization=normalization)
        - denormalize_action(action=baseline, normalization=normalization)
    )
    return {
        "normalized_total": float(normalized.mean()),
        "normalized_joint": float(normalized[..., :JOINT_DIMS].mean()),
        "normalized_gripper": float(normalized[..., JOINT_DIMS:].mean()),
        "physical_total": float(physical.mean()),
        "physical_joint": float(physical[..., :JOINT_DIMS].mean()),
        "physical_gripper": float(physical[..., JOINT_DIMS:].mean()),
    }


def per_sample_physical_total(
    *,
    pred: np.ndarray,
    baseline: np.ndarray,
    normalization: NormalizationStats,
) -> np.ndarray:
    physical = np.abs(
        denormalize_action(action=pred, normalization=normalization)
        - denormalize_action(action=baseline, normalization=normalization)
    )
    return physical.mean(axis=(1, 2))


def compute_summary(*, latents: dict, normalization: NormalizationStats) -> dict:
    preds = latents["preds"]
    baseline = preds[BASELINE_VARIANT]

    per_sample: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, float]] = {}
    for name, pred in preds.items():
        if name == BASELINE_VARIANT:
            continue
        metrics[name] = l1_metrics(pred=pred, baseline=baseline, normalization=normalization)
        per_sample[name] = per_sample_physical_total(
            pred=pred,
            baseline=baseline,
            normalization=normalization,
        )

    return {
        "n_samples": int(latents["mu"].shape[0]),
        "action_chunk_size": int(preds[BASELINE_VARIANT].shape[1]),
        "metrics": metrics,
        "per_sample_physical_total": per_sample,
    }


def save_artifacts(*, output_dir: Path, latents: dict, summary: dict) -> None:
    np.savez_compressed(
        output_dir / "posterior.npz",
        demo_name=latents["demo_name"],
        timestep=latents["timestep"],
        mu=latents["mu"],
        log_sigma_x2=latents["log_sigma_x2"],
        shuffled_index=latents["shuffled_index"],
        z_prior=latents["z_prior"],
    )

    names = list(latents["preds"].keys())
    np.savez_compressed(
        output_dir / "preds.npz",
        variant_names=np.asarray(names, dtype=object),
        **{f"pred_{name}": latents["preds"][name] for name in names},
    )

    metrics_payload = {
        "n_samples": summary["n_samples"],
        "action_chunk_size": summary["action_chunk_size"],
        "metrics": summary["metrics"],
    }
    with (output_dir / "summary.json").open("w") as file:
        json.dump(metrics_payload, file, indent=2, sort_keys=True)
        file.write("\n")


def plot_action_change_hist(*, output_dir: Path, per_sample: dict[str, np.ndarray]) -> None:
    plt.figure(figsize=(7, 4))
    for name in sorted(per_sample):
        plt.hist(per_sample[name], bins=50, histtype="step", label=name)
    plt.title("Per-sample physical action L1 vs z = 0")
    plt.xlabel("mean physical action L1")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "action_change_hist.png")
    plt.close()


def plot_sensitivity_bar(*, output_dir: Path, metrics: dict[str, dict[str, float]]) -> None:
    names = sorted(metrics)
    joint = [metrics[name]["physical_joint"] for name in names]
    gripper = [metrics[name]["physical_gripper"] for name in names]

    positions = np.arange(len(names))
    width = 0.4
    plt.figure(figsize=(8, 4))
    plt.bar(positions - width / 2, joint, width, label="joint")
    plt.bar(positions + width / 2, gripper, width, label="gripper")
    plt.xticks(positions, names, rotation=30, ha="right")
    plt.title("Mean physical action L1 vs z = 0")
    plt.ylabel("physical L1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "sensitivity_bar.png")
    plt.close()


def save_metadata(
    *,
    output_dir: Path,
    run_name: str,
    args: argparse.Namespace,
    num_samples: int,
    dataset_num_samples: int,
    device: torch.device,
    timestamp: str,
) -> None:
    metadata = {
        "run_name": run_name,
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "num_samples": num_samples,
        "dataset_num_samples": dataset_num_samples,
        "limit": args.limit,
        "n_prior": args.n_prior,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": str(device),
        "timestamp": timestamp,
        "z_variants": variant_names(n_prior=args.n_prior),
        "baseline_variant": BASELINE_VARIANT,
    }
    with (output_dir / "meta.json").open("w") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)
        file.write("\n")


def main() -> None:
    args = parse_args()
    run_name = args.run_name or args.checkpoint.parent.name
    output_dir = make_output_dir(
        output_root=args.output_dir,
        run_name=run_name,
        checkpoint_path=args.checkpoint,
    )
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device => {device}")

    model, normalization = load_model(checkpoint_path=args.checkpoint, device=device)
    dataset = CanPhDataset(file=str(args.dataset), k=ACTION_CHUNK_SIZE)

    indices = make_subset_indices(num_samples=len(dataset), limit=args.limit, seed=args.seed)
    print(f"analyzing {len(indices)} / {len(dataset)} samples")

    latents = collect_sensitivity(
        model=model,
        dataset=dataset,
        indices=indices,
        n_prior=args.n_prior,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
    )

    summary = compute_summary(latents=latents, normalization=normalization)
    save_artifacts(output_dir=output_dir, latents=latents, summary=summary)
    plot_action_change_hist(
        output_dir=output_dir,
        per_sample=summary["per_sample_physical_total"],
    )
    plot_sensitivity_bar(output_dir=output_dir, metrics=summary["metrics"])

    save_metadata(
        output_dir=output_dir,
        run_name=run_name,
        args=args,
        num_samples=int(latents["mu"].shape[0]),
        dataset_num_samples=int(len(dataset)),
        device=device,
        timestamp=timestamp,
    )

    print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
