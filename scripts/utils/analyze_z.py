"""Measure how the ACT v2 CVAE latent varies across dataset samples.

Loads an existing ACT v2 checkpoint and runs the training-time forward pass over
a seeded random subset of CanPhDataset samples, recording the CVAE encoder
outputs (mu, log sigma^2) per sample alongside (demo_name, timestep).

The deterministic latent is z = mu (epsilon = 0), per notes/z-debug.md.
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

from scripts.dataset import CanPhDataset
from scripts.models.act_v2.config import (
    ACTION_CHUNK_SIZE,
    D_MODEL,
    N_HEAD,
    NUM_LAYERS,
    PROPRIO_DIMS,
    Z_DIMS,
)
from scripts.models.act_v2.model import ACTV2

DEFAULT_DATASET = Path("datasets/can/ph/2026-09-19_03-14-50_act_agentview.hdf5")
DEFAULT_OUTPUT_DIR = Path("debug/z-debug")


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
    parser.add_argument("--limit", type=int, default=5000, help="number of samples to analyze (<=0 for all)")
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_model(*, checkpoint_path: Path, device: torch.device) -> ACTV2:
    model = ACTV2(
        d_model=D_MODEL,
        nhead=N_HEAD,
        num_layers=NUM_LAYERS,
        z_dims=Z_DIMS,
        proprio_dims=PROPRIO_DIMS,
        action_chunk_size=ACTION_CHUNK_SIZE,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device=device)
    model.eval()
    return model


def make_subset_indices(*, num_samples: int, limit: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    size = num_samples if limit <= 0 else min(limit, num_samples)
    indices = rng.choice(num_samples, size=size, replace=False)
    return np.sort(indices)


def per_row_kl(*, mu: np.ndarray, log_sigma_x2: np.ndarray) -> np.ndarray:
    d = mu.shape[1]
    sum_mu_sq = np.sum(mu**2, axis=1)
    sum_sigma = np.sum(np.exp(log_sigma_x2), axis=1)
    sum_log_sigma = np.sum(log_sigma_x2, axis=1)
    return 0.5 * (sum_mu_sq + sum_sigma - d - sum_log_sigma)


def collect_latents(*, model: ACTV2, dataset: CanPhDataset, indices: np.ndarray, batch_size: int, device: torch.device) -> dict:
    subset = Subset(dataset, indices.tolist())
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)

    demo_names: list[str] = []
    timesteps: list[int] = []
    mu_rows: list[np.ndarray] = []
    log_sigma_x2_rows: list[np.ndarray] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="forward"):
            proprio = batch["proprio"].to(device=device)
            actions = batch["target_actions"].to(device=device)
            img = batch["image"].to(device=device)

            _, mu, log_sigma_x2 = model(proprio=proprio, actions=actions, img=img)

            mu_rows.append(mu.squeeze(1).detach().cpu().numpy())
            log_sigma_x2_rows.append(log_sigma_x2.squeeze(1).detach().cpu().numpy())
            demo_names.extend(batch["demo_name"])
            timesteps.extend(int(t) for t in batch["timestep"])

    mu = np.concatenate(mu_rows, axis=0)
    log_sigma_x2 = np.concatenate(log_sigma_x2_rows, axis=0)
    return {
        "demo_name": np.asarray(demo_names, dtype=object),
        "timestep": np.asarray(timesteps, dtype=np.int64),
        "mu": mu,
        "log_sigma_x2": log_sigma_x2,
    }


def save_artifact(*, output_dir: Path, latents: dict) -> None:
    mu = latents["mu"]
    log_sigma_x2 = latents["log_sigma_x2"]
    mu_norm = np.linalg.norm(mu, axis=1)
    mean_var = np.exp(log_sigma_x2).mean(axis=1)
    kl = per_row_kl(mu=mu, log_sigma_x2=log_sigma_x2)

    np.savez_compressed(
        output_dir / "latents.npz",
        demo_name=latents["demo_name"],
        timestep=latents["timestep"],
        mu=mu,
        log_sigma_x2=log_sigma_x2,
        mu_norm=mu_norm,
        mean_var=mean_var,
        kl=kl,
    )


def demo_codes(*, demo_names: np.ndarray) -> np.ndarray:
    unique = sorted(set(demo_names.tolist()), key=lambda name: int(name.split("_")[1]))
    mapping = {name: i for i, name in enumerate(unique)}
    return np.asarray([mapping[name] for name in demo_names])


def plot_histogram(*, values: np.ndarray, path: Path, title: str, xlabel: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=60)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_pca(*, mu: np.ndarray, demo_names: np.ndarray, path: Path) -> None:
    centered = mu - mu.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T

    codes = demo_codes(demo_names=demo_names)
    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=codes, cmap="viridis", s=6, alpha=0.7)
    plt.colorbar(scatter, label="demo index")
    plt.title("PCA of mu, colored by demo")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_dim_boxplot(*, mu: np.ndarray, path: Path) -> None:
    plt.figure(figsize=(10, 4))
    plt.boxplot([mu[:, d] for d in range(mu.shape[1])])
    plt.title("mu per latent dimension")
    plt.xlabel("latent dim")
    plt.ylabel("mu")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def make_plots(*, output_dir: Path, latents: dict) -> None:
    mu = latents["mu"]
    mu_norm = np.linalg.norm(mu, axis=1)
    mean_var = np.exp(latents["log_sigma_x2"]).mean(axis=1)

    plot_histogram(
        values=mu_norm,
        path=output_dir / "mu_norm_hist.png",
        title="Histogram of ||mu||",
        xlabel="||mu||",
    )
    plot_histogram(
        values=mean_var,
        path=output_dir / "mean_var_hist.png",
        title="Histogram of mean exp(log sigma^2)",
        xlabel="mean variance",
    )
    plot_pca(mu=mu, demo_names=latents["demo_name"], path=output_dir / "mu_pca.png")
    plot_dim_boxplot(mu=mu, path=output_dir / "mu_boxplot.png")


def compute_summary(*, latents: dict) -> dict:
    mu = latents["mu"]
    log_sigma_x2 = latents["log_sigma_x2"]
    demo_names = latents["demo_name"]

    dim_std = mu.std(axis=0)
    near_zero_dims = int((dim_std < 0.01).sum())

    per_demo_means = {}
    per_demo_stds = {}
    for i, name in enumerate(demo_names):
        per_demo_means.setdefault(name, []).append(mu[i])
    for name, rows in per_demo_means.items():
        arr = np.stack(rows)
        per_demo_stds[name] = arr.std(axis=0)

    within_demo_std = np.stack(list(per_demo_stds.values())).mean(axis=0).mean()
    demo_mean_std = np.stack([arr.mean(axis=0) for arr in per_demo_stds.values()]).std(axis=0).mean()

    return {
        "n_samples": int(mu.shape[0]),
        "z_dims": int(mu.shape[1]),
        "mu_norm_mean": float(np.linalg.norm(mu, axis=1).mean()),
        "mu_norm_std": float(np.linalg.norm(mu, axis=1).std()),
        "mean_var_mean": float(np.exp(log_sigma_x2).mean()),
        "mean_var_std": float(np.exp(log_sigma_x2).mean(axis=1).std()),
        "kl_mean": float(per_row_kl(mu=mu, log_sigma_x2=log_sigma_x2).mean()),
        "dim_std_min": float(dim_std.min()),
        "dim_std_max": float(dim_std.max()),
        "dim_std_mean": float(dim_std.mean()),
        "near_zero_std_dims": near_zero_dims,
        "within_demo_std_mean": float(within_demo_std),
        "across_demo_std_mean": float(demo_mean_std),
    }


def make_output_dir(*, output_root: Path, run_name: str, checkpoint_path: Path) -> Path:
    output_dir = output_root / run_name / checkpoint_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_metadata(
    *,
    output_dir: Path,
    run_name: str,
    checkpoint_path: Path,
    dataset_path: Path,
    num_samples: int,
    dataset_num_samples: int,
    limit: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    timestamp: str,
) -> None:
    metadata = {
        "run_name": run_name,
        "checkpoint": str(checkpoint_path),
        "dataset": str(dataset_path),
        "num_samples": num_samples,
        "dataset_num_samples": dataset_num_samples,
        "limit": limit,
        "batch_size": batch_size,
        "seed": seed,
        "device": str(device),
        "timestamp": timestamp,
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

    model = load_model(checkpoint_path=args.checkpoint, device=device)
    dataset = CanPhDataset(file=str(args.dataset), k=ACTION_CHUNK_SIZE)

    indices = make_subset_indices(num_samples=len(dataset), limit=args.limit, seed=args.seed)
    print(f"analyzing {len(indices)} / {len(dataset)} samples")

    latents = collect_latents(
        model=model,
        dataset=dataset,
        indices=indices,
        batch_size=args.batch_size,
        device=device,
    )

    save_artifact(output_dir=output_dir, latents=latents)
    make_plots(output_dir=output_dir, latents=latents)

    summary = compute_summary(latents=latents)
    with (output_dir / "summary.json").open("w") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
        file.write("\n")

    save_metadata(
        output_dir=output_dir,
        run_name=run_name,
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset,
        num_samples=int(len(indices)),
        dataset_num_samples=int(len(dataset)),
        limit=args.limit,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
        timestamp=timestamp,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
