"""Measure whether the ACT v2 CVAE posterior encodes example-dependent information.

Item 1 of the z-debug investigation: load a checkpoint, run the training-time
posterior q(z | proprio, demonstrated_action_chunk) over a seeded,
demo-stratified subset of the dataset, and report

  * per-example KL to N(0, I) (distribution, not just the mean),
  * per-dimension KL and across-example mean/variance structure,
  * whether across-example differences exceed posterior sampling noise,
  * the aggregate posterior (does it match the prior?),
  * breakdowns by demonstration and task progress.

The posterior q(z | X) is conditioned on X = (proprio, action chunk).  KL to the
standard-normal prior is averaged over examples; that expected conditional KL
upper-bounds the mutual information I(X; Z) between the encoder inputs and a
sampled latent.  Using the identity

    E_X[KL(q(z|X) || p(z))] = I(X; Z) + KL(q(z) || p(z)),

we also report the mutual information directly from the aggregate posterior.
A uniformly tiny expected KL (and hence tiny I(X; Z)) is strong evidence of
posterior collapse.

Only the CVAE encoder is exercised: ``model.posterior`` skips the image encoder
and the action decoder.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.dataset import NormalizationStats, build_action_chunk, build_proprio
from scripts.models.act_v2.config import (
    ACTION_CHUNK_SIZE,
    D_MODEL,
    N_HEAD,
    NUM_LAYERS,
    PROPRIO_DIMS,
    Z_DIMS,
)
from scripts.models.act_v2.model import ACTV2
from scripts.utils.analyze_z import normalization_from_checkpoint

DEFAULT_OUTPUT_DIR = Path("debug/cvae-collapse")
DEFAULT_DATASET = Path("datasets/can/ph/2026-09-29_02-01-42_agentview_robot0_eye_in_hand.hdf5")
LOCAL_DATASET_ROOT = Path("datasets/can/ph")

# Per-example KL thresholds used to express "how collapsed" a run is.
KL_THRESHOLDS = (1e-3, 1e-2, 5e-2, 1e-1, 5e-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="ACT v2 .pt checkpoint")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=f"CAN PH HDF5 file; defaults to the checkpoint's dataset, else {DEFAULT_DATASET}",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--samples", type=int, default=5000, help="number of action chunks to analyze")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def resolve_dataset_path(*, checkpoint: dict, override: Path | None) -> Path:
    """Resolve the dataset path, mapping the cluster /workspace path to a local one."""
    if override is not None:
        return override

    stored = checkpoint.get("dataset")
    if isinstance(stored, str) and stored:
        stored_path = Path(stored)
        if stored_path.exists():
            return stored_path
        local_candidate = LOCAL_DATASET_ROOT / stored_path.name
        if local_candidate.exists():
            return local_candidate

    return DEFAULT_DATASET


def load_model_from_checkpoint(
    *,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[ACTV2, dict, NormalizationStats]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    train_config = checkpoint["config"]

    model = ACTV2(
        d_model=D_MODEL,
        nhead=N_HEAD,
        num_layers=NUM_LAYERS,
        z_dims=Z_DIMS,
        proprio_dims=PROPRIO_DIMS,
        action_chunk_size=ACTION_CHUNK_SIZE,
        use_z=bool(train_config.get("use_z", True)),
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device=device)
    model.eval()

    normalization = normalization_from_checkpoint(checkpoint=checkpoint)
    return model, train_config, normalization


def build_demo_lengths(*, dataset_path: Path) -> dict[str, int]:
    """Map demo name -> number of valid action-chunk start indices."""
    with h5py.File(dataset_path, "r") as hdf5:
        data = hdf5["data"]
        demo_names = sorted(data.keys(), key=lambda name: int(name.split("_")[1]))
        lengths: dict[str, int] = {}
        for demo_name in demo_names:
            num_timesteps = int(data[demo_name].attrs["num_samples"])
            lengths[demo_name] = num_timesteps - ACTION_CHUNK_SIZE
    return lengths


def sample_demo_stratified(
    *,
    demo_lengths: dict[str, int],
    total: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Sample start indices proportionally to each demonstration's length."""
    rng = np.random.default_rng(seed)
    demo_names = list(demo_lengths.keys())
    lengths = np.asarray([demo_lengths[name] for name in demo_names], dtype=np.int64)
    available = int(lengths.sum())
    target = min(total, available)

    quotas = lengths / lengths.sum() * target
    counts = np.floor(quotas).astype(np.int64)
    remainder = target - int(counts.sum())
    if remainder > 0:
        fractional = quotas - counts
        for index in np.argsort(-fractional)[:remainder]:
            counts[index] += 1
    counts = np.minimum(counts, lengths)

    sampled: dict[str, np.ndarray] = {}
    for demo_name, count, length in zip(demo_names, counts, lengths):
        if count <= 0:
            continue
        indices = rng.choice(length, size=int(count), replace=False)
        sampled[demo_name] = np.sort(indices)
    return sampled


def gather_posterior_inputs(
    *,
    hdf5: h5py.File,
    demo_name: str,
    timesteps: np.ndarray,
    normalization: NormalizationStats,
) -> tuple[np.ndarray, np.ndarray]:
    demo = hdf5[f"data/{demo_name}"]

    # Low-dimensional arrays are small; read each demo once and index in numpy
    # because h5py does not support fancy indexing with 2D index arrays.
    proprio = build_proprio(
        joint_pos=demo["obs/robot0_joint_pos"][:][timesteps],
        gripper_qpos=demo["obs/robot0_gripper_qpos"][:][timesteps],
    )
    action_offsets = timesteps[:, None] + np.arange(ACTION_CHUNK_SIZE)
    actions = build_action_chunk(
        joint_pos=demo["next_obs/robot0_joint_pos"][:][action_offsets],
        gripper_qpos=demo["next_obs/robot0_gripper_qpos"][:][action_offsets],
    )

    proprio = normalization.normalize_proprio(value=proprio)
    actions = normalization.normalize_action(value=actions)
    return proprio, actions


def collect_posterior(
    *,
    model: ACTV2,
    dataset_path: Path,
    sampled: dict[str, np.ndarray],
    normalization: NormalizationStats,
    batch_size: int,
    device: torch.device,
) -> dict:
    """Run the posterior over the sampled action chunks and gather encoder outputs."""
    demo_names: list[str] = []
    timesteps: list[int] = []
    mu_rows: list[np.ndarray] = []
    log_sigma_x2_rows: list[np.ndarray] = []

    with h5py.File(dataset_path, "r") as hdf5, torch.no_grad():
        for demo_name, demo_timesteps in sampled.items():
            for start in range(0, len(demo_timesteps), batch_size):
                chunk = demo_timesteps[start : start + batch_size]
                proprio, actions = gather_posterior_inputs(
                    hdf5=hdf5,
                    demo_name=demo_name,
                    timesteps=chunk,
                    normalization=normalization,
                )
                proprio_t = torch.from_numpy(proprio).unsqueeze(1).to(device=device)
                actions_t = torch.from_numpy(actions).to(device=device)

                mu, log_sigma_x2 = model.posterior(proprio=proprio_t, actions=actions_t)

                mu_rows.append(mu.squeeze(1).detach().cpu().numpy())
                log_sigma_x2_rows.append(log_sigma_x2.squeeze(1).detach().cpu().numpy())
                demo_names.extend([demo_name] * len(chunk))
                timesteps.extend(int(timestep) for timestep in chunk)

    return {
        "demo_name": np.asarray(demo_names, dtype=object),
        "timestep": np.asarray(timesteps, dtype=np.int64),
        "mu": np.concatenate(mu_rows, axis=0).astype(np.float64),
        "log_sigma_x2": np.concatenate(log_sigma_x2_rows, axis=0).astype(np.float64),
    }


def per_example_kl(*, mu: np.ndarray, log_sigma_x2: np.ndarray) -> np.ndarray:
    """KL(q(z|X) || N(0, I)) per example, summed over latent dimensions.

    Uses ``expm1`` for numerical stability near sigma^2 = 1.
    """
    return 0.5 * np.sum(mu**2 + np.expm1(log_sigma_x2) - log_sigma_x2, axis=1)


def aggregate_posterior_kl(*, mu: np.ndarray, log_sigma_x2: np.ndarray) -> tuple[float, np.ndarray]:
    """KL(q(z) || N(0, I)) for the aggregate posterior q(z) = E_X[q(z|X)].

    Returns the scalar KL and the aggregate per-dimension mean/variance.  The
    aggregate variance is computed with the law of total variance.
    """
    aggregate_mean = mu.mean(axis=0)
    aggregate_var = (mu**2 + np.exp(log_sigma_x2)).mean(axis=0) - aggregate_mean**2
    aggregate_var = np.maximum(aggregate_var, 1e-12)
    kl = 0.5 * np.sum(aggregate_mean**2 + aggregate_var - 1.0 - np.log(aggregate_var))
    return float(kl), aggregate_var


def compute_stats(*, latents: dict, demo_lengths: dict[str, int]) -> dict:
    mu = latents["mu"]
    log_sigma_x2 = latents["log_sigma_x2"]
    num_examples, num_dims = mu.shape

    kl_per_example = per_example_kl(mu=mu, log_sigma_x2=log_sigma_x2)
    kl_per_dimension = 0.5 * np.mean(mu**2 + np.expm1(log_sigma_x2) - log_sigma_x2, axis=0)

    variance_of_mean = mu.var(axis=0)
    mean_posterior_variance = np.exp(log_sigma_x2).mean(axis=0)
    # Between-example mean variation relative to within-posterior sampling noise.
    signal_to_noise = variance_of_mean / np.maximum(mean_posterior_variance, 1e-12)
    log_sigma_variation = log_sigma_x2.std(axis=0)

    aggregate_kl, aggregate_var = aggregate_posterior_kl(mu=mu, log_sigma_x2=log_sigma_x2)
    expected_kl = float(kl_per_example.mean())
    # Exact decomposition of the expected conditional KL.  The aggregate
    # posterior is approximated as a Gaussian with matched mean/variance, so
    # the difference is clamped at zero to absorb floating-point cancellation.
    mutual_information = max(0.0, expected_kl - aggregate_kl)

    progress = np.asarray(
        [
            timestep / max(demo_lengths[demo_name] - 1, 1)
            for demo_name, timestep in zip(latents["demo_name"], latents["timestep"])
        ],
        dtype=np.float64,
    )

    demo_kl: dict[str, float] = {}
    for demo_name in sorted(set(latents["demo_name"].tolist())):
        mask = latents["demo_name"] == demo_name
        demo_kl[demo_name] = float(kl_per_example[mask].mean())

    progress_bins = np.linspace(0.0, 1.0, 6)
    kl_by_progress: list[float] = []
    for lower, upper in zip(progress_bins[:-1], progress_bins[1:]):
        mask = (progress >= lower) & (progress < upper if upper < 1.0 else progress <= upper)
        kl_by_progress.append(float(kl_per_example[mask].mean()) if mask.any() else float("nan"))

    return {
        "num_examples": int(num_examples),
        "num_dims": int(num_dims),
        "expected_conditional_kl": expected_kl,
        "mutual_information_nats": float(mutual_information),
        "mutual_information_bits": float(mutual_information / np.log(2.0)),
        "aggregate_kl": aggregate_kl,
        "kl_quantiles": {
            "min": float(np.min(kl_per_example)),
            "p25": float(np.percentile(kl_per_example, 25)),
            "median": float(np.median(kl_per_example)),
            "p75": float(np.percentile(kl_per_example, 75)),
            "p90": float(np.percentile(kl_per_example, 90)),
            "p99": float(np.percentile(kl_per_example, 99)),
            "max": float(np.max(kl_per_example)),
        },
        "fraction_examples_below_kl": {
            f"{threshold:g}": float(np.mean(kl_per_example < threshold)) for threshold in KL_THRESHOLDS
        },
        "fraction_dims_below_kl": {
            f"{threshold:g}": float(np.mean(kl_per_dimension < threshold)) for threshold in KL_THRESHOLDS
        },
        "kl_per_dimension": kl_per_dimension.tolist(),
        "signal_to_noise_per_dimension": signal_to_noise.tolist(),
        "variance_of_mean_per_dimension": variance_of_mean.tolist(),
        "mean_posterior_variance_per_dimension": mean_posterior_variance.tolist(),
        "log_sigma_variation_per_dimension": log_sigma_variation.tolist(),
        "aggregate_posterior_variance_per_dimension": aggregate_var.tolist(),
        "demo_kl": demo_kl,
        "demo_kl_spread": {
            "min": float(min(demo_kl.values())),
            "median": float(np.median(list(demo_kl.values()))),
            "max": float(max(demo_kl.values())),
        },
        "kl_by_progress_bin": kl_by_progress,
        "progress_bin_edges": progress_bins.tolist(),
    }


def make_plots(*, stats: dict, kl_per_example: np.ndarray, output_dir: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(np.log10(np.maximum(kl_per_example, 1e-12)), bins=60)
    plt.title("Per-example KL(q(z|X) || N(0,I))")
    plt.xlabel("log10(KL, nats)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(output_dir / "kl_per_example_hist.png")
    plt.close()

    kl_per_dimension = np.asarray(stats["kl_per_dimension"])
    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(len(kl_per_dimension)), kl_per_dimension)
    plt.title("Mean KL per latent dimension")
    plt.xlabel("latent dim")
    plt.ylabel("mean KL (nats)")
    plt.tight_layout()
    plt.savefig(output_dir / "kl_per_dimension.png")
    plt.close()

    variance_of_mean = np.asarray(stats["variance_of_mean_per_dimension"])
    mean_posterior_variance = np.asarray(stats["mean_posterior_variance_per_dimension"])
    plt.figure(figsize=(6, 4))
    plt.scatter(mean_posterior_variance, np.maximum(variance_of_mean, 1e-12), s=16)
    plt.xscale("log")
    plt.yscale("log")
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    plt.title("Between-example mean variance vs posterior noise")
    plt.xlabel("mean posterior variance E[sigma^2]")
    plt.ylabel("Var_X(mu)")
    plt.tight_layout()
    plt.savefig(output_dir / "variance_vs_noise.png")
    plt.close()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, train_config, normalization = load_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        device=device,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    dataset_path = resolve_dataset_path(checkpoint=checkpoint, override=args.dataset)

    run_name = checkpoint.get("run_name", args.checkpoint.parent.name)
    output_dir = args.output_dir / run_name / args.checkpoint.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    demo_lengths = build_demo_lengths(dataset_path=dataset_path)
    sampled = sample_demo_stratified(
        demo_lengths=demo_lengths,
        total=args.samples,
        seed=args.seed,
    )
    num_sampled = int(sum(len(indices) for indices in sampled.values()))

    print(f"checkpoint       => {args.checkpoint}")
    print(f"global_step      => {checkpoint.get('global_step')}")
    print(f"use_z            => {train_config.get('use_z')}")
    print(f"image_keys       => {train_config.get('image_keys')}")
    print(f"dataset          => {dataset_path}")
    print(f"demos            => {len(demo_lengths)}")
    print(f"sampled examples => {num_sampled} (requested {args.samples}, seed {args.seed})")
    print(f"device           => {device}")

    latents = collect_posterior(
        model=model,
        dataset_path=dataset_path,
        sampled=sampled,
        normalization=normalization,
        batch_size=args.batch_size,
        device=device,
    )
    stats = compute_stats(latents=latents, demo_lengths=demo_lengths)

    summary = {
        "run_name": run_name,
        "checkpoint": str(args.checkpoint),
        "global_step": checkpoint.get("global_step"),
        "use_z": bool(train_config.get("use_z", True)),
        "image_keys": train_config.get("image_keys"),
        "dataset": str(dataset_path),
        "samples": num_sampled,
        "seed": args.seed,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stats": stats,
    }

    with open(output_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    np.savez_compressed(
        output_dir / "latents.npz",
        demo_name=latents["demo_name"],
        timestep=latents["timestep"],
        mu=latents["mu"],
        log_sigma_x2=latents["log_sigma_x2"],
        kl_per_example=per_example_kl(mu=latents["mu"], log_sigma_x2=latents["log_sigma_x2"]),
    )
    make_plots(
        stats=stats,
        kl_per_example=per_example_kl(mu=latents["mu"], log_sigma_x2=latents["log_sigma_x2"]),
        output_dir=output_dir,
    )

    print("\n=== posterior-collapse diagnostics ===")
    print(f"expected conditional KL : {stats['expected_conditional_kl']:.6f} nats")
    print(f"mutual information I(X;Z): {stats['mutual_information_nats']:.6f} nats "
          f"({stats['mutual_information_bits']:.6f} bits)")
    print(f"aggregate KL(q(z)||p)    : {stats['aggregate_kl']:.6f} nats")
    print(f"KL quantiles             : {stats['kl_quantiles']}")
    print(f"fraction of examples with KL < threshold: {stats['fraction_examples_below_kl']}")
    print(f"fraction of dims with mean KL < threshold: {stats['fraction_dims_below_kl']}")
    print(f"KL by progress bin       : {[round(value, 5) for value in stats['kl_by_progress_bin']]}")
    print(f"per-demo mean KL spread  : {stats['demo_kl_spread']}")
    print(f"\nwrote artifacts to {output_dir}")


if __name__ == "__main__":
    main()
