"""Training script for ACT v2 (CVAE + action chunking)."""

import argparse
import json
import tomllib
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from scripts.dataset import CanPhDataset, NormalizationStats
from scripts.models.act_v2.config import (
    ACTION_CHUNK_SIZE,
    D_MODEL,
    JOINT_DIMS,
    N_HEAD,
    NUM_LAYERS,
    PROPRIO_DIMS,
    Z_DIMS,
)
from scripts.models.act_v2.model import ACTV2
from scripts.train_v1 import (
    compute_adam_moment_norms,
    compute_global_l2_norm,
    compute_global_update_norm,
    make_dataloader_generator,
    make_dataloader_worker_init_fn,
    seed_everything,
    snapshot_parameters,
)

RUNS_ROOT = Path("runs/act_v2")
CHECKPOINTS_ROOT = Path("checkpoints/act_v2")
DEFAULT_DATASET = Path(
    "datasets/can/ph/2026-09-29_02-01-42_agentview_robot0_eye_in_hand.hdf5"
)
SENSITIVITY_PROBE_SIZE = 16
CONFIG_VERSION = "v3"

CONFIG_KEYS = (
    "action_loss",
    "batch_size",
    "steps",
    "image_keys",
    "lr",
    "seed",
    "beta",
    "checkpoint_every",
    "version",
)

ACTION_LOSS_CHOICES = frozenset({"l1", "l2"})

ACTIVATION_HOOK_TAGS = (
    "image_encoder",
    "cvae_encoder",
    "transformer_encoder",
    "transformer_decoder",
    "action_head",
)

RUN_NAME_FIELDS = (
    ("batch_size", "bs", "{:d}"),
    ("lr", "lr", "{:.0e}"),
    ("beta", "beta", "{:g}"),
    ("beta_warmup_steps", "wu", "{:d}"),
    ("steps", "st", "{:d}"),
    ("image_keys", "img", "{:s}"),
    ("use_z", "use_z", "{:d}"),
    ("action_loss", "", "{:s}"),
)

RUN_NAME_OMIT_KEYS = frozenset({
    "seed",
    "checkpoint_every",
    "beta_start",
    "version",
})


def validate_config(*, config: dict) -> dict:
    missing = [key for key in CONFIG_KEYS if key not in config]
    if missing:
        raise KeyError(f"config missing required keys: {missing}")

    version = config["version"]
    if version != CONFIG_VERSION:
        raise ValueError(
            f"config version must be {CONFIG_VERSION!r}, got {version!r}"
        )

    config.setdefault("beta_start", 0.0)
    config.setdefault("beta_warmup_steps", 0)
    config.setdefault("use_z", True)

    if config["steps"] <= 0:
        raise ValueError(f"steps must be > 0, got {config['steps']}")
    if config["checkpoint_every"] <= 0:
        raise ValueError(
            f"checkpoint_every must be > 0, got {config['checkpoint_every']}"
        )
    if config["beta_warmup_steps"] < 0:
        raise ValueError(
            f"beta_warmup_steps must be >= 0, got {config['beta_warmup_steps']}"
        )
    if config["beta_start"] < 0.0:
        raise ValueError(f"beta_start must be >= 0, got {config['beta_start']}")

    action_loss = config["action_loss"]
    if action_loss not in ACTION_LOSS_CHOICES:
        allowed = ", ".join(sorted(ACTION_LOSS_CHOICES))
        raise ValueError(f"action_loss must be one of {{{allowed}}}, got {action_loss!r}")

    image_keys = config["image_keys"]
    if not isinstance(image_keys, list) or len(image_keys) == 0:
        raise ValueError("image_keys must be a non-empty list of HDF5 obs key names")
    if not all(isinstance(key, str) and key for key in image_keys):
        raise ValueError("image_keys must contain only non-empty strings")

    return config


def load_config(*, path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")

    with path.open("rb") as file:
        config = tomllib.load(file)

    return validate_config(config=config)


def make_action_loss_fn(*, action_loss: str) -> nn.Module:
    if action_loss == "l1":
        return nn.L1Loss(reduction="mean")
    if action_loss == "l2":
        return nn.MSELoss(reduction="mean")
    allowed = ", ".join(sorted(ACTION_LOSS_CHOICES))
    raise ValueError(f"action_loss must be one of {{{allowed}}}, got {action_loss!r}")


def beta_at_step(
    *,
    step: int,
    beta_start: float,
    beta: float,
    beta_warmup_steps: int) -> float:
    if beta_warmup_steps <= 0 or step >= beta_warmup_steps:
        return beta
    progress = step / beta_warmup_steps
    return beta_start + (beta - beta_start) * progress


def save_run_config(
    *,
    run_dir: Path,
    run_name: str,
    config: dict) -> None:
    payload = {"run_name": run_name, "config": config}
    path = run_dir / "config.json"
    with path.open("w") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")


def format_config_value(
    *,
    value: object,
    template: str) -> str:
    if isinstance(value, bool):
        value = int(value)
    if isinstance(value, (list, tuple)):
        value = "-".join(str(item) for item in value)
    rendered = template.format(value)
    return "".join(
        character if character.isalnum() or character in "._-" else "-"
        for character in rendered
    )


def make_run_name(*, config: dict) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parts = [timestamp]
    handled_keys: set[str] = set(RUN_NAME_OMIT_KEYS)
    for key, label, template in RUN_NAME_FIELDS:
        if key not in config:
            continue
        handled_keys.add(key)
        rendered = format_config_value(value=config[key], template=template)
        parts.append(f"{label}{rendered}" if label else rendered)
    for key in sorted(key for key in config if key not in handled_keys):
        rendered = format_config_value(value=config[key], template="{}")
        parts.append(f"{key}-{rendered}")
    return "_".join(parts)


def register_activation_norm_hooks(*, model: ACTV2) -> tuple[dict[str, float], list]:
    norms: dict[str, float] = {}
    hook_targets = {
        "image_encoder": model.image_encoder,
        "cvae_encoder": model.cvae_encoder,
        "transformer_encoder": model.transformer_encoder,
        "transformer_decoder": model.transformer_decoder,
        "action_head": model.action_head,
    }
    handles = []
    for tag, module in hook_targets.items():
        def make_hook(name: str):
            def hook(_module, _inputs, output) -> None:
                tensor = output[0] if isinstance(output, tuple) else output
                norms[name] = tensor.detach().float().norm().item()

            return hook

        handles.append(module.register_forward_hook(make_hook(tag)))
    return norms, handles


def get_kl_loss(
    *,
    mu: torch.Tensor,
    log_sigma_x2: torch.Tensor) -> torch.Tensor:
    _, _, d = mu.shape

    sum_mu_x2 = torch.sum(mu.pow(2), dim=2)
    sum_sigma_x2 = torch.sum(log_sigma_x2.exp(), dim=2)
    sum_log_sigma_x2 = torch.sum(log_sigma_x2, dim=2)

    kl_loss = 0.5 * (sum_mu_x2 + sum_sigma_x2 - d - sum_log_sigma_x2)

    return kl_loss.mean()


def measure_observation_sensitivity(
    *, model: ACTV2, images: torch.Tensor, proprio: torch.Tensor) -> tuple[float, float]:
    """Measure normalized action changes from swapping one observation modality."""
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            baseline = model.infer(proprio=proprio, img=images)
            swapped_images = model.infer(proprio=proprio, img=images.roll(1, dims=0))
            swapped_proprio = model.infer(proprio=proprio.roll(1, dims=0), img=images)
            image_l1 = (baseline - swapped_images).abs().mean().item()
            proprio_l1 = (baseline - swapped_proprio).abs().mean().item()
    finally:
        model.train(was_training)

    return image_l1, proprio_l1


def checkpoint_path_for_step(*, checkpoint_dir: Path, global_step: int) -> Path:
    return checkpoint_dir / f"step_{global_step:09d}.pt"


def save_checkpoint(
    *,
    path: Path,
    run_name: str,
    config: dict,
    dataset: str,
    epoch: int,
    global_step: int,
    model: ACTV2,
    loss_running_avg: float,
    normalization: NormalizationStats) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "run_name": run_name,
            "config": dict(config),
            "dataset": dataset,
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "loss_running_avg": loss_running_avg,
            "normalization": normalization.as_checkpoint_dict(),
        },
        path,
    )


def maybe_save_step_checkpoint(
    *,
    checkpoint_dir: Path,
    checkpoint_every: int,
    run_name: str,
    config: dict,
    dataset: str,
    epoch: int,
    global_step: int,
    model: ACTV2,
    loss_running_avg: float,
    normalization: NormalizationStats) -> Path | None:
    if global_step <= 0 or global_step % checkpoint_every != 0:
        return None
    snapshot_path = checkpoint_path_for_step(
        checkpoint_dir=checkpoint_dir,
        global_step=global_step,
    )
    save_checkpoint(
        path=snapshot_path,
        run_name=run_name,
        config=config,
        dataset=dataset,
        epoch=epoch,
        global_step=global_step,
        model=model,
        loss_running_avg=loss_running_avg,
        normalization=normalization,
    )
    return snapshot_path


def train(
    *,
    config: dict,
    dataset: Path) -> None:
    seed = config["seed"]
    batch_size = config["batch_size"]
    lr = config["lr"]

    seed_everything(seed=seed)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for training but PyTorch could not initialize it. "
            "Verify the NVIDIA driver and CUDA_VISIBLE_DEVICES before rerunning."
        )
    device = torch.device("cuda")

    can_ph_dataset = CanPhDataset(
        file=str(dataset),
        image_keys=tuple(config["image_keys"]),
        k=ACTION_CHUNK_SIZE,
    )
    dataloader_generator = make_dataloader_generator(seed=seed)
    can_ph_dataloader = DataLoader(
        dataset=can_ph_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=device.type == "cuda",
        generator=dataloader_generator,
        worker_init_fn=make_dataloader_worker_init_fn(base_seed=seed),
    )

    model = ACTV2(
        d_model=D_MODEL,
        nhead=N_HEAD,
        num_layers=NUM_LAYERS,
        z_dims=Z_DIMS,
        proprio_dims=PROPRIO_DIMS,
        action_chunk_size=ACTION_CHUNK_SIZE,
        use_z=config["use_z"],
    ).to(device=device)

    action_loss_kind = config["action_loss"]
    action_loss_fn = make_action_loss_fn(action_loss=action_loss_kind)
    denorm_l1_fn = nn.L1Loss(reduction="mean")
    optimizer = optim.Adam(params=model.parameters(), lr=lr, betas=(0.9, 0.999))
    model_parameters = list(model.parameters())
    activation_norms, activation_hook_handles = register_activation_norm_hooks(model=model)
    normalization = can_ph_dataset.normalization
    action_mean = torch.from_numpy(normalization.action_mean).to(device=device).view(1, 1, -1)
    action_std = torch.from_numpy(normalization.action_std).to(device=device).view(1, 1, -1)

    target_steps = config["steps"]
    global_step = 0

    beta = config["beta"]
    beta_start = config["beta_start"]
    beta_warmup_steps = config["beta_warmup_steps"]
    checkpoint_every = config["checkpoint_every"]

    run_name = make_run_name(config=config)
    dataset_path = str(dataset.resolve())
    run_dir = RUNS_ROOT / run_name
    checkpoint_dir = CHECKPOINTS_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(run_dir=run_dir, run_name=run_name, config=config)
    writer = SummaryWriter(log_dir=str(run_dir))
    print(f"seed => {seed}")
    print(f"device => {device}")
    print(f"gpu => {torch.cuda.get_device_name(device)}")
    print(f"tensorboard logs => {run_dir.resolve()}")
    print(f"checkpoints => {checkpoint_dir.resolve()}")
    if beta_warmup_steps > 0:
        print(
            f"beta schedule => linear warmup from {beta_start:g} to {beta:g} "
            f"over steps 0–{beta_warmup_steps - 1}"
        )
    else:
        print(f"beta => {beta:g}")
    print(f"target_steps => {target_steps}")
    print(f"checkpoint_every => {checkpoint_every} steps")
    print(f"action_loss => {action_loss_kind}")
    print(f"use_z => {config['use_z']}")

    probe_images = None
    probe_proprio = None
    last_loss_running_avg = 0.0
    epoch = 0
    while global_step < target_steps:
        epoch_loss = 0.0
        epoch_action_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_weighted_kl_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            can_ph_dataloader,
            desc=f"epoch {epoch + 1} step {global_step}/{target_steps}",
        )
        for batch_dict in pbar:
            beta_t = beta_at_step(
                step=global_step,
                beta_start=beta_start,
                beta=beta,
                beta_warmup_steps=beta_warmup_steps,
            )
            optimizer.zero_grad()

            img = batch_dict["images"].to(device, non_blocking=True)
            proprio = batch_dict["proprio"].to(device, non_blocking=True)
            actions = batch_dict["target_actions"].to(device, non_blocking=True)
            if probe_images is None:
                # Fix real observations from the first batch for every epoch.
                probe_images = img[:SENSITIVITY_PROBE_SIZE].detach().clone()
                probe_proprio = proprio[:SENSITIVITY_PROBE_SIZE].detach().clone()

            pred_actions, mu, log_sigma_x2 = model(proprio=proprio, actions=actions, img=img)

            action_loss = action_loss_fn(pred_actions, actions)
            kl_loss = get_kl_loss(mu=mu, log_sigma_x2=log_sigma_x2)
            loss = action_loss + beta_t * kl_loss

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float("inf"),
            ).item()
            params_before = snapshot_parameters(parameters=model_parameters)
            optimizer.step()

            batch_loss = loss.item()
            batch_action_loss = action_loss.item()
            batch_kl_loss = kl_loss.item()
            batch_weighted_kl_loss = beta_t * batch_kl_loss

            epoch_loss += batch_loss
            epoch_action_loss += batch_action_loss
            epoch_kl_loss += batch_kl_loss
            epoch_weighted_kl_loss += batch_weighted_kl_loss
            num_batches += 1
            pbar.set_postfix(loss=f"{batch_loss:.4f}")

            with torch.no_grad():
                learning_rate = optimizer.param_groups[0]["lr"]
                adam_exp_avg_norm, adam_exp_avg_sq_norm = compute_adam_moment_norms(
                    optimizer=optimizer,
                )
                param_norm = compute_global_l2_norm(tensors=model_parameters)
                update_norm = compute_global_update_norm(
                    parameters=model_parameters,
                    before=params_before,
                )

                pred_physical = pred_actions.detach() * action_std + action_mean
                target_physical = actions * action_std + action_mean
                pred_joint = pred_physical[..., :JOINT_DIMS]
                pred_gripper = pred_physical[..., JOINT_DIMS:]
                target_joint = target_physical[..., :JOINT_DIMS]
                target_gripper = target_physical[..., JOINT_DIMS:]

                batch_l1_joint = denorm_l1_fn(pred_joint, target_joint).item()
                batch_l1_gripper = denorm_l1_fn(pred_gripper, target_gripper).item()

                pred_joint_min = pred_joint.min().item()
                pred_joint_max = pred_joint.max().item()
                pred_gripper_min = pred_gripper.min().item()
                pred_gripper_max = pred_gripper.max().item()
                target_joint_min = target_joint.min().item()
                target_joint_max = target_joint.max().item()
                target_gripper_min = target_gripper.min().item()
                target_gripper_max = target_gripper.max().item()

                mu_norm = mu.detach().float().norm().item()
                log_sigma_x2_mean = log_sigma_x2.detach().mean().item()
                sigma_mean = log_sigma_x2.detach().exp().mean().item()

                if batch_loss > 0.0:
                    batch_kl_fraction = batch_weighted_kl_loss / batch_loss
                else:
                    batch_kl_fraction = 0.0

            writer.add_scalar("batch_metrics/loss", batch_loss, global_step)
            writer.add_scalar("batch_metrics/action_loss", batch_action_loss, global_step)
            writer.add_scalar("batch_metrics/kl_loss", batch_kl_loss, global_step)
            writer.add_scalar("batch_metrics/weighted_kl_loss", batch_weighted_kl_loss, global_step)
            writer.add_scalar("batch_metrics/kl_fraction", batch_kl_fraction, global_step)
            writer.add_scalar("hyperparams/beta", beta_t, global_step)
            writer.add_scalar("optimizer/grad_norm_global", grad_norm, global_step)
            writer.add_scalar("optimizer/param_norm_global", param_norm, global_step)
            writer.add_scalar("optimizer/update_norm_global", update_norm, global_step)
            writer.add_scalar("optimizer/lr", learning_rate, global_step)
            writer.add_scalar("optimizer/adam_exp_avg_norm", adam_exp_avg_norm, global_step)
            writer.add_scalar("optimizer/adam_exp_avg_sq_norm", adam_exp_avg_sq_norm, global_step)
            writer.add_scalar("denorm_l1/joint", batch_l1_joint, global_step)
            writer.add_scalar("denorm_l1/gripper", batch_l1_gripper, global_step)
            writer.add_scalar("ranges/pred_min_joint", pred_joint_min, global_step)
            writer.add_scalar("ranges/pred_max_joint", pred_joint_max, global_step)
            writer.add_scalar("ranges/pred_min_gripper", pred_gripper_min, global_step)
            writer.add_scalar("ranges/pred_max_gripper", pred_gripper_max, global_step)
            writer.add_scalar("ranges/target_min_joint", target_joint_min, global_step)
            writer.add_scalar("ranges/target_max_joint", target_joint_max, global_step)
            writer.add_scalar("ranges/target_min_gripper", target_gripper_min, global_step)
            writer.add_scalar("ranges/target_max_gripper", target_gripper_max, global_step)
            writer.add_scalar("latent/mu_norm", mu_norm, global_step)
            writer.add_scalar("latent/log_sigma_x2_mean", log_sigma_x2_mean, global_step)
            writer.add_scalar("latent/sigma_mean", sigma_mean, global_step)
            for tag in ACTIVATION_HOOK_TAGS:
                writer.add_scalar(
                    f"activations/{tag}",
                    activation_norms[tag],
                    global_step,
                )
            global_step += 1

            running_avg_loss = epoch_loss / num_batches
            last_loss_running_avg = running_avg_loss
            snapshot_path = maybe_save_step_checkpoint(
                checkpoint_dir=checkpoint_dir,
                checkpoint_every=checkpoint_every,
                run_name=run_name,
                config=config,
                dataset=dataset_path,
                epoch=epoch,
                global_step=global_step,
                model=model,
                loss_running_avg=running_avg_loss,
                normalization=normalization,
            )
            if snapshot_path is not None:
                print(f"saved snapshot => {snapshot_path.resolve()}")

            if global_step >= target_steps:
                break

        if num_batches == 0:
            break

        avg_loss = epoch_loss / num_batches
        avg_action_loss = epoch_action_loss / num_batches
        avg_kl_loss = epoch_kl_loss / num_batches
        avg_weighted_kl_loss = epoch_weighted_kl_loss / num_batches
        epoch_beta = beta_at_step(
            step=max(global_step - 1, 0),
            beta_start=beta_start,
            beta=beta,
            beta_warmup_steps=beta_warmup_steps,
        )
        if avg_loss > 0.0:
            kl_fraction = avg_weighted_kl_loss / avg_loss
        else:
            kl_fraction = 0.0

        writer.add_scalar("epoch_metrics/loss", avg_loss, epoch)
        writer.add_scalar("epoch_metrics/action_loss", avg_action_loss, epoch)
        writer.add_scalar("epoch_metrics/kl_loss", avg_kl_loss, epoch)
        writer.add_scalar("epoch_metrics/weighted_kl_loss", avg_weighted_kl_loss, epoch)
        writer.add_scalar("epoch_metrics/kl_fraction", kl_fraction, epoch)
        writer.add_scalar("epoch_metrics/beta", epoch_beta, epoch)
        image_l1, proprio_l1 = measure_observation_sensitivity(
            model=model, images=probe_images, proprio=probe_proprio,
        )
        writer.add_scalar("sensitivity/image_swap_l1", image_l1, epoch)
        writer.add_scalar("sensitivity/proprio_swap_l1", proprio_l1, epoch)

        epoch += 1

    if global_step > 0 and global_step % checkpoint_every != 0:
        snapshot_path = checkpoint_path_for_step(
            checkpoint_dir=checkpoint_dir,
            global_step=global_step,
        )
        save_checkpoint(
            path=snapshot_path,
            run_name=run_name,
            config=config,
            dataset=dataset_path,
            epoch=epoch - 1,
            global_step=global_step,
            model=model,
            loss_running_avg=last_loss_running_avg,
            normalization=normalization,
        )
        print(f"saved final snapshot => {snapshot_path.resolve()}")

    for handle in activation_hook_handles:
        handle.remove()
    writer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ACT v2 (CVAE + action chunking).")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/act_v2_instance_bs8_beta0p01_wu80_l1_img_agentview_eyeinhand.toml"
        ),
        help="path to the training hyperparameter config file",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="path to the CAN PH HDF5 training dataset",
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="override total training steps",
    )
    args = parser.parse_args()

    return args


def main() -> None:
    args = parse_args()

    config = load_config(path=args.config)

    if args.steps is not None:
        config["steps"] = args.steps

    train(config=config, dataset=args.dataset)


if __name__ == "__main__":
    main()
