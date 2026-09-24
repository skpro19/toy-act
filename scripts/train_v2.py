"""Training script for ACT v2 (CVAE + action chunking)."""

import argparse
import tomllib
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
    make_run_name,
    seed_everything,
    snapshot_parameters,
)

RUNS_ROOT = Path("runs/act_v2")
CHECKPOINTS_ROOT = Path("checkpoints/act_v2")

CONFIG_KEYS = (
    "batch_size",
    "epochs",
    "lr",
    "seed",
    "beta",
    "checkpoint_every",
)

ACTIVATION_HOOK_TAGS = (
    "image_encoder",
    "cvae_encoder",
    "transformer_encoder",
    "transformer_decoder",
    "action_head",
)


def load_config(*, path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")

    with path.open("rb") as file:
        config = tomllib.load(file)

    missing = [key for key in CONFIG_KEYS if key not in config]
    if missing:
        raise KeyError(f"config missing required keys: {missing}")

    return config


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


def get_kl_loss(*, mu: torch.Tensor, log_sigma_x2: torch.Tensor) -> torch.Tensor:
    _, _, d = mu.shape

    sum_mu_x2 = torch.sum(mu.pow(2), dim=2)
    sum_sigma_x2 = torch.sum(log_sigma_x2.exp(), dim=2)
    sum_log_sigma_x2 = torch.sum(log_sigma_x2, dim=2)

    kl_loss = 0.5 * (sum_mu_x2 + sum_sigma_x2 - d - sum_log_sigma_x2)

    return kl_loss.mean()


def save_checkpoint(
    *,
    path: Path,
    epoch: int,
    global_step: int,
    model: ACTV2,
    optimizer: optim.Optimizer,
    loss_epoch: float,
    normalization: NormalizationStats,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss_epoch": loss_epoch,
            "normalization": normalization.as_checkpoint_dict(),
        },
        path,
    )


def train(*, config: dict) -> None:
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
        file="datasets/can/ph/2026-09-19_03-14-50_act_agentview.hdf5",
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
    ).to(device=device)

    l1_loss_fn = nn.L1Loss(reduction="mean")
    optimizer = optim.Adam(params=model.parameters(), lr=lr, betas=(0.9, 0.999))
    model_parameters = list(model.parameters())
    activation_norms, activation_hook_handles = register_activation_norm_hooks(model=model)
    normalization = can_ph_dataset.normalization
    action_mean = torch.from_numpy(normalization.action_mean).to(device=device).view(1, 1, -1)
    action_std = torch.from_numpy(normalization.action_std).to(device=device).view(1, 1, -1)

    run_name = make_run_name(batch_size=batch_size, lr=lr)
    run_dir = RUNS_ROOT / run_name
    checkpoint_dir = CHECKPOINTS_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))
    print(f"seed => {seed}")
    print(f"device => {device}")
    print(f"gpu => {torch.cuda.get_device_name(device)}")
    print(f"tensorboard logs => {run_dir.resolve()}")
    print(f"checkpoints => {checkpoint_dir.resolve()}")

    global_step = 0
    for epoch in range(config["epochs"]):
        epoch_loss = 0.0
        epoch_l1_loss = 0.0
        epoch_kl_loss = 0.0
        num_batches = 0

        pbar = tqdm(can_ph_dataloader, desc=f"epoch {epoch + 1}/{config['epochs']}")
        for batch_dict in pbar:
            optimizer.zero_grad()

            img = batch_dict["image"].to(device, non_blocking=True)
            proprio = batch_dict["proprio"].to(device, non_blocking=True)
            actions = batch_dict["target_actions"].to(device, non_blocking=True)

            pred_actions, mu, log_sigma_x2 = model(proprio=proprio, actions=actions, img=img)

            l1_loss = l1_loss_fn(pred_actions, actions)
            kl_loss = get_kl_loss(mu=mu, log_sigma_x2=log_sigma_x2)
            loss = l1_loss + config["beta"] * kl_loss

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float("inf"),
            ).item()
            params_before = snapshot_parameters(parameters=model_parameters)
            optimizer.step()

            batch_loss = loss.item()
            batch_l1_loss = l1_loss.item()
            batch_kl_loss = kl_loss.item()
            batch_weighted_kl_loss = config["beta"] * batch_kl_loss

            epoch_loss += batch_loss
            epoch_l1_loss += batch_l1_loss
            epoch_kl_loss += batch_kl_loss
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

                batch_l1_joint = l1_loss_fn(pred_joint, target_joint).item()
                batch_l1_gripper = l1_loss_fn(pred_gripper, target_gripper).item()

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

            writer.add_scalar("debug/batch_loss", batch_loss, global_step)
            writer.add_scalar("debug/l1_loss", batch_l1_loss, global_step)
            writer.add_scalar("debug/kl_loss", batch_kl_loss, global_step)
            writer.add_scalar("debug/weighted_kl_loss", batch_weighted_kl_loss, global_step)
            writer.add_scalar("debug/grad_norm_global", grad_norm, global_step)
            writer.add_scalar("debug/param_norm_global", param_norm, global_step)
            writer.add_scalar("debug/update_norm_global", update_norm, global_step)
            writer.add_scalar("debug/lr", learning_rate, global_step)
            writer.add_scalar("debug/adam_exp_avg_norm", adam_exp_avg_norm, global_step)
            writer.add_scalar("debug/adam_exp_avg_sq_norm", adam_exp_avg_sq_norm, global_step)
            writer.add_scalar("debug/l1_joint", batch_l1_joint, global_step)
            writer.add_scalar("debug/l1_gripper", batch_l1_gripper, global_step)
            writer.add_scalar("debug/pred_min_joint", pred_joint_min, global_step)
            writer.add_scalar("debug/pred_max_joint", pred_joint_max, global_step)
            writer.add_scalar("debug/pred_min_gripper", pred_gripper_min, global_step)
            writer.add_scalar("debug/pred_max_gripper", pred_gripper_max, global_step)
            writer.add_scalar("debug/target_min_joint", target_joint_min, global_step)
            writer.add_scalar("debug/target_max_joint", target_joint_max, global_step)
            writer.add_scalar("debug/target_min_gripper", target_gripper_min, global_step)
            writer.add_scalar("debug/target_max_gripper", target_gripper_max, global_step)
            writer.add_scalar("debug/kl_fraction", batch_kl_fraction, global_step)
            writer.add_scalar("debug/mu_norm", mu_norm, global_step)
            writer.add_scalar("debug/log_sigma_x2_mean", log_sigma_x2_mean, global_step)
            writer.add_scalar("debug/sigma_mean", sigma_mean, global_step)
            for tag in ACTIVATION_HOOK_TAGS:
                writer.add_scalar(
                    f"debug/activations/{tag}",
                    activation_norms[tag],
                    global_step,
                )
            global_step += 1

        avg_loss = epoch_loss / num_batches
        avg_l1_loss = epoch_l1_loss / num_batches
        avg_kl_loss = epoch_kl_loss / num_batches
        avg_weighted_kl_loss = config["beta"] * avg_kl_loss
        if avg_loss > 0.0:
            kl_fraction = avg_weighted_kl_loss / avg_loss
        else:
            kl_fraction = 0.0

        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("train/l1_loss", avg_l1_loss, epoch)
        writer.add_scalar("train/kl_loss", avg_kl_loss, epoch)
        writer.add_scalar("train/weighted_kl_loss", avg_weighted_kl_loss, epoch)
        writer.add_scalar("train/kl_fraction", kl_fraction, epoch)

        if (epoch + 1) % config["checkpoint_every"] == 0:
            snapshot_path = checkpoint_dir / f"epoch_{epoch + 1:03d}.pt"
            save_checkpoint(
                path=snapshot_path,
                epoch=epoch,
                global_step=global_step,
                model=model,
                optimizer=optimizer,
                loss_epoch=avg_loss,
                normalization=normalization,
            )
            print(f"saved snapshot => {snapshot_path.resolve()}")

    for handle in activation_hook_handles:
        handle.remove()
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ACT v2 (CVAE + action chunking).")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/act_v2_bs250.toml"),
        help="path to the training hyperparameter config file",
    )
    args = parser.parse_args()

    config = load_config(path=args.config)
    train(config=config)
