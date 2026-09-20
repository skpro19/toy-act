""" Training script for act-v1. """
import json
from collections import deque
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from scripts.dataset import CanPhDataset
from scripts.models.act_v1.config import (
    ACTION_CHUNK_SIZE,
    D_MODEL,
    JOINT_DIMS,
    PROPRIO_DIMS,
    NUM_LAYERS,
    N_HEAD,
)
from scripts.models.act_v1 import ACTV1


BATCH_SIZE = 250
EPOCHS = 200
LR = 1e-4
CHECKPOINT_EVERY = 10
RUNS_ROOT = Path("runs/act_v1")
CHECKPOINTS_ROOT = Path("checkpoints/act_v1")

ACTIVATION_HOOK_TAGS = (
    "img_encoder",
    "transformer_encoder",
    "transformer_decoder",
    "action_head",
)
SPIKE_MEDIAN_WINDOW = 100
SPIKE_MIN_BATCHES = 20
SPIKE_RATIO = 5.0


def make_run_name(*, batch_size: int, lr: float) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}_bs{batch_size}_lr{lr:.0e}"


def save_checkpoint(
    *,
    path: Path,
    epoch: int,
    global_step: int,
    model: ACTV1,
    optimizer: optim.Optimizer,
    loss_epoch: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss_epoch": loss_epoch,
        },
        path,
    )


def compute_global_l2_norm(*, tensors) -> float:
    total = 0.0
    for tensor in tensors:
        total += tensor.detach().pow(2).sum().item()
    return total ** 0.5


def snapshot_parameters(*, parameters) -> list[torch.Tensor]:
    return [param.detach().clone() for param in parameters]


def compute_global_update_norm(*, parameters, before: list[torch.Tensor]) -> float:
    total = 0.0
    for param, prev in zip(parameters, before, strict=True):
        total += (param.detach() - prev).pow(2).sum().item()
    return total ** 0.5


def compute_adam_moment_norms(*, optimizer: optim.Optimizer) -> tuple[float, float]:
    exp_avg_total = 0.0
    exp_avg_sq_total = 0.0
    for state in optimizer.state.values():
        exp_avg_total += state["exp_avg"].detach().pow(2).sum().item()
        exp_avg_sq_total += state["exp_avg_sq"].detach().pow(2).sum().item()
    return exp_avg_total ** 0.5, exp_avg_sq_total ** 0.5


def compute_rolling_median(*, recent_losses: deque[float]) -> float:
    sorted_losses = sorted(recent_losses)
    count = len(sorted_losses)
    mid = count // 2
    if count % 2 == 1:
        return sorted_losses[mid]
    return (sorted_losses[mid - 1] + sorted_losses[mid]) / 2.0


def build_sample_records(*, demo_names: list[str], timesteps: torch.Tensor) -> list[list[str | int]]:
    return [
        [demo_name, int(timestep)]
        for demo_name, timestep in zip(demo_names, timesteps.tolist(), strict=True)
    ]


def save_spike_anomaly(*, path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def register_activation_norm_hooks(*, model: ACTV1) -> tuple[dict[str, float], list]:
    norms: dict[str, float] = {}
    hook_targets = {
        "img_encoder": model.img_encoder,
        "transformer_encoder": model.encoder,
        "transformer_decoder": model.decoder,
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


def train() -> None:
    can_ph_dataset = CanPhDataset(file="datasets/can/ph/2026-09-19_03-14-50_act_agentview.hdf5", k=ACTION_CHUNK_SIZE)
    train_dataloader = DataLoader(
        dataset=can_ph_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
    )

    device = torch.device("cuda")
    model = ACTV1(
        d_model=D_MODEL,
        nhead=N_HEAD,
        num_layers=NUM_LAYERS,
        action_chunk_size=ACTION_CHUNK_SIZE,
        proprio_dims=PROPRIO_DIMS,
    ).to(device=device)

    optimizer = optim.Adam(params=model.parameters(), lr=LR, betas=(0.9, 0.999))
    loss_fn = nn.MSELoss()
    model_parameters = list(model.parameters())
    activation_norms, activation_hook_handles = register_activation_norm_hooks(model=model)

    run_name = make_run_name(batch_size=BATCH_SIZE, lr=LR)
    run_dir = RUNS_ROOT / run_name
    checkpoint_dir = CHECKPOINTS_ROOT / run_name
    anomalies_dir = run_dir / "anomalies"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))
    print(f"tensorboard logs => {run_dir.resolve()}")
    print(f"checkpoints => {checkpoint_dir.resolve()}")
    print(f"loss spike logs => {anomalies_dir.resolve()}")

    global_step = 0
    recent_losses: deque[float] = deque(maxlen=SPIKE_MEDIAN_WINDOW)
    prev_sample_records: list[list[str | int]] | None = None
    prev_global_step: int | None = None
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        epoch_joint_mse = 0.0
        epoch_gripper_mse = 0.0
        num_batches = 0
        pbar = tqdm(train_dataloader, desc=f"epoch {epoch + 1}/{EPOCHS}")

        for batch in pbar:
            demo_names = batch["demo_name"]
            timesteps = batch["timestep"]
            sample_records = build_sample_records(demo_names=demo_names, timesteps=timesteps)

            img_obs = batch["image"].to(device)
            proprio_obs = batch["proprio"].to(device)
            target_actions = batch["target_actions"].to(device)

            optimizer.zero_grad()
            pred = model(img_tensor=img_obs, proprio_tensor=proprio_obs)
            loss = loss_fn(pred, target_actions)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float("inf"),
            ).item()
            params_before = snapshot_parameters(parameters=model_parameters)
            optimizer.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss

            with torch.no_grad():
                learning_rate = optimizer.param_groups[0]["lr"]
                adam_exp_avg_norm, adam_exp_avg_sq_norm = compute_adam_moment_norms(optimizer=optimizer)
                param_norm = compute_global_l2_norm(tensors=model_parameters)
                update_norm = compute_global_update_norm(
                    parameters=model_parameters,
                    before=params_before,
                )
                pred_joint = pred[..., :JOINT_DIMS]
                pred_gripper = pred[..., JOINT_DIMS:]
                target_joint = target_actions[..., :JOINT_DIMS]
                target_gripper = target_actions[..., JOINT_DIMS:]

                batch_joint_mse = loss_fn(pred_joint, target_joint).item()
                batch_gripper_mse = loss_fn(pred_gripper, target_gripper).item()
                epoch_joint_mse += batch_joint_mse
                epoch_gripper_mse += batch_gripper_mse

                pred_joint_min = pred_joint.min().item()
                pred_joint_max = pred_joint.max().item()
                pred_gripper_min = pred_gripper.min().item()
                pred_gripper_max = pred_gripper.max().item()
                target_joint_min = target_joint.min().item()
                target_joint_max = target_joint.max().item()
                target_gripper_min = target_gripper.min().item()
                target_gripper_max = target_gripper.max().item()

                per_sample_mse = ((pred - target_actions) ** 2).mean(dim=(1, 2))
                worst_sample_index = int(per_sample_mse.argmax().item())
                worst_sample = sample_records[worst_sample_index]
                worst_sample_mse = float(per_sample_mse[worst_sample_index].item())

            rolling_median_loss = None
            if len(recent_losses) >= SPIKE_MIN_BATCHES:
                rolling_median_loss = compute_rolling_median(recent_losses=recent_losses)
                if batch_loss > SPIKE_RATIO * rolling_median_loss:
                    anomaly_path = anomalies_dir / f"step_{global_step:06d}.json"
                    debug_metrics = {
                        "batch_loss": batch_loss,
                        "rolling_median_loss": rolling_median_loss,
                        "spike_ratio": batch_loss / rolling_median_loss,
                        "mse_joint": batch_joint_mse,
                        "mse_gripper": batch_gripper_mse,
                        "grad_norm_global": grad_norm,
                        "param_norm_global": param_norm,
                        "update_norm_global": update_norm,
                        "lr": learning_rate,
                        "adam_exp_avg_norm": adam_exp_avg_norm,
                        "adam_exp_avg_sq_norm": adam_exp_avg_sq_norm,
                        "pred_min_joint": pred_joint_min,
                        "pred_max_joint": pred_joint_max,
                        "pred_min_gripper": pred_gripper_min,
                        "pred_max_gripper": pred_gripper_max,
                        "target_min_joint": target_joint_min,
                        "target_max_joint": target_joint_max,
                        "target_min_gripper": target_gripper_min,
                        "target_max_gripper": target_gripper_max,
                        "worst_sample_mse": worst_sample_mse,
                    }
                    for tag in ACTIVATION_HOOK_TAGS:
                        debug_metrics[f"act_{tag}"] = activation_norms[tag]

                    payload = {
                        "global_step": global_step,
                        "epoch": epoch,
                        "batch_loss": batch_loss,
                        "rolling_median_loss": rolling_median_loss,
                        "spike_ratio_threshold": SPIKE_RATIO,
                        "spike_ratio_actual": batch_loss / rolling_median_loss,
                        "worst_sample": worst_sample,
                        "samples": sample_records,
                        "prev_global_step": prev_global_step,
                        "prev_samples": prev_sample_records,
                        "debug": debug_metrics,
                    }
                    save_spike_anomaly(path=anomaly_path, payload=payload)
                    print(
                        f"SPIKE global_step={global_step} epoch={epoch + 1} "
                        f"loss={batch_loss:.4f} median={rolling_median_loss:.4f} "
                        f"worst={worst_sample[0]}@{worst_sample[1]} "
                        f"=> {anomaly_path.resolve()}"
                    )

            num_batches += 1
            pbar.set_postfix(loss=f"{batch_loss:.4f}")
            writer.add_scalar("debug/batch_loss", batch_loss, global_step)
            writer.add_scalar("debug/grad_norm_global", grad_norm, global_step)
            writer.add_scalar("debug/param_norm_global", param_norm, global_step)
            writer.add_scalar("debug/update_norm_global", update_norm, global_step)
            writer.add_scalar("debug/lr", learning_rate, global_step)
            writer.add_scalar("debug/adam_exp_avg_norm", adam_exp_avg_norm, global_step)
            writer.add_scalar("debug/adam_exp_avg_sq_norm", adam_exp_avg_sq_norm, global_step)
            for tag in ACTIVATION_HOOK_TAGS:
                writer.add_scalar(f"debug/act/{tag}", activation_norms[tag], global_step)
            writer.add_scalar("debug/mse_joint", batch_joint_mse, global_step)
            writer.add_scalar("debug/mse_gripper", batch_gripper_mse, global_step)
            writer.add_scalar("debug/pred_min_joint", pred_joint_min, global_step)
            writer.add_scalar("debug/pred_max_joint", pred_joint_max, global_step)
            writer.add_scalar("debug/pred_min_gripper", pred_gripper_min, global_step)
            writer.add_scalar("debug/pred_max_gripper", pred_gripper_max, global_step)
            writer.add_scalar("debug/target_min_joint", target_joint_min, global_step)
            writer.add_scalar("debug/target_max_joint", target_joint_max, global_step)
            writer.add_scalar("debug/target_min_gripper", target_gripper_min, global_step)
            writer.add_scalar("debug/target_max_gripper", target_gripper_max, global_step)
            prev_sample_records = sample_records
            prev_global_step = global_step
            recent_losses.append(batch_loss)
            global_step += 1

        avg_loss = epoch_loss / num_batches
        writer.add_scalar("train/loss_epoch", avg_loss, epoch)
        writer.add_scalar("train/action_mse_joint", epoch_joint_mse / num_batches, epoch)
        writer.add_scalar("train/action_mse_gripper", epoch_gripper_mse / num_batches, epoch)

        save_checkpoint(
            path=checkpoint_dir / "last.pt",
            epoch=epoch,
            global_step=global_step,
            model=model,
            optimizer=optimizer,
            loss_epoch=avg_loss,
        )
        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            snapshot_path = checkpoint_dir / f"epoch_{epoch + 1:03d}.pt"
            save_checkpoint(
                path=snapshot_path,
                epoch=epoch,
                global_step=global_step,
                model=model,
                optimizer=optimizer,
                loss_epoch=avg_loss,
            )
            print(f"saved snapshot => {snapshot_path.resolve()}")

    for handle in activation_hook_handles:
        handle.remove()
    writer.close()


if __name__ == "__main__":
    train()
