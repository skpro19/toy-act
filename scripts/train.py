""" Training script for act-v1. """
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
CHECKPOINT_EVERY = 5
RUNS_ROOT = Path("runs/act_v1")
CHECKPOINTS_ROOT = Path("checkpoints/act_v1")


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

    run_name = make_run_name(batch_size=BATCH_SIZE, lr=LR)
    run_dir = RUNS_ROOT / run_name
    checkpoint_dir = CHECKPOINTS_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))
    print(f"tensorboard logs => {run_dir.resolve()}")
    print(f"checkpoints => {checkpoint_dir.resolve()}")

    global_step = 0
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        epoch_joint_mse = 0.0
        epoch_gripper_mse = 0.0
        num_batches = 0
        pbar = tqdm(train_dataloader, desc=f"epoch {epoch + 1}/{EPOCHS}")

        for batch in pbar:
            img_obs = batch["image"].to(device)
            proprio_obs = batch["proprio"].to(device)
            target_actions = batch["target_actions"].to(device)

            optimizer.zero_grad()
            pred = model(img_tensor=img_obs, proprio_tensor=proprio_obs)
            loss = loss_fn(pred, target_actions)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss

            with torch.no_grad():
                epoch_joint_mse += loss_fn(
                    pred[..., :JOINT_DIMS],
                    target_actions[..., :JOINT_DIMS],
                ).item()
                epoch_gripper_mse += loss_fn(
                    pred[..., JOINT_DIMS:],
                    target_actions[..., JOINT_DIMS:],
                ).item()

            num_batches += 1
            pbar.set_postfix(loss=f"{batch_loss:.4f}")
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

    writer.close()


if __name__ == "__main__":
    train()
