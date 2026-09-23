"""Training script for ACT v2 (CVAE + action chunking)."""

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
    N_HEAD,
    NUM_LAYERS,
    PROPRIO_DIMS,
    Z_DIMS,
)
from scripts.models.act_v2.model import ACTV2
from scripts.train_v1 import (
    make_dataloader_generator,
    make_dataloader_worker_init_fn,
    make_run_name,
    seed_everything,
)

BATCH_SIZE = 200
EPOCHS = 1
LR = 1e-4
SEED = 0
BETA = 10.0
RUNS_ROOT = Path("runs/act_v2")
CHECKPOINTS_ROOT = Path("checkpoints/act_v2")


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


def train() -> None:
    seed_everything(seed=SEED)

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
    dataloader_generator = make_dataloader_generator(seed=SEED)
    can_ph_dataloader = DataLoader(
        dataset=can_ph_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=device.type == "cuda",
        generator=dataloader_generator,
        worker_init_fn=make_dataloader_worker_init_fn(base_seed=SEED),
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
    optimizer = optim.Adam(params=model.parameters(), lr=LR, betas=(0.9, 0.999))
    normalization = can_ph_dataset.normalization

    run_name = make_run_name(batch_size=BATCH_SIZE, lr=LR)
    run_dir = RUNS_ROOT / run_name
    checkpoint_dir = CHECKPOINTS_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))
    print(f"seed => {SEED}")
    print(f"device => {device}")
    print(f"gpu => {torch.cuda.get_device_name(device)}")
    print(f"tensorboard logs => {run_dir.resolve()}")
    print(f"checkpoints => {checkpoint_dir.resolve()}")

    global_step = 0
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        epoch_l1_loss = 0.0
        epoch_kl_loss = 0.0
        num_batches = 0

        pbar = tqdm(can_ph_dataloader, desc=f"epoch {epoch + 1}/{EPOCHS}")
        for batch_dict in pbar:
            optimizer.zero_grad()

            img = batch_dict["image"].to(device, non_blocking=True)
            proprio = batch_dict["proprio"].to(device, non_blocking=True)
            actions = batch_dict["target_actions"].to(device, non_blocking=True)

            pred_actions, mu, log_sigma_x2 = model(proprio=proprio, actions=actions, img=img)

            l1_loss = l1_loss_fn(pred_actions, actions)
            kl_loss = get_kl_loss(mu=mu, log_sigma_x2=log_sigma_x2)
            loss = l1_loss + BETA * kl_loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_l1_loss += l1_loss.item()
            epoch_kl_loss += kl_loss.item()
            num_batches += 1
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / num_batches
        avg_l1_loss = epoch_l1_loss / num_batches
        avg_kl_loss = epoch_kl_loss / num_batches
        avg_weighted_kl_loss = BETA * avg_kl_loss
        if avg_loss > 0.0:
            kl_fraction = avg_weighted_kl_loss / avg_loss
            l1_fraction = avg_l1_loss / avg_loss
        else:
            kl_fraction = 0.0
            l1_fraction = 0.0

        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("train/l1_loss", avg_l1_loss, epoch)
        writer.add_scalar("train/kl_loss", avg_kl_loss, epoch)
        writer.add_scalar("train/weighted_kl_loss", avg_weighted_kl_loss, epoch)
        writer.add_scalar("train/kl_fraction", kl_fraction, epoch)
        writer.add_scalar("train/l1_fraction", l1_fraction, epoch)

        save_checkpoint(
            path=checkpoint_dir / "last.pt",
            epoch=epoch,
            global_step=global_step,
            model=model,
            optimizer=optimizer,
            loss_epoch=avg_loss,
            normalization=normalization,
        )

    writer.close()


if __name__ == "__main__":
    train()
