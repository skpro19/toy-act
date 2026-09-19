"""PyTorch dataset for ACT training on slim CAN PH HDF5 files."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

import h5py

from scripts.models.act_v1.config import ACTION_CHUNK_SIZE, PROPRIO_DIMS


def gripper_aperture(gripper_qpos: np.ndarray) -> np.ndarray:
    return gripper_qpos[..., :1] - gripper_qpos[..., 1:2]


def build_proprio(*, joint_pos: np.ndarray, gripper_qpos: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [joint_pos, gripper_aperture(gripper_qpos)],
        axis=-1,
    ).astype(np.float32)


def build_action_chunk(
    *,
    joint_pos: np.ndarray,
    gripper_qpos: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        [joint_pos, gripper_aperture(gripper_qpos)],
        axis=-1,
    ).astype(np.float32)


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    # HDF5 stores RGB as (H, W, 3) uint8; the model expects (3, H, W) float32.
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).to(dtype=torch.float32)
    return image_tensor / 255.0


class CanPhDataset(Dataset):
    def __init__(self, *, file: str, k: int = ACTION_CHUNK_SIZE) -> None:
        self.hdf5 = h5py.File(file, "r")
        self.k = k
        self.samples = self._build_sample_index()

    def _build_sample_index(self) -> list[tuple[str, int]]:
        data = self.hdf5["data"]
        demo_names = sorted(data.keys(), key=lambda name: int(name.split("_")[1]))

        samples: list[tuple[str, int]] = []
        for demo_name in demo_names:
            demo = data[demo_name]
            num_timesteps = int(demo.attrs["num_samples"])
            for timestep in range(num_timesteps - self.k):
                samples.append((demo_name, timestep))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        demo_name, timestep = self.samples[idx]
        demo = self.hdf5[f"data/{demo_name}"]

        image = demo["obs/agentview_image"][timestep]
        joint_pos = demo["obs/robot0_joint_pos"][timestep]
        gripper_qpos = demo["obs/robot0_gripper_qpos"][timestep]

        target_joint_pos = demo["next_obs/robot0_joint_pos"][timestep : timestep + self.k]
        target_gripper_qpos = demo["next_obs/robot0_gripper_qpos"][timestep : timestep + self.k]

        proprio = build_proprio(joint_pos=joint_pos, gripper_qpos=gripper_qpos)
        target_actions = build_action_chunk(
            joint_pos=target_joint_pos,
            gripper_qpos=target_gripper_qpos,
        )

        if proprio.shape != (PROPRIO_DIMS,):
            raise ValueError(f"expected proprio shape ({PROPRIO_DIMS},), got {proprio.shape}")
        if target_actions.shape != (self.k, PROPRIO_DIMS):
            raise ValueError(
                f"expected target_actions shape ({self.k}, {PROPRIO_DIMS}), got {target_actions.shape}"
            )

        return {
            "image": image_to_tensor(image),
            "proprio": torch.from_numpy(proprio).unsqueeze(0),
            "target_actions": torch.from_numpy(target_actions),
        }
