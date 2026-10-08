"""PyTorch dataset for ACT training on slim CAN PH HDF5 files."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

import h5py

from scripts.models.act_v1.config import ACTION_CHUNK_SIZE, PROPRIO_DIMS


NORMALIZATION_EPSILON = 1e-6


class NormalizationStats:
    def __init__(
        self,
        *,
        proprio_mean: np.ndarray,
        proprio_std: np.ndarray,
        action_mean: np.ndarray,
        action_std: np.ndarray,
    ) -> None:
        self.proprio_mean = proprio_mean.astype(np.float32)
        self.proprio_std = proprio_std.astype(np.float32)
        self.action_mean = action_mean.astype(np.float32)
        self.action_std = action_std.astype(np.float32)

    def normalize_proprio(self, *, value: np.ndarray) -> np.ndarray:
        return ((value - self.proprio_mean) / self.proprio_std).astype(np.float32)

    def normalize_action(self, *, value: np.ndarray) -> np.ndarray:
        return ((value - self.action_mean) / self.action_std).astype(np.float32)

    def as_checkpoint_dict(self) -> dict[str, list[float]]:
        return {
            "proprio_mean": self.proprio_mean.tolist(),
            "proprio_std": self.proprio_std.tolist(),
            "action_mean": self.action_mean.tolist(),
            "action_std": self.action_std.tolist(),
        }


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


def images_to_tensor(*, images: list[np.ndarray]) -> torch.Tensor:
    if len(images) == 0:
        raise ValueError("images must contain at least one frame")
    return torch.stack([image_to_tensor(image) for image in images], dim=0)


def validate_image_keys(*, hdf5: h5py.File, image_keys: tuple[str, ...]) -> None:
    if len(image_keys) == 0:
        raise ValueError("image_keys must contain at least one observation key")

    data = hdf5["data"]
    demo_names = sorted(data.keys(), key=lambda name: int(name.split("_")[1]))
    if len(demo_names) == 0:
        raise ValueError("dataset contains no demos under data/")

    for demo_name in demo_names:
        obs = data[demo_name]["obs"]
        for key in image_keys:
            if key not in obs:
                raise ValueError(
                    f"demo {demo_name} is missing obs/{key}; "
                    f"available image keys: {sorted(name for name in obs.keys() if name.endswith('_image'))}"
                )


def compute_normalization_stats(*, hdf5: h5py.File) -> NormalizationStats:
    proprio_values = []
    action_values = []

    for demo in hdf5["data"].values():
        num_timesteps = int(demo.attrs["num_samples"])
        num_samples = num_timesteps

        proprio_values.append(
            build_proprio(
                joint_pos=demo["obs/robot0_joint_pos"][:num_samples],
                gripper_qpos=demo["obs/robot0_gripper_qpos"][:num_samples],
            )
        )

        next_actions = build_action_chunk(
            joint_pos=demo["next_obs/robot0_joint_pos"][:],
            gripper_qpos=demo["next_obs/robot0_gripper_qpos"][:],
        )
        action_values.append(next_actions[:num_samples])

    proprio = np.concatenate(proprio_values, axis=0).astype(np.float64)
    actions = np.concatenate(action_values, axis=0).astype(np.float64)
    return NormalizationStats(
        proprio_mean=proprio.mean(axis=0),
        proprio_std=np.maximum(proprio.std(axis=0), NORMALIZATION_EPSILON),
        action_mean=actions.mean(axis=0),
        action_std=np.maximum(actions.std(axis=0), NORMALIZATION_EPSILON),
    )


class CanPhDataset(Dataset):
    def __init__(
        self,
        *,
        file: str,
        image_keys: tuple[str, ...],
        k: int = ACTION_CHUNK_SIZE,
    ) -> None:
        self.file = file
        self.image_keys = image_keys
        self.k = k
        self._hdf5: h5py.File | None = None
        with h5py.File(file, "r") as hdf5:
            validate_image_keys(hdf5=hdf5, image_keys=image_keys)
            self.demo_num_timesteps = self._build_demo_num_timesteps(hdf5=hdf5)
            self.samples = self._build_sample_index(hdf5=hdf5)
            self.normalization = compute_normalization_stats(hdf5=hdf5)

    def _get_hdf5(self) -> h5py.File:
        # Open lazily so each DataLoader worker gets its own fork-safe handle.
        if self._hdf5 is None:
            self._hdf5 = h5py.File(self.file, "r")
        return self._hdf5

    def _build_demo_num_timesteps(self, *, hdf5: h5py.File) -> dict[str, int]:
        data = hdf5["data"]
        demo_names = sorted(data.keys(), key=lambda name: int(name.split("_")[1]))
        lengths: dict[str, int] = {}
        for demo_name in demo_names:
            demo = data[demo_name]
            lengths[demo_name] = int(demo.attrs["num_samples"])
        return lengths

    def _build_sample_index(self, *, hdf5: h5py.File) -> list[tuple[str, int]]:
        data = hdf5["data"]
        demo_names = sorted(data.keys(), key=lambda name: int(name.split("_")[1]))

        samples: list[tuple[str, int]] = []
        for demo_name in demo_names:
            demo = data[demo_name]
            num_timesteps = int(demo.attrs["num_samples"])
            # for timestep in range(num_timesteps - self.k):
            for timestep in range(num_timesteps):
                samples.append((demo_name, timestep))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int]:
        hdf5 = self._get_hdf5()
        demo_name, timestep = self.samples[idx]
        demo = hdf5[f"data/{demo_name}"]

        obs = demo["obs"]
        image_frames = [obs[key][timestep] for key in self.image_keys]
        joint_pos = demo["obs/robot0_joint_pos"][timestep]
        gripper_qpos = demo["obs/robot0_gripper_qpos"][timestep]

        target_joint_pos = demo["next_obs/robot0_joint_pos"][timestep : timestep + self.k]
        target_gripper_qpos = demo["next_obs/robot0_gripper_qpos"][timestep : timestep + self.k]

        proprio = build_proprio(joint_pos=joint_pos, gripper_qpos=gripper_qpos)
        target_actions = build_action_chunk(
            joint_pos=target_joint_pos,
            gripper_qpos=target_gripper_qpos,
        )

        if target_actions.shape[0] < self.k:
            pad = np.zeros((1, PROPRIO_DIMS), dtype=np.float32)
            diff = self.k - target_actions.shape[0]
            target_actions = np.concatenate(
                [target_actions, np.repeat(pad, repeats=diff, axis=0)],
                axis=0,
            )

        if proprio.shape != (PROPRIO_DIMS,):
            raise ValueError(f"expected proprio shape ({PROPRIO_DIMS},), got {proprio.shape}")
        if target_actions.shape != (self.k, PROPRIO_DIMS):
            raise ValueError(
                f"expected target_actions shape ({self.k}, {PROPRIO_DIMS}), got {target_actions.shape}"
            )

        proprio = self.normalization.normalize_proprio(value=proprio)
        target_actions = self.normalization.normalize_action(value=target_actions)

        action_mask = torch.ones(self.k)
        N = self.demo_num_timesteps[demo_name]

        # N=3, idx=0 , k=4
        if timestep + self.k > N:
            overflow = timestep  + self.k - N
            action_mask[-overflow:] = 0

        return {
            "images": images_to_tensor(images=image_frames),
            "proprio": torch.from_numpy(proprio).unsqueeze(0),
            "target_actions": torch.from_numpy(target_actions),
            "demo_name": demo_name,
            "timestep": timestep,
            "action_mask": action_mask
        }
