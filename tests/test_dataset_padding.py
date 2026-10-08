"""Terminal observations retain valid targets without changing normalization."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import torch
from torch import nn

from scripts import train_v2
from scripts.dataset import CanPhDataset, build_action_chunk, build_proprio
from scripts.models.act_v2.config import D_MODEL
from scripts.models.act_v2.cvae_encoder import CVAEEncoder
from scripts.utils.analyze_z import collect_latents


class DatasetPaddingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "dataset.hdf5"
        self.proprio_rows = []
        self.action_rows = []
        with h5py.File(self.path, "w") as hdf5:
            for index, length in enumerate((3, 5)):
                demo = hdf5.create_group(f"data/demo_{index}")
                demo.attrs["num_samples"] = length
                joints = np.arange(length * 7, dtype=np.float32).reshape(length, 7) + index * 100
                gripper = np.stack((np.arange(length), -np.arange(length)), axis=1).astype(np.float32)
                demo.create_dataset("obs/robot0_joint_pos", data=joints)
                demo.create_dataset("obs/robot0_gripper_qpos", data=gripper)
                demo.create_dataset("next_obs/robot0_joint_pos", data=joints + 1)
                demo.create_dataset("next_obs/robot0_gripper_qpos", data=gripper + [1, -1])
                demo.create_dataset("obs/agentview_image", data=np.zeros((length, 4, 4, 3), dtype=np.uint8))
                self.proprio_rows.append(build_proprio(joint_pos=joints, gripper_qpos=gripper))
                self.action_rows.append(build_action_chunk(joint_pos=joints + 1, gripper_qpos=gripper + [1, -1]))

    def make_dataset(self, *, k: int) -> CanPhDataset:
        dataset = CanPhDataset(file=str(self.path), image_keys=("agentview_image",), k=k)

        def close_dataset() -> None:
            if dataset._hdf5 is not None:
                dataset._hdf5.close()

        self.addCleanup(close_dataset)
        return dataset

    def test_all_observations_and_short_demonstrations_are_retained(self) -> None:
        for k in (1, 3, 7):
            with self.subTest(k=k):
                dataset = self.make_dataset(k=k)
                self.assertEqual(len(dataset), 8)
                for index, (name, timestep) in enumerate(dataset.samples):
                    sample = dataset[index]
                    length = dataset.demo_num_timesteps[name]
                    valid = min(k, length - timestep)
                    self.assertEqual(sample["target_actions"].shape, (k, 8))
                    torch.testing.assert_close(sample["action_mask"], (torch.arange(k) < valid).float())
                    demo_index = int(name.split("_")[1])
                    expected = dataset.normalization.normalize_action(
                        value=self.action_rows[demo_index][timestep:timestep + valid],
                    )
                    np.testing.assert_allclose(sample["target_actions"][:valid].numpy(), expected)

    def test_normalization_counts_each_real_timestep_once_for_every_k(self) -> None:
        proprio = np.concatenate(self.proprio_rows).astype(np.float64)
        actions = np.concatenate(self.action_rows).astype(np.float64)
        for k in (1, 3, 7):
            with self.subTest(k=k):
                stats = self.make_dataset(k=k).normalization
                for actual, expected in (
                    (stats.proprio_mean, proprio.mean(axis=0)),
                    (stats.proprio_std, proprio.std(axis=0)),
                    (stats.action_mean, actions.mean(axis=0)),
                    (stats.action_std, actions.std(axis=0)),
                ):
                    np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_masked_l1_and_l2_ignore_padding_and_its_gradients(self) -> None:
        for kind, expected in (("l1", 2.), ("l2", 5.)):
            with self.subTest(kind=kind):
                predictions = torch.tensor([[[1., 3.], [100., 200.]]], requires_grad=True)
                mask = torch.tensor([[1., 0.]])
                loss_fn = train_v2.make_action_loss_fn(action_loss=kind)
                loss = train_v2.get_masked_action_loss(loss_fn(predictions, torch.zeros_like(predictions)), mask)
                torch.testing.assert_close(loss, torch.tensor(expected))
                loss.backward()
                torch.testing.assert_close(predictions.grad[:, 1], torch.zeros(1, 2))
                full_mask = torch.ones_like(mask)
                full_loss = train_v2.get_masked_action_loss(loss_fn(predictions, torch.zeros_like(predictions)), full_mask)
                torch.testing.assert_close(full_loss, loss_fn(predictions, torch.zeros_like(predictions)).mean())

    def test_latent_collection_passes_dataset_masks_to_posterior(self) -> None:
        dataset = self.make_dataset(k=7)

        class PosteriorProbe(nn.Module):
            use_z = True

            def posterior(self, *, proprio: torch.Tensor, actions: torch.Tensor,
                          mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                valid = mask.unsqueeze(-1)
                mean = (actions * valid).sum(dim=1, keepdim=True) / valid.sum(dim=1, keepdim=True)
                return mean, torch.zeros_like(mean)

        model = PosteriorProbe()
        with patch.object(model, "posterior", wraps=model.posterior) as posterior:
            result = collect_latents(model=model, dataset=dataset, indices=np.array([2]),
                                     batch_size=1, device=torch.device("cpu"))
        torch.testing.assert_close(posterior.call_args.kwargs["mask"], torch.tensor([[1., 0., 0., 0., 0., 0., 0.]]))
        np.testing.assert_allclose(result["mu"][0], dataset[2]["target_actions"][0].numpy())
        model.use_z = False
        with self.assertRaisesRegex(ValueError, "use_z=True"):
            collect_latents(model=model, dataset=dataset, indices=np.array([2]),
                            batch_size=1, device=torch.device("cpu"))

    def test_real_cvae_attention_ignores_padded_action_tokens(self) -> None:
        encoder = CVAEEncoder(num_layers=1).eval()
        src = torch.randn(1, 5, D_MODEL)
        mask = torch.tensor([[False, False, False, True, True]])
        changed = src.clone()
        changed[:, 3:] = 1000.
        with torch.no_grad():
            baseline = encoder(src=src, mask=mask)
            actual = encoder(src=changed, mask=mask)
        for expected, observed in zip(baseline, actual):
            torch.testing.assert_close(expected, observed)


if __name__ == "__main__":
    unittest.main()
