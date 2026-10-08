"""Exercise latent TensorBoard logging through a tiny CPU training step."""

import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from scripts import train_v2


class TinyLatentModel(nn.Module):
    def __init__(self, *, use_z: bool, latent_repeats: int) -> None:
        super().__init__()
        self.use_z = use_z
        self.latent_repeats = latent_repeats
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(
        self, *, proprio: torch.Tensor, actions: torch.Tensor,
        img: torch.Tensor,
        action_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        predictions = self.weight * torch.ones_like(actions)
        if not self.use_z:
            return predictions, None, None
        mu = torch.tensor([3.0, -4.0]).repeat(actions.shape[0], 1, self.latent_repeats)
        return predictions, mu, torch.zeros_like(mu)


class LatentLoggingTest(unittest.TestCase):
    def run_training_step(
        self, *, batch_size: int = 2, latent_repeats: int = 1,
        use_z: bool = True, log_latent: bool = True) -> dict[str, float]:
        config = {
            "seed": 0, "batch_size": batch_size, "lr": 1e-4,
            "action_chunk_size": 2, "image_keys": ["image"],
            "action_loss": "l1", "steps": 1, "beta": 0.01,
            "beta_start": 0.0, "beta_warmup_steps": 0,
            "checkpoint_every": 1, "use_z": use_z,
            "tensorboard": {
                name: name == "latent" and log_latent
                for name in train_v2.TENSORBOARD_NAMESPACES
            },
            "rollout": {"episodes": 1, "horizon": 2, "seed": 0, "n_action_steps": 1},
        }
        batch = {
            "images": torch.zeros(batch_size, 1, 1),
            "proprio": torch.zeros(batch_size, 1, 7),
            "target_actions": torch.zeros(batch_size, 2, 7),
            "action_mask": torch.ones(batch_size, 2),
        }
        normalization = SimpleNamespace(
            action_mean=np.zeros(7, dtype=np.float32),
            action_std=np.ones(7, dtype=np.float32),
        )
        model = TinyLatentModel(use_z=use_z, latent_repeats=latent_repeats)
        cpu = torch.device("cpu")
        with TemporaryDirectory() as temp:
            root = Path(temp)
            with (
                patch.object(train_v2.torch.cuda, "is_available", return_value=True),
                patch.object(train_v2.torch.cuda, "get_device_name", return_value="test CPU"),
                patch.object(train_v2.torch, "device", return_value=cpu),
                patch.object(
                    train_v2, "CanPhDataset",
                    return_value=SimpleNamespace(normalization=normalization),
                ),
                patch.object(train_v2, "DataLoader", return_value=[batch]),
                patch.object(train_v2, "ACTV2", return_value=model),
                patch.object(train_v2, "RUNS_ROOT", root / "runs"),
                patch.object(train_v2, "CHECKPOINTS_ROOT", root / "checkpoints"),
                patch.object(train_v2, "make_run_name", return_value="test-run"),
                patch.object(train_v2, "save_run_config"),
                patch.object(train_v2, "maybe_save_step_checkpoint", return_value=None),
            ):
                train_v2.train(config=config, dataset=root / "dataset.hdf5")
            events = EventAccumulator(str(root / "runs" / "test-run"))
            events.Reload()
            return {tag: events.Scalars(tag)[0].value for tag in events.Tags()["scalars"]}

    def test_logs_rms_and_retains_existing_latent_metrics(self) -> None:
        values = self.run_training_step()
        self.assertEqual(set(values), {
            "latent/mu_norm", "latent/mu_rms",
            "latent/log_sigma_x2_mean", "latent/sigma_mean",
        })
        self.assertAlmostEqual(values["latent/mu_rms"], math.sqrt(12.5), places=6)
        self.assertAlmostEqual(values["latent/mu_norm"], math.sqrt(50), places=6)
        self.assertEqual(values["latent/log_sigma_x2_mean"], 0.0)
        self.assertEqual(values["latent/sigma_mean"], 1.0)

    def test_rms_is_invariant_to_batch_size_and_latent_dimension(self) -> None:
        baseline = self.run_training_step()
        for batch_size, latent_repeats in ((1, 1), (4, 1), (2, 3)):
            with self.subTest(batch_size=batch_size, latent_repeats=latent_repeats):
                values = self.run_training_step(batch_size=batch_size, latent_repeats=latent_repeats)
                self.assertAlmostEqual(values["latent/mu_rms"], baseline["latent/mu_rms"])
                self.assertNotAlmostEqual(values["latent/mu_norm"], baseline["latent/mu_norm"])

    def test_no_latent_metrics_when_namespace_is_disabled(self) -> None:
        self.assertEqual(self.run_training_step(log_latent=False), {})

    def test_no_latent_metrics_when_use_z_is_disabled(self) -> None:
        self.assertEqual(self.run_training_step(use_z=False), {})


if __name__ == "__main__":
    unittest.main()
