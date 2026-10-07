"""Checkpoint evaluation preserves training state and records useful metrics."""

import tempfile
import unittest
import random
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from scripts import train_v2
from scripts.dataset import NormalizationStats


class CheckpointEvaluationTest(unittest.TestCase):
    def test_evaluation_logs_throughput_without_changing_training_rng(self) -> None:
        model = torch.nn.Linear(1, 1)
        model.train()
        normalization = NormalizationStats(
            proprio_mean=np.zeros(8),
            proprio_std=np.ones(8),
            action_mean=np.zeros(8),
            action_std=np.ones(8),
        )
        env = object()
        seeds = []

        def fake_rollout(**kwargs):
            self.assertIs(kwargs["env"], env)
            self.assertFalse(kwargs["model"].training)
            seeds.append(np.random.get_state()[1][0])
            np.random.random()
            random.random()
            torch.rand(1)
            return {"success": len(seeds) % 2 == 0, "return": 1.0,
                    "horizon": 10, "truncated": False}

        np.random.seed(123)
        random.seed(234)
        torch.manual_seed(456)
        expected_numpy = np.random.get_state()
        expected_python = random.getstate()
        expected_torch = torch.get_rng_state().clone()

        with tempfile.TemporaryDirectory() as directory:
            with SummaryWriter(log_dir=directory) as writer:
                with patch.object(train_v2, "run_rollout", side_effect=fake_rollout):
                    result = train_v2.evaluate_and_log_checkpoint(
                        model=model,
                        env=env,
                        dataset=Path("unused.hdf5"),
                        image_keys=("agentview_image", "robot0_eye_in_hand_image"),
                        normalization=normalization,
                        device=torch.device("cpu"),
                        writer=writer,
                        global_step=2000,
                        run_started=train_v2.time.monotonic() - 60,
                        train_seconds=10,
                        train_steps=20,
                        train_samples=160,
                        checkpoint_seconds=2,
                        tensorboard={
                            namespace: True
                            for namespace in train_v2.TENSORBOARD_NAMESPACES
                        },
                        episodes=30,
                        horizon=250,
                        seed=0,
                        action_chunk_size=10,
                    )

            events = EventAccumulator(directory)
            events.Reload()
            self.assertIs(result, env)
            self.assertTrue(model.training)
            self.assertTrue(np.array_equal(np.random.get_state()[1], expected_numpy[1]))
            self.assertEqual(random.getstate(), expected_python)
            self.assertTrue(torch.equal(torch.get_rng_state(), expected_torch))
            self.assertEqual(seeds, list(range(30)))
            self.assertEqual(events.Scalars("eval/success_rate")[0].value, 0.5)
            self.assertEqual(events.Scalars("throughput/train_steps_per_sec")[0].value, 2)
            self.assertEqual(events.Scalars("throughput/train_samples_per_sec")[0].value, 16)
            self.assertGreater(events.Scalars("throughput/rollout_env_steps_per_sec")[0].value, 0)
            self.assertEqual(events.Scalars("timing/checkpoint_seconds")[0].step, 2000)


class DatasetResolutionTest(unittest.TestCase):
    def base_config(self) -> dict:
        return {
            "version": "v4",
            "action_chunk_size": 10,
            "action_loss": "l1",
            "batch_size": 8,
            "steps": 100000,
            "image_keys": ["agentview_image", "robot0_eye_in_hand_image"],
            "dataset": "datasets/can/ph_mh_better/example.hdf5",
            "lr": 1e-4,
            "seed": 0,
            "beta": 0.01,
            "beta_start": 0.0,
            "beta_warmup_steps": 0,
            "use_z": True,
            "checkpoint_every": 2000,
            "tensorboard": {
                namespace: True
                for namespace in train_v2.TENSORBOARD_NAMESPACES
            },
            "rollout": {"episodes": 30, "horizon": 250, "seed": 0},
        }

    def test_resolve_dataset_prefers_cli_then_config(self) -> None:
        cli_dataset = Path("custom.hdf5")
        config = {"dataset": "datasets/config.hdf5"}

        self.assertEqual(
            train_v2.resolve_dataset(cli_dataset=cli_dataset, config=config), cli_dataset)
        self.assertEqual(
            train_v2.resolve_dataset(cli_dataset=None, config=config),
            Path("datasets/config.hdf5"),
        )

    def test_validate_config_requires_dataset(self) -> None:
        config = self.base_config()
        del config["dataset"]

        with self.assertRaises(KeyError):
            train_v2.validate_config(config=config)

    def test_validate_config_accepts_dataset(self) -> None:
        validated = train_v2.validate_config(config=self.base_config())

        self.assertEqual(validated["dataset"], "datasets/can/ph_mh_better/example.hdf5")

    def test_validate_config_rejects_empty_or_non_string_dataset(self) -> None:
        for bad_dataset in ("", 123):
            with self.subTest(dataset=bad_dataset):
                config = self.base_config()
                config["dataset"] = bad_dataset
                with self.assertRaises(ValueError):
                    train_v2.validate_config(config=config)

    def test_dataset_is_encoded_in_run_name(self) -> None:
        config = self.base_config()
        config["dataset"] = "datasets/can/ph_mh_better/example.hdf5"

        slug = train_v2.make_run_slug(config=config)

        self.assertIn("ds_ph_mh_better_example", slug)

    def test_run_name_fields_take_precedence_over_omit_keys(self) -> None:
        config = self.base_config()
        fields = train_v2.RUN_NAME_FIELDS + (("checkpoint_every", "ce", "{:d}"),)

        with patch.object(train_v2, "RUN_NAME_FIELDS", fields):
            slug = train_v2.make_run_slug(config=config)

        # checkpoint_every is in RUN_NAME_OMIT_KEYS but also in RUN_NAME_FIELDS,
        # so the field list wins and it must still appear in the slug.
        self.assertIn("ce2000", slug)

    def test_validate_config_requires_rollout_section(self) -> None:
        config = self.base_config()
        del config["rollout"]

        with self.assertRaises(KeyError):
            train_v2.validate_config(config=config)

    def test_validate_config_rejects_unknown_rollout_key(self) -> None:
        config = self.base_config()
        config["rollout"]["episodes_per_checkpoint"] = 5

        with self.assertRaises(ValueError):
            train_v2.validate_config(config=config)

    def test_validate_config_rejects_invalid_rollout_values(self) -> None:
        cases = {
            "episodes": [0, -1, 1.5, True],
            "horizon": [0, -1, 1.5, True],
            "seed": [-1, 1.5, True],
        }
        for key, bad_values in cases.items():
            for bad_value in bad_values:
                with self.subTest(key=key, value=bad_value):
                    config = self.base_config()
                    config["rollout"][key] = bad_value
                    with self.assertRaises(ValueError):
                        train_v2.validate_config(config=config)

    def test_rollout_is_excluded_from_run_name(self) -> None:
        config = self.base_config()
        config["rollout"] = {"episodes": 5, "horizon": 50, "seed": 7}

        run_name = train_v2.make_run_name(config=config)

        self.assertNotIn("rollout", run_name)


if __name__ == "__main__":
    unittest.main()
