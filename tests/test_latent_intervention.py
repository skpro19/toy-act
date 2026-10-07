"""Unit tests for deterministic latent interventions and metric aggregation."""

import unittest

import numpy as np
import torch

from scripts.debug.latent_intervention import make_latent_variants, prediction_metrics, summarize_rows


class LatentInterventionTests(unittest.TestCase):
    def test_variants_and_seed(self) -> None:
        mu = torch.ones(3, 1, 2)
        shuffled = mu * 2
        first = make_latent_variants(mu=mu, shuffled_mu=shuffled, prior_samples=3,
                                     generator=torch.Generator().manual_seed(7))
        second = make_latent_variants(mu=mu, shuffled_mu=shuffled, prior_samples=3,
                                      generator=torch.Generator().manual_seed(7))
        self.assertEqual(len(first), 6)
        self.assertTrue(torch.equal(first["posterior_mean"], mu))
        self.assertTrue(torch.equal(first["shuffled_mean"], shuffled))
        self.assertEqual(first["zero"].abs().sum().item(), 0)
        for name in first:
            self.assertTrue(torch.equal(first[name], second[name]))
        self.assertFalse(torch.equal(first["prior_00"], first["prior_01"]))

    def test_native_units_and_baseline(self) -> None:
        baseline = torch.zeros(2, 3, 8)
        prediction = torch.ones_like(baseline)
        target = prediction * 3
        std = torch.tensor([2.] * 7 + [0.1])
        metrics = prediction_metrics(prediction=prediction, baseline=baseline, target=target, action_std=std)
        np.testing.assert_allclose(metrics["prediction_delta_normalized"], 1)
        np.testing.assert_allclose(metrics["action_l1_normalized"], 2)
        np.testing.assert_allclose(metrics["prediction_delta_joint_rad"], 2)
        np.testing.assert_allclose(metrics["prediction_delta_gripper_m"], 0.1)
        np.testing.assert_allclose(metrics["action_l1_joint_rad"], 4)
        np.testing.assert_allclose(metrics["action_l1_gripper_m"], 0.2)
        same = prediction_metrics(prediction=baseline, baseline=baseline, target=target, action_std=std)
        np.testing.assert_array_equal(same["prediction_delta_normalized"], 0)

    def test_summary_weights_examples_not_batches(self) -> None:
        rows = []
        for variant, deltas, errors in [("posterior_mean", [0, 0, 0], [1, 1, 1]),
                                        ("zero", [1, 2, 6], [2, 3, 4])]:
            for timestep, (delta, error) in enumerate(zip(deltas, errors)):
                rows.append({"demo_name": "demo_0", "timestep": timestep, "variant": variant,
                             "prediction_delta_normalized": delta, "action_l1_normalized": error})
        summary = summarize_rows(rows=rows, action_variation=2)
        self.assertEqual(summary["zero"]["metrics"]["prediction_delta_normalized"]["mean"], 3)
        self.assertEqual(summary["zero"]["delta_over_target_variation"], 1.5)
        self.assertEqual(summary["zero"]["action_l1_change_from_baseline"], 2)

    def test_invalid_variants(self) -> None:
        with self.assertRaises(ValueError):
            make_latent_variants(mu=torch.zeros(2, 1, 3), shuffled_mu=torch.zeros(1, 1, 3),
                                 prior_samples=1, generator=torch.Generator())
        with self.assertRaises(ValueError):
            make_latent_variants(mu=torch.zeros(2, 1, 3), shuffled_mu=torch.zeros(2, 1, 3),
                                 prior_samples=0, generator=torch.Generator())


if __name__ == "__main__":
    unittest.main()
