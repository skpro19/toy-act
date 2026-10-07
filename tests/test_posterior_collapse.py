"""Numerical and artifact checks for the posterior-only diagnostic."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.debug.posterior_collapse import compute_diagnostics, make_plots, save_dimension_csv


class PosteriorCollapseTests(unittest.TestCase):
    def test_prior_has_zero_kl(self) -> None:
        mu = np.zeros((10, 32))
        diagnostics = compute_diagnostics(mu=mu, log_variance=np.zeros_like(mu))
        np.testing.assert_array_equal(diagnostics["mean_kl"], 0)
        np.testing.assert_array_equal(diagnostics["sigma"], 1)
        np.testing.assert_array_equal(diagnostics["mu_variance"], 0)

    def test_mean_offset_and_variation_are_distinct(self) -> None:
        mu = np.array([[2, -1], [2, 1]], dtype=float)
        diagnostics = compute_diagnostics(mu=mu, log_variance=np.zeros_like(mu))
        np.testing.assert_allclose(diagnostics["mu_rms"], [2, 1])
        np.testing.assert_allclose(diagnostics["mu_variance"], [0, 1])
        np.testing.assert_allclose(diagnostics["mean_kl"], [2, 0.5])
        np.testing.assert_allclose(diagnostics["kl_per_example"], [2.5, 2.5])

    def test_sigma_is_standard_deviation_not_variance(self) -> None:
        log_variance = np.log(np.array([[0.25, 4.0]]))
        diagnostics = compute_diagnostics(mu=np.zeros((1, 2)), log_variance=log_variance)
        np.testing.assert_allclose(diagnostics["sigma"], [[0.5, 2]])
        self.assertTrue((diagnostics["mean_kl"] > 0).all())

    def test_invalid_inputs(self) -> None:
        for mu, log_variance in [
            (np.zeros((0, 32)), np.zeros((0, 32))),
            (np.zeros((2, 32)), np.zeros((2, 31))),
            (np.array([[np.nan]]), np.zeros((1, 1))),
        ]:
            with self.assertRaises(ValueError):
                compute_diagnostics(mu=mu, log_variance=log_variance)

    def test_six_plots_and_dimension_csv(self) -> None:
        rng = np.random.default_rng(0)
        mu = rng.normal(scale=0.01, size=(40, 32))
        diagnostics = compute_diagnostics(mu=mu, log_variance=np.zeros_like(mu))
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            make_plots(mu=mu, diagnostics=diagnostics, output_dir=output_dir)
            save_dimension_csv(diagnostics=diagnostics, output_dir=output_dir)
            self.assertEqual(len(list(output_dir.glob("*.png"))), 6)
            with (output_dir / "per_dimension.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 32)
            self.assertEqual(rows[-1]["dimension"], "31")
            self.assertEqual(float(rows[0]["sigma_p50"]), 1)


if __name__ == "__main__":
    unittest.main()
