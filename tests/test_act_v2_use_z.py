"""Latent switching preserves checkpoint structure and loss semantics."""

import unittest
from unittest.mock import patch

import torch
from torch import nn

from scripts import train_v2
from scripts.models.act_v2 import model as act_v2


class TinyPosterior(nn.Module):
    """Small posterior for testing policy plumbing without a full CVAE."""

    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(8, 4)

    def forward(self, *, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        stats = self.head(src.mean(dim=1, keepdim=True))
        return stats[..., :2], stats[..., 2:]


class LatentSwitchTest(unittest.TestCase):
    def make_model(self, *, use_z: bool) -> act_v2.ACTV2:
        # Avoid image-backbone downloads and keep these tests CPU-friendly.
        with patch.object(act_v2, "ImageEncoder", side_effect=lambda **kwargs: nn.Identity()), \
                patch.object(act_v2, "CVAEEncoder", side_effect=lambda **kwargs: TinyPosterior()):
            return act_v2.ACTV2(
                d_model=8,
                nhead=2,
                num_layers=1,
                z_dims=2,
                proprio_dims=3,
                action_chunk_size=2,
                use_z=use_z,
            ).eval()

    def observations(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.randn(2, 1, 3), torch.randn(2, 2, 3), torch.randn(2, 4, 8)

    def test_strict_checkpoint_loading_between_modes(self) -> None:
        enabled = self.make_model(use_z=True)
        disabled = self.make_model(use_z=False)
        self.assertEqual(set(enabled.state_dict()), set(disabled.state_dict()))
        disabled.load_state_dict(enabled.state_dict(), strict=True)
        enabled.load_state_dict(disabled.state_dict(), strict=True)
        for name in ("cvae_encoder", "action_encoder", "z_encoder"):
            self.assertIn(name, dict(disabled.named_children()))

    def test_disabled_mode_omits_latent_and_has_no_cvae_gradients(self) -> None:
        model = self.make_model(use_z=False)
        proprio, targets, images = self.observations()
        with patch.object(model.z_encoder, "forward", side_effect=AssertionError("latent used")), \
                patch.object(model.transformer_encoder, "forward", wraps=model.transformer_encoder.forward) as encoder:
            predictions, mu, log_variance = model(proprio=proprio, actions=targets, img=images)
        self.assertEqual(encoder.call_args.args[0].shape[1], 5)
        action_loss = nn.functional.l1_loss(predictions, targets)
        loss, kl = train_v2.get_training_loss(
            action_loss=action_loss, mu=mu, log_sigma_x2=log_variance, use_z=False, beta=100,
        )
        self.assertIs(loss, action_loss)
        self.assertEqual(kl.item(), 0)
        self.assertFalse(kl.requires_grad)
        loss.backward()
        for module in (model.cvae_encoder, model.action_encoder, model.z_encoder):
            self.assertTrue(all(parameter.grad is None for parameter in module.parameters()))
        self.assertIsNone(model.cls.grad)
        self.assertIsNotNone(model.proprio_encoder.project.weight.grad)

    def test_disabled_forward_skips_posterior(self) -> None:
        model = self.make_model(use_z=False)
        proprio, targets, images = self.observations()
        with patch.object(
            model, "posterior", side_effect=AssertionError("posterior called"),
        ):
            predictions, mu, log_variance = model(
                proprio=proprio, actions=targets, img=images,
            )
        self.assertIsNone(mu)
        self.assertIsNone(log_variance)
        self.assertEqual(predictions.shape, targets.shape)

    def test_disabled_decode_ignores_supplied_latent(self) -> None:
        model = self.make_model(use_z=False)
        proprio, _, images = self.observations()
        with patch.object(model.z_encoder, "forward", side_effect=AssertionError("latent used")):
            baseline = model.decode(proprio=proprio, img=images)
            supplied = model.decode(proprio=proprio, img=images, z=torch.randn(2, 1, 2))
            inferred = model.infer(proprio=proprio, img=images)
        torch.testing.assert_close(baseline, supplied)
        torch.testing.assert_close(baseline, inferred)

    def test_enabled_training_samples_and_inference_uses_zero(self) -> None:
        model = self.make_model(use_z=True)
        proprio, targets, images = self.observations()
        with patch.object(torch, "randn_like", side_effect=lambda tensor: torch.ones_like(tensor)), \
                patch.object(model.z_encoder, "forward", wraps=model.z_encoder.forward) as projection:
            _, mu, log_variance = model(proprio=proprio, actions=targets, img=images)
            torch.testing.assert_close(
                projection.call_args.args[0], mu + (0.5 * log_variance).exp(),
            )
            model.infer(proprio=proprio, img=images)
            torch.testing.assert_close(projection.call_args.args[0], torch.zeros(2, 1, 2))

    def test_disabled_loss_accepts_absent_posterior(self) -> None:
        action_loss = torch.tensor(2.0, requires_grad=True)
        with patch.object(train_v2, "get_kl_loss", side_effect=AssertionError("KL computed")):
            loss, kl = train_v2.get_training_loss(
                action_loss=action_loss, mu=None, log_sigma_x2=None, use_z=False, beta=1,
            )
        self.assertIs(loss, action_loss)
        self.assertEqual(kl.item(), 0)
        self.assertEqual(kl.device, action_loss.device)
        self.assertEqual(kl.dtype, action_loss.dtype)

    def test_enabled_loss_includes_weighted_kl(self) -> None:
        action_loss = torch.tensor(2.0, requires_grad=True)
        mu = torch.ones(2, 1, 2, requires_grad=True)
        log_variance = torch.zeros_like(mu)
        loss, kl = train_v2.get_training_loss(
            action_loss=action_loss, mu=mu, log_sigma_x2=log_variance, use_z=True, beta=0.25,
        )
        torch.testing.assert_close(kl, torch.tensor(1.0))
        torch.testing.assert_close(loss, torch.tensor(2.25))
        loss.backward()
        self.assertIsNotNone(mu.grad)
        with self.assertRaisesRegex(ValueError, "posterior outputs"):
            train_v2.get_training_loss(
                action_loss=action_loss, mu=None, log_sigma_x2=None, use_z=True, beta=1,
            )

    def test_disabled_activation_hooks_exclude_posterior(self) -> None:
        model = self.make_model(use_z=False)
        norms, handles = train_v2.register_activation_norm_hooks(model=model)
        try:
            proprio, targets, images = self.observations()
            model(proprio=proprio, actions=targets, img=images)
            self.assertNotIn("cvae_encoder", norms)
            self.assertIn("transformer_encoder", norms)
        finally:
            for handle in handles:
                handle.remove()


if __name__ == "__main__":
    unittest.main()
