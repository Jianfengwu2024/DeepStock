from __future__ import annotations

import unittest

import torch

from bnhmm.sector_flow.regime_model import (
    BayesianSectorRegimeHMM,
    SectorRegimeModelConfig,
)


class SectorRegimeModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = BayesianSectorRegimeHMM(
            SectorRegimeModelConfig(
                input_channels=4,
                hidden_channels=8,
                latent_states=5,
                coarse_height=2,
                coarse_minutes=3,
                residual_blocks=2,
                dropout=0.0,
            )
        )

    def test_forward_shapes_transition_and_gradient(self) -> None:
        images = torch.randn(3, 4, 4, 6, 8)

        output = self.model(images, sample=True)

        self.assertEqual(output.regime_logits.shape, (3, 3))
        self.assertEqual(output.latent_probabilities.shape, (3, 4, 5))
        self.assertEqual(output.next_latent_probability.shape, (3, 5))
        self.assertEqual(output.next_image.shape, (3, 4, 2, 3))
        self.assertTrue(
            torch.allclose(
                output.latent_probabilities.sum(dim=-1),
                torch.ones(3, 4),
                atol=1e-5,
            )
        )
        transition = self.model.transition_probability(sample=False)
        self.assertTrue(
            torch.allclose(transition.sum(dim=-1), torch.ones(5), atol=1e-6)
        )
        loss = output.regime_logits.square().mean() + output.next_image.square().mean()
        loss.backward()
        self.assertIsNotNone(self.model.transition.weight_mu.grad)
        self.assertGreater(float(self.model.transition.weight_mu.grad.norm()), 0.0)

    def test_filter_combines_previous_state_with_current_evidence(self) -> None:
        evidence = torch.tensor(
            [[[0.9, 0.1], [0.2, 0.8]]], dtype=torch.float32
        )
        transition = torch.tensor(
            [[0.95, 0.05], [0.05, 0.95]], dtype=torch.float32
        )

        filtered = BayesianSectorRegimeHMM.filter_latent(evidence, transition)

        self.assertEqual(filtered.shape, evidence.shape)
        self.assertTrue(torch.allclose(filtered[:, 0], evidence[:, 0], atol=1e-6))
        expected_prior = evidence[:, 0] @ transition
        expected = expected_prior * evidence[:, 1]
        expected = expected / expected.sum(dim=-1, keepdim=True)
        self.assertTrue(torch.allclose(filtered[:, 1], expected, atol=1e-6))
        self.assertLess(float(filtered[0, 1, 1]), float(evidence[0, 1, 1]))
        self.assertTrue(
            torch.allclose(filtered.sum(dim=-1), torch.ones(1, 2), atol=1e-6)
        )

    def test_mc_posterior_separates_uncertainty_and_state_image(self) -> None:
        posterior = self.model.predict_posterior(
            torch.randn(2, 3, 4, 6, 8), mc_samples=5
        )

        self.assertEqual(posterior.probability_mean.shape, (2, 3))
        self.assertEqual(posterior.probability_variance.shape, (2, 3))
        self.assertEqual(posterior.next_image_mean.shape, (2, 4, 2, 3))
        self.assertEqual(posterior.next_image_variance.shape, (2, 4, 2, 3))
        self.assertTrue(
            torch.allclose(
                posterior.probability_mean.sum(dim=-1), torch.ones(2), atol=1e-5
            )
        )
        self.assertTrue(torch.all(posterior.aleatoric_variance >= 0.0))
        self.assertTrue(torch.all(posterior.epistemic_variance >= 0.0))
        self.assertTrue(torch.all(posterior.next_image_variance >= 0.0))


if __name__ == "__main__":
    unittest.main()
