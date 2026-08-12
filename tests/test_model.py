from __future__ import annotations

import unittest

import torch

from bnhmm.model import BNHMMNetwork, evidential_classification_loss


def profiles(batch: int = 2, sequence: int = 6, bins: int = 16) -> torch.Tensor:
    values = torch.rand(batch, sequence, bins)
    return values / values.sum(dim=-1, keepdim=True)


class BNHMMNetworkTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = BNHMMNetwork(
            profile_bins=16,
            d_model=24,
            motif_dim=8,
            heads=4,
            layers=2,
            feedforward_dim=48,
            dropout=0.0,
            max_sequence_length=12,
        )

    def test_forward_shapes_probabilities_and_gradient(self) -> None:
        values = profiles()
        timeframes = torch.tensor([[0, 0, 0, 1, 0, 1]]).repeat(2, 1)
        output = self.model(values, timeframes, sample=True)
        self.assertEqual(output.alpha.shape, (2, 6, 5))
        self.assertEqual(output.latent_probabilities.shape, (2, 6, 10))
        self.assertTrue(torch.all(output.alpha > 1.0))
        self.assertTrue(
            torch.allclose(
                output.latent_probabilities.sum(dim=-1), torch.ones(2, 6), atol=1e-5
            )
        )
        targets = torch.randint(0, 5, (2, 6))
        loss = evidential_classification_loss(output, targets)
        loss.total.backward()
        self.assertTrue(torch.isfinite(loss.total))
        self.assertTrue(any(parameter.grad is not None for parameter in self.model.parameters()))

    def test_posterior_separates_uncertainty(self) -> None:
        values = profiles(batch=1)
        timeframes = torch.tensor([[0, 0, 0, 1, 0, 1]])
        posterior = self.model.predict_posterior(
            values, timeframes, mc_samples=4
        )
        self.assertTrue(
            torch.allclose(
                posterior.probability_mean.sum(dim=-1), torch.ones(1, 6), atol=1e-5
            )
        )
        self.assertTrue(torch.all(posterior.aleatoric_variance >= 0))
        self.assertTrue(torch.all(posterior.epistemic_variance >= 0))
        self.assertTrue(torch.all(posterior.expected_score <= 2.0))
        self.assertTrue(torch.all(posterior.expected_score >= -2.0))

    def test_future_profile_does_not_change_past_output(self) -> None:
        self.model.eval()
        original = profiles(batch=1)
        changed = original.clone()
        changed[:, 4:] = profiles(batch=1, sequence=2)
        timeframes = torch.tensor([[0, 0, 0, 1, 0, 1]])
        first = self.model(original, timeframes, sample=False)
        second = self.model(changed, timeframes, sample=False)
        self.assertTrue(torch.allclose(first.alpha[:, :4], second.alpha[:, :4], atol=1e-6))
        self.assertTrue(
            torch.allclose(
                first.latent_probabilities[:, :4],
                second.latent_probabilities[:, :4],
                atol=1e-6,
            )
        )


if __name__ == "__main__":
    unittest.main()

