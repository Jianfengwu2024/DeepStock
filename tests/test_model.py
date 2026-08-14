from __future__ import annotations

import unittest

import torch

from bnhmm import PROFILE_CHANNELS
from bnhmm.model import BNHMMNetwork, SinglePassOutput, evidential_classification_loss


def profiles(
    batch: int = 2,
    sequence: int = 6,
    bins: int = 16,
    channels: int = PROFILE_CHANNELS,
) -> torch.Tensor:
    values = torch.rand(batch, sequence, bins, channels)
    return values / values.sum(dim=-2, keepdim=True)


def state_features(batch: int = 2, sequence: int = 6) -> torch.Tensor:
    return torch.randn(batch, sequence, 9)


class BNHMMNetworkTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = BNHMMNetwork(
            max_profile_bins=32,
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
        features = state_features()
        timeframes = torch.tensor([[0, 0, 0, 1, 0, 1]]).repeat(2, 1)
        output = self.model(values, timeframes, features, sample=True)
        self.assertEqual(output.alpha.shape, (2, 6, 3))
        self.assertEqual(output.latent_probabilities.shape, (2, 6, 50))
        self.assertTrue(torch.all(output.alpha > 1.0))
        self.assertTrue(
            torch.allclose(
                output.latent_probabilities.sum(dim=-1), torch.ones(2, 6), atol=1e-5
            )
        )
        targets = torch.randint(0, 3, (2, 6))
        loss = evidential_classification_loss(output, targets)
        loss.total.backward()
        self.assertTrue(torch.isfinite(loss.total))
        self.assertLess(float(loss.kl.detach()), 10.0)
        self.assertGreaterEqual(float(loss.ordinal.detach()), 0.0)
        self.assertTrue(any(parameter.grad is not None for parameter in self.model.parameters()))

    def test_posterior_separates_uncertainty(self) -> None:
        values = profiles(batch=1)
        features = state_features(batch=1)
        timeframes = torch.tensor([[0, 0, 0, 1, 0, 1]])
        posterior = self.model.predict_posterior(
            values, timeframes, features, mc_samples=4
        )
        self.assertTrue(
            torch.allclose(
                posterior.probability_mean.sum(dim=-1), torch.ones(1, 6), atol=1e-5
            )
        )
        self.assertTrue(torch.all(posterior.aleatoric_variance >= 0))
        self.assertTrue(torch.all(posterior.epistemic_variance >= 0))
        self.assertTrue(torch.all(posterior.expected_score <= 1.0))
        self.assertTrue(torch.all(posterior.expected_score >= -1.0))
        self.assertEqual(self.model.regime_scores.tolist(), [-1.0, 0.0, 1.0])

    def test_ordinal_loss_penalizes_distant_state_errors(self) -> None:
        common = {
            "latent_probabilities": torch.full((1, 1, 50), 0.02),
            "context": torch.zeros(1, 1, 4),
            "kl_divergence": torch.tensor(0.0),
        }
        near = SinglePassOutput(
            alpha=torch.tensor([[[2.0, 9.0, 2.0]]]), **common
        )
        far = SinglePassOutput(
            alpha=torch.tensor([[[2.0, 2.0, 9.0]]]), **common
        )
        target = torch.zeros((1, 1), dtype=torch.long)

        near_loss = evidential_classification_loss(near, target).ordinal
        far_loss = evidential_classification_loss(far, target).ordinal

        self.assertGreater(float(far_loss), float(near_loss))

    def test_future_profile_does_not_change_past_output(self) -> None:
        self.model.eval()
        original = profiles(batch=1)
        changed = original.clone()
        changed[:, 4:] = profiles(batch=1, sequence=2)
        features = state_features(batch=1)
        timeframes = torch.tensor([[0, 0, 0, 1, 0, 1]])
        first = self.model(original, timeframes, features, sample=False)
        second = self.model(changed, timeframes, features, sample=False)
        self.assertTrue(torch.allclose(first.alpha[:, :4], second.alpha[:, :4], atol=1e-6))
        self.assertTrue(
            torch.allclose(
                first.latent_probabilities[:, :4],
                second.latent_probabilities[:, :4],
                atol=1e-6,
            )
        )

    def test_motif_location_changes_daily_encoding(self) -> None:
        values = profiles(batch=1, sequence=1)
        left_coordinates = torch.arange(-12, 4).view(1, 1, 16).float()
        right_coordinates = torch.arange(4, 20).view(1, 1, 16).float()
        mask = torch.ones(1, 1, 16, dtype=torch.bool)
        features = torch.zeros(1, 1, 9)

        left_token = self.model.tokenizer(
            values, left_coordinates, mask, features, sample=False
        )
        right_token = self.model.tokenizer(
            values, right_coordinates, mask, features, sample=False
        )

        self.assertFalse(torch.allclose(left_token, right_token, atol=1e-5))


if __name__ == "__main__":
    unittest.main()

