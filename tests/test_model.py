from __future__ import annotations

import unittest

import torch

from bnhmm import PROFILE_CHANNELS
from bnhmm.model import (
    BNHMMNetwork,
    SinglePassOutput,
    student_t_regime_probabilities,
    return_distribution_loss,
)


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
        self.assertEqual(output.return_mean.shape, (2, 6))
        self.assertEqual(output.return_log_variance.shape, (2, 6))
        self.assertEqual(output.latent_probabilities.shape, (2, 6, 20))
        self.assertEqual(output.regime_logits.shape, (2, 6, 4))
        self.assertFalse(hasattr(self.model, "output_hidden"))
        self.assertEqual(self.model.return_distribution.weight_mu.shape, (2, 20))
        self.assertTrue(torch.all(output.return_log_variance <= 4.0))
        self.assertTrue(
            torch.allclose(
                output.latent_probabilities.sum(dim=-1), torch.ones(2, 6), atol=1e-5
            )
        )
        targets = torch.randint(0, 4, (2, 6))
        returns = torch.randn(2, 6) * 0.01
        loss = return_distribution_loss(output, returns, targets)
        loss.total.backward()
        self.assertTrue(torch.isfinite(loss.total))
        self.assertLess(float(loss.kl.detach()), 10.0)
        self.assertGreaterEqual(float(loss.ordinal.detach()), 0.0)
        self.assertTrue(any(parameter.grad is not None for parameter in self.model.parameters()))
        self.assertGreater(float(self.model.transition_mu.grad.norm()), 1e-4)
        self.assertGreater(
            float(self.model.latent_emission.weight_mu.grad.norm()), 1e-3
        )
        self.assertIsNotNone(self.model.tokenizer.auxiliary_gate_logits.grad)
        self.assertTrue(
            torch.all(self.model.tokenizer.effective_auxiliary_gates() < 0.05)
        )

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
        self.assertTrue(torch.all(posterior.expected_score <= 2.0))
        self.assertTrue(torch.all(posterior.expected_score >= -2.0))
        self.assertEqual(self.model.regime_scores.tolist(), [-2.0, -1.0, 1.0, 2.0])

    def test_initial_evidence_stays_close_to_training_class_prior(self) -> None:
        prior = torch.tensor([0.1, 0.2, 0.5, 0.2])
        self.model.set_class_prior(prior)
        output = self.model(
            profiles(batch=1),
            torch.ones(1, 6, dtype=torch.long),
            state_features(batch=1),
            sample=False,
        )

        probability = torch.softmax(output.regime_logits, dim=-1)
        expected = prior.view(1, 1, 4).expand_as(probability)
        self.assertLess(float((probability - expected).abs().max().detach()), 0.1)
        self.assertGreater(float(self.model.class_weights[0]), 1.0)
        self.assertLess(float(self.model.class_weights[2]), 1.0)

    def test_low_rank_transition_dimensions_match_configuration(self) -> None:
        self.assertEqual(self.model.transition_rank, 4)
        self.assertEqual(
            self.model.transition_source_context.weight_mu.shape,
            (self.model.latent_states * 4, self.model.d_model),
        )
        self.assertEqual(
            self.model.transition_destination_context.weight_mu.shape,
            (self.model.latent_states * 4, self.model.d_model),
        )
        self.assertEqual(
            self.model.transition_mu.shape,
            (self.model.latent_states, self.model.latent_states),
        )

    def test_return_loss_penalizes_distant_errors(self) -> None:
        common = {
            "latent_probabilities": torch.full((1, 1, 20), 0.05),
            "regime_logits": torch.zeros(1, 1, 4),
            "context": torch.zeros(1, 1, 4),
            "kl_divergence": torch.tensor(0.0),
            "return_log_variance": torch.full((1, 1), 0.0),
        }
        near = SinglePassOutput(
            return_mean=torch.tensor([[0.01]]), **common
        )
        far = SinglePassOutput(
            return_mean=torch.tensor([[3.0]]), **common
        )
        targets = torch.ones((1, 1), dtype=torch.long)
        returns = torch.zeros((1, 1))

        near_loss = return_distribution_loss(near, returns, targets).return_nll
        far_loss = return_distribution_loss(far, returns, targets).return_nll

        self.assertGreater(float(far_loss), float(near_loss))

    def test_student_t_probabilities_are_symmetric_and_normalized(self) -> None:
        probability = student_t_regime_probabilities(
            torch.tensor([0.0]), torch.tensor([1.0])
        )

        self.assertTrue(torch.allclose(probability.sum(dim=-1), torch.ones(1)))
        self.assertTrue(torch.allclose(probability[:, 0], probability[:, 3]))
        self.assertTrue(torch.allclose(probability[:, 1], probability[:, 2]))
        self.assertIn(int(probability.argmax(dim=-1)), (1, 2))

    def test_supervised_heads_start_small_nonzero_and_centered(self) -> None:
        self.assertGreater(
            float(self.model.return_distribution.weight_mu.norm().detach()), 0.0
        )
        self.assertLess(
            float(self.model.return_distribution.weight_mu.std().detach()), 0.02
        )
        self.assertGreater(
            float(self.model.regime_evidence.weight_mu.norm().detach()), 0.0
        )
        self.assertLess(
            float(self.model.regime_evidence.weight_mu.std().detach()), 0.02
        )
        self.assertTrue(torch.equal(
            self.model.return_distribution.bias_mu,
            torch.zeros_like(self.model.return_distribution.bias_mu),
        ))
        posterior_std = torch.nn.functional.softplus(
            self.model.return_distribution.weight_rho
        )
        self.assertLess(float(posterior_std.max().detach()), 0.003)

    def test_future_profile_does_not_change_past_output(self) -> None:
        self.model.eval()
        original = profiles(batch=1)
        changed = original.clone()
        changed[:, 4:] = profiles(batch=1, sequence=2)
        features = state_features(batch=1)
        timeframes = torch.tensor([[0, 0, 0, 1, 0, 1]])
        first = self.model(original, timeframes, features, sample=False)
        second = self.model(changed, timeframes, features, sample=False)
        self.assertTrue(
            torch.allclose(
                first.return_mean[:, :4], second.return_mean[:, :4], atol=1e-6
            )
        )
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

    def test_price_only_mode_ignores_volume_and_velocity_channels(self) -> None:
        model = BNHMMNetwork(
            max_profile_bins=32,
            feature_mode="price-only",
            d_model=24,
            motif_dim=8,
            heads=4,
            layers=1,
            feedforward_dim=48,
            dropout=0.0,
            max_sequence_length=12,
        )
        original = profiles(batch=1)
        changed = original.clone()
        changed[..., 1:3] = torch.rand_like(changed[..., 1:3])
        changed[..., 4:] = torch.rand_like(changed[..., 4:])
        features = state_features(batch=1)
        changed_features = features.clone()
        changed_features[..., 6] = changed_features[..., 6] + 100.0
        timeframes = torch.ones(1, 6, dtype=torch.long)

        first = model(original, timeframes, features, sample=False)
        second = model(changed, timeframes, changed_features, sample=False)

        self.assertTrue(
            torch.allclose(first.return_mean, second.return_mean, atol=1e-6)
        )

    def test_zero_auxiliary_features_produce_zero_residual(self) -> None:
        values = torch.zeros(1, 1, 16, 2)
        coordinates = torch.arange(-8, 8).view(1, 1, 16).float()
        positions = self.model.tokenizer._coordinate_features(
            coordinates, values.dtype
        )
        pair_positions = self.model.tokenizer._coordinate_features(
            (coordinates[..., :-1] + coordinates[..., 1:]) / 2.0,
            values.dtype,
        )
        triple_positions = self.model.tokenizer._coordinate_features(
            coordinates[..., 1:-1], values.dtype
        )

        residual = self.model.tokenizer.volume_branch(
            values,
            positions,
            pair_positions,
            triple_positions,
            sample=False,
        )

        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))


if __name__ == "__main__":
    unittest.main()

