from __future__ import annotations

import unittest

import numpy as np
import torch

from bnhmm.stock_transformer.model import StockTransformerConfig
from bnhmm.stock_transformer.tokenization import (
    TOKEN_FEATURE_DIM,
    ConservationToken,
    TokenizedState,
)
from bnhmm.stock_transformer.training import (
    StockTrainerConfig,
    StockTransformerTrainer,
    StockWindow,
    collate_stock_windows,
)


def window(index: int, target: int) -> StockWindow:
    states = []
    for day in range(3):
        tokens = []
        for token_index in range(2 + day % 2):
            features = np.zeros(TOKEN_FEATURE_DIM, dtype=np.float64)
            features[0] = 1.0 / (2 + day % 2)
            features[1] = (token_index - 0.5) * 0.01
            features[2] = -0.5 + token_index * 0.5
            features[6] = 0.02 * (target - 1.5)
            features[-1] = float(token_index == 0)
            tokens.append(
                ConservationToken(token_index * 2, token_index * 2 + 2, 0, features)
            )
        states.append(
            TokenizedState(
                "000001", np.datetime64("2024-01-01") + np.timedelta64(day, "D"), tuple(tokens)
            )
        )
    return StockWindow(
        symbol="000001",
        states=tuple(states),
        signal_time=np.datetime64("2024-02-01") + np.timedelta64(index, "D"),
        execution_time=np.datetime64("2024-02-02") + np.timedelta64(index, "D"),
        target_time=np.datetime64("2024-02-02") + np.timedelta64(index, "D"),
        target=target,
        forward_return=(target - 1.5) * 0.01,
        standardized_return=(target - 1.5) * 0.4,
        return_location=0.0,
        return_scale=0.02,
    )


class StockTransformerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = StockTransformerConfig(
            d_model=16,
            heads=4,
            layers=1,
            feedforward_dim=32,
            maximum_days=5,
            maximum_tokens=8,
        )

    def test_forward_and_mc_posterior_shapes(self) -> None:
        samples = [window(0, 0), window(1, 3)]
        batch = collate_stock_windows(samples)
        model = self.config.build()
        output = model(batch["token_features"], batch["token_mask"])
        posterior = model.predict_posterior(
            batch["token_features"], batch["token_mask"], mc_samples=3
        )
        self.assertEqual(output.regime_logits.shape, (2, 4))
        self.assertEqual(output.conservation_prediction.shape[:2], (2, 3))
        self.assertEqual(posterior.probability_mean.shape, (2, 4))
        self.assertTrue(
            torch.allclose(posterior.probability_mean.sum(dim=-1), torch.ones(2))
        )

    def test_training_uses_conservation_auxiliary_objective(self) -> None:
        samples = [window(index, index % 4) for index in range(12)]
        model = self.config.build(seed=3)
        trainer = StockTransformerTrainer(
            model,
            StockTrainerConfig(epochs=1, batch_size=4, seed=3),
            device="cpu",
        )
        result = trainer.fit(samples[:8], samples[8:])
        self.assertEqual(result.best_epoch, 1)
        self.assertIn("train_conservation", result.history[0])
        self.assertTrue(np.isfinite(result.best_validation_loss))


if __name__ == "__main__":
    unittest.main()
