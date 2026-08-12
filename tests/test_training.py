from __future__ import annotations

import unittest

import numpy as np

from bnhmm.data.dataset import ProfileSequence
from bnhmm.training import BNHMMTrainer, ModelConfig, TrainerConfig


def sequence(seed: int, length: int = 8, bins: int = 16) -> ProfileSequence:
    random = np.random.default_rng(seed)
    profiles = random.random((length, bins), dtype=np.float32)
    profiles /= profiles.sum(axis=-1, keepdims=True)
    signal_times = np.arange(length).astype("timedelta64[D]") + np.datetime64("2026-01-01")
    return ProfileSequence(
        symbol="000001",
        profiles=profiles,
        timeframe_ids=np.arange(length, dtype=np.int64) % 2,
        targets=random.integers(0, 5, length, dtype=np.int64),
        signal_times=signal_times.astype("datetime64[ns]"),
        execution_times=(signal_times + np.timedelta64(1, "h")).astype("datetime64[ns]"),
        target_times=(signal_times + np.timedelta64(1, "D")).astype("datetime64[ns]"),
        entry_prices=np.full(length, 10.0),
        forward_returns=random.normal(0, 0.01, length),
    )


class TrainerTests(unittest.TestCase):
    def test_one_epoch_updates_and_validates_model(self) -> None:
        model = ModelConfig(
            profile_bins=16,
            d_model=16,
            motif_dim=4,
            heads=4,
            layers=1,
            feedforward_dim=24,
            dropout=0.0,
            max_sequence_length=8,
        ).build()
        trainer = BNHMMTrainer(
            model,
            TrainerConfig(epochs=1, batch_size=2),
        )

        result = trainer.fit([sequence(1), sequence(2)], [sequence(3)])

        self.assertEqual(result.best_epoch, 1)
        self.assertTrue(np.isfinite(result.best_validation_loss))
        self.assertEqual(len(result.history), 1)

    def test_fixed_epoch_training_does_not_require_validation(self) -> None:
        model = ModelConfig(
            profile_bins=16,
            d_model=16,
            motif_dim=4,
            heads=4,
            layers=1,
            feedforward_dim=24,
            dropout=0.0,
            max_sequence_length=8,
        ).build()
        result = BNHMMTrainer(
            model, TrainerConfig(epochs=1, batch_size=1)
        ).fit_fixed([sequence(4)])
        self.assertEqual(result.best_epoch, 1)
        self.assertTrue(np.isnan(result.best_validation_loss))


if __name__ == "__main__":
    unittest.main()
