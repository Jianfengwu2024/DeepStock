from __future__ import annotations

import unittest

import numpy as np
import torch

from bnhmm.data.dataset import ProfileSequence
from bnhmm.training import (
    BNHMMTrainer,
    ModelConfig,
    TrainerConfig,
    collate_profile_sequences,
)


def sequence(seed: int, length: int = 8, bins: int = 16) -> ProfileSequence:
    random = np.random.default_rng(seed)
    profiles: list[np.ndarray] = []
    coordinates: list[np.ndarray] = []
    for index in range(length):
        state_bins = bins + index % 3
        profile = random.random((state_bins, 3), dtype=np.float32)
        profile[:, :2] /= profile[:, :2].sum(axis=0, keepdims=True)
        profiles.append(profile)
        coordinate_start = -(state_bins // 2)
        coordinates.append(
            np.arange(coordinate_start, coordinate_start + state_bins, dtype=np.float32)
        )
    state_features = random.normal(0, 1, (length, 9)).astype(np.float32)
    signal_times = np.arange(length).astype("timedelta64[D]") + np.datetime64("2026-01-01")
    return ProfileSequence(
        symbol="000001",
        profiles=tuple(profiles),
        bin_coordinates=tuple(coordinates),
        state_features=state_features,
        timeframe_ids=np.arange(length, dtype=np.int64) % 2,
        targets=random.integers(0, 5, length, dtype=np.int64),
        signal_times=signal_times.astype("datetime64[ns]"),
        execution_times=(signal_times + np.timedelta64(1, "h")).astype("datetime64[ns]"),
        target_times=(signal_times + np.timedelta64(1, "D")).astype("datetime64[ns]"),
        entry_prices=np.full(length, 10.0),
        forward_returns=random.normal(0, 0.01, length),
    )


def learnable_sequence(symbol: str, length: int = 25) -> ProfileSequence:
    targets = np.arange(length, dtype=np.int64) % 5
    profiles: list[np.ndarray] = []
    coordinates: list[np.ndarray] = []
    state_features = np.zeros((length, 9), dtype=np.float32)
    for index, target in enumerate(targets):
        profile = np.full((16, 3), 1e-4, dtype=np.float32)
        position = 1 + 3 * target
        profile[position : position + 2, :2] = 1.0
        profile[:, :2] /= profile[:, :2].sum(axis=0, keepdims=True)
        profile[position : position + 2, 2] = 2.0
        profiles.append(profile)
        coordinates.append(np.arange(-8, 8, dtype=np.float32))
        state_features[index, target] = 2.0
    signal_times = np.arange(length).astype("timedelta64[D]") + np.datetime64(
        "2026-01-01"
    )
    return ProfileSequence(
        symbol=symbol,
        profiles=tuple(profiles),
        bin_coordinates=tuple(coordinates),
        state_features=state_features,
        timeframe_ids=np.ones(length, dtype=np.int64),
        targets=targets,
        signal_times=signal_times.astype("datetime64[ns]"),
        execution_times=(signal_times + np.timedelta64(1, "D")).astype(
            "datetime64[ns]"
        ),
        target_times=(signal_times + np.timedelta64(2, "D")).astype(
            "datetime64[ns]"
        ),
        entry_prices=np.ones(length),
        forward_returns=np.zeros(length),
    )


class TrainerTests(unittest.TestCase):
    def test_collation_masks_variable_days_and_sequence_lengths(self) -> None:
        items = [sequence(1, length=8), sequence(2, length=5)]

        batch = collate_profile_sequences(items)

        self.assertEqual(batch["profiles"].ndim, 4)
        self.assertTrue(batch["profile_mask"][0, 0].any())
        self.assertFalse(batch["profile_mask"][1, 5:].any())
        self.assertFalse(batch["valid_mask"][1, 5:].any())

    def test_clear_five_state_patterns_can_be_overfit(self) -> None:
        sequences = [learnable_sequence(str(index)) for index in range(5)]
        model = ModelConfig(
            max_profile_bins=32,
            d_model=16,
            motif_dim=8,
            heads=4,
            layers=1,
            feedforward_dim=32,
            dropout=0.0,
            max_sequence_length=25,
        ).build()
        trainer = BNHMMTrainer(
            model,
            TrainerConfig(
                epochs=50,
                batch_size=5,
                learning_rate=0.01,
                weight_decay=0.0,
                kl_weight=0.0,
                latent_balance_weight=0.0,
                gradient_clip=5.0,
            ),
        )

        trainer.fit_fixed(sequences)
        batch = collate_profile_sequences(sequences)
        with torch.no_grad():
            output = model(
                batch["profiles"],
                batch["timeframe_ids"],
                batch["state_features"],
                batch["bin_coordinates"],
                batch["profile_mask"],
                valid_mask=batch["valid_mask"],
                sample=False,
            )
        accuracy = (
            (output.alpha.argmax(dim=-1) == batch["targets"])[batch["valid_mask"]]
            .float()
            .mean()
        )
        self.assertGreater(float(accuracy), 0.95)

    def test_one_epoch_updates_and_validates_model(self) -> None:
        model = ModelConfig(
            max_profile_bins=32,
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
            max_profile_bins=32,
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
