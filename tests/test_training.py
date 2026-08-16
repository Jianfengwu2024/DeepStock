from __future__ import annotations

import unittest

import numpy as np
import torch

from bnhmm import PROFILE_CHANNELS
from bnhmm.data.dataset import ProfileSequence
from bnhmm.training import (
    BNHMMTrainer,
    ModelConfig,
    TrainerConfig,
    collate_profile_sequences,
    profile_summary_targets,
)


def sequence(seed: int, length: int = 8, bins: int = 16) -> ProfileSequence:
    random = np.random.default_rng(seed)
    profiles: list[np.ndarray] = []
    coordinates: list[np.ndarray] = []
    for index in range(length):
        state_bins = bins + index % 3
        profile = random.random((state_bins, PROFILE_CHANNELS), dtype=np.float32)
        profile[:, :2] /= profile[:, :2].sum(axis=0, keepdims=True)
        profiles.append(profile)
        coordinate_start = -(state_bins // 2)
        coordinates.append(
            np.arange(coordinate_start, coordinate_start + state_bins, dtype=np.float32)
        )
    state_features = random.normal(0, 1, (length, 9)).astype(np.float32)
    forward_returns = random.normal(0, 0.01, length)
    signal_times = np.arange(length).astype("timedelta64[D]") + np.datetime64("2026-01-01")
    return ProfileSequence(
        symbol="000001",
        profiles=tuple(profiles),
        bin_coordinates=tuple(coordinates),
        state_features=state_features,
        timeframe_ids=np.arange(length, dtype=np.int64) % 2,
        targets=random.integers(0, 4, length, dtype=np.int64),
        signal_times=signal_times.astype("datetime64[ns]"),
        execution_times=(signal_times + np.timedelta64(1, "h")).astype("datetime64[ns]"),
        target_times=(signal_times + np.timedelta64(1, "D")).astype("datetime64[ns]"),
        entry_prices=np.full(length, 10.0),
        forward_returns=forward_returns,
        standardized_returns=forward_returns / 0.01,
        return_locations=np.zeros(length),
        return_scales=np.full(length, 0.01),
    )


def learnable_sequence(
    symbol: str, length: int = 25, target_offset: int = 0
) -> ProfileSequence:
    targets = (np.arange(length, dtype=np.int64) + target_offset) % 4
    profiles: list[np.ndarray] = []
    coordinates: list[np.ndarray] = []
    state_features = np.zeros((length, 9), dtype=np.float32)
    for index, target in enumerate(targets):
        profile = np.full((16, PROFILE_CHANNELS), 1e-4, dtype=np.float32)
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
        forward_returns=np.asarray((-0.02, -0.0025, 0.0025, 0.02))[targets],
        standardized_returns=np.asarray((-2.0, -0.25, 0.25, 2.0))[targets],
        return_locations=np.zeros(length),
        return_scales=np.full(length, 0.01),
    )


class TrainerTests(unittest.TestCase):
    def test_self_supervised_pretraining_records_both_objectives(self) -> None:
        model = ModelConfig(
            max_profile_bins=32,
            d_model=16,
            motif_dim=4,
            heads=4,
            layers=1,
            feedforward_dim=24,
            dropout=0.0,
            max_sequence_length=8,
        ).build(seed=7)
        result = BNHMMTrainer(
            model,
            TrainerConfig(
                epochs=1,
                pretrain_epochs=2,
                batch_size=2,
                learning_rate=1e-3,
            ),
        ).fit_fixed([sequence(1), sequence(2)])

        self.assertEqual(len(result.pretraining_history), 2)
        self.assertGreaterEqual(
            result.pretraining_history[-1]["masked_state_loss"], 0.0
        )
        self.assertGreaterEqual(
            result.pretraining_history[-1]["next_state_loss"], 0.0
        )
        self.assertGreaterEqual(
            result.pretraining_history[-1]["masked_profile_loss"], 0.0
        )
        self.assertGreaterEqual(
            result.pretraining_history[-1]["next_profile_loss"], 0.0
        )

    def test_seeded_model_build_is_reproducible(self) -> None:
        config = ModelConfig(
            max_profile_bins=32,
            d_model=16,
            motif_dim=4,
            heads=4,
            layers=1,
            feedforward_dim=24,
            max_sequence_length=8,
        )

        first = config.build(seed=17)
        second = config.build(seed=17)
        different = config.build(seed=18)

        first_state = first.state_dict()
        second_state = second.state_dict()
        different_state = different.state_dict()
        self.assertTrue(
            all(torch.equal(first_state[key], second_state[key]) for key in first_state)
        )
        self.assertTrue(
            any(
                not torch.equal(first_state[key], different_state[key])
                for key in first_state
            )
        )

    def test_deterministic_training_disables_weight_sampling(self) -> None:
        model = ModelConfig(
            max_profile_bins=32,
            d_model=16,
            motif_dim=4,
            heads=4,
            layers=1,
            feedforward_dim=24,
            dropout=0.0,
            max_sequence_length=8,
        ).build(seed=7)
        calls: list[bool] = []
        original_forward = model.forward

        def recording_forward(*args: object, **kwargs: object):
            calls.append(bool(kwargs.get("sample")))
            return original_forward(*args, **kwargs)

        model.forward = recording_forward  # type: ignore[method-assign]
        BNHMMTrainer(
            model,
            TrainerConfig(
                epochs=1,
                pretrain_epochs=0,
                batch_size=1,
                sample_during_training=False,
            ),
        ).fit_fixed([sequence(4)])

        self.assertEqual(calls, [False])

    def test_profile_summaries_are_finite_and_fixed_width(self) -> None:
        batch = collate_profile_sequences(
            [sequence(1, length=8), sequence(2, length=5)]
        )

        summaries = profile_summary_targets(batch)

        self.assertEqual(summaries.shape, (2, 8, PROFILE_CHANNELS * 3))
        self.assertTrue(torch.isfinite(summaries).all())

    def test_effective_batch_size_caps_daily_state_tokens(self) -> None:
        model = ModelConfig(
            max_profile_bins=32,
            d_model=16,
            motif_dim=4,
            heads=4,
            layers=1,
            feedforward_dim=24,
            max_sequence_length=128,
        ).build(seed=7)
        trainer = BNHMMTrainer(
            model,
            TrainerConfig(batch_size=32, max_tokens_per_batch=256),
        )

        self.assertEqual(
            trainer._effective_batch_size([sequence(1, length=80)]),
            3,
        )

    def test_collation_masks_variable_days_and_sequence_lengths(self) -> None:
        items = [sequence(1, length=8), sequence(2, length=5)]

        batch = collate_profile_sequences(items)

        self.assertEqual(batch["profiles"].ndim, 4)
        self.assertTrue(batch["profile_mask"][0, 0].any())
        self.assertFalse(batch["profile_mask"][1, 5:].any())
        self.assertFalse(batch["valid_mask"][1, 5:].any())
        self.assertEqual(int(batch["target_mask"].sum()), 2)
        self.assertTrue(batch["target_mask"][0, 7])
        self.assertTrue(batch["target_mask"][1, 4])

    def test_clear_four_state_patterns_can_be_overfit(self) -> None:
        sequences = [
            learnable_sequence(str(index), target_offset=index % 4)
            for index in range(16)
        ]
        model = ModelConfig(
            max_profile_bins=32,
            d_model=16,
            motif_dim=8,
            heads=4,
            layers=1,
            feedforward_dim=32,
            dropout=0.0,
            max_sequence_length=25,
        ).build(seed=7)
        trainer = BNHMMTrainer(
            model,
            TrainerConfig(
                epochs=200,
                batch_size=16,
                learning_rate=0.003,
                weight_decay=0.0,
                kl_weight=0.0,
                latent_information_weight=0.0,
                gradient_clip=5.0,
                sample_during_training=False,
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
        probability = torch.softmax(output.regime_logits, dim=-1)
        accuracy = (
            (probability.argmax(dim=-1) == batch["targets"])[batch["target_mask"]]
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

    def test_auxiliary_gate_warmup_is_recorded_and_then_released(self) -> None:
        model = ModelConfig(
            max_profile_bins=32,
            d_model=16,
            motif_dim=4,
            heads=4,
            layers=1,
            feedforward_dim=32,
            max_sequence_length=8,
        ).build(seed=7)
        result = BNHMMTrainer(
            model,
            TrainerConfig(
                epochs=2,
                batch_size=1,
                pretrain_epochs=0,
                auxiliary_warmup_epochs=1,
            ),
        ).fit_fixed([sequence(4)])

        self.assertEqual(result.history[0]["auxiliary_warmup"], 1.0)
        self.assertEqual(result.history[0]["relative_volume_gate"], 0.0)
        self.assertEqual(result.history[0]["joint_velocity_gate"], 1.0)
        self.assertEqual(result.history[1]["auxiliary_warmup"], 0.0)
        self.assertEqual(result.history[1]["relative_volume_gate"], 0.0)
        self.assertGreaterEqual(result.history[1]["joint_velocity_gate"], 0.25)

    def test_z2_mirror_is_an_involution(self) -> None:
        model = ModelConfig(
            max_profile_bins=32,
            d_model=16,
            motif_dim=4,
            heads=4,
            layers=1,
            feedforward_dim=32,
            max_sequence_length=8,
        ).build(seed=7)
        trainer = BNHMMTrainer(model, TrainerConfig(epochs=1, batch_size=1))
        batch = collate_profile_sequences([sequence(4)])
        first = trainer._augment_z2_batch(batch)
        mirrored = {
            key: value[1:].clone() for key, value in first.items()
        }
        second = trainer._augment_z2_batch(mirrored)

        for key in (
            "profiles",
            "bin_coordinates",
            "state_features",
            "targets",
            "standardized_returns",
        ):
            self.assertTrue(torch.allclose(second[key][1:], batch[key]))


if __name__ == "__main__":
    unittest.main()
