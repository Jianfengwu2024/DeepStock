from __future__ import annotations

import unittest
from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import torch

from bnhmm import (
    BarBatch,
    HolographicProfileBuilder,
    ProfileConfig,
    SequenceConfig,
    SequenceWindowBuilder,
)
from bnhmm.config import HOLOGRAPHIC_FIELD_COUNT
from bnhmm.data import ProfileLabelBuilder
from bnhmm.training import BNHMMTrainer, ModelConfig, collate_profile_sequences
from bnhmm.walk_forward import _state_diagnostics


def synthetic_sessions(session_count: int = 25) -> BarBatch:
    timestamps: list[np.datetime64] = []
    prices: list[float] = []
    volumes: list[float] = []
    for session_index in range(session_count):
        day = date(2026, 1, 5) + timedelta(days=session_index)
        for hour, minute in ((9, 30), (13, 0)):
            start = np.datetime64(f"{day.isoformat()}T{hour:02d}:{minute:02d}")
            for offset in range(1, 121):
                phase = (session_index * 240 + len(prices) % 240) / 17.0
                timestamps.append(start + np.timedelta64(offset, "m"))
                prices.append(10.0 + session_index * 0.01 + 0.05 * np.sin(phase))
                volumes.append(1000.0 + 300.0 * (1.0 + np.cos(phase * 0.7)))
    close = np.asarray(prices, dtype=np.float64)
    return BarBatch(
        symbol="000001",
        timestamp=np.asarray(timestamps, dtype="datetime64[ns]"),
        open=close - 0.005,
        high=close + 0.01,
        low=close - 0.01,
        close=close,
        volume=np.asarray(volumes, dtype=np.float64),
    )


def wide_range_sessions(session_count: int = 8) -> BarBatch:
    bars = synthetic_sessions(session_count)
    sessions = bars.timestamp.astype("datetime64[D]")
    last_session = np.unique(sessions)[-1]
    last_mask = sessions == last_session
    phase = np.linspace(-1.0, 1.0, int(last_mask.sum()))
    close = bars.close.copy()
    close[last_mask] = 10.0 * np.exp(0.12 * phase)
    return BarBatch(
        symbol=bars.symbol,
        timestamp=bars.timestamp,
        open=close * 0.999,
        high=close * 1.001,
        low=close * 0.999,
        close=close,
        volume=bars.volume,
    )


class HolographicProfileTests(unittest.TestCase):
    def test_builds_finite_two_dimensional_fields(self) -> None:
        config = ProfileConfig(
            relative_volume_min_observations=3,
            relative_volume_lookback_sessions=3,
            holographic_volume_bins=5,
        )
        states = HolographicProfileBuilder(config).build(synthetic_sessions(8))

        self.assertEqual(len(states), 5)
        for state in states:
            self.assertEqual(
                state.fields.shape,
                (
                    len(state.bin_coordinates),
                    5,
                    HOLOGRAPHIC_FIELD_COUNT,
                ),
            )
            self.assertAlmostEqual(float(state.fields[..., 0].sum()), 1.0)
            self.assertGreater(state.flow_samples, 0)
            self.assertTrue(np.isfinite(state.fields).all())
            self.assertGreater(float(np.abs(state.fields[..., 7]).sum()), 0.0)

    def test_wide_price_range_keeps_derivatives_finite_after_position_clipping(self) -> None:
        config = ProfileConfig(
            relative_volume_min_observations=3,
            relative_volume_lookback_sessions=3,
            holographic_volume_bins=5,
            feature_clip=4.0,
        )

        state = HolographicProfileBuilder(config).build(wide_range_sessions())[-1]

        self.assertGreater(np.count_nonzero(state.normalized_bin_coordinates == -1.0), 1)
        self.assertGreater(np.count_nonzero(state.normalized_bin_coordinates == 1.0), 1)
        self.assertTrue(np.isfinite(state.fields).all())

    def test_gradient_falls_back_for_repeated_coordinates(self) -> None:
        values = np.arange(6, dtype=np.float64).reshape(3, 2)

        gradient = HolographicProfileBuilder._gradient(
            values, np.asarray((-1.0, -1.0, 1.0)), axis=0
        )

        self.assertTrue(np.isfinite(gradient).all())
        np.testing.assert_allclose(gradient, 2.0)

    def test_label_builder_flattens_volume_cells_without_losing_price_axis(self) -> None:
        config = ProfileConfig(
            relative_volume_min_observations=3,
            relative_volume_lookback_sessions=3,
            holographic_volume_bins=5,
        )
        states = HolographicProfileBuilder(config).build(synthetic_sessions())
        observations = ProfileLabelBuilder().build(states)

        self.assertGreater(len(observations), 0)
        self.assertEqual(
            observations.profiles[0].shape,
            (len(states[0].bin_coordinates), 5 * HOLOGRAPHIC_FIELD_COUNT),
        )

    def test_vorticity_operator_detects_counterclockwise_rotation(self) -> None:
        coordinates = np.linspace(-1.0, 1.0, 7)
        x, y = np.meshgrid(coordinates, coordinates, indexing="ij")
        price_current = -y
        volume_current = x

        vorticity = HolographicProfileBuilder._gradient(
            volume_current, coordinates, axis=0
        ) - HolographicProfileBuilder._gradient(
            price_current, coordinates, axis=1
        )

        np.testing.assert_allclose(vorticity, 2.0, atol=1e-12)

    def test_current_session_does_not_change_its_normalization_baseline(self) -> None:
        bars = synthetic_sessions(8)
        sessions = bars.timestamp.astype("datetime64[D]")
        last = np.unique(sessions)[-1]
        changed_volume = bars.volume.copy()
        changed_volume[sessions == last] *= 20.0
        changed = BarBatch(
            symbol=bars.symbol,
            timestamp=bars.timestamp,
            open=bars.open,
            high=bars.high,
            low=bars.low,
            close=bars.close,
            volume=changed_volume,
        )
        config = ProfileConfig(
            relative_volume_min_observations=3,
            relative_volume_lookback_sessions=3,
        )

        original = HolographicProfileBuilder(config).build(bars)[-1]
        modified = HolographicProfileBuilder(config).build(changed)[-1]

        self.assertAlmostEqual(original.average_minute_volume, modified.average_minute_volume)
        self.assertFalse(np.allclose(original.fields, modified.fields))

    def test_holographic_encoder_preserves_bayesian_output_contract(self) -> None:
        config = ModelConfig(
            max_profile_bins=32,
            profile_channels=5 * HOLOGRAPHIC_FIELD_COUNT,
            latent_states=50,
            tokenizer_kind="holographic-2d",
            holographic_volume_bins=5,
            d_model=24,
            heads=4,
            layers=1,
            feedforward_dim=48,
            max_sequence_length=5,
        )
        model = config.build(seed=7)
        values = torch.randn(2, 5, 12, 5 * HOLOGRAPHIC_FIELD_COUNT)
        grid = values.reshape(2, 5, 12, 5, HOLOGRAPHIC_FIELD_COUNT)
        grid[..., 0] = torch.rand(2, 5, 12, 5) + 0.01
        values = grid.reshape_as(values)
        coordinates = torch.arange(12, dtype=torch.float32).view(1, 1, 12).expand(2, 5, 12)
        mask = torch.ones(2, 5, 12, dtype=torch.bool)
        output = model(
            values,
            torch.ones(2, 5, dtype=torch.long),
            torch.randn(2, 5, 9),
            coordinates,
            mask,
            sample=True,
        )

        self.assertEqual(output.latent_probabilities.shape, (2, 5, 50))
        self.assertEqual(output.regime_logits.shape, (2, 5, 4))
        output.return_mean.sum().backward()
        self.assertTrue(any(item.grad is not None for item in model.parameters()))

    def test_holographic_z2_mirror_uses_field_parity(self) -> None:
        profile_config = ProfileConfig(
            relative_volume_min_observations=3,
            relative_volume_lookback_sessions=3,
            holographic_volume_bins=5,
        )
        states = HolographicProfileBuilder(profile_config).build(synthetic_sessions())
        observations = ProfileLabelBuilder().build(states)
        sequence = SequenceWindowBuilder(SequenceConfig(length=5)).build(observations)[0]
        batch = collate_profile_sequences([sequence])
        model = ModelConfig(
            max_profile_bins=512,
            profile_channels=5 * HOLOGRAPHIC_FIELD_COUNT,
            latent_states=50,
            tokenizer_kind="holographic-2d",
            holographic_volume_bins=5,
            max_sequence_length=5,
        ).build(seed=7)
        mirrored_batch = BNHMMTrainer(model)._augment_z2_batch(batch)
        valid = int(batch["profile_mask"][0, 0].sum())
        original = batch["profiles"][0, 0, :valid].reshape(
            valid, 5, HOLOGRAPHIC_FIELD_COUNT
        )
        mirrored = mirrored_batch["profiles"][1, 0, :valid].reshape(
            valid, 5, HOLOGRAPHIC_FIELD_COUNT
        )

        for index in (0, 2, 3, 4, 6):
            torch.testing.assert_close(mirrored[..., index], original.flip(0)[..., index])
        for index in (1, 5, 7):
            torch.testing.assert_close(mirrored[..., index], -original.flip(0)[..., index])
        torch.testing.assert_close(
            mirrored_batch["bin_coordinates"][1, 0, :valid],
            -batch["bin_coordinates"][0, 0, :valid].flip(0),
        )

    def test_holographic_state_diagnostics_use_two_dimensional_fields(self) -> None:
        profile_config = ProfileConfig(
            relative_volume_min_observations=3,
            relative_volume_lookback_sessions=3,
            holographic_volume_bins=5,
        )
        states = HolographicProfileBuilder(profile_config).build(synthetic_sessions())
        observations = ProfileLabelBuilder().build(states)
        sequence = SequenceWindowBuilder(SequenceConfig(length=5)).build(observations)[0]
        model = ModelConfig(
            max_profile_bins=512,
            profile_channels=5 * HOLOGRAPHIC_FIELD_COUNT,
            latent_states=50,
            tokenizer_kind="holographic-2d",
            holographic_volume_bins=5,
            max_sequence_length=5,
        ).build(seed=7)
        prediction = SimpleNamespace(
            probability_mean=(0.25, 0.25, 0.25, 0.25),
            prior_probability=(0.25, 0.25, 0.25, 0.25),
        )

        diagnostics = _state_diagnostics(
            model, [sequence], [prediction], device="cpu"
        )

        self.assertGreater(diagnostics["observed_holographic_cells"], 0)
        self.assertGreaterEqual(diagnostics["mean_flow_magnitude"], 0.0)
        self.assertGreaterEqual(diagnostics["mean_absolute_vorticity"], 0.0)
        self.assertNotIn("mean_z2_flow_order", diagnostics)


if __name__ == "__main__":
    unittest.main()
