from __future__ import annotations

import unittest

from bnhmm import (
    HOLOGRAPHIC_LITE_FIELD_NAMES,
    PROFILE_CHANNELS,
    PROFILE_PROTOCOL,
    SEQUENCE_PROTOCOL,
    REGIME_NAMES,
    REGIME_PROTOCOL,
    BNHMMConfig,
    LabelConfig,
    ProfileConfig,
    SequenceConfig,
)


class ConfigTests(unittest.TestCase):
    def test_model_defaults_to_twenty_hidden_states_and_four_outputs(self) -> None:
        config = BNHMMConfig()
        self.assertEqual(config.latent_states, 20)
        self.assertEqual(len(config.output_states), 4)
        self.assertEqual(
            REGIME_NAMES,
            ("strong_bear", "weak_bear", "weak_bull", "strong_bull"),
        )
        self.assertEqual(
            REGIME_PROTOCOL, "volatility-scaled-prior-evidence-v5"
        )
        with self.assertRaises(ValueError):
            BNHMMConfig(latent_states=PROFILE_CHANNELS - 1)

    def test_invalid_profile_scale_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ProfileConfig(max_profile_bins=4)
        with self.assertRaises(ValueError):
            ProfileConfig(bin_width_bps=0)

    def test_daily_states_are_default_and_timeframes_are_validated(self) -> None:
        self.assertEqual(ProfileConfig().state_timeframes, ("1d",))
        with self.assertRaises(ValueError):
            ProfileConfig(state_timeframes=())
        with self.assertRaises(ValueError):
            ProfileConfig(state_timeframes=("5m",))

    def test_phase_space_defaults_use_ten_prior_sessions(self) -> None:
        config = ProfileConfig()
        self.assertEqual(config.relative_volume_lookback_sessions, 10)
        self.assertEqual(config.relative_volume_min_observations, 10)
        self.assertEqual(config.velocity_bins, 3)
        self.assertEqual(config.velocity_aggregation_minutes, 10)
        self.assertEqual(config.velocity_smoothing, 0.25)
        self.assertEqual(config.profile_channels, PROFILE_CHANNELS)
        self.assertEqual(PROFILE_PROTOCOL, "phase-space-10m-local-flow-v8")
        self.assertEqual(SEQUENCE_PROTOCOL, "fixed-recent-endpoint-v1")
        with self.assertRaises(ValueError):
            ProfileConfig(velocity_bins=8)
        with self.assertRaises(ValueError):
            ProfileConfig(velocity_aggregation_minutes=7)

    def test_holographic_lite_has_fifteen_ordered_channels(self) -> None:
        config = ProfileConfig(
            holographic_volume_bins=3,
            holographic_fields=HOLOGRAPHIC_LITE_FIELD_NAMES,
        )

        self.assertEqual(config.holographic_profile_channels, 15)
        with self.assertRaises(ValueError):
            ProfileConfig(holographic_fields=("price_current", "density"))
        with self.assertRaises(ValueError):
            ProfileConfig(holographic_fields=("density", "density"))

    def test_label_standardization_is_validated(self) -> None:
        with self.assertRaises(ValueError):
            LabelConfig(strong_threshold_sigma=0.0)
        with self.assertRaises(ValueError):
            LabelConfig(
                standardization_lookback_sessions=5,
                standardization_min_observations=6,
            )
        self.assertEqual(LabelConfig().thresholds, (-0.75, 0.0, 0.75))

    def test_sequence_defaults_to_five_recent_trading_days(self) -> None:
        config = SequenceConfig()
        self.assertEqual((config.length, config.stride), (5, 1))
        with self.assertRaises(ValueError):
            SequenceConfig(length=21)


if __name__ == "__main__":
    unittest.main()

