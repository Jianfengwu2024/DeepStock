from __future__ import annotations

import unittest

from bnhmm import (
    PROFILE_CHANNELS,
    PROFILE_PROTOCOL,
    REGIME_NAMES,
    BNHMMConfig,
    LabelConfig,
    ProfileConfig,
)


class ConfigTests(unittest.TestCase):
    def test_model_defaults_to_fifty_hidden_states_and_three_outputs(self) -> None:
        config = BNHMMConfig()
        self.assertEqual(config.latent_states, 50)
        self.assertEqual(len(config.output_states), 3)
        self.assertEqual(REGIME_NAMES, ("bear", "flat", "bull"))
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
        self.assertEqual(config.velocity_bins, 9)
        self.assertEqual(config.profile_channels, PROFILE_CHANNELS)
        self.assertEqual(PROFILE_PROTOCOL, "phase-space-log-v3")
        with self.assertRaises(ValueError):
            ProfileConfig(velocity_bins=8)

    def test_label_thresholds_must_be_ordered(self) -> None:
        with self.assertRaises(ValueError):
            LabelConfig(bear_threshold=0.01, bull_threshold=0.005)
        self.assertEqual(LabelConfig().thresholds, (-0.005, 0.005))


if __name__ == "__main__":
    unittest.main()

