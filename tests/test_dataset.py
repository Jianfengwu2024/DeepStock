from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from bnhmm import LabelConfig, PriceVolumeProfileBuilder, ProfileConfig, SequenceConfig
from bnhmm.data.dataset import ProfileLabelBuilder, SequenceWindowBuilder
from test_profile import synthetic_sessions


class ProfileDatasetTests(unittest.TestCase):
    def test_daily_signals_execute_on_next_sessions_first_tradable_minute(self) -> None:
        states = PriceVolumeProfileBuilder().build(synthetic_sessions(13))
        observations = ProfileLabelBuilder(LabelConfig(forward_sessions=1)).build(states)

        self.assertEqual(len(observations), 2)
        self.assertEqual(len(observations.profiles), 2)
        self.assertEqual(len(observations.bin_coordinates), 2)
        self.assertTrue(all(profile.shape[1] == 14 for profile in observations.profiles))
        self.assertTrue(
            all(
                len(profile) == len(coordinates)
                for profile, coordinates in zip(
                    observations.profiles, observations.bin_coordinates, strict=True
                )
            )
        )
        self.assertEqual(observations.state_features.shape, (2, 9))
        self.assertTrue(np.all(np.isfinite(observations.state_features)))
        self.assertTrue(np.all(observations.timeframe_ids == 1))
        self.assertTrue(np.all(observations.execution_times > observations.signal_times))
        self.assertTrue(
            np.all(
                observations.execution_times.astype("datetime64[D]")
                > observations.signal_times.astype("datetime64[D]")
            )
        )

    def test_labels_use_state_end_price_and_future_session_close(self) -> None:
        states = PriceVolumeProfileBuilder(
            ProfileConfig(state_timeframes=("60m", "1d"))
        ).build(synthetic_sessions(13))
        dates = sorted({state.period_start.astype("datetime64[D]") for state in states})
        closes = dict(zip(dates, (101.0, 97.0, 100.0), strict=True))
        states = [
            replace(
                state,
                start_price=closes[state.period_start.astype("datetime64[D]")],
                end_price=closes[state.period_start.astype("datetime64[D]")],
            )
            for state in states
        ]

        observations = ProfileLabelBuilder(LabelConfig(forward_sessions=1)).build(states)

        self.assertEqual(len(observations), 10)
        np.testing.assert_array_equal(observations.targets[:5], [0, 0, 0, 1, 1])
        np.testing.assert_array_equal(observations.targets[5:], [2, 2, 2, 1, 1])
        self.assertTrue(
            np.all(observations.execution_times > observations.signal_times)
        )

    def test_temporal_split_purges_labels_crossing_cutoff(self) -> None:
        states = PriceVolumeProfileBuilder(
            ProfileConfig(state_timeframes=("60m", "1d"))
        ).build(synthetic_sessions(14))
        observations = ProfileLabelBuilder().build(states)
        dates = np.unique(observations.signal_times.astype("datetime64[D]"))

        train, validation = observations.temporal_split(dates[2])

        self.assertEqual(len(train), 5)
        self.assertEqual(len(validation), 5)
        self.assertTrue(np.all(train.target_times < dates[2]))
        self.assertTrue(np.all(validation.signal_times >= dates[2]))

    def test_sequence_windows_include_the_tail_once(self) -> None:
        states = PriceVolumeProfileBuilder(
            ProfileConfig(state_timeframes=("60m", "1d"))
        ).build(synthetic_sessions(15))
        observations = ProfileLabelBuilder().build(states)
        windows = SequenceWindowBuilder(
            SequenceConfig(length=7, stride=5, minimum_length=4)
        ).build(observations)

        self.assertTrue(windows)
        self.assertLessEqual(max(map(len, windows)), 7)
        self.assertEqual(windows[-1].signal_times[-1], observations.signal_times[-1])


if __name__ == "__main__":
    unittest.main()
