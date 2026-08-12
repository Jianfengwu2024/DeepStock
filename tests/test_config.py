from __future__ import annotations

import unittest

from bnhmm import BNHMMConfig, LabelConfig, ProfileConfig


class ConfigTests(unittest.TestCase):
    def test_model_shape_is_fixed(self) -> None:
        config = BNHMMConfig()
        self.assertEqual(config.latent_states, 10)
        self.assertEqual(len(config.output_states), 5)

    def test_invalid_profile_bins_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ProfileConfig(bins=4)

    def test_label_thresholds_must_be_ordered(self) -> None:
        with self.assertRaises(ValueError):
            LabelConfig(weak_bear_threshold=0.03)


if __name__ == "__main__":
    unittest.main()

