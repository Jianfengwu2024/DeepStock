from __future__ import annotations

import unittest

from bnhmm import PriceVolumeProfileBuilder
from bnhmm.data.dataset import ProfileLabelBuilder
from bnhmm.walk_forward import WalkForwardConfig, build_walk_forward_folds
from test_profile import synthetic_sessions


class WalkForwardTests(unittest.TestCase):
    def test_folds_keep_inner_tuning_before_outer_test(self) -> None:
        states = PriceVolumeProfileBuilder().build(synthetic_sessions(24))
        observations = ProfileLabelBuilder().build(states)
        config = WalkForwardConfig(
            minimum_train_sessions=10,
            tuning_sessions=3,
            test_sessions=4,
            step_sessions=4,
            max_folds=2,
        )

        folds = build_walk_forward_folds([observations], config)

        self.assertEqual(len(folds), 2)
        for fold in folds:
            self.assertLess(fold.tuning_start, fold.test_start)
            self.assertLess(fold.test_start, fold.test_end)
        self.assertLessEqual(folds[0].test_end, folds[1].test_start)


if __name__ == "__main__":
    unittest.main()
