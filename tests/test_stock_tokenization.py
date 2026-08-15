from __future__ import annotations

import unittest

import numpy as np

from bnhmm.stock_transformer.conservation import (
    ConservationConfig,
    ConservationLawDiscoverer,
    DensityFrame,
)
from bnhmm.stock_transformer.tokenization import (
    TOKEN_FEATURE_DIM,
    ConservationTokenizer,
    TokenizationConfig,
)


class StockTokenizationTests(unittest.TestCase):
    def setUp(self) -> None:
        config = ConservationConfig(grid_points=32)
        self.discoverer = ConservationLawDiscoverer(config)
        grid = self.discoverer.grid
        self.frames = []
        for day, center in enumerate((-0.35, -0.2, 0.0, 0.2, 0.35), start=1):
            density = (
                np.exp(-0.5 * ((grid - center) / 0.12) ** 2)
                + 0.7 * np.exp(-0.5 * ((grid + 0.45) / 0.09) ** 2)
            )
            density /= density.sum()
            self.frames.append(
                DensityFrame(
                    "000001", np.datetime64(f"2024-01-{day:02d}"), density
                )
            )
        self.law = self.discoverer.fit([self.frames])

    def test_tokens_cover_profile_without_fixed_width_chunks(self) -> None:
        transition = self.discoverer.transform(
            self.frames[-2], self.frames[-1], self.law
        )
        tokenizer = ConservationTokenizer(
            TokenizationConfig(
                minimum_token_bins=2, maximum_tokens=8, boundary_threshold=0.5
            )
        )
        state = tokenizer.tokenize(transition, self.law)

        self.assertGreaterEqual(len(state.tokens), 2)
        self.assertLessEqual(len(state.tokens), 8)
        self.assertEqual(state.tokens[0].start, 0)
        self.assertEqual(state.tokens[-1].stop, 32)
        self.assertEqual(state.features.shape[1], TOKEN_FEATURE_DIM)
        self.assertAlmostEqual(float(state.features[:, 0].sum()), 1.0, places=6)
        widths = {token.stop - token.start for token in state.tokens}
        self.assertGreater(len(widths), 1)

    def test_sentence_ids_are_monotone_and_first_token_starts_sentence(self) -> None:
        transition = self.discoverer.transform(
            self.frames[-2], self.frames[-1], self.law
        )
        state = ConservationTokenizer(
            TokenizationConfig(boundary_threshold=0.5)
        ).tokenize(transition, self.law)
        self.assertEqual(float(state.tokens[0].features[-1]), 1.0)
        self.assertTrue(np.all(np.diff(state.sentence_ids) >= 0))


if __name__ == "__main__":
    unittest.main()
