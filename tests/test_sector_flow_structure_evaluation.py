from __future__ import annotations

import unittest

import numpy as np

from bnhmm.sector_flow import MonthlyRiverTopology
from bnhmm.sector_flow.structure_evaluation import (
    STRUCTURAL_ABLATIONS,
    evaluate_topology_structure,
    migration_rate,
)


def structure_topology(bands: tuple[int, ...] = (0, 0, 1, 1)) -> MonthlyRiverTopology:
    similarity = np.array(
        [
            [1.0, 0.9, 0.1, 0.1],
            [0.9, 1.0, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.8],
            [0.1, 0.1, 0.8, 1.0],
        ]
    )
    graph = similarity.copy()
    np.fill_diagonal(graph, 0.0)
    return MonthlyRiverTopology(
        month=np.datetime64("2025-04", "M"),
        training_cutoff=np.datetime64("2025-03-31T15:00"),
        symbols=("A", "B", "C", "D"),
        assigned=np.ones(4, dtype=np.bool_),
        band_ids=np.asarray(bands, dtype=np.int64),
        row_order=np.arange(4, dtype=np.int64),
        common_similarity=similarity,
        residual_similarity=similarity,
        fused_similarity=similarity,
        graph=graph,
        cohesion=np.array([0.9, 0.9, 0.8, 0.8]),
        centrality=graph.sum(axis=1) / 3.0,
    )


class StructureEvaluationTests(unittest.TestCase):
    def test_planted_bands_beat_matched_random_grouping(self) -> None:
        metrics = evaluate_topology_structure(structure_topology(), random_samples=128)

        self.assertGreater(metrics["within_band_similarity"], 0.8)
        self.assertGreater(metrics["synchronization_lift"], 0.2)
        self.assertEqual(metrics["band_count"], 2)

    def test_migration_uses_stable_band_identifiers(self) -> None:
        previous = structure_topology()
        current = structure_topology((0, 1, 1, 1))

        self.assertEqual(migration_rate(previous, current), 0.25)

    def test_declared_ablations_are_fixed(self) -> None:
        self.assertEqual(
            STRUCTURAL_ABLATIONS,
            ("v1", "random", "common", "residual", "unstable", "full"),
        )


if __name__ == "__main__":
    unittest.main()
