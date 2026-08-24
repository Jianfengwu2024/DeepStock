from __future__ import annotations

import unittest

import numpy as np

from bnhmm.sector_flow import (
    DailyRiverSlice,
    MonthlyRiverTopology,
    SectorGraphStateModel,
)


def graph_topology() -> MonthlyRiverTopology:
    graph = np.array(
        [[0.0, 0.9, 0.0], [0.9, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    return MonthlyRiverTopology(
        month=np.datetime64("2025-02", "M"),
        training_cutoff=np.datetime64("2025-01-31T15:00"),
        symbols=("A", "B", "C"),
        assigned=np.ones(3, dtype=np.bool_),
        band_ids=np.zeros(3, dtype=np.int64),
        row_order=np.arange(3, dtype=np.int64),
        common_similarity=np.eye(3),
        residual_similarity=np.eye(3),
        fused_similarity=np.eye(3),
        graph=graph,
        cohesion=np.array([0.9, 0.9, 0.0]),
        centrality=graph.sum(axis=1) / 2.0,
    )


def graph_slice(
    values: tuple[float, float, float],
    *,
    activity: float = 1.0,
    session: str = "2025-02-05",
) -> DailyRiverSlice:
    slots = np.concatenate((np.arange(571, 691), np.arange(781, 901)))
    timestamps = np.asarray(
        [np.datetime64(session, "D") + np.timedelta64(int(slot), "m") for slot in slots],
        dtype="datetime64[ns]",
    )
    velocity = np.repeat(np.asarray(values, dtype=np.float64)[:, None], 240, axis=1)
    return DailyRiverSlice(
        session=np.datetime64(session, "D"),
        topology_month=np.datetime64(session, "M"),
        symbols=("A", "B", "C"),
        timestamps=timestamps,
        velocity=velocity,
        residual_velocity=velocity.copy(),
        activity=np.full_like(velocity, activity),
        valid=np.ones_like(velocity, dtype=np.bool_),
    )


class SectorGraphStateModelTests(unittest.TestCase):
    def test_graph_prior_smooths_connected_stocks(self) -> None:
        state = SectorGraphStateModel().infer_state(
            graph_topology(), graph_slice((1.0, 0.0, -1.0))
        )

        self.assertLess(
            abs(state.stock_mean[0] - state.stock_mean[1]),
            abs(state.stock_mean[1] - state.stock_mean[2]),
        )

    def test_activity_changes_uncertainty_without_flipping_direction(self) -> None:
        model = SectorGraphStateModel()
        low = model.infer_state(
            graph_topology(), graph_slice((1.0, 0.2, -1.0), activity=0.5)
        )
        high = model.infer_state(
            graph_topology(), graph_slice((1.0, 0.2, -1.0), activity=2.0)
        )

        np.testing.assert_array_equal(np.sign(low.stock_mean), np.sign(high.stock_mean))
        self.assertTrue(np.all(high.stock_variance < low.stock_variance))

    def test_fit_and_predict_preserve_next_session_timing(self) -> None:
        model = SectorGraphStateModel()
        topology = graph_topology()
        training = [
            (topology, graph_slice((1.0, 0.0, -1.0), session="2025-02-03"), np.array([0.02, 0.0, -0.02])),
            (topology, graph_slice((0.5, 0.1, -0.6), session="2025-02-04"), np.array([0.01, 0.0, -0.01])),
            (topology, graph_slice((-0.5, 0.0, 0.5), session="2025-02-05"), np.array([-0.01, 0.0, 0.01])),
        ]
        parameters = model.fit(
            training,
            training_cutoff=np.datetime64("2025-02-05T15:00"),
        )

        prediction = model.predict(
            parameters,
            topology,
            graph_slice((0.4, 0.1, -0.5), session="2025-02-06"),
            available_at=np.datetime64("2025-02-07T09:31"),
        )

        self.assertEqual(prediction.signal_time, np.datetime64("2025-02-06T15:00"))
        self.assertEqual(prediction.available_at, np.datetime64("2025-02-07T09:31"))
        self.assertLess(prediction.training_cutoff, prediction.signal_time)
        self.assertEqual(prediction.expected_relative_return.shape, (3,))
        self.assertTrue(np.all(prediction.predictive_variance > 0.0))


if __name__ == "__main__":
    unittest.main()
