from __future__ import annotations

import unittest

import numpy as np

from bnhmm.sector_flow import (
    AttributionConfig,
    FlowAttributionModel,
    NormalizedSectorPanel,
    SectorFlowPosterior,
)


def attribution_fixture(seed: int = 17) -> tuple[NormalizedSectorPanel, SectorFlowPosterior]:
    rng = np.random.default_rng(seed)
    rows, symbols = 900, 6
    factor = rng.normal(0.0, 1.0, rows)
    velocity = rng.normal(0.0, 1.0, (rows, symbols))
    velocity[:-3, 0] = factor[3:] + rng.normal(0.0, 0.08, rows - 3)
    velocity[2:, 1] = factor[:-2] + rng.normal(0.0, 0.08, rows - 2)
    timestamps = (
        np.datetime64("2025-01-02T09:31")
        + np.arange(rows).astype("timedelta64[m]")
    ).astype("datetime64[ns]")
    valid = np.ones_like(velocity, dtype=np.bool_)
    activity = np.ones_like(velocity)
    panel = NormalizedSectorPanel(
        symbols=tuple(f"S{index}" for index in range(symbols)),
        timestamps=timestamps,
        raw_velocity=velocity.copy(),
        velocity=velocity,
        activity=activity,
        velocity_scale=np.ones_like(velocity),
        volume_baseline=np.ones_like(velocity),
        valid=valid,
        activity_valid=valid.copy(),
    )
    leave_one_out = np.repeat(factor[:, None], symbols, axis=1)
    flow = SectorFlowPosterior(
        symbols=panel.symbols,
        timestamps=timestamps,
        factor_mean=factor,
        factor_variance=np.full(rows, 0.05),
        leave_one_out_mean=leave_one_out,
        leave_one_out_variance=np.full_like(leave_one_out, 0.05),
        activity=np.ones(rows),
        breadth=np.ones(rows),
        coherence=np.ones(rows),
        dispersion=np.zeros(rows),
        flow_strength=factor.copy(),
        valid=np.ones(rows, dtype=np.bool_),
    )
    return panel, flow


class FlowAttributionModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = FlowAttributionModel(
            AttributionConfig(max_lag_minutes=5, ridge_precision=2.0, minimum_rows=100)
        )

    def test_recovers_planted_leader_and_responder_lags(self) -> None:
        panel, flow = attribution_fixture()
        parameters = self.model.fit(panel, flow, np.arange(700))
        summaries = self.model.summarize(parameters)

        self.assertEqual(summaries[0].leader_horizon_minutes, 3)
        self.assertGreater(summaries[0].leader_probability, 0.99)
        self.assertEqual(summaries[1].response_delay_minutes, 2)
        self.assertGreater(summaries[1].response_probability, 0.99)

    def test_unrelated_noise_is_less_directional_and_more_uncertain(self) -> None:
        panel, flow = attribution_fixture()
        summaries = self.model.summarize(
            self.model.fit(panel, flow, np.arange(700))
        )

        leader = summaries[0]
        noise = summaries[4]
        self.assertGreater(leader.leader_probability, noise.leader_probability)
        self.assertLess(leader.leader_coefficient_std, noise.leader_coefficient_std)

    def test_fitted_parameters_ignore_rows_outside_training_selection(self) -> None:
        panel, flow = attribution_fixture()
        training_rows = np.arange(500)
        original = self.model.fit(panel, flow, training_rows)
        changed_velocity = panel.velocity.copy()
        changed_velocity[500:] *= -50.0
        changed_panel = NormalizedSectorPanel(
            symbols=panel.symbols,
            timestamps=panel.timestamps,
            raw_velocity=changed_velocity.copy(),
            velocity=changed_velocity,
            activity=panel.activity,
            velocity_scale=panel.velocity_scale,
            volume_baseline=panel.volume_baseline,
            valid=panel.valid,
            activity_valid=panel.activity_valid,
        )
        changed = self.model.fit(changed_panel, flow, training_rows)

        np.testing.assert_allclose(
            original.leader_coefficient_mean,
            changed.leader_coefficient_mean,
        )
        np.testing.assert_allclose(
            original.response_coefficient_mean,
            changed.response_coefficient_mean,
        )

    def test_online_scores_do_not_change_before_a_future_perturbation(self) -> None:
        panel, flow = attribution_fixture()
        parameters = self.model.fit(panel, flow, np.arange(500))
        original = self.model.score(panel, flow, parameters)
        changed_factor = flow.leave_one_out_mean.copy()
        changed_factor[750:] *= -20.0
        changed_flow = SectorFlowPosterior(
            symbols=flow.symbols,
            timestamps=flow.timestamps,
            factor_mean=flow.factor_mean,
            factor_variance=flow.factor_variance,
            leave_one_out_mean=changed_factor,
            leave_one_out_variance=flow.leave_one_out_variance,
            activity=flow.activity,
            breadth=flow.breadth,
            coherence=flow.coherence,
            dispersion=flow.dispersion,
            flow_strength=flow.flow_strength,
            valid=flow.valid,
        )
        changed = self.model.score(panel, changed_flow, parameters)

        np.testing.assert_allclose(
            original.response_mean[:750],
            changed.response_mean[:750],
            equal_nan=True,
        )


if __name__ == "__main__":
    unittest.main()
