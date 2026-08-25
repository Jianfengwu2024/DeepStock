from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from bnhmm.web.app import _job_payload, config
from bnhmm.web.jobs import JobManager, JobRecord, WalkForwardJobRequest


class WebSectorFlowConfigTests(unittest.TestCase):
    def test_config_exposes_boundary_flow_specific_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = JobManager(directory, Path.cwd())
            request = SimpleNamespace(
                app=SimpleNamespace(
                    state=SimpleNamespace(
                        jobs=manager,
                        sector_datasets=SimpleNamespace(list=lambda: []),
                    )
                )
            )

            payload = config(request)
            manager.close()

        defaults = payload["strategy_defaults"]["boundary-flow-fk"]
        self.assertEqual(defaults["strategy"], "boundary-flow-fk")
        self.assertEqual(defaults["topology_lookback_sessions"], 60)
        self.assertEqual(defaults["minimum_train_sessions"], 120)
        self.assertEqual(defaults["regime_neutral_sigma"], 0.35)
        self.assertEqual(defaults["boundary_fk_paths"], 10_000)
        self.assertEqual(defaults["boundary_drift_half_life_sessions"], 60.0)
        self.assertEqual(defaults["boundary_minimum_direction_probability"], 0.60)
        self.assertEqual(defaults["boundary_minimum_signal_to_noise"], 0.10)
        self.assertEqual(defaults["boundary_minimum_survival_probability"], 0.35)
        self.assertEqual(defaults["device"], "cpu")
        for obsolete in (
            "boundary_branch_count",
            "boundary_viscosity_window",
            "boundary_min_raw_bullish_probability",
            "boundary_min_survival_bullish_probability",
            "boundary_min_trend_survival_probability",
            "boundary_max_bullish_probability_std",
            "boundary_max_thickness_risk_quantile",
            "sector_flow_top_k",
        ):
            self.assertNotIn(obsolete, defaults)

    def test_config_exposes_sector_flow_specific_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = JobManager(directory, Path.cwd())
            request = SimpleNamespace(
                app=SimpleNamespace(
                    state=SimpleNamespace(
                        jobs=manager,
                        sector_datasets=SimpleNamespace(list=lambda: []),
                    )
                )
            )

            payload = config(request)
            manager.close()

        defaults = payload["strategy_defaults"]["sector-flow-bayes"]
        self.assertEqual(defaults["strategy"], "sector-flow-bayes")
        self.assertEqual(defaults["normalization_lookback_sessions"], 20)
        self.assertEqual(defaults["maximum_lag_minutes"], 30)
        self.assertEqual(defaults["sector_flow_path_windows"], [5, 15, 30, 60])
        self.assertEqual(defaults["sector_flow_protocol"], "river-regime-v3")
        self.assertEqual(defaults["sector_index_symbol"], "881121")
        self.assertEqual(defaults["regime_context_sessions"], 5)
        self.assertEqual(defaults["regime_hidden_channels"], 16)
        self.assertEqual(defaults["regime_latent_states"], 12)
        self.assertEqual(defaults["regime_neutral_sigma"], 0.35)
        self.assertEqual(defaults["regime_confidence_threshold"], 0.6)

    def test_job_payload_discovers_sector_flow_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = JobManager(directory, Path.cwd())
            output = Path(directory, "flow")
            output.mkdir()
            for name in (
                "sector-flow.png",
                "sector-flow-report.md",
                "sector-flow-v2-report.md",
                "structure-evaluation.json",
                "ranking.csv",
                "evaluation.json",
                "predictions.json",
                "sector-regime-report.md",
                "sector-regime-predictions.json",
                "sector-regime-evaluation.json",
                "sector-policy-evaluation.json",
                "sector-state-forecasts.npz",
            ):
                (output / name).write_text("test", encoding="utf-8")
            request = WalkForwardJobRequest(
                strategy="sector-flow-bayes",
                training_scope="sector-pooled",
                dataset_id="semiconductor",
            )
            record = JobRecord(
                id="flow",
                status="completed",
                created_at=datetime.now(timezone.utc),
                output_dir=str(output),
                request=request,
            )
            with manager._lock:
                manager._jobs[record.id] = record

            payload = _job_payload(manager, record, detail=False)
            manager.close()

        self.assertIn("sector-flow.png", payload["artifacts"])
        self.assertIn("sector-flow-report.md", payload["artifacts"])
        self.assertIn("sector-flow-v2-report.md", payload["artifacts"])
        self.assertIn("structure-evaluation.json", payload["artifacts"])
        self.assertIn("ranking.csv", payload["artifacts"])
        self.assertIn("sector-regime-report.md", payload["artifacts"])
        self.assertIn("sector-regime-evaluation.json", payload["artifacts"])
        self.assertIn("sector-policy-evaluation.json", payload["artifacts"])


if __name__ == "__main__":
    unittest.main()
