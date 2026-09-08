from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from bnhmm.web.app import _job_payload, app, config
from bnhmm.web.jobs import JobManager, JobRecord, WalkForwardJobRequest


class WebSectorFlowConfigTests(unittest.TestCase):
    def test_removed_strategy_routes_and_artifacts_are_not_exposed(self) -> None:
        routes = {route.path for route in app.routes}
        self.assertFalse(any("/river" in path for path in routes))

        with tempfile.TemporaryDirectory() as directory:
            manager = JobManager(directory, Path.cwd())
            output = Path(directory, "bnhmm")
            output.mkdir()
            (output / "evaluation.json").write_text("{}", encoding="utf-8")
            (output / "percolation-evaluation.json").write_text(
                "{}", encoding="utf-8"
            )
            (output / "sector-flow-report.md").write_text("old", encoding="utf-8")
            record = JobRecord(
                id="bnhmm",
                status="completed",
                created_at=datetime.now(timezone.utc),
                output_dir=str(output),
                request=WalkForwardJobRequest(),
            )
            with manager._lock:
                manager._jobs[record.id] = record

            payload = _job_payload(manager, record, detail=False)
            manager.close()

        self.assertIn("evaluation.json", payload["artifacts"])
        self.assertNotIn("percolation-evaluation.json", payload["artifacts"])
        self.assertNotIn("sector-flow-report.md", payload["artifacts"])

    def test_config_exposes_only_current_strategy_defaults(self) -> None:
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

        self.assertEqual(
            set(payload["strategy_defaults"]),
            {"bnhmm", "boundary-flow-fk"},
        )
        self.assertNotIn("repair_variants", payload)

    def test_config_has_no_direct_path_bypass(self) -> None:
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

        self.assertNotIn("path-efficiency-direct", payload["strategy_defaults"])

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
        self.assertNotIn("topology_lookback_sessions", defaults)
        self.assertNotIn("boundary_cache_lookbacks", payload)
        self.assertEqual(defaults["minimum_train_sessions"], 120)
        self.assertEqual(defaults["regime_neutral_sigma"], 0.35)
        self.assertEqual(defaults["boundary_fk_paths"], 10_000)
        self.assertEqual(defaults["boundary_maximum_drawdown"], 0.05)
        self.assertEqual(defaults["boundary_drift_half_life_sessions"], 60.0)
        self.assertEqual(defaults["boundary_minimum_direction_probability"], 0.51)
        self.assertEqual(defaults["boundary_minimum_signal_to_noise"], 0.10)
        self.assertEqual(defaults["boundary_minimum_survival_probability"], 0.35)
        self.assertEqual(
            defaults["boundary_relative_strength_lookback_sessions"], 20
        )
        self.assertNotIn("convection_lookback_sessions", defaults)
        self.assertEqual(defaults["device"], "cpu")
        for obsolete in (
            "boundary_branch_count",
            "boundary_viscosity_window",
            "boundary_min_raw_bullish_probability",
            "boundary_min_survival_bullish_probability",
            "boundary_min_trend_survival_probability",
            "boundary_max_bullish_probability_std",
            "boundary_max_thickness_risk_quantile",
            "boundary_maximum_branch_drawdown",
            "sector_flow_top_k",
        ):
            self.assertNotIn(obsolete, defaults)



    def test_job_payload_discovers_one_day_potential_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = JobManager(directory, Path.cwd())
            output = Path(directory, "boundary")
            output.mkdir()
            for name in (
                "path-efficiency-convection-potential-report.md",
                "convection-potential-evaluation.json",
            ):
                (output / name).write_text("test", encoding="utf-8")
            request = WalkForwardJobRequest(
                strategy="boundary-flow-fk",
                training_scope="sector-pooled",
                dataset_id="semiconductor",
            )
            record = JobRecord(
                id="boundary",
                status="completed",
                created_at=datetime.now(timezone.utc),
                output_dir=str(output),
                request=request,
            )
            with manager._lock:
                manager._jobs[record.id] = record

            payload = _job_payload(manager, record, detail=False)
            manager.close()

        self.assertIn(
            "path-efficiency-convection-potential-report.md", payload["artifacts"]
        )
        self.assertIn("convection-potential-evaluation.json", payload["artifacts"])

    def test_job_payload_exposes_sector_direction_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = JobManager(directory, Path.cwd())
            output = Path(directory, "sector-direction")
            output.mkdir()
            (output / "sector-direction-evaluation.json").write_text(
                "{}", encoding="utf-8"
            )
            record = JobRecord(
                id="sector-direction",
                status="completed",
                created_at=datetime.now(timezone.utc),
                output_dir=str(output),
                request=WalkForwardJobRequest(
                    strategy="boundary-flow-fk",
                    training_scope="sector-pooled",
                    dataset_id="semiconductor",
                ),
            )
            with manager._lock:
                manager._jobs[record.id] = record

            payload = _job_payload(manager, record, detail=False)
            manager.close()

        self.assertIn("sector-direction-evaluation.json", payload["artifacts"])


if __name__ == "__main__":
    unittest.main()
