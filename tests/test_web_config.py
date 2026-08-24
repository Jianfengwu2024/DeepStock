from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from bnhmm.web.app import _job_payload, config
from bnhmm.web.jobs import JobManager, JobRecord, WalkForwardJobRequest


class WebSectorFlowConfigTests(unittest.TestCase):
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

    def test_job_payload_discovers_sector_flow_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = JobManager(directory, Path.cwd())
            output = Path(directory, "flow")
            output.mkdir()
            for name in (
                "sector-flow.png",
                "sector-flow-report.md",
                "sector-flow-v2-report.md",
                "ranking.csv",
                "evaluation.json",
                "predictions.json",
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
        self.assertIn("ranking.csv", payload["artifacts"])


if __name__ == "__main__":
    unittest.main()
