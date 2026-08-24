from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from fastapi import HTTPException

from bnhmm.web.app import (
    river_manifest,
    river_migration,
    river_slice,
    river_symbol,
    river_topology,
)
from bnhmm.web.jobs import JobManager, JobRecord, WalkForwardJobRequest


class RiverWebTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.manager = JobManager(self.directory.name, Path.cwd())
        self.output = Path(self.directory.name, "river-job")
        river = self.output / "river"
        (river / "topologies").mkdir(parents=True)
        (river / "slices").mkdir()
        np.savez_compressed(
            river / "topologies" / "2025-04.npz",
            month=np.datetime64("2025-04", "M"),
            symbols=np.asarray(("A", "B"), dtype=np.str_),
            row_order=np.asarray((1, 0), dtype=np.int64),
            band_ids=np.asarray((0, 0), dtype=np.int64),
            graph=np.asarray(((0.0, 0.8), (0.8, 0.0))),
        )
        np.savez_compressed(
            river / "slices" / "2025-04-08.npz",
            session=np.datetime64("2025-04-08", "D"),
            symbols=np.asarray(("B", "A"), dtype=np.str_),
            timestamps=np.asarray(
                ("2025-04-08T09:31", "2025-04-08T09:32"),
                dtype="datetime64[m]",
            ),
            velocity=np.asarray(((0.2, -0.1), (0.1, 0.3))),
            valid=np.ones((2, 2), dtype=np.bool_),
        )
        (river / "manifest.json").write_text(
            json.dumps(
                {
                    "protocol": "river-topology-v2",
                    "topologies": [
                        {
                            "month": "2025-04",
                            "file": "topologies/2025-04.npz",
                        }
                    ],
                    "slices": [
                        {
                            "session": "2025-04-08",
                            "topology_month": "2025-04",
                            "file": "slices/2025-04-08.npz",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        (self.output / "evaluation.json").write_text(
            json.dumps(
                {
                    "predictions": [
                        {
                            "symbol": "A",
                            "signal_time": "2025-04-08T15:00:00",
                            "expected_relative_return": 0.01,
                        },
                        {
                            "symbol": "B",
                            "signal_time": "2025-04-08T15:00:00",
                            "expected_relative_return": -0.01,
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        request = WalkForwardJobRequest(
            strategy="sector-flow-bayes",
            training_scope="sector-pooled",
            dataset_id="semiconductor",
        )
        self.record = JobRecord(
            id="river-job",
            status="completed",
            created_at=datetime.now(timezone.utc),
            output_dir=str(self.output),
            request=request,
        )
        with self.manager._lock:
            self.manager._jobs[self.record.id] = self.record
        self.request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(jobs=self.manager))
        )

    def tearDown(self) -> None:
        self.manager.close()
        self.directory.cleanup()

    def test_manifest_and_npz_are_served_as_bounded_json(self) -> None:
        manifest = river_manifest(self.record.id, self.request)
        topology = river_topology(self.record.id, "2025-04", self.request)
        daily = river_slice(self.record.id, "2025-04-08", self.request)

        self.assertEqual(manifest["protocol"], "river-topology-v2")
        self.assertEqual(topology["symbols"], ["A", "B"])
        self.assertEqual(topology["month"], "2025-04")
        self.assertEqual(daily["velocity"][0], [0.2, -0.1])
        self.assertEqual(daily["timestamps"][0], "2025-04-08T09:31")

    def test_river_artifact_cannot_escape_job_directory(self) -> None:
        self.assertIsNotNone(
            self.manager.river_artifact_path(
                self.record.id, "topology", "2025-04"
            )
        )
        self.assertIsNone(
            self.manager.river_artifact_path(
                self.record.id, "topology", "../job.json"
            )
        )

    def test_missing_day_slice_returns_not_found(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            river_slice(self.record.id, "1999-01-01", self.request)

        self.assertEqual(raised.exception.status_code, 404)

    def test_migration_and_symbol_history_remain_compact(self) -> None:
        migration = river_migration(self.record.id, self.request)
        history = river_symbol(self.record.id, "A", self.request)

        self.assertEqual(migration["months"], ["2025-04"])
        self.assertEqual(migration["transitions"], [])
        self.assertEqual(len(history["predictions"]), 1)
        self.assertEqual(history["predictions"][0]["symbol"], "A")


if __name__ == "__main__":
    unittest.main()
