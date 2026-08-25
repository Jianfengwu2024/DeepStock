from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from bnhmm.web.app import (
    delete_comparison,
    get_comparison,
    list_comparisons,
    submit_comparison,
)
from bnhmm.web.comparisons import ComparisonManager, WindowComparisonRequest
from bnhmm.web.jobs import WalkForwardJobRequest

from test_web_comparisons import RecordingJobs


class WindowComparisonApiTests(unittest.TestCase):
    def test_submit_replaces_database_and_symbols_from_sector_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = RecordingJobs(root)
            comparisons = ComparisonManager(root, jobs)
            dataset = SimpleNamespace(
                dataset_id="semiconductor",
                database_path="/data/semiconductor.duckdb",
                symbols=("000001", "000002"),
            )
            application = SimpleNamespace(
                state=SimpleNamespace(
                    jobs=jobs,
                    comparisons=comparisons,
                    sector_datasets=SimpleNamespace(
                        get=lambda identifier: dataset if identifier == dataset.dataset_id else None
                    ),
                )
            )
            request = SimpleNamespace(app=application)
            payload = WindowComparisonRequest(
                name="PathEfficiencyFK-Compare-001",
                request=WalkForwardJobRequest(
                    strategy="boundary-flow-fk",
                    training_scope="sector-pooled",
                    dataset_id="semiconductor",
                    database_path="ignored.duckdb",
                    symbols=["ignored"],
                ),
            )

            created = submit_comparison(payload, request)
            listed = list_comparisons(request)
            detail = get_comparison(created["id"], request)
            deleted = delete_comparison(created["id"], request)

        self.assertEqual(len(listed), 1)
        self.assertEqual(detail["id"], created["id"])
        self.assertTrue(deleted["deleted"])
        for record in jobs.records.values():
            self.assertEqual(record.request.database_path, dataset.database_path)
            self.assertEqual(record.request.symbols, list(dataset.symbols))


if __name__ == "__main__":
    unittest.main()
