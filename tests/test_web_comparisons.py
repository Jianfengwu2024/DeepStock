from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from bnhmm.web.comparisons import (
    COMPARISON_WINDOWS,
    ComparisonManager,
    WindowComparisonRequest,
)
from bnhmm.web.jobs import JobRecord, WalkForwardJobRequest


class RecordingJobs:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.records: dict[str, JobRecord] = {}

    def submit(self, request: WalkForwardJobRequest) -> JobRecord:
        identifier = f"job-{len(self.records) + 1}"
        output = self.root / identifier
        output.mkdir()
        record = JobRecord(
            id=identifier,
            status="queued",
            created_at=datetime.now(timezone.utc),
            output_dir=str(output),
            request=request,
        )
        self.records[identifier] = record
        return record.model_copy(deep=True)

    def get(self, identifier: str) -> JobRecord | None:
        record = self.records.get(identifier)
        return record.model_copy(deep=True) if record else None

    def summary(self, identifier: str) -> dict | None:
        return None

    def artifact_path(self, identifier: str, name: str) -> Path | None:
        record = self.records.get(identifier)
        if record is None:
            return None
        path = Path(record.output_dir) / name
        return path if path.is_file() else None


class WindowComparisonTests(unittest.TestCase):
    def request(self) -> WindowComparisonRequest:
        return WindowComparisonRequest(
            name="PathEfficiencyFK-Compare-001",
            request=WalkForwardJobRequest(
                strategy="boundary-flow-fk",
                name="ignored-child-name",
                training_scope="sector-pooled",
                dataset_id="semiconductor",
            ),
        )

    def test_create_submits_seven_fixed_windows_with_unique_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = RecordingJobs(root)
            manager = ComparisonManager(root, jobs)

            comparison = manager.create(self.request())

        self.assertEqual(COMPARISON_WINDOWS, (120, 90, 60, 40, 30, 20, 10))
        self.assertEqual(
            [child.window_sessions for child in comparison.children],
            list(COMPARISON_WINDOWS),
        )
        child_requests = [jobs.records[child.job_id].request for child in comparison.children]
        self.assertEqual(
            [item.name for item in child_requests],
            [f"PathEfficiencyFK-Compare-001-{window}d" for window in COMPARISON_WINDOWS],
        )
        self.assertEqual(
            [item.topology_lookback_sessions for item in child_requests],
            list(COMPARISON_WINDOWS),
        )
        reference = child_requests[0].model_dump(exclude={"name", "topology_lookback_sessions"})
        for child_request in child_requests[1:]:
            self.assertEqual(
                child_request.model_dump(exclude={"name", "topology_lookback_sessions"}),
                reference,
            )

    def test_extend_submits_only_windows_missing_from_legacy_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = RecordingJobs(root)
            manager = ComparisonManager(root, jobs)
            comparison = manager.create(self.request())
            legacy = comparison.model_copy(update={"children": comparison.children[1:]})
            comparison_path = root / "_comparisons" / f"{comparison.id}.json"
            comparison_path.write_text(
                legacy.model_dump_json(indent=2),
                encoding="utf-8",
            )
            reloaded = ComparisonManager(root, jobs)

            extended = reloaded.extend(comparison.id)

        self.assertEqual(len(extended.children), 7)
        self.assertEqual(extended.children[0].window_sessions, 120)
        self.assertEqual(len(jobs.records), 8)
        new_job = jobs.records[extended.children[0].job_id]
        self.assertEqual(new_job.request.name, "PathEfficiencyFK-Compare-001-120d")
        self.assertEqual(new_job.request.topology_lookback_sessions, 120)

    def test_extend_retries_failed_window_without_repeating_successes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = RecordingJobs(root)
            manager = ComparisonManager(root, jobs)
            comparison = manager.create(self.request())
            failed = comparison.children[0]
            jobs.records[failed.job_id].status = "failed"

            extended = manager.extend(comparison.id)

        self.assertEqual(len(extended.children), 7)
        self.assertNotEqual(extended.children[0].job_id, failed.job_id)
        self.assertEqual(len(jobs.records), 8)
        retry_job = jobs.records[extended.children[0].job_id]
        self.assertRegex(
            retry_job.request.name,
            r"^PathEfficiencyFK-Compare-001-120d-retry-[0-9a-f]{6}$",
        )
        self.assertEqual(
            [child.job_id for child in extended.children[1:]],
            [child.job_id for child in comparison.children[1:]],
        )

    def test_records_survive_manager_reload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = RecordingJobs(root)
            created = ComparisonManager(root, jobs).create(self.request())

            reloaded = ComparisonManager(root, jobs).get(created.id)

        self.assertIsNotNone(reloaded)
        self.assertEqual(reloaded.model_dump(mode="json"), created.model_dump(mode="json"))

    def test_payload_aggregates_partial_failures_without_hiding_successes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = RecordingJobs(root)
            manager = ComparisonManager(root, jobs)
            comparison = manager.create(self.request())
            statuses = (
                "completed", "failed", "running", "queued", "cancelled", "completed", "completed"
            )
            for child, status in zip(comparison.children, statuses, strict=True):
                jobs.records[child.job_id].status = status

            payload = manager.payload(comparison.id)

        self.assertEqual(payload["status"], "running")
        self.assertEqual(payload["completed_count"], 3)
        self.assertEqual(payload["finished_count"], 5)
        self.assertEqual(
            [item["status"] for item in payload["windows"]],
            list(statuses),
        )

    def test_only_path_efficiency_fk_can_create_comparison(self) -> None:
        invalid = self.request().model_copy(
            update={"request": WalkForwardJobRequest(strategy="bnhmm")}
        )
        with tempfile.TemporaryDirectory() as directory:
            manager = ComparisonManager(Path(directory), RecordingJobs(Path(directory)))
            with self.assertRaisesRegex(ValueError, "PathEfficiencyFK"):
                manager.create(invalid)

    def test_payload_reads_metrics_and_builds_cost_adjusted_equity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = RecordingJobs(root)
            manager = ComparisonManager(root, jobs)
            comparison = manager.create(self.request())
            child = comparison.children[0]
            record = jobs.records[child.job_id]
            record.status = "completed"
            output = Path(record.output_dir)
            (output / "boundary-policy-evaluation.json").write_text(
                """[{"metrics":{"net_total_return":0.12,"maximum_drawdown":0.04,"drift_direction_accuracy":0.55,"balanced_accuracy":0.48,"average_exposure":0.5,"average_turnover":0.2}}]""",
                encoding="utf-8",
            )
            (output / "feynman-kac-predictions.json").write_text(
                """[{"signal_time":"2025-01-02","gross_strategy_return":0.02,"turnover":0.5},{"signal_time":"2025-01-03","gross_strategy_return":-0.01,"turnover":0.25}]""",
                encoding="utf-8",
            )

            payload = manager.payload(comparison.id)

        result = payload["windows"][0]
        self.assertEqual(result["metrics"]["net_total_return"], 0.12)
        self.assertEqual(result["metrics"]["maximum_drawdown"], 0.04)
        self.assertEqual(
            [point["signal_time"] for point in result["equity_curve"]],
            ["2025-01-02", "2025-01-03"],
        )
        expected = (1 + 0.02 - 0.0005 * 0.5) * (1 - 0.01 - 0.0005 * 0.25)
        self.assertAlmostEqual(result["equity_curve"][-1]["value"], expected)


if __name__ == "__main__":
    unittest.main()
