from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError

from bnhmm.web.jobs import (
    ActiveJobDeletionError,
    JobManager,
    JobRecord,
    WalkForwardJobRequest,
    build_command,
    gpu_memory_for_pid,
)


class WebJobTests(unittest.TestCase):
    def test_request_checks_related_model_and_window_values(self) -> None:
        with self.assertRaises(ValidationError):
            WalkForwardJobRequest(d_model=95, heads=4)
        with self.assertRaises(ValidationError):
            WalkForwardJobRequest(minimum_train_sessions=50, tuning_sessions=60)

    def test_command_contains_optional_policy_parameters(self) -> None:
        request = WalkForwardJobRequest(
            symbols=["000001", "000002"], max_folds=3, allow_short=True
        )
        command = build_command(request, Path("/tmp/result"))
        self.assertIn("000001,000002", command)
        self.assertIn("--max-folds", command)
        self.assertIn("--allow-short", command)
        self.assertEqual(request.state_timeframe, "daily")
        self.assertIn("--state-timeframe", command)
        self.assertIn("daily", command)
        self.assertIn("--bin-width-bps", command)
        self.assertIn("--max-profile-bins", command)
        self.assertIn("--kl-weight", command)
        self.assertIn("--ordinal-weight", command)
        self.assertIn("--pretrain-epochs", command)
        self.assertIn("--pretrain-mask-probability", command)
        self.assertNotIn("--deterministic-training", command)
        self.assertNotIn("shell=True", command)

        deterministic = build_command(
            request.model_copy(update={"deterministic_training": True}),
            Path("/tmp/result"),
        )
        self.assertIn("--deterministic-training", deterministic)

    def test_gpu_memory_is_summed_across_matching_process_rows(self) -> None:
        output = "123, 512\n456, 900\n123, 256\n"
        with patch(
            "bnhmm.web.jobs.subprocess.run",
            return_value=SimpleNamespace(stdout=output),
        ):
            self.assertEqual(gpu_memory_for_pid(123), 768)

    def test_worker_persists_completed_job(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def fake_command(_request, output_dir):
                script = (
                    "from pathlib import Path; "
                    f"Path({str(output_dir / 'run-summary.json')!r}).write_text("
                    "'{\"oos_predictions\": 1}', encoding='utf-8')"
                )
                return [sys.executable, "-c", script]

            with patch("bnhmm.web.jobs.build_command", side_effect=fake_command):
                manager = JobManager(root, Path.cwd())
                record = manager.submit(WalkForwardJobRequest(name="test"))
                deadline = time.time() + 10
                while time.time() < deadline:
                    current = manager.get(record.id)
                    if current and current.status in {"completed", "failed"}:
                        break
                    time.sleep(0.05)
                manager.close()

            self.assertIsNotNone(current)
            self.assertEqual(current.status, "completed")
            stored = json.loads((Path(current.output_dir) / "job.json").read_text())
            self.assertEqual(stored["status"], "completed")

            output_dir = Path(current.output_dir)
            self.assertTrue(manager.delete(current.id))
            self.assertIsNone(manager.get(current.id))
            self.assertFalse(output_dir.exists())

    def test_active_job_cannot_be_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = JobManager(directory, Path.cwd())
            output_dir = Path(directory, "active")
            output_dir.mkdir()
            record = JobRecord(
                id="active",
                status="running",
                created_at=datetime.now(timezone.utc),
                output_dir=str(output_dir),
                request=WalkForwardJobRequest(name="active"),
            )
            with manager._lock:
                manager._jobs[record.id] = record

            with self.assertRaises(ActiveJobDeletionError):
                manager.delete(record.id)

            with manager._lock:
                manager._jobs[record.id].status = "cancelled"
            self.assertTrue(manager.delete(record.id))
            manager.close()

    def test_jobs_without_state_mode_are_loaded_as_legacy_mixed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job_dir = root / "legacy-job"
            job_dir.mkdir()
            request = WalkForwardJobRequest().model_dump(mode="json")
            request.pop("state_timeframe")
            request.pop("profile_protocol")
            request.pop("bin_width_bps")
            request.pop("max_profile_bins")
            payload = {
                "id": "legacy-job",
                "status": "completed",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "output_dir": str(job_dir),
                "request": request,
            }
            (job_dir / "job.json").write_text(json.dumps(payload), encoding="utf-8")

            manager = JobManager(root, Path.cwd())
            loaded = manager.get("legacy-job")
            manager.close()

            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.request.state_timeframe, "mixed")
            self.assertEqual(loaded.request.profile_protocol, "legacy-fixed-bins")


if __name__ == "__main__":
    unittest.main()
