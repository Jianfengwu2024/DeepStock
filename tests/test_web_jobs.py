from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError

from bnhmm.web.jobs import (
    JobManager,
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
        self.assertNotIn("shell=True", command)

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


if __name__ == "__main__":
    unittest.main()
