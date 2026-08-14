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
    DuplicateJobNameError,
    JobManager,
    JobRecord,
    WalkForwardJobRequest,
    build_command,
    gpu_memory_for_pid,
)


class WebJobTests(unittest.TestCase):
    def test_job_names_are_case_insensitive_and_unique_until_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = JobManager(directory, Path.cwd())
            job_dir = Path(directory, "existing")
            job_dir.mkdir()
            record = JobRecord(
                id="existing",
                status="completed",
                created_at=datetime.now(timezone.utc),
                output_dir=str(job_dir),
                request=WalkForwardJobRequest(name="Experiment"),
            )
            with manager._lock:
                manager._jobs[record.id] = record

            with self.assertRaises(DuplicateJobNameError):
                manager.submit(WalkForwardJobRequest(name="experiment"))

            self.assertTrue(manager.delete(record.id))
            def fake_command(_request, output_dir):
                script = (
                    "from pathlib import Path; "
                    f"Path({str(output_dir / 'run-summary.json')!r}).write_text("
                    "'{}', encoding='utf-8')"
                )
                return [sys.executable, "-c", script]

            with patch("bnhmm.web.jobs.build_command", side_effect=fake_command):
                replacement = manager.submit(
                    WalkForwardJobRequest(name="experiment")
                )
                deadline = time.time() + 10
                while time.time() < deadline:
                    current = manager.get(replacement.id)
                    if current and current.status in {"completed", "failed"}:
                        break
                    time.sleep(0.01)
            self.assertEqual(current.status, "completed")
            manager.close()

    def test_request_checks_related_model_and_window_values(self) -> None:
        with self.assertRaises(ValidationError):
            WalkForwardJobRequest(d_model=95, heads=4)
        with self.assertRaises(ValidationError):
            WalkForwardJobRequest(minimum_train_sessions=50, tuning_sessions=60)
        with self.assertRaises(ValidationError):
            WalkForwardJobRequest(velocity_bins=8)
        with self.assertRaises(ValidationError):
            WalkForwardJobRequest(latent_states=12)
        with self.assertRaises(ValidationError):
            WalkForwardJobRequest(latent_states=50, transition_rank=51)
        with self.assertRaises(ValidationError):
            WalkForwardJobRequest(forward_sessions=2)
        with self.assertRaises(ValidationError):
            WalkForwardJobRequest(sequence_stride=2)
        with self.assertRaises(ValidationError):
            WalkForwardJobRequest(state_timeframe="mixed")
        self.assertEqual(WalkForwardJobRequest(sequence_length=2).sequence_length, 2)
        self.assertEqual(WalkForwardJobRequest(sequence_length=20).sequence_length, 20)
        with self.assertRaises(ValidationError):
            WalkForwardJobRequest(sequence_length=21)

    def test_command_contains_model_and_policy_parameters(self) -> None:
        request = WalkForwardJobRequest(
            symbols=["000001", "000002"], max_folds=3
        )
        command = build_command(request, Path("/tmp/result"))
        self.assertIn("000001,000002", command)
        self.assertIn("--max-folds", command)
        self.assertNotIn("--allow-short", command)
        self.assertIn("--latent-states", command)
        self.assertIn("--transition-rank", command)
        self.assertIn("20", command)
        self.assertEqual(request.state_timeframe, "daily")
        self.assertIn("--state-timeframe", command)
        self.assertIn("daily", command)
        self.assertIn("--bin-width-bps", command)
        self.assertIn("--max-profile-bins", command)
        self.assertIn("--normalization-lookback-sessions", command)
        self.assertIn("--velocity-bins", command)
        self.assertIn("--velocity-clip", command)
        self.assertIn("--velocity-smoothing", command)
        self.assertIn("--feature-mode", command)
        self.assertIn("--kl-weight", command)
        self.assertIn("--ordinal-weight", command)
        self.assertIn("--pretrain-epochs", command)
        self.assertIn("--pretrain-mask-probability", command)
        self.assertNotIn("--deterministic-training", command)
        self.assertNotIn("shell=True", command)
        self.assertEqual(request.training_scope, "per-symbol")
        self.assertEqual(request.profile_protocol, "phase-space-joint-return-v6")
        self.assertEqual(request.feature_mode, "price-only")
        self.assertEqual(request.sequence_length, 5)
        self.assertEqual(request.sequence_stride, 1)
        self.assertEqual(request.sequence_protocol, "fixed-recent-endpoint-v1")
        self.assertEqual(request.transition_rank, 4)
        self.assertEqual(request.strong_threshold_sigma, 0.75)
        self.assertIn("--strong-threshold-sigma", command)
        self.assertIn("--return-standardization-lookback", command)

        custom_threshold = WalkForwardJobRequest(strong_threshold_sigma=1.0)
        custom_command = build_command(custom_threshold, Path("/tmp/result"))
        threshold_index = custom_command.index("--strong-threshold-sigma") + 1
        self.assertEqual(custom_command[threshold_index], "1.0")

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

    def test_symbol_artifacts_are_scoped_to_the_requested_symbol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = JobManager(directory, Path.cwd())
            job_dir = Path(directory, "completed")
            symbol_dir = job_dir / "symbols" / "000001"
            symbol_dir.mkdir(parents=True)
            report = symbol_dir / "evaluation.json"
            report.write_text("{}", encoding="utf-8")
            record = JobRecord(
                id="completed",
                status="completed",
                created_at=datetime.now(timezone.utc),
                output_dir=str(job_dir),
                request=WalkForwardJobRequest(symbols=["000001"]),
            )
            with manager._lock:
                manager._jobs[record.id] = record

            self.assertEqual(
                manager.symbol_artifact_path(
                    "completed", "000001", "evaluation.json"
                ),
                report.resolve(),
            )
            self.assertIsNone(
                manager.symbol_artifact_path("completed", "../000001", "evaluation.json")
            )
            self.assertIsNone(
                manager.symbol_artifact_path("completed", "000001", "../job.json")
            )
            manager.close()

    def test_jobs_without_state_mode_are_loaded_as_legacy_mixed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job_dir = root / "legacy-job"
            job_dir.mkdir()
            request = WalkForwardJobRequest().model_dump(mode="json")
            request.pop("state_timeframe")
            request.pop("profile_protocol")
            request.pop("sequence_protocol")
            request.pop("training_scope")
            request.pop("latent_states")
            request.pop("feature_mode")
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
            self.assertEqual(
                loaded.request.sequence_protocol, "legacy-overlapping-window-v1"
            )
            self.assertEqual(loaded.request.training_scope, "shared-legacy")
            self.assertEqual(loaded.request.latent_states, 10)
            self.assertEqual(loaded.request.feature_mode, "legacy-entangled")


if __name__ == "__main__":
    unittest.main()
