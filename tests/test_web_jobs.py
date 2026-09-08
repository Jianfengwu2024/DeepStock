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
    def test_percolation_job_requires_sector_dataset_and_builds_cpu_command(self) -> None:
        request = WalkForwardJobRequest(
            strategy="percolation-branching",
            training_scope="sector-pooled",
            dataset_id="semiconductor",
            symbols=[f"S{index}" for index in range(20)],
            maximum_lag_minutes=3,
            topology_lookback_sessions=60,
        )

        command = build_command(request, Path("/tmp/percolation"))

        self.assertIn("percolation-branching-walk-forward", command)
        self.assertEqual(command[command.index("--maximum-lag-minutes") + 1], "3")
        self.assertEqual(command[command.index("--topology-lookback-sessions") + 1], "60")
        self.assertEqual(command[command.index("--null-permutations") + 1], "500")
        self.assertEqual(command[command.index("--posterior-paths") + 1], "10000")
        self.assertNotIn("--device", command)
        self.assertNotIn("--epochs", command)

        with self.assertRaises(ValidationError):
            WalkForwardJobRequest(strategy="percolation-branching")
        with self.assertRaises(ValidationError):
            request.model_copy(update={"maximum_lag_minutes": 2}).model_validate(
                request.model_copy(update={"maximum_lag_minutes": 2}).model_dump()
            )

    def test_completed_artifact_directory_without_job_metadata_is_discovered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory, "20260826-PathEfficiencyFK-StrongestThird-20D-5BP")
            output.mkdir()
            (output / "run-summary.json").write_text(
                json.dumps(
                    {
                        "strategy": "boundary-flow-fk",
                        "dataset_id": "semiconductor",
                        "symbols": 96,
                        "topology_lookback_sessions": 60,
                        "relative_strength_lookback_sessions": 20,
                    }
                ),
                encoding="utf-8",
            )
            (output / "boundary-policy-evaluation.json").write_text(
                json.dumps(
                    [
                        {"model_name": "PathEfficiencyFK-60D-Beta0PlusC1-Baseline"},
                        {"model_name": "PathEfficiencyFK-60D-StrongestThird"},
                    ]
                ),
                encoding="utf-8",
            )
            (output / "feynman-kac-predictions.json").write_text("[]", encoding="utf-8")

            manager = JobManager(directory, Path.cwd())
            jobs = manager.list()

            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0].id, output.name)
            self.assertEqual(jobs[0].status, "completed")
            self.assertEqual(jobs[0].request.strategy, "boundary-flow-fk")
            self.assertEqual(jobs[0].request.dataset_id, "semiconductor")
            self.assertEqual(jobs[0].request.topology_lookback_sessions, 60)
            self.assertEqual(
                jobs[0].request.boundary_relative_strength_lookback_sessions, 20
            )
            manager.close()

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
        self.assertEqual(WalkForwardJobRequest().transaction_cost_bps, 5.0)
        self.assertEqual(
            WalkForwardJobRequest().boundary_minimum_direction_probability,
            0.51,
        )
        with self.assertRaises(ValidationError):
            WalkForwardJobRequest(training_scope="sector-pooled")
        sector_request = WalkForwardJobRequest(
            training_scope="sector-pooled", dataset_id="semiconductor"
        )
        self.assertEqual(sector_request.dataset_id, "semiconductor")
        with self.assertRaises(ValidationError):
            WalkForwardJobRequest(d_model=95, heads=4)
        with self.assertRaises(ValidationError):
            WalkForwardJobRequest(minimum_train_sessions=50, tuning_sessions=60)
        with self.assertRaises(ValidationError):
            WalkForwardJobRequest(velocity_bins=8)
        with self.assertRaises(ValidationError):
            WalkForwardJobRequest(velocity_aggregation_minutes=7)
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
        with self.assertRaises(ValidationError):
            WalkForwardJobRequest(
                auxiliary_gate_floor=0.8, auxiliary_gate_initial=0.7
            )
        self.assertNotIn(
            "convection_lookback_sessions", WalkForwardJobRequest.model_fields
        )

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
        self.assertIn("--velocity-aggregation-minutes", command)
        self.assertIn("--velocity-clip", command)
        self.assertIn("--velocity-smoothing", command)
        self.assertIn("--feature-mode", command)
        self.assertIn("--kl-weight", command)
        self.assertIn("--ordinal-weight", command)
        self.assertIn("--pretrain-epochs", command)
        self.assertIn("--pretrain-mask-probability", command)
        self.assertNotIn("--deterministic-training", command)
        self.assertNotIn("shell=True", command)
        self.assertEqual(request.strategy, "bnhmm")
        self.assertEqual(request.training_scope, "per-symbol")
        self.assertEqual(request.profile_protocol, "phase-space-10m-local-flow-v8")
        self.assertEqual(request.velocity_aggregation_minutes, 10)
        self.assertEqual(request.feature_mode, "price-velocity")
        self.assertEqual(request.auxiliary_gate_floor, 0.25)
        self.assertEqual(request.auxiliary_gate_initial, 0.75)
        self.assertEqual(request.auxiliary_warmup_epochs, 3)
        self.assertIn("--auxiliary-gate-floor", command)
        self.assertIn("--auxiliary-gate-initial", command)
        self.assertIn("--auxiliary-warmup-epochs", command)
        self.assertIn("--symmetry-weight", command)
        self.assertEqual(request.symmetry_weight, 0.2)
        self.assertEqual(request.sequence_length, 5)
        self.assertEqual(request.sequence_stride, 1)
        self.assertEqual(request.sequence_protocol, "fixed-recent-endpoint-v1")
        self.assertEqual(request.transition_rank, 4)
        self.assertEqual(request.strong_threshold_sigma, 0.75)
        self.assertIn("--strong-threshold-sigma", command)
        self.assertIn("--return-standardization-lookback", command)

        boundary = WalkForwardJobRequest(
            strategy="boundary-flow-fk",
            training_scope="sector-pooled",
            dataset_id="semiconductor",
        )
        boundary_command = build_command(boundary, Path("/tmp/boundary-result"))
        self.assertNotIn("--convection-lookback-sessions", boundary_command)

        sector = WalkForwardJobRequest(
            training_scope="sector-pooled", dataset_id="semiconductor"
        )
        sector_command = build_command(sector, Path("/tmp/result"))
        self.assertIn("--training-scope", sector_command)
        self.assertIn("sector-pooled", sector_command)
        self.assertIn("--dataset-id", sector_command)
        self.assertIn("semiconductor", sector_command)

        custom_threshold = WalkForwardJobRequest(strong_threshold_sigma=1.0)
        custom_command = build_command(custom_threshold, Path("/tmp/result"))
        threshold_index = custom_command.index("--strong-threshold-sigma") + 1
        self.assertEqual(custom_command[threshold_index], "1.0")

        deterministic = build_command(
            request.model_copy(update={"deterministic_training": True}),
            Path("/tmp/result"),
        )
        self.assertIn("--deterministic-training", deterministic)

    def test_stock_transformer_command_uses_isolated_pipeline(self) -> None:
        request = WalkForwardJobRequest(
            strategy="stock-transformer",
            training_scope="sector-pooled",
            dataset_id="semiconductor",
        )
        command = build_command(request, Path("/tmp/stock-result"))
        self.assertIn("stock-transformer-walk-forward", command)
        self.assertIn("--conservation-grid-points", command)
        self.assertIn("--boundary-threshold", command)
        self.assertIn("--conservation-weight", command)
        self.assertNotIn("--latent-states", command)

    def test_holographic_bnhmm_command_selects_two_dimensional_profile(self) -> None:
        request = WalkForwardJobRequest(
            strategy="holo-bnhmm",
            training_scope="sector-pooled",
            dataset_id="semiconductor",
            profile_protocol="price-volume-holographic-flow-v1",
            latent_states=50,
        )
        command = build_command(request, Path("/tmp/holo-result"))

        self.assertIn("walk-forward", command)
        self.assertIn("--profile-kind", command)
        self.assertIn("holographic", command)
        self.assertIn("--holographic-volume-bins", command)
        self.assertIn("--holographic-smoothing", command)
        self.assertIn("50", command)

        with self.assertRaises(ValidationError):
            request.model_copy(update={"latent_states": 39}).model_validate(
            request.model_copy(update={"latent_states": 39}).model_dump()
            )

    def test_holo_repair_requires_an_implemented_named_variant(self) -> None:
        request = WalkForwardJobRequest(
            strategy="holo-repair",
            repair_variant="HoloProbe-Image",
            training_scope="sector-pooled",
            dataset_id="semiconductor",
            profile_protocol="price-volume-holographic-flow-v1",
            latent_states=50,
        )

        command = build_command(request, Path("/tmp/holo-repair-result"))

        index = command.index("--repair-variant")
        self.assertEqual(command[index + 1], "HoloProbe-Image")
        with self.assertRaises(ValidationError):
            WalkForwardJobRequest(
                strategy="holo-repair",
                training_scope="sector-pooled",
                dataset_id="semiconductor",
                profile_protocol="price-volume-holographic-flow-v1",
                latent_states=50,
            )
        phase = WalkForwardJobRequest(
            strategy="holo-repair",
            repair_variant="HoloPhase",
            training_scope="sector-pooled",
            dataset_id="semiconductor",
            profile_protocol="price-volume-holographic-flow-v1",
            latent_states=50,
        )
        self.assertIn("HoloPhase", build_command(phase, Path("/tmp/holo-phase")))
        resnet = phase.model_copy(
            update={"name": "HoloResNet-001", "repair_variant": "HoloResNet"}
        ).model_validate(
            phase.model_copy(
                update={"name": "HoloResNet-001", "repair_variant": "HoloResNet"}
            ).model_dump()
        )
        self.assertIn("HoloResNet", build_command(resnet, Path("/tmp/holo-resnet")))
        restcn = phase.model_copy(
            update={"name": "HoloResTCN-001", "repair_variant": "HoloResTCN"}
        ).model_validate(
            phase.model_copy(
                update={"name": "HoloResTCN-001", "repair_variant": "HoloResTCN"}
            ).model_dump()
        )
        self.assertIn("HoloResTCN", build_command(restcn, Path("/tmp/holo-restcn")))
        with self.assertRaises(ValidationError):
            WalkForwardJobRequest(
                strategy="holo-repair",
                repair_variant="HoloSticky",
                training_scope="sector-pooled",
                dataset_id="semiconductor",
                profile_protocol="price-volume-holographic-flow-v1",
                latent_states=50,
            )

    def test_holographic_lite_command_uses_fixed_fifteen_channel_contract(self) -> None:
        fields = (
            "density",
            "price_current",
            "volume_current",
            "divergence",
            "vorticity",
        )
        request = WalkForwardJobRequest(
            strategy="holo-bnhmm-lite",
            training_scope="sector-pooled",
            dataset_id="semiconductor",
            profile_protocol="price-volume-holographic-lite-v1",
            holographic_volume_bins=3,
            holographic_fields=fields,
            holographic_conv_layers=1,
            d_model=16,
            latent_states=16,
            layers=1,
            feedforward_dim=32,
        )

        command = build_command(request, Path("/tmp/holo-lite-result"))

        self.assertIn("--holographic-fields", command)
        self.assertIn(",".join(fields), command)
        self.assertIn("--holographic-conv-layers", command)
        self.assertEqual(request.holographic_volume_bins * len(fields), 15)
        with self.assertRaises(ValidationError):
            WalkForwardJobRequest(
                strategy="holo-bnhmm-lite",
                training_scope="sector-pooled",
                dataset_id="semiconductor",
                profile_protocol="price-volume-holographic-lite-v1",
                holographic_volume_bins=3,
                holographic_fields=fields,
                latent_states=14,
            )

    def test_stock_transformer_rejects_non_sector_scope(self) -> None:
        with self.assertRaises(ValidationError):
            WalkForwardJobRequest(strategy="stock-transformer")

    def test_fk_stock_command_contains_only_closed_form_parameters(self) -> None:
        request = WalkForwardJobRequest(
            strategy="fk-stock",
            training_scope="sector-pooled",
            dataset_id="semiconductor",
        )
        command = build_command(request, Path("/tmp/fk-result"))
        self.assertIn("fk-stock-walk-forward", command)
        self.assertNotIn("--fk-drift-lookback-sessions", command)
        self.assertNotIn("--fk-noise-potential-weight", command)
        self.assertNotIn("--conservation-grid-points", command)
        self.assertNotIn("--epochs", command)
        self.assertNotIn("--device", command)
        self.assertNotIn("--d-model", command)
        self.assertNotIn("--latent-states", command)
        self.assertNotIn("--maximum-tokens", command)

    def test_fk_stock_rejects_non_sector_scope(self) -> None:
        with self.assertRaises(ValidationError):
            WalkForwardJobRequest(strategy="fk-stock")

    def test_sector_flow_command_uses_only_causal_flow_parameters(self) -> None:
        request = WalkForwardJobRequest(
            strategy="sector-flow-bayes",
            training_scope="sector-pooled",
            dataset_id="semiconductor",
            minimum_train_sessions=120,
            normalization_lookback_sessions=20,
            normalization_minimum_observations=10,
            maximum_lag_minutes=30,
            factor_process_variance=0.2,
            sector_flow_path_windows=(5, 15, 30, 60),
            sector_flow_ranking_window=60,
            sector_flow_top_k=10,
            sector_index_symbol="881121",
            regime_context_sessions=5,
            regime_hidden_channels=16,
            regime_latent_states=12,
            regime_neutral_sigma=0.35,
            regime_confidence_threshold=0.6,
            regime_epochs=20,
            regime_mc_samples=16,
        )

        command = build_command(request, Path("/tmp/sector-flow"))

        self.assertIn("sector-flow-walk-forward", command)
        self.assertEqual(
            command[command.index("--protocol") + 1], "river-regime-v3"
        )
        self.assertEqual(
            command[command.index("--topology-lookback-sessions") + 1], "60"
        )
        self.assertIn("--maximum-lag-minutes", command)
        self.assertIn("--factor-process-variance", command)
        self.assertIn("--top-k", command)
        self.assertEqual(
            command[command.index("--sector-index-symbol") + 1], "881121"
        )
        self.assertEqual(
            command[command.index("--regime-context-sessions") + 1], "5"
        )
        self.assertEqual(
            command[command.index("--regime-latent-states") + 1], "12"
        )
        self.assertEqual(command[command.index("--regime-epochs") + 1], "20")
        self.assertEqual(
            command[command.index("--regime-mc-samples") + 1], "16"
        )
        self.assertIn("--device", command)
        self.assertNotIn("--d-model", command)
        self.assertNotIn("--latent-states", command)
        with self.assertRaises(ValidationError):
            request.model_copy(
                update={"sector_flow_ranking_window": 45}
            ).model_validate(
                request.model_copy(
                    update={"sector_flow_ranking_window": 45}
                ).model_dump()
            )

    def test_boundary_flow_command_uses_independent_cpu_pipeline(self) -> None:
        request = WalkForwardJobRequest(
            strategy="boundary-flow-fk",
            training_scope="sector-pooled",
            dataset_id="semiconductor",
            minimum_train_sessions=120,
            topology_lookback_sessions=30,
            boundary_maximum_drawdown=0.03,
        )

        command = build_command(request, Path("/tmp/boundary-flow"))

        self.assertIn("boundary-flow-fk-walk-forward", command)
        self.assertEqual(command[command.index("--fk-paths") + 1], "10000")
        self.assertNotIn("--topology-lookback-sessions", command)
        self.assertEqual(command[command.index("--maximum-drawdown") + 1], "0.03")
        self.assertNotIn("--maximum-branch-drawdown", command)
        self.assertIn("--cache-root", command)
        self.assertNotIn("--branch-count", command)
        self.assertNotIn("--viscosity-window", command)
        self.assertNotIn("--minimum-raw-bullish-probability", command)
        self.assertNotIn("--maximum-thickness-risk-quantile", command)
        self.assertNotIn("--top-k", command)
        self.assertNotIn("--epochs", command)
        self.assertNotIn("--device", command)
        self.assertNotIn("--layers", command)
        self.assertNotIn("--latent-states", command)

    def test_direct_path_bypass_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            WalkForwardJobRequest(
                strategy="path-efficiency-direct",
                training_scope="sector-pooled",
                dataset_id="semiconductor",
            )

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
