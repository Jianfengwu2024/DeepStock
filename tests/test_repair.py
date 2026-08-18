from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from bnhmm.config import HOLOGRAPHIC_FIELD_COUNT
from bnhmm.repair import RepairExperimentPipeline, RepairVariantCatalog
from bnhmm.training import ModelConfig
from bnhmm.web.app import config as web_config


def holographic_model(**changes):
    values = {
        "max_profile_bins": 16,
        "profile_channels": 5 * HOLOGRAPHIC_FIELD_COUNT,
        "latent_states": 50,
        "tokenizer_kind": "holographic-2d",
        "holographic_volume_bins": 5,
        "d_model": 16,
        "heads": 4,
        "layers": 1,
        "feedforward_dim": 32,
        "dropout": 0.0,
        "max_sequence_length": 5,
        **changes,
    }
    return ModelConfig(**values).build(seed=7)


def inputs():
    profiles = torch.randn(2, 5, 8, 5 * HOLOGRAPHIC_FIELD_COUNT)
    grid = profiles.reshape(2, 5, 8, 5, HOLOGRAPHIC_FIELD_COUNT)
    grid[..., 0] = torch.rand(2, 5, 8, 5) + 0.01
    profiles = grid.reshape_as(profiles)
    timeframes = torch.ones(2, 5, dtype=torch.long)
    metadata = torch.randn(2, 5, 9)
    coordinates = torch.arange(8, dtype=torch.float32).view(1, 1, 8).expand(2, 5, 8)
    mask = torch.ones(2, 5, 8, dtype=torch.bool)
    return profiles, timeframes, metadata, coordinates, mask


class RepairControlTests(unittest.TestCase):
    def test_catalog_is_ordered_and_rejects_unknown_names(self) -> None:
        names = RepairVariantCatalog.names()

        self.assertEqual(len(names), 11)
        self.assertEqual(names[:4], (
            "HoloProbe-Prior",
            "HoloProbe-Metadata",
            "HoloProbe-Image",
            "HoloProbe-Shuffled",
        ))
        self.assertEqual(
            [item.stage for item in RepairVariantCatalog.all()], list(range(11))
        )
        with self.assertRaises(ValueError):
            RepairVariantCatalog.resolve("does-not-exist")

    def test_prior_control_returns_exact_training_prior(self) -> None:
        model = holographic_model(
            repair_variant="HoloProbe-Prior", control_mode="prior-only"
        )
        prior = torch.tensor([0.1, 0.2, 0.3, 0.4])
        model.set_class_prior(prior)

        output = model(*inputs(), sample=False)

        expected = prior.view(1, 1, 4).expand(2, 5, 4)
        self.assertTrue(
            torch.allclose(torch.softmax(output.regime_logits, dim=-1), expected)
        )
        self.assertTrue(torch.equal(output.return_mean, torch.zeros(2, 5)))

    def test_metadata_control_is_invariant_to_profile_values(self) -> None:
        model = holographic_model(
            repair_variant="HoloProbe-Metadata", control_mode="metadata-only"
        )
        values = inputs()
        changed = (torch.randn_like(values[0]) * 100.0, *values[1:])

        first = model(*values, sample=False)
        second = model(*changed, sample=False)

        self.assertTrue(torch.allclose(first.return_mean, second.return_mean))
        self.assertTrue(torch.allclose(first.regime_logits, second.regime_logits))

    def test_image_control_is_invariant_to_metadata(self) -> None:
        model = holographic_model(
            repair_variant="HoloProbe-Image",
            control_mode="image-only",
            include_metadata=False,
            direct_autoregression=False,
        )
        values = inputs()
        changed = (values[0], values[1], values[2] * 100.0, values[3], values[4])

        first = model(*values, sample=False)
        second = model(*changed, sample=False)

        self.assertTrue(torch.allclose(first.return_mean, second.return_mean))
        self.assertTrue(torch.allclose(first.regime_logits, second.regime_logits))

    def test_shuffled_control_reassigns_images_across_batch(self) -> None:
        shuffled = holographic_model(
            repair_variant="HoloProbe-Shuffled",
            control_mode="shuffled-image",
            include_metadata=False,
            direct_autoregression=False,
        )
        full = holographic_model(
            repair_variant="HoloProbe-Shuffled",
            control_mode="full",
            include_metadata=False,
            direct_autoregression=False,
        )
        full.load_state_dict(shuffled.state_dict())
        values = inputs()

        shuffled_output = shuffled(*values, sample=False)
        full_output = full(*values, sample=False)

        self.assertTrue(
            torch.allclose(shuffled_output.return_mean[0], full_output.return_mean[1])
        )
        self.assertTrue(
            torch.allclose(shuffled_output.regime_logits[0], full_output.regime_logits[1])
        )


class RepairReportingTests(unittest.TestCase):
    @staticmethod
    def summary(name: str, accuracy: float) -> dict[str, object]:
        return {
            "repair_variant": name,
            "dataset_id": "semiconductor",
            "symbols": ["000001", "000002"],
            "folds": [{"fold": 0}],
            "oos_predictions": 20,
            "accuracy": accuracy,
            "majority_baseline": 0.30,
            "balanced_accuracy": 0.28,
            "macro_f1": 0.27,
            "brier_score": 0.72,
            "log_loss": 1.40,
            "baselines": {
                "class_prior_brier": 0.75,
                "class_prior_log_loss": 1.45,
            },
            "return_residual": {"correlation": 0.05},
            "backtest": {"total_return": 0.02, "benchmark_return": 0.01},
            "model_config": {"control_mode": "prior-only", "max_sequence_length": 5},
            "trainer_config": {"epochs": 1, "batch_size": 128, "seed": 7},
            "walk_forward_config": {"minimum_train_sessions": 240},
            "class_support": [5, 5, 5, 5],
            "predicted_class_counts": [5, 5, 5, 5],
            "validation_gate": {"passed": False},
        }

    def test_writes_stage_and_comparison_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stages = []
            for index, name in enumerate(RepairVariantCatalog.names()[:2]):
                stage = root / name
                stage.mkdir()
                (stage / "run-summary.json").write_text(
                    json.dumps(self.summary(name, 0.31 + index * 0.01)),
                    encoding="utf-8",
                )
                stages.append(stage)

            stage_report = RepairExperimentPipeline.write_stage_report(stages[0])
            final_report = RepairExperimentPipeline.write_report(
                reversed(stages), root / "final-report.md"
            )

            self.assertIn("HoloProbe-Prior Stage Report", stage_report.read_text())
            comparison = final_report.read_text()
            self.assertLess(
                comparison.index("HoloProbe-Prior"),
                comparison.index("HoloProbe-Metadata"),
            )
            self.assertIn("Interpretation Rule", comparison)

    def test_web_config_exposes_only_completed_stage_as_implemented(self) -> None:
        jobs = SimpleNamespace(list=lambda: [], root=Path("/tmp/jobs"))
        sectors = SimpleNamespace(list=lambda: [])
        request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(jobs=jobs, sector_datasets=sectors))
        )

        with patch("bnhmm.web.app._devices", return_value=[]):
            payload = web_config(request)

        self.assertEqual(payload["repair_variants"][0]["name"], "HoloProbe-Prior")
        self.assertTrue(all(item["implemented"] for item in payload["repair_variants"][:4]))
        self.assertTrue(all(not item["implemented"] for item in payload["repair_variants"][4:]))


if __name__ == "__main__":
    unittest.main()
