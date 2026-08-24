from __future__ import annotations

import unittest

import numpy as np
import torch

from bnhmm.sector_flow.regime_data import SectorRegimeSample, SectorRegimeState
from bnhmm.sector_flow.regime_model import SectorRegimeModelConfig
from bnhmm.sector_flow.regime_training import (
    SectorRegimeTrainer,
    SectorRegimeTrainerConfig,
    state_reconstruction_loss,
)


def sample(index: int, state: SectorRegimeState) -> SectorRegimeSample:
    day = np.datetime64("2025-01-01", "D") + np.timedelta64(index, "D")
    target_day = day + np.timedelta64(1, "D")
    values = np.full((2, 4, 6, 8), index / 10.0, dtype=np.float32)
    values[:, 3] = 1.0
    target = np.full((4, 6, 8), (index + 1) / 10.0, dtype=np.float32)
    target[3] = 1.0
    return SectorRegimeSample(
        images=values,
        next_image=target,
        target_state=state,
        previous_state=SectorRegimeState((int(state) + 2) % 3),
        target_return=(-0.01, 0.0, 0.01)[int(state)],
        return_scale=0.01,
        signal_time=day.astype("datetime64[ns]") + np.timedelta64(15, "h"),
        available_at=target_day.astype("datetime64[ns]") + np.timedelta64(9, "h"),
        target_time=target_day.astype("datetime64[ns]") + np.timedelta64(15, "h"),
    )


class SectorRegimeTrainingTests(unittest.TestCase):
    def test_state_loss_uses_observed_mask(self) -> None:
        predicted = torch.zeros(1, 4, 2, 2)
        target = torch.ones(1, 4, 2, 2)
        target[:, 3] = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])

        observed = state_reconstruction_loss(predicted, target)
        hidden_changed = target.clone()
        hidden_changed[:, :3, 0, 1:] = 100.0
        hidden_changed[:, :3, 1, :] = 100.0

        self.assertTrue(
            torch.allclose(
                observed,
                state_reconstruction_loss(predicted, hidden_changed),
                atol=1e-6,
            )
        )

    def test_fit_uses_only_matured_targets_and_predicts_posterior(self) -> None:
        samples = tuple(
            sample(index, SectorRegimeState(index % 3)) for index in range(8)
        )
        cutoff = samples[4].target_time
        trainer = SectorRegimeTrainer(
            SectorRegimeTrainerConfig(
                epochs=1,
                batch_size=2,
                learning_rate=1e-3,
                state_loss_weight=0.2,
                kl_weight=1e-5,
                mc_samples=3,
                device="cpu",
                seed=7,
            )
        )

        result = trainer.fit(
            samples,
            SectorRegimeModelConfig(
                input_channels=4,
                hidden_channels=8,
                latent_states=5,
                coarse_height=2,
                coarse_minutes=3,
                residual_blocks=1,
                dropout=0.0,
            ),
            training_cutoff=cutoff,
        )
        records = trainer.predict(result, samples[5:])

        self.assertEqual(result.training_count, 5)
        self.assertEqual(len(result.loss_history), 1)
        self.assertAlmostEqual(float(result.class_prior.sum()), 1.0)
        self.assertEqual(len(records), 2)
        self.assertTrue(all(item.training_cutoff == cutoff for item in records))
        self.assertTrue(
            all(np.isclose(item.probability_mean.sum(), 1.0) for item in records)
        )
        self.assertEqual(records[0].next_image_mean.shape, (4, 2, 3))


if __name__ == "__main__":
    unittest.main()
