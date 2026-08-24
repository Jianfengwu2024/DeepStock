"""Training and posterior prediction for the sector-regime BNHMM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.nn import functional as F

from .regime_data import SectorRegimeSample
from .regime_evaluation import MODEL_VERSION, SectorRegimePredictionRecord
from .regime_model import BayesianSectorRegimeHMM, SectorRegimeModelConfig


@dataclass(frozen=True, slots=True)
class SectorRegimeTrainerConfig:
    epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 3e-4
    kl_weight: float = 1e-4
    state_loss_weight: float = 0.25
    gradient_clip: float = 5.0
    mc_samples: int = 16
    device: str = "cpu"
    seed: int = 7

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.learning_rate <= 0.0:
            raise ValueError("sector regime training settings must be positive")
        if self.kl_weight < 0.0 or self.state_loss_weight < 0.0:
            raise ValueError("sector regime loss weights cannot be negative")
        if self.gradient_clip <= 0.0 or self.mc_samples < 2:
            raise ValueError("sector regime gradient and posterior settings are invalid")


@dataclass(frozen=True, slots=True)
class SectorRegimeTrainingResult:
    model: BayesianSectorRegimeHMM
    training_cutoff: np.datetime64
    training_count: int
    class_prior: NDArray[np.float64]
    loss_history: tuple[float, ...]


def state_reconstruction_loss(predicted: Tensor, target: Tensor) -> Tensor:
    if predicted.shape != target.shape or predicted.ndim != 4 or predicted.shape[1] != 4:
        raise ValueError("sector state reconstruction tensors must be aligned four-channel images")
    mask = target[:, 3:4].clamp(0.0, 1.0)
    flow_error = F.smooth_l1_loss(predicted[:, :3], target[:, :3], reduction="none")
    observed = mask.expand_as(flow_error)
    flow_loss = (flow_error * observed).sum() / observed.sum().clamp_min(1.0)
    mask_loss = F.binary_cross_entropy_with_logits(predicted[:, 3:4], mask)
    return flow_loss + 0.1 * mask_loss


class SectorRegimeTrainer:
    def __init__(self, config: SectorRegimeTrainerConfig | None = None) -> None:
        self.config = config or SectorRegimeTrainerConfig()

    def fit(
        self,
        samples: Sequence[SectorRegimeSample],
        model_config: SectorRegimeModelConfig,
        *,
        training_cutoff: np.datetime64,
    ) -> SectorRegimeTrainingResult:
        cutoff = np.datetime64(training_cutoff, "ns")
        matured = tuple(item for item in samples if item.target_time <= cutoff)
        if len(matured) < 2:
            raise ValueError("sector regime training requires at least two matured targets")
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        device = torch.device(self.config.device)
        model = BayesianSectorRegimeHMM(model_config).to(device)
        images = torch.as_tensor(np.stack([item.images for item in matured]), device=device)
        next_images = torch.as_tensor(
            np.stack([item.next_image for item in matured]), device=device
        )
        targets = torch.as_tensor(
            [int(item.target_state) for item in matured], device=device, dtype=torch.long
        )
        counts = torch.bincount(targets, minlength=3).float()
        class_prior = (counts / counts.sum()).detach().cpu().numpy().astype(np.float64)
        weights = counts.sum() / (3.0 * counts.clamp_min(1.0))
        weights = weights / weights.mean()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
        history: list[float] = []
        model.train()
        for _ in range(self.config.epochs):
            permutation = torch.randperm(len(matured), generator=generator)
            losses: list[float] = []
            for start in range(0, len(matured), self.config.batch_size):
                indices = permutation[start : start + self.config.batch_size].to(device)
                output = model(images[indices], sample=True)
                coarse_target = F.adaptive_avg_pool2d(
                    next_images[indices],
                    (model_config.coarse_height, model_config.coarse_minutes),
                )
                classification = F.cross_entropy(
                    output.regime_logits, targets[indices], weight=weights
                )
                reconstruction = state_reconstruction_loss(
                    output.next_image, coarse_target
                )
                loss = (
                    classification
                    + self.config.state_loss_weight * reconstruction
                    + self.config.kl_weight
                    * output.kl_divergence
                    / float(len(matured))
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            history.append(float(np.mean(losses)))
        return SectorRegimeTrainingResult(
            model=model,
            training_cutoff=cutoff,
            training_count=len(matured),
            class_prior=class_prior,
            loss_history=tuple(history),
        )

    def predict(
        self,
        result: SectorRegimeTrainingResult,
        samples: Sequence[SectorRegimeSample],
    ) -> tuple[SectorRegimePredictionRecord, ...]:
        eligible = tuple(
            item for item in samples if item.signal_time > result.training_cutoff
        )
        if not eligible:
            return ()
        device = next(result.model.parameters()).device
        images = torch.as_tensor(np.stack([item.images for item in eligible]), device=device)
        posterior = result.model.predict_posterior(
            images, mc_samples=self.config.mc_samples
        )
        probability = posterior.probability_mean.detach().cpu().numpy()
        probability_variance = posterior.probability_variance.detach().cpu().numpy()
        next_image = posterior.next_image_mean.detach().cpu().numpy()
        next_image_variance = posterior.next_image_variance.detach().cpu().numpy()
        return tuple(
            SectorRegimePredictionRecord(
                signal_time=item.signal_time,
                available_at=item.available_at,
                target_time=item.target_time,
                training_cutoff=result.training_cutoff,
                model_version=MODEL_VERSION,
                probability_mean=np.asarray(probability[index], dtype=np.float64),
                probability_variance=np.asarray(
                    probability_variance[index], dtype=np.float64
                ),
                previous_state=item.previous_state,
                realized_state=item.target_state,
                realized_return=item.target_return,
                return_scale=item.return_scale,
                next_image_mean=np.asarray(next_image[index], dtype=np.float64),
                next_image_variance=np.asarray(
                    next_image_variance[index], dtype=np.float64
                ),
            )
            for index, item in enumerate(eligible)
        )

