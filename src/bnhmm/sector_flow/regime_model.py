"""Compact Bayesian convolutional HMM for sector river images."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..model.bayesian import BayesianLinear, module_kl_divergence


@dataclass(frozen=True, slots=True)
class SectorRegimeModelConfig:
    input_channels: int = 4
    hidden_channels: int = 16
    latent_states: int = 12
    coarse_height: int = 12
    coarse_minutes: int = 30
    residual_blocks: int = 2
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.input_channels < 1 or self.hidden_channels < 4:
            raise ValueError("sector regime channels are too small")
        if self.latent_states < 3:
            raise ValueError("sector regime HMM requires at least three latent states")
        if self.coarse_height < 2 or self.coarse_minutes < 2:
            raise ValueError("coarse state dimensions must be at least two")
        if self.residual_blocks < 1:
            raise ValueError("sector regime encoder requires a residual block")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("sector regime dropout must be in [0, 1)")


@dataclass(frozen=True, slots=True)
class SectorRegimePass:
    regime_logits: Tensor
    latent_probabilities: Tensor
    next_latent_probability: Tensor
    next_image: Tensor
    kl_divergence: Tensor


@dataclass(frozen=True, slots=True)
class SectorRegimePosterior:
    probability_mean: Tensor
    probability_variance: Tensor
    aleatoric_variance: Tensor
    epistemic_variance: Tensor
    latent_probability_mean: Tensor
    next_image_mean: Tensor
    next_image_variance: Tensor

    @property
    def state_index(self) -> Tensor:
        return self.probability_mean.argmax(dim=-1)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        groups = 4 if channels % 4 == 0 else 1
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, values: Tensor) -> Tensor:
        hidden = self.conv1(F.gelu(self.norm1(values)))
        hidden = self.conv2(self.dropout(F.gelu(self.norm2(hidden))))
        return values + hidden


class BayesianSectorRegimeHMM(nn.Module):
    """Encode daily images, filter latent states, and predict the next session."""

    def __init__(self, config: SectorRegimeModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or SectorRegimeModelConfig()
        groups = 4 if self.config.hidden_channels % 4 == 0 else 1
        self.stem = nn.Sequential(
            nn.Conv2d(
                self.config.input_channels,
                self.config.hidden_channels,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            nn.GroupNorm(groups, self.config.hidden_channels),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            ResidualConvBlock(self.config.hidden_channels, self.config.dropout)
            for _ in range(self.config.residual_blocks)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.emission = BayesianLinear(
            self.config.hidden_channels, self.config.latent_states
        )
        self.transition = BayesianLinear(
            self.config.latent_states,
            self.config.latent_states,
            bias=False,
        )
        self.regime_head = BayesianLinear(self.config.latent_states, 3)
        self.state_decoder = BayesianLinear(
            self.config.latent_states,
            self.config.input_channels
            * self.config.coarse_height
            * self.config.coarse_minutes,
        )

    def transition_probability(self, *, sample: bool) -> Tensor:
        identity = torch.eye(
            self.config.latent_states,
            device=self.transition.weight_mu.device,
            dtype=self.transition.weight_mu.dtype,
        )
        return torch.softmax(self.transition(identity, sample=sample), dim=-1)

    @staticmethod
    def filter_latent(evidence: Tensor, transition: Tensor) -> Tensor:
        if evidence.ndim != 3 or transition.shape != (
            evidence.shape[-1],
            evidence.shape[-1],
        ):
            raise ValueError("latent evidence and transition dimensions do not align")
        posterior = evidence[:, 0] / evidence[:, 0].sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)
        sequence = [posterior]
        for position in range(1, evidence.shape[1]):
            prior = posterior @ transition
            posterior = prior * evidence[:, position]
            posterior = posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(
                1e-12
            )
            sequence.append(posterior)
        return torch.stack(sequence, dim=1)

    def forward(self, images: Tensor, *, sample: bool = True) -> SectorRegimePass:
        if images.ndim != 5 or images.shape[2] != self.config.input_channels:
            raise ValueError("sector regime input must be batch by context by channels by image")
        batch, context, channels, height, minutes = images.shape
        encoded = images.reshape(batch * context, channels, height, minutes)
        encoded = self.stem(encoded)
        for block in self.blocks:
            encoded = block(encoded)
        encoded = self.pool(encoded).flatten(1)
        emission_logits = self.emission(encoded, sample=sample).reshape(
            batch, context, self.config.latent_states
        )
        evidence = torch.softmax(emission_logits, dim=-1)
        transition = self.transition_probability(sample=sample)
        filtered = self.filter_latent(evidence, transition)
        next_latent = filtered[:, -1] @ transition
        regime_logits = self.regime_head(next_latent, sample=sample)
        decoded = self.state_decoder(next_latent, sample=sample).reshape(
            batch,
            self.config.input_channels,
            self.config.coarse_height,
            self.config.coarse_minutes,
        )
        return SectorRegimePass(
            regime_logits=regime_logits,
            latent_probabilities=filtered,
            next_latent_probability=next_latent,
            next_image=decoded,
            kl_divergence=module_kl_divergence(self),
        )

    def predict_posterior(
        self, images: Tensor, *, mc_samples: int = 16
    ) -> SectorRegimePosterior:
        if mc_samples < 2:
            raise ValueError("posterior prediction requires at least two samples")
        training = self.training
        self.eval()
        probabilities: list[Tensor] = []
        latent: list[Tensor] = []
        next_images: list[Tensor] = []
        with torch.no_grad():
            for _ in range(mc_samples):
                output = self(images, sample=True)
                probabilities.append(torch.softmax(output.regime_logits, dim=-1))
                latent.append(output.next_latent_probability)
                next_images.append(output.next_image)
        self.train(training)
        probability_samples = torch.stack(probabilities)
        image_samples = torch.stack(next_images)
        probability_mean = probability_samples.mean(dim=0)
        epistemic = probability_samples.var(dim=0, unbiased=False)
        return SectorRegimePosterior(
            probability_mean=probability_mean,
            probability_variance=epistemic,
            aleatoric_variance=(
                probability_samples * (1.0 - probability_samples)
            ).mean(dim=0),
            epistemic_variance=epistemic,
            latent_probability_mean=torch.stack(latent).mean(dim=0),
            next_image_mean=image_samples.mean(dim=0),
            next_image_variance=image_samples.var(dim=0, unbiased=False),
        )

