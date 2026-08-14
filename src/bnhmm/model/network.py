"""Bayesian profile Transformer with configurable conditional latent states."""

from __future__ import annotations

from dataclasses import dataclass
from math import isqrt

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..config import (
    PROFILE_CHANNELS,
    REGIME_COUNT,
    REGIME_SCORES,
)
from .bayesian import BayesianLinear, DeterministicLinear, module_kl_divergence
from .transformer import BayesianTransformerBlock


def student_t_regime_probabilities(
    mean: Tensor,
    variance: Tensor,
    strong_bear_threshold: float = -0.015,
    direction_threshold: float = 0.0,
    strong_bull_threshold: float = 0.015,
    degrees_of_freedom: float = 4.0,
) -> Tensor:
    """Convert a Student-t(4) return posterior into regime probabilities."""
    if degrees_of_freedom != 4.0:
        raise ValueError("exact regime probabilities currently require Student-t df=4")
    scale = torch.sqrt(variance.clamp_min(1e-12) / 2.0)

    def cdf(value: Tensor | float) -> Tensor:
        standardized = (value - mean) / scale
        sine = standardized / torch.sqrt(standardized.square() + 4.0)
        return 0.5 + 0.75 * sine - 0.25 * sine.pow(3)

    strong_bear = cdf(strong_bear_threshold)
    below_direction = cdf(direction_threshold)
    below_strong_bull = cdf(strong_bull_threshold)
    weak_bear = (below_direction - strong_bear).clamp_min(0.0)
    weak_bull = (below_strong_bull - below_direction).clamp_min(0.0)
    strong_bull = (1.0 - below_strong_bull).clamp_min(0.0)
    probability = torch.stack(
        (strong_bear, weak_bear, weak_bull, strong_bull), dim=-1
    )
    return probability / probability.sum(dim=-1, keepdim=True).clamp_min(1e-12)


@dataclass(frozen=True, slots=True)
class SinglePassOutput:
    return_mean: Tensor
    return_log_variance: Tensor
    latent_probabilities: Tensor
    context: Tensor
    kl_divergence: Tensor


@dataclass(frozen=True, slots=True)
class PosteriorOutput:
    probability_mean: Tensor
    probability_variance: Tensor
    aleatoric_variance: Tensor
    epistemic_variance: Tensor
    latent_probabilities: Tensor
    expected_score: Tensor
    expected_return: Tensor
    return_variance: Tensor
    aleatoric_return_variance: Tensor
    epistemic_return_variance: Tensor

    @property
    def state_index(self) -> Tensor:
        return self.probability_mean.argmax(dim=-1)


class BayesianMotifBranch(nn.Module):
    """Encode single, pair, and triple motifs for one feature family."""

    def __init__(
        self,
        channels: int,
        d_model: int,
        motif_dim: int,
        *,
        include_coordinates: bool,
        bias: bool,
        bayesian: bool,
    ) -> None:
        super().__init__()
        self.include_coordinates = include_coordinates
        coordinate_features = 5 if include_coordinates else 0
        linear = BayesianLinear if bayesian else DeterministicLinear
        self.single = linear(
            channels + coordinate_features, motif_dim, bias=bias
        )
        self.pair = linear(
            channels * 2 + coordinate_features, motif_dim, bias=bias
        )
        self.triple = linear(
            channels * 3 + coordinate_features, motif_dim, bias=bias
        )
        self.output = linear(motif_dim * 3, d_model, bias=bias)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        values: Tensor,
        positions: Tensor,
        pair_positions: Tensor,
        triple_positions: Tensor,
        *,
        sample: bool,
    ) -> Tensor:
        single = F.gelu(
            self.single(self._input(values, positions), sample=sample)
        )
        pair_values = values.unfold(-2, 2, 1).transpose(-1, -2).flatten(-2)
        pairs = F.gelu(
            self.pair(
                self._input(pair_values, pair_positions), sample=sample
            )
        )
        pairs = F.pad(pairs, (0, 0, 0, 1))
        triple_values = values.unfold(-2, 3, 1).transpose(-1, -2).flatten(-2)
        triples = F.gelu(
            self.triple(
                self._input(triple_values, triple_positions), sample=sample
            )
        )
        triples = F.pad(triples, (0, 0, 1, 1))
        output = F.gelu(
            self.output(torch.cat((single, pairs, triples), dim=-1), sample=sample)
        )
        return self.output_norm(output)

    def _input(self, values: Tensor, positions: Tensor) -> Tensor:
        return (
            torch.cat((values, positions), dim=-1)
            if self.include_coordinates
            else values
        )


class BayesianMotifTokenizer(nn.Module):
    """Use price as the backbone and auxiliary families as gated residuals."""

    def __init__(
        self,
        max_profile_bins: int,
        profile_channels: int,
        state_feature_dim: int,
        d_model: int,
        motif_dim: int,
        feature_mode: str,
        *,
        bayesian: bool,
    ) -> None:
        super().__init__()
        joint_velocity_channels = profile_channels - 4
        velocity_bins = isqrt(joint_velocity_channels)
        if velocity_bins < 3 or velocity_bins**2 != joint_velocity_channels:
            raise ValueError(
                "profile channels must contain a square joint velocity grid"
            )
        if feature_mode not in {
            "gated-full",
            "price-only",
            "price-volume",
            "price-velocity",
        }:
            raise ValueError("unsupported feature mode")
        self.max_profile_bins = max_profile_bins
        self.profile_channels = profile_channels
        self.state_feature_dim = state_feature_dim
        self.feature_mode = feature_mode
        self.joint_velocity_channels = joint_velocity_channels
        linear = BayesianLinear if bayesian else DeterministicLinear
        self.price_branch = BayesianMotifBranch(
            2,
            d_model,
            motif_dim,
            include_coordinates=True,
            bias=True,
            bayesian=bayesian,
        )
        self.volume_branch = BayesianMotifBranch(
            2,
            d_model,
            motif_dim,
            include_coordinates=False,
            bias=False,
            bayesian=bayesian,
        )
        self.joint_velocity_branch = BayesianMotifBranch(
            self.joint_velocity_channels,
            d_model,
            motif_dim,
            include_coordinates=False,
            bias=False,
            bayesian=bayesian,
        )
        self.auxiliary_gate_logits = nn.Parameter(torch.full((2,), -3.0))
        enabled = {
            "gated-full": (1.0, 1.0),
            "price-only": (0.0, 0.0),
            "price-volume": (1.0, 0.0),
            "price-velocity": (0.0, 1.0),
        }[feature_mode]
        self.enabled_auxiliary_indices = tuple(
            index for index, value in enumerate(enabled) if value
        )
        self.register_buffer(
            "auxiliary_enabled", torch.tensor(enabled), persistent=True
        )
        self.pool_score = linear(d_model, 1)
        self.metadata = linear(state_feature_dim, d_model)
        self.combine = linear(d_model * 2, d_model)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        profiles: Tensor,
        bin_coordinates: Tensor,
        profile_mask: Tensor,
        state_features: Tensor,
        *,
        sample: bool,
        state_mask: Tensor | None = None,
    ) -> Tensor:
        if (
            profiles.ndim != 4
            or profiles.shape[-1] != self.profile_channels
        ):
            raise ValueError(
                "profiles must have shape "
                f"[batch, sequence, bins, {self.profile_channels}]"
            )
        if profiles.shape[-2] > self.max_profile_bins:
            raise ValueError("profile exceeds configured maximum bin count")
        if bin_coordinates.shape != profiles.shape[:3]:
            raise ValueError("bin_coordinates must match profile batch, sequence, and bins")
        if profile_mask.shape != profiles.shape[:3] or profile_mask.dtype != torch.bool:
            raise ValueError("profile_mask must be boolean and align with profile bins")
        if state_mask is None:
            state_mask = torch.ones(
                profiles.shape[:2], dtype=torch.bool, device=profiles.device
            )
        if state_mask.shape != profiles.shape[:2] or state_mask.dtype != torch.bool:
            raise ValueError("state_mask must be boolean and align with daily states")
        if not torch.all(profile_mask.any(dim=-1) | ~state_mask):
            raise ValueError("every valid daily profile must contain at least one valid bin")
        if state_features.shape != (*profiles.shape[:2], self.state_feature_dim):
            raise ValueError("state_features must match batch, sequence, and feature dimensions")

        positions = self._coordinate_features(bin_coordinates, profiles.dtype)
        pair_coordinates = (
            bin_coordinates[..., :-1] + bin_coordinates[..., 1:]
        ) / 2.0
        pair_positions = self._coordinate_features(pair_coordinates, profiles.dtype)
        triple_positions = self._coordinate_features(
            bin_coordinates[..., 1:-1], profiles.dtype
        )
        price_values = torch.stack((profiles[..., 0], profiles[..., 3]), dim=-1)
        volume_values = profiles[..., 1:3]
        joint_velocity_values = profiles[..., 4:]
        branch_inputs = (
            volume_values,
            joint_velocity_values,
        )
        branches = (
            self.volume_branch,
            self.joint_velocity_branch,
        )
        tokens = self.price_branch(
            price_values,
            positions,
            pair_positions,
            triple_positions,
            sample=sample,
        )
        gates = self.effective_auxiliary_gates()
        for index in self.enabled_auxiliary_indices:
            branch = branches[index]
            values = branch_inputs[index]
            residual = branch(
                values,
                positions,
                pair_positions,
                triple_positions,
                sample=sample,
            )
            tokens = tokens + gates[index] * residual
        tokens = tokens * profile_mask.unsqueeze(-1)
        scores = self.pool_score(torch.tanh(tokens), sample=sample).squeeze(-1)
        scores = scores.masked_fill(~profile_mask, torch.finfo(scores.dtype).min)
        attention = F.softmax(scores, dim=-1)
        profile_summary = (tokens * attention.unsqueeze(-1)).sum(dim=-2)
        metadata_features = state_features.clone()
        metadata_features[..., 6] = state_features[..., 6] * gates[0]
        metadata = F.gelu(self.metadata(metadata_features, sample=sample))
        combined = self.combine(
            torch.cat((profile_summary, metadata), dim=-1), sample=sample
        )
        return self.output_norm(combined)

    def effective_auxiliary_gates(self) -> Tensor:
        return torch.sigmoid(self.auxiliary_gate_logits) * self.auxiliary_enabled

    @staticmethod
    def _coordinate_features(coordinates: Tensor, dtype: torch.dtype) -> Tensor:
        coordinates = coordinates.to(dtype=dtype)
        return torch.stack(
            (
                coordinates / 256.0,
                torch.sin(coordinates * torch.pi / 8.0),
                torch.cos(coordinates * torch.pi / 8.0),
                torch.sin(coordinates * torch.pi / 32.0),
                torch.cos(coordinates * torch.pi / 32.0),
            ),
            dim=-1,
        )


class BNHMMNetwork(nn.Module):
    def __init__(
        self,
        *,
        max_profile_bins: int = 512,
        profile_channels: int = PROFILE_CHANNELS,
        state_feature_dim: int = 9,
        latent_states: int = 50,
        output_states: int = REGIME_COUNT,
        feature_mode: str = "gated-full",
        bayesian_backbone: bool = False,
        transition_rank: int = 4,
        strong_bear_threshold: float = -0.015,
        direction_threshold: float = 0.0,
        strong_bull_threshold: float = 0.015,
        d_model: int = 96,
        motif_dim: int = 32,
        heads: int = 4,
        layers: int = 3,
        feedforward_dim: int = 192,
        dropout: float = 0.1,
        max_sequence_length: int = 512,
    ) -> None:
        super().__init__()
        if latent_states < profile_channels:
            raise ValueError("latent states cannot be fewer than profile channels")
        if latent_states > 512:
            raise ValueError("latent states cannot exceed 512")
        if output_states != REGIME_COUNT:
            raise ValueError("BNHMM requires four directional output regimes")
        if max_sequence_length < 2:
            raise ValueError("max_sequence_length must be at least two")
        if transition_rank < 1 or transition_rank > latent_states:
            raise ValueError("transition rank must be between one and latent states")
        if not strong_bear_threshold < direction_threshold < strong_bull_threshold:
            raise ValueError("return regime thresholds must be ordered")
        self.max_profile_bins = max_profile_bins
        self.profile_channels = profile_channels
        self.state_feature_dim = state_feature_dim
        self.latent_states = latent_states
        self.output_states = output_states
        self.feature_mode = feature_mode
        self.bayesian_backbone = bayesian_backbone
        self.transition_rank = transition_rank
        self.strong_bear_threshold = float(strong_bear_threshold)
        self.direction_threshold = float(direction_threshold)
        self.strong_bull_threshold = float(strong_bull_threshold)
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model

        self.tokenizer = BayesianMotifTokenizer(
            max_profile_bins,
            profile_channels,
            state_feature_dim,
            d_model,
            motif_dim,
            feature_mode,
            bayesian=bayesian_backbone,
        )
        self.timeframe_embedding = nn.Embedding(2, d_model)
        self.position_embedding = nn.Parameter(
            torch.zeros(max_sequence_length, d_model)
        )
        nn.init.normal_(self.position_embedding, std=0.02)
        self.blocks = nn.ModuleList(
            BayesianTransformerBlock(
                d_model,
                heads,
                feedforward_dim,
                dropout,
                bayesian=bayesian_backbone,
            )
            for _ in range(layers)
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.latent_emission = BayesianLinear(d_model, latent_states)
        self.transition_source_context = BayesianLinear(
            d_model, latent_states * transition_rank
        )
        self.transition_destination_context = BayesianLinear(
            d_model, latent_states * transition_rank
        )
        transition = torch.eye(latent_states) * 2.0
        self.transition_mu = nn.Parameter(transition)
        self.transition_rho = nn.Parameter(torch.full_like(transition, -6.0))
        self.return_distribution = BayesianLinear(latent_states, 2)
        nn.init.zeros_(self.return_distribution.weight_mu)
        nn.init.zeros_(self.return_distribution.bias_mu)
        pretraining_dim = d_model + latent_states
        self.masked_state_reconstruction = nn.Linear(
            pretraining_dim, state_feature_dim
        )
        self.next_state_prediction = nn.Linear(
            pretraining_dim, state_feature_dim
        )
        profile_summary_dim = profile_channels * 3
        self.masked_profile_reconstruction = nn.Linear(
            pretraining_dim, profile_summary_dim
        )
        self.next_profile_prediction = nn.Linear(
            pretraining_dim, profile_summary_dim
        )
        self.register_buffer(
            "regime_scores", torch.tensor(REGIME_SCORES), persistent=False
        )
        self.kl_distribution_count = sum(
            module.weight_mu.numel()
            + (module.bias_mu.numel() if module.bias_mu is not None else 0)
            for module in self.modules()
            if isinstance(module, BayesianLinear)
        ) + self.transition_mu.numel()

    def forward(
        self,
        profiles: Tensor,
        timeframe_ids: Tensor,
        state_features: Tensor | None = None,
        bin_coordinates: Tensor | None = None,
        profile_mask: Tensor | None = None,
        *,
        valid_mask: Tensor | None = None,
        sample: bool = True,
    ) -> SinglePassOutput:
        if profiles.ndim != 4:
            raise ValueError("profiles must be rank four")
        batch, sequence, bins, channels = profiles.shape
        if bins > self.max_profile_bins or channels != self.profile_channels:
            raise ValueError("profile maximum bins or channel count does not match the model")
        if sequence > self.max_sequence_length:
            raise ValueError("sequence exceeds max_sequence_length")
        if timeframe_ids.shape != (batch, sequence):
            raise ValueError("timeframe_ids must match the first two profile dimensions")
        if valid_mask is None:
            valid_mask = torch.ones(
                batch, sequence, dtype=torch.bool, device=profiles.device
            )
        if valid_mask.shape != (batch, sequence) or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be boolean [batch, sequence]")
        if not torch.all(valid_mask[:, 0]):
            raise ValueError("the first sequence element must be valid")
        if state_features is None:
            state_features = torch.zeros(
                batch,
                sequence,
                self.state_feature_dim,
                dtype=profiles.dtype,
                device=profiles.device,
            )
        if bin_coordinates is None:
            coordinates = torch.arange(bins, device=profiles.device, dtype=profiles.dtype)
            bin_coordinates = coordinates.view(1, 1, bins).expand(batch, sequence, bins)
        if profile_mask is None:
            profile_mask = torch.ones(
                batch, sequence, bins, dtype=torch.bool, device=profiles.device
            )

        tokens = self.tokenizer(
            profiles,
            bin_coordinates,
            profile_mask,
            state_features,
            sample=sample,
            state_mask=valid_mask,
        )
        tokens = tokens + self.timeframe_embedding(timeframe_ids)
        tokens = tokens + self.position_embedding[:sequence]
        tokens = tokens * valid_mask.unsqueeze(-1)
        for block in self.blocks:
            tokens = block(tokens, valid_mask=valid_mask, sample=sample)
        context = self.output_norm(tokens) * valid_mask.unsqueeze(-1)

        emissions = F.softmax(self.latent_emission(context, sample=sample), dim=-1)
        transition_source = self.transition_source_context(
            context, sample=sample
        ).view(batch, sequence, self.latent_states, self.transition_rank)
        transition_destination = self.transition_destination_context(
            context, sample=sample
        ).view(batch, sequence, self.latent_states, self.transition_rank)
        transition_delta = torch.einsum(
            "bsir,bsjr->bsij", transition_source, transition_destination
        ) / self.transition_rank**0.5
        transition_std = F.softplus(self.transition_rho)
        transition_base = self.transition_mu
        if sample:
            transition_base = transition_base + transition_std * torch.randn_like(
                transition_base
            )
        latent = self._filter_latent(
            emissions, transition_base, transition_delta, valid_mask
        )
        latent_features = latent * float(self.latent_states)
        return_parameters = self.return_distribution(
            latent_features, sample=sample
        )
        return_mean = 0.1 * torch.tanh(return_parameters[..., 0])
        return_log_variance = torch.clamp(
            return_parameters[..., 1] - 8.4, min=-12.0, max=-3.0
        )
        kl = (
            module_kl_divergence(self) + self._transition_kl(transition_std)
        ) / self.kl_distribution_count
        return SinglePassOutput(
            return_mean, return_log_variance, latent, context, kl
        )

    def pretraining_features(self, output: SinglePassOutput) -> Tensor:
        return torch.cat((output.context, output.latent_probabilities), dim=-1)

    def _filter_latent(
        self,
        emissions: Tensor,
        transition_base: Tensor,
        transition_delta: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        posteriors: list[Tensor] = []
        previous = emissions[:, 0]
        posteriors.append(previous)
        for index in range(1, emissions.shape[1]):
            transition = F.softmax(
                transition_base.unsqueeze(0) + transition_delta[:, index], dim=-1
            )
            predicted = torch.einsum("bi,bij->bj", previous, transition)
            posterior = predicted * emissions[:, index]
            posterior = posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            previous = torch.where(
                valid_mask[:, index, None], posterior, previous
            )
            posteriors.append(previous)
        return torch.stack(posteriors, dim=1)

    def _transition_kl(self, std: Tensor) -> Tensor:
        return (
            torch.log(1.0 / std)
            + (std.square() + self.transition_mu.square()) / 2.0
            - 0.5
        ).sum()

    @torch.no_grad()
    def predict_posterior(
        self,
        profiles: Tensor,
        timeframe_ids: Tensor,
        state_features: Tensor | None = None,
        bin_coordinates: Tensor | None = None,
        profile_mask: Tensor | None = None,
        *,
        valid_mask: Tensor | None = None,
        mc_samples: int = 16,
    ) -> PosteriorOutput:
        if mc_samples < 2:
            raise ValueError("mc_samples must be at least two")
        was_training = self.training
        self.eval()
        probability_samples: list[Tensor] = []
        return_means: list[Tensor] = []
        return_variances: list[Tensor] = []
        latent: list[Tensor] = []
        for _ in range(mc_samples):
            output = self(
                profiles,
                timeframe_ids,
                state_features,
                bin_coordinates,
                profile_mask,
                valid_mask=valid_mask,
                sample=True,
            )
            return_variance = output.return_log_variance.exp()
            probability = student_t_regime_probabilities(
                output.return_mean,
                return_variance,
                self.strong_bear_threshold,
                self.direction_threshold,
                self.strong_bull_threshold,
            )
            probability_samples.append(probability)
            return_means.append(output.return_mean)
            return_variances.append(return_variance)
            latent.append(output.latent_probabilities)
        probability_stack = torch.stack(probability_samples)
        probability_mean = probability_stack.mean(dim=0)
        aleatoric_variance = (
            probability_stack * (1.0 - probability_stack)
        ).mean(dim=0)
        epistemic_variance = probability_stack.var(dim=0, unbiased=False)
        return_mean_stack = torch.stack(return_means)
        expected_return = return_mean_stack.mean(dim=0)
        aleatoric_return_variance = torch.stack(return_variances).mean(dim=0)
        epistemic_return_variance = return_mean_stack.var(
            dim=0, unbiased=False
        )
        expected_score = (probability_mean * self.regime_scores).sum(dim=-1)
        if was_training:
            self.train()
        return PosteriorOutput(
            probability_mean=probability_mean,
            probability_variance=aleatoric_variance + epistemic_variance,
            aleatoric_variance=aleatoric_variance,
            epistemic_variance=epistemic_variance,
            latent_probabilities=torch.stack(latent).mean(dim=0),
            expected_score=expected_score,
            expected_return=expected_return,
            return_variance=(
                aleatoric_return_variance + epistemic_return_variance
            ),
            aleatoric_return_variance=aleatoric_return_variance,
            epistemic_return_variance=epistemic_return_variance,
        )

