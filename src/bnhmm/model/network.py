"""Bayesian profile Transformer with ten conditional latent states."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .bayesian import BayesianLinear, module_kl_divergence
from .transformer import BayesianTransformerBlock


@dataclass(frozen=True, slots=True)
class SinglePassOutput:
    alpha: Tensor
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

    @property
    def state_index(self) -> Tensor:
        return self.probability_mean.argmax(dim=-1)


class BayesianMotifTokenizer(nn.Module):
    """Encode position-aligned bin motifs without averaging their locations away."""

    def __init__(
        self,
        max_profile_bins: int,
        profile_channels: int,
        state_feature_dim: int,
        d_model: int,
        motif_dim: int,
    ) -> None:
        super().__init__()
        self.max_profile_bins = max_profile_bins
        self.profile_channels = profile_channels
        self.state_feature_dim = state_feature_dim
        coordinate_features = 5
        self.single = BayesianLinear(
            profile_channels + coordinate_features, motif_dim
        )
        self.pair = BayesianLinear(
            profile_channels * 2 + coordinate_features, motif_dim
        )
        self.triple = BayesianLinear(
            profile_channels * 3 + coordinate_features, motif_dim
        )
        self.output = BayesianLinear(motif_dim * 3, d_model)
        self.pool_score = BayesianLinear(d_model, 1)
        self.metadata = BayesianLinear(state_feature_dim, d_model)
        self.combine = BayesianLinear(d_model * 2, d_model)
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
        single = F.gelu(
            self.single(torch.cat((profiles, positions), dim=-1), sample=sample)
        )

        pair_values = profiles.unfold(-2, 2, 1).transpose(-1, -2).flatten(-2)
        pair_coordinates = (
            bin_coordinates[..., :-1] + bin_coordinates[..., 1:]
        ) / 2.0
        pair_positions = self._coordinate_features(pair_coordinates, profiles.dtype)
        pairs = F.gelu(
            self.pair(
                torch.cat((pair_values, pair_positions), dim=-1), sample=sample
            )
        )
        pairs = F.pad(pairs, (0, 0, 0, 1))

        triple_values = profiles.unfold(-2, 3, 1).transpose(-1, -2).flatten(-2)
        triple_positions = self._coordinate_features(
            bin_coordinates[..., 1:-1], profiles.dtype
        )
        triples = F.gelu(
            self.triple(
                torch.cat((triple_values, triple_positions), dim=-1), sample=sample
            )
        )
        triples = F.pad(triples, (0, 0, 1, 1))

        tokens = F.gelu(
            self.output(torch.cat((single, pairs, triples), dim=-1), sample=sample)
        )
        tokens = tokens * profile_mask.unsqueeze(-1)
        scores = self.pool_score(torch.tanh(tokens), sample=sample).squeeze(-1)
        scores = scores.masked_fill(~profile_mask, torch.finfo(scores.dtype).min)
        attention = F.softmax(scores, dim=-1)
        profile_summary = (tokens * attention.unsqueeze(-1)).sum(dim=-2)
        metadata = F.gelu(self.metadata(state_features, sample=sample))
        combined = self.combine(
            torch.cat((profile_summary, metadata), dim=-1), sample=sample
        )
        return self.output_norm(combined)

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
        profile_channels: int = 3,
        state_feature_dim: int = 9,
        latent_states: int = 10,
        output_states: int = 5,
        d_model: int = 96,
        motif_dim: int = 32,
        heads: int = 4,
        layers: int = 3,
        feedforward_dim: int = 192,
        dropout: float = 0.1,
        max_sequence_length: int = 512,
    ) -> None:
        super().__init__()
        if latent_states != 10:
            raise ValueError("BNHMM v2 requires ten unnamed latent states")
        if output_states != 5:
            raise ValueError("BNHMM v2 requires five output regimes")
        if max_sequence_length < 2:
            raise ValueError("max_sequence_length must be at least two")
        self.max_profile_bins = max_profile_bins
        self.profile_channels = profile_channels
        self.state_feature_dim = state_feature_dim
        self.latent_states = latent_states
        self.output_states = output_states
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model

        self.tokenizer = BayesianMotifTokenizer(
            max_profile_bins,
            profile_channels,
            state_feature_dim,
            d_model,
            motif_dim,
        )
        self.timeframe_embedding = nn.Embedding(2, d_model)
        self.position_embedding = nn.Parameter(
            torch.zeros(max_sequence_length, d_model)
        )
        nn.init.normal_(self.position_embedding, std=0.02)
        self.blocks = nn.ModuleList(
            BayesianTransformerBlock(
                d_model, heads, feedforward_dim, dropout
            )
            for _ in range(layers)
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.latent_emission = BayesianLinear(d_model, latent_states)
        self.transition_context = BayesianLinear(
            d_model, latent_states * latent_states
        )
        transition = torch.eye(latent_states) * 2.0
        self.transition_mu = nn.Parameter(transition)
        self.transition_rho = nn.Parameter(torch.full_like(transition, -4.5))
        self.evidence = BayesianLinear(d_model + latent_states, output_states)
        pretraining_dim = d_model + latent_states
        self.masked_state_reconstruction = nn.Linear(
            pretraining_dim, state_feature_dim
        )
        self.next_state_prediction = nn.Linear(
            pretraining_dim, state_feature_dim
        )
        self.register_buffer(
            "regime_scores", torch.linspace(-2.0, 2.0, output_states), persistent=False
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
        transition_delta = self.transition_context(context, sample=sample).view(
            batch, sequence, self.latent_states, self.latent_states
        )
        transition_std = F.softplus(self.transition_rho)
        transition_base = self.transition_mu
        if sample:
            transition_base = transition_base + transition_std * torch.randn_like(
                transition_base
            )
        latent = self._filter_latent(
            emissions, transition_base, transition_delta, valid_mask
        )
        evidence = F.softplus(
            self.evidence(torch.cat((context, latent), dim=-1), sample=sample)
        )
        alpha = evidence + 1.0
        kl = (
            module_kl_divergence(self) + self._transition_kl(transition_std)
        ) / self.kl_distribution_count
        return SinglePassOutput(alpha, latent, context, kl)

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
        means: list[Tensor] = []
        aleatoric: list[Tensor] = []
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
            concentration = output.alpha.sum(dim=-1, keepdim=True)
            mean = output.alpha / concentration
            variance = (
                output.alpha * (concentration - output.alpha)
                / (concentration.square() * (concentration + 1.0))
            )
            means.append(mean)
            aleatoric.append(variance)
            latent.append(output.latent_probabilities)
        mean_stack = torch.stack(means)
        aleatoric_variance = torch.stack(aleatoric).mean(dim=0)
        epistemic_variance = mean_stack.var(dim=0, unbiased=False)
        probability_mean = mean_stack.mean(dim=0)
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
        )

