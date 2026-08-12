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
    """Encode bin values and adjacent two/three-bin local shapes."""

    def __init__(self, profile_bins: int, d_model: int, motif_dim: int) -> None:
        super().__init__()
        self.profile_bins = profile_bins
        self.single = BayesianLinear(2, motif_dim)
        self.pair = BayesianLinear(2, motif_dim)
        self.triple = BayesianLinear(3, motif_dim)
        self.output = BayesianLinear(motif_dim * 3, d_model)
        self.register_buffer(
            "bin_positions", torch.linspace(-1.0, 1.0, profile_bins), persistent=False
        )

    def forward(self, profiles: Tensor, *, sample: bool) -> Tensor:
        if profiles.ndim != 3 or profiles.shape[-1] != self.profile_bins:
            raise ValueError(
                f"profiles must have shape [batch, sequence, {self.profile_bins}]"
            )
        positions = self.bin_positions.to(dtype=profiles.dtype).expand_as(profiles)
        single_input = torch.stack((profiles, positions), dim=-1)
        single = F.gelu(self.single(single_input, sample=sample)).mean(dim=-2)
        pairs = F.gelu(self.pair(profiles.unfold(-1, 2, 1), sample=sample)).mean(dim=-2)
        triples = F.gelu(
            self.triple(profiles.unfold(-1, 3, 1), sample=sample)
        ).mean(dim=-2)
        return self.output(torch.cat((single, pairs, triples), dim=-1), sample=sample)


class BNHMMNetwork(nn.Module):
    def __init__(
        self,
        *,
        profile_bins: int = 48,
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
        self.profile_bins = profile_bins
        self.latent_states = latent_states
        self.output_states = output_states
        self.max_sequence_length = max_sequence_length

        self.tokenizer = BayesianMotifTokenizer(profile_bins, d_model, motif_dim)
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
        self.register_buffer(
            "regime_scores", torch.linspace(-2.0, 2.0, output_states), persistent=False
        )

    def forward(
        self,
        profiles: Tensor,
        timeframe_ids: Tensor,
        *,
        valid_mask: Tensor | None = None,
        sample: bool = True,
    ) -> SinglePassOutput:
        if profiles.ndim != 3:
            raise ValueError("profiles must be rank three")
        batch, sequence, _ = profiles.shape
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

        tokens = self.tokenizer(profiles, sample=sample)
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
        kl = module_kl_divergence(self) + self._transition_kl(transition_std)
        return SinglePassOutput(alpha, latent, context, kl)

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
                profiles, timeframe_ids, valid_mask=valid_mask, sample=True
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

