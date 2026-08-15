"""Continuous posterior objectives for the Bayesian market-state model."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor
from torch.nn import functional as F

from .network import SinglePassOutput, student_t_regime_probabilities


@dataclass(frozen=True, slots=True)
class LossBreakdown:
    total: Tensor
    return_nll: Tensor
    return_mae: Tensor
    classification: Tensor
    brier: Tensor
    ordinal: Tensor
    consistency: Tensor
    kl: Tensor
    latent_information: Tensor


def return_distribution_loss(
    output: SinglePassOutput,
    forward_returns: Tensor,
    targets: Tensor,
    *,
    valid_mask: Tensor | None = None,
    kl_weight: float = 1e-6,
    classification_weight: float = 1.0,
    brier_weight: float = 0.0,
    ordinal_weight: float = 0.0,
    consistency_weight: float = 0.0,
    latent_information_weight: float = 0.05,
    class_weights: Tensor | None = None,
    student_df: float = 4.0,
    strong_bear_threshold: float = -0.75,
    direction_threshold: float = 0.0,
    strong_bull_threshold: float = 0.75,
) -> LossBreakdown:
    if forward_returns.shape != output.return_mean.shape:
        raise ValueError("forward returns must match posterior mean dimensions")
    if targets.shape != forward_returns.shape:
        raise ValueError("targets must match forward returns")
    if valid_mask is None:
        valid_mask = torch.ones_like(targets, dtype=torch.bool)
    if valid_mask.shape != targets.shape:
        raise ValueError("valid_mask must match targets")
    if student_df <= 2.0:
        raise ValueError("Student-t degrees of freedom must exceed two")

    error = forward_returns - output.return_mean
    variance = output.return_log_variance.exp().clamp_min(1e-12)
    # The network reports predictive variance. StudentT is parameterized by
    # scale, whose variance is scale^2 * df / (df - 2).
    scale_squared = variance * ((student_df - 2.0) / student_df)
    constant = (
        torch.lgamma(torch.as_tensor(student_df / 2.0, device=error.device))
        - torch.lgamma(
            torch.as_tensor((student_df + 1.0) / 2.0, device=error.device)
        )
        + 0.5 * math.log(student_df * math.pi)
    )
    nll_per_item = (
        constant
        + 0.5 * output.return_log_variance
        + (student_df + 1.0)
        / 2.0
        * torch.log1p(error.square() / (student_df * scale_squared))
    )
    continuous_probability = student_t_regime_probabilities(
        output.return_mean,
        variance,
        strong_bear_threshold,
        direction_threshold,
        strong_bull_threshold,
        student_df,
    )
    probability = F.softmax(output.regime_logits, dim=-1)
    one_hot = F.one_hot(targets, num_classes=probability.shape[-1]).to(
        probability.dtype
    )
    classification_per_item = F.cross_entropy(
        output.regime_logits.transpose(1, 2),
        targets,
        weight=class_weights,
        reduction="none",
    )
    brier_per_item = (probability - one_hot).square().sum(dim=-1)
    ordinal_per_item = (
        probability.cumsum(dim=-1)[..., :-1]
        - one_hot.cumsum(dim=-1)[..., :-1]
    ).square().mean(dim=-1)
    consistency_per_item = (probability - continuous_probability).square().mean(dim=-1)

    mask = valid_mask.to(probability.dtype)
    denominator = mask.sum().clamp_min(1.0)
    return_nll = (nll_per_item * mask).sum() / denominator
    return_mae = (error.abs() * mask).sum() / denominator
    classification = (classification_per_item * mask).sum() / denominator
    brier = (brier_per_item * mask).sum() / denominator
    ordinal = (ordinal_per_item * mask).sum() / denominator
    consistency = (consistency_per_item * mask).sum() / denominator

    latent_mask = mask.unsqueeze(-1)
    usage = (output.latent_probabilities * latent_mask).sum(dim=(0, 1))
    usage = usage / latent_mask.sum().clamp_min(1.0)
    sample_entropy = -(
        output.latent_probabilities.clamp_min(1e-8).log()
        * output.latent_probabilities
        * latent_mask
    ).sum() / denominator
    aggregate_entropy = -(usage * usage.clamp_min(1e-8).log()).sum()
    latent_information = (sample_entropy - aggregate_entropy) / math.log(usage.numel())
    kl = output.kl_divergence
    total = (
        return_nll
        + classification_weight * classification
        + brier_weight * brier
        + ordinal_weight * ordinal
        + consistency_weight * consistency
        + kl_weight * kl
        + latent_information_weight * latent_information
    )
    return LossBreakdown(
        total,
        return_nll,
        return_mae,
        classification,
        brier,
        ordinal,
        consistency,
        kl,
        latent_information,
    )
