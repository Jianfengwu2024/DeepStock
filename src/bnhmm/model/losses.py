"""Training objective for four-state evidential Bayesian classification."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .network import SinglePassOutput


@dataclass(frozen=True, slots=True)
class LossBreakdown:
    total: Tensor
    evidential_nll: Tensor
    brier: Tensor
    ordinal: Tensor
    kl: Tensor
    latent_balance: Tensor


def evidential_classification_loss(
    output: SinglePassOutput,
    targets: Tensor,
    *,
    valid_mask: Tensor | None = None,
    kl_weight: float = 1e-6,
    brier_weight: float = 0.25,
    ordinal_weight: float = 0.0,
    latent_balance_weight: float = 0.05,
) -> LossBreakdown:
    if targets.shape != output.alpha.shape[:2]:
        raise ValueError("targets must match alpha batch and sequence dimensions")
    if valid_mask is None:
        valid_mask = torch.ones_like(targets, dtype=torch.bool)
    if valid_mask.shape != targets.shape:
        raise ValueError("valid_mask must match targets")

    concentration = output.alpha.sum(dim=-1)
    selected_alpha = output.alpha.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    expected_nll = torch.digamma(concentration) - torch.digamma(selected_alpha)
    probability = output.alpha / concentration.unsqueeze(-1)
    one_hot = F.one_hot(targets, num_classes=output.alpha.shape[-1]).to(
        probability.dtype
    )
    brier_per_item = (probability - one_hot).square().sum(dim=-1)
    cumulative_probability = probability.cumsum(dim=-1)[..., :-1]
    cumulative_target = one_hot.cumsum(dim=-1)[..., :-1]
    ordinal_per_item = (
        cumulative_probability - cumulative_target
    ).square().mean(dim=-1)

    mask = valid_mask.to(probability.dtype)
    denominator = mask.sum().clamp_min(1.0)
    nll = (expected_nll * mask).sum() / denominator
    brier = (brier_per_item * mask).sum() / denominator
    ordinal = (ordinal_per_item * mask).sum() / denominator

    latent_mask = mask.unsqueeze(-1)
    usage = (output.latent_probabilities * latent_mask).sum(dim=(0, 1))
    usage = usage / latent_mask.sum().clamp_min(1.0)
    uniform = torch.full_like(usage, 1.0 / usage.numel())
    latent_balance = (usage - uniform).square().sum()
    kl = output.kl_divergence
    total = (
        nll
        + brier_weight * brier
        + ordinal_weight * ordinal
        + kl_weight * kl
        + latent_balance_weight * latent_balance
    )
    return LossBreakdown(total, nll, brier, ordinal, kl, latent_balance)

