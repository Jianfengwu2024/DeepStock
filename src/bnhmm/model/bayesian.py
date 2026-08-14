"""Variational Bayesian building blocks with analytic Gaussian KL terms."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class DeterministicLinear(nn.Linear):
    """Linear layer sharing the Bayesian layer call contract."""

    def forward(self, values: Tensor, *, sample: bool = False) -> Tensor:
        del sample
        return F.linear(values, self.weight, self.bias)


class BayesianLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        prior_std: float = 1.0,
        posterior_rho_init: float = -6.0,
    ) -> None:
        super().__init__()
        if in_features < 1 or out_features < 1:
            raise ValueError("BayesianLinear dimensions must be positive")
        if prior_std <= 0:
            raise ValueError("prior_std must be positive")
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = float(prior_std)
        self.posterior_rho_init = float(posterior_rho_init)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias_mu = nn.Parameter(torch.empty(out_features))
            self.bias_rho = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias_mu", None)
            self.register_parameter("bias_rho", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.constant_(self.weight_rho, self.posterior_rho_init)
        if self.bias_mu is not None:
            nn.init.uniform_(self.bias_mu, -bound, bound)
            nn.init.constant_(self.bias_rho, self.posterior_rho_init)

    @staticmethod
    def _sample(mu: Tensor, rho: Tensor, sample: bool) -> Tensor:
        if not sample:
            return mu
        return mu + F.softplus(rho) * torch.randn_like(mu)

    def forward(self, values: Tensor, *, sample: bool = True) -> Tensor:
        weight = self._sample(self.weight_mu, self.weight_rho, sample)
        bias = None
        if self.bias_mu is not None and self.bias_rho is not None:
            bias = self._sample(self.bias_mu, self.bias_rho, sample)
        return F.linear(values, weight, bias)

    def kl_divergence(self) -> Tensor:
        result = self._normal_kl(self.weight_mu, F.softplus(self.weight_rho))
        if self.bias_mu is not None and self.bias_rho is not None:
            result = result + self._normal_kl(
                self.bias_mu, F.softplus(self.bias_rho)
            )
        return result

    def _normal_kl(self, mu: Tensor, std: Tensor) -> Tensor:
        prior_variance = self.prior_std**2
        return (
            torch.log(torch.as_tensor(self.prior_std, device=mu.device, dtype=mu.dtype) / std)
            + (std.square() + mu.square()) / (2.0 * prior_variance)
            - 0.5
        ).sum()


def module_kl_divergence(module: nn.Module) -> Tensor:
    total: Tensor | None = None
    for child in module.modules():
        if isinstance(child, BayesianLinear):
            value = child.kl_divergence()
            total = value if total is None else total + value
    if total is not None:
        return total
    parameter = next(module.parameters(), None)
    return torch.zeros((), device=parameter.device if parameter is not None else None)

