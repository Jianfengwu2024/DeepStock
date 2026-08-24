"""Five-gate long-only policy for BoundaryFlowFK forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .feynman_kac import BoundaryFeynmanKacForecast


@dataclass(frozen=True, slots=True)
class BoundaryPolicyConfig:
    minimum_raw_bullish_probability: float = 0.55
    minimum_survival_bullish_probability: float = 0.55
    minimum_trend_survival_probability: float = 0.60
    maximum_bullish_probability_std: float = 0.15
    maximum_thickness_risk_quantile: float = 0.90
    top_k: int = 10

    def __post_init__(self) -> None:
        probabilities = (
            self.minimum_raw_bullish_probability,
            self.minimum_survival_bullish_probability,
            self.minimum_trend_survival_probability,
            self.maximum_bullish_probability_std,
            self.maximum_thickness_risk_quantile,
        )
        if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("boundary policy probability thresholds must lie in [0, 1]")
        if self.top_k < 1:
            raise ValueError("boundary policy top_k must be positive")


@dataclass(frozen=True, slots=True)
class BoundaryPolicyDecision:
    open_position: bool
    gate_reason: str
    survival_conditional_bullish_probability: float
    selected_symbols: tuple[str, ...] = ()
    target_weights: tuple[float, ...] = ()


class BoundaryPolicy:
    def __init__(
        self,
        config: BoundaryPolicyConfig | None = None,
        *,
        ranker: Any | None = None,
    ) -> None:
        self.config = config or BoundaryPolicyConfig()
        if ranker is None:
            from bnhmm.sector_flow.ranking import SourceProximityRanker

            ranker = SourceProximityRanker()
        self.ranker = ranker

    def decide(
        self,
        forecast: BoundaryFeynmanKacForecast,
        *,
        paths: tuple[Any, ...],
        attributions: tuple[Any, ...],
        thickness_risk_quantile: float,
    ) -> BoundaryPolicyDecision:
        if not np.isfinite(thickness_risk_quantile) or not 0.0 <= thickness_risk_quantile <= 1.0:
            raise ValueError("thickness risk quantile must lie in [0, 1]")
        survival = forecast.trend_survival_probability
        conditional_bullish = (
            forecast.survival_joint_probabilities[2] / survival
            if survival > 0.0
            else 0.0
        )
        gates = (
            (
                forecast.raw_state_probabilities[2]
                >= self.config.minimum_raw_bullish_probability,
                "raw_bullish_probability",
            ),
            (
                conditional_bullish
                >= self.config.minimum_survival_bullish_probability,
                "survival_bullish_probability",
            ),
            (
                survival >= self.config.minimum_trend_survival_probability,
                "trend_survival_probability",
            ),
            (
                forecast.bullish_probability_std
                <= self.config.maximum_bullish_probability_std,
                "bullish_probability_uncertainty",
            ),
            (
                thickness_risk_quantile
                < self.config.maximum_thickness_risk_quantile,
                "boundary_thickening_risk",
            ),
        )
        for passed, reason in gates:
            if not passed:
                return BoundaryPolicyDecision(False, reason, conditional_bullish)

        ranked = self.ranker.rank(paths, attributions)
        selected = tuple(item.symbol for item in ranked[: self.config.top_k])
        if not selected:
            return BoundaryPolicyDecision(False, "no_ranked_constituents", conditional_bullish)
        weight = 1.0 / len(selected)
        return BoundaryPolicyDecision(
            True,
            "all_gates_passed",
            conditional_bullish,
            selected,
            (weight,) * len(selected),
        )
