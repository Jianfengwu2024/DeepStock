"""Causal exposure gate derived from opposing constituent minute flow."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .types import TwoBranchDataset


@dataclass(frozen=True, slots=True)
class ConvectionGateDecision:
    strength: float | None
    band: str
    exposure_cap: float
    reason: str


class ConvectionPersistenceGate:
    """Require repeated one-day convection before limiting exposure."""

    def __init__(self) -> None:
        self._streaks = [0, 0]

    def apply(
        self,
        decisions: tuple[ConvectionGateDecision, ConvectionGateDecision],
    ) -> tuple[ConvectionGateDecision, ConvectionGateDecision]:
        resolved: list[ConvectionGateDecision] = []
        for branch, decision in enumerate(decisions):
            if decision.exposure_cap < 0.5:
                self._streaks[branch] += 1
            else:
                self._streaks[branch] = 0
            streak = self._streaks[branch]
            cap = 0.0 if streak >= 3 else 0.25 if streak == 2 else 0.5
            resolved.append(
                replace(
                    decision,
                    exposure_cap=cap,
                    reason=f"one_day_convection_streak_{streak}",
                )
            )
        return resolved[0], resolved[1]


@dataclass(frozen=True, slots=True)
class ConvectionGate:
    thresholds: tuple[
        tuple[float, float, float] | None,
        tuple[float, float, float] | None,
    ]
    training_rows: tuple[int, int]
    lookback_sessions: int = 3

    @classmethod
    def fit(
        cls,
        data: TwoBranchDataset,
        *,
        training_cutoff: np.datetime64,
        minimum_rows: int = 30,
        lookback_sessions: int = 3,
    ) -> "ConvectionGate":
        if minimum_rows < 1:
            raise ValueError("minimum convection rows must be positive")
        cutoff = np.datetime64(training_cutoff, "ns")
        try:
            window_index = data.convection_lookbacks.index(lookback_sessions)
        except ValueError as exc:
            raise ValueError(
                f"convection lookback must be one of {data.convection_lookbacks}"
            ) from exc
        matured = data.labels.valid & (data.labels.target_time[:, None] <= cutoff)
        thresholds: list[tuple[float, float, float] | None] = []
        row_counts: list[int] = []
        for branch in range(2):
            selected = (
                data.convection_valid[:, branch, window_index] & matured[:, branch]
            )
            values = data.convection_strength[selected, branch, window_index]
            row_counts.append(int(len(values)))
            if len(values) < minimum_rows:
                thresholds.append(None)
                continue
            quantiles = np.quantile(values, (0.50, 0.75, 0.90))
            thresholds.append(tuple(float(item) for item in quantiles))
        return cls(
            thresholds=(thresholds[0], thresholds[1]),
            training_rows=(row_counts[0], row_counts[1]),
            lookback_sessions=lookback_sessions,
        )

    def decide(
        self,
        data: TwoBranchDataset,
        *,
        row: int,
        branch: int,
    ) -> ConvectionGateDecision:
        if branch not in (0, 1):
            raise ValueError("convection branch must be zero or one")
        resolved_row = row if row >= 0 else len(data.states.sessions) + row
        if resolved_row < 0 or resolved_row >= len(data.states.sessions):
            raise IndexError("convection row is out of range")
        threshold = self.thresholds[branch]
        try:
            window_index = data.convection_lookbacks.index(self.lookback_sessions)
        except ValueError as exc:
            raise ValueError(
                f"convection lookback must be one of {data.convection_lookbacks}"
            ) from exc
        if (
            threshold is None
            or not data.convection_valid[resolved_row, branch, window_index]
        ):
            return ConvectionGateDecision(None, "unavailable", 0.5, "no_causal_calibration")

        strength = float(
            data.convection_strength[resolved_row, branch, window_index]
        )
        q50, q75, q90 = threshold
        if strength <= q50:
            return ConvectionGateDecision(strength, "low", 0.5, "below_median")
        if strength <= q75:
            return ConvectionGateDecision(strength, "elevated", 0.375, "above_q50")
        if strength <= q90:
            return ConvectionGateDecision(strength, "high", 0.25, "above_q75")
        return ConvectionGateDecision(strength, "extreme", 0.0, "above_q90")
