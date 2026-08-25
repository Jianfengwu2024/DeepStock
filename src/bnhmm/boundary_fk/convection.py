"""Causal exposure gate derived from opposing constituent minute flow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import TwoBranchDataset


@dataclass(frozen=True, slots=True)
class ConvectionGateDecision:
    strength: float | None
    band: str
    exposure_cap: float
    reason: str


@dataclass(frozen=True, slots=True)
class ConvectionGate:
    thresholds: tuple[
        tuple[float, float, float] | None,
        tuple[float, float, float] | None,
    ]
    training_rows: tuple[int, int]

    @classmethod
    def fit(
        cls,
        data: TwoBranchDataset,
        *,
        training_cutoff: np.datetime64,
        minimum_rows: int = 30,
    ) -> "ConvectionGate":
        if minimum_rows < 1:
            raise ValueError("minimum convection rows must be positive")
        cutoff = np.datetime64(training_cutoff, "ns")
        matured = data.labels.valid & (data.labels.target_time[:, None] <= cutoff)
        thresholds: list[tuple[float, float, float] | None] = []
        row_counts: list[int] = []
        for branch in range(2):
            selected = data.convection_valid[:, branch] & matured[:, branch]
            values = data.convection_strength[selected, branch]
            row_counts.append(int(len(values)))
            if len(values) < minimum_rows:
                thresholds.append(None)
                continue
            quantiles = np.quantile(values, (0.50, 0.75, 0.90))
            thresholds.append(tuple(float(item) for item in quantiles))
        return cls(
            thresholds=(thresholds[0], thresholds[1]),
            training_rows=(row_counts[0], row_counts[1]),
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
        if threshold is None or not data.convection_valid[resolved_row, branch]:
            return ConvectionGateDecision(None, "unavailable", 0.5, "no_causal_calibration")

        strength = float(data.convection_strength[resolved_row, branch])
        q50, q75, q90 = threshold
        if strength <= q50:
            return ConvectionGateDecision(strength, "low", 0.5, "below_median")
        if strength <= q75:
            return ConvectionGateDecision(strength, "elevated", 0.375, "above_q50")
        if strength <= q90:
            return ConvectionGateDecision(strength, "high", 0.25, "above_q75")
        return ConvectionGateDecision(strength, "extreme", 0.0, "above_q90")
