"""Causal daily states for two frozen synchronization branches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..sector_flow.types import MonthlyRiverTopology
from .types import (
    MonthlyBranchMembership,
    NormalizedBoundaryPanel,
    TwoBranchDailyStates,
    TwoBranchDataset,
    TwoBranchLabels,
)


BRANCH_FEATURE_NAMES = (
    "return_1d",
    "return_mean_5",
    "return_mean_20",
    "volatility_20",
    "coherence_1d",
    "signed_path_efficiency",
)


@dataclass(frozen=True, slots=True)
class TwoBranchDailyStateConfig:
    minimum_valid_minutes: int = 30
    history_sessions: int = 20
    minimum_member_fraction: float = 0.8

    def __post_init__(self) -> None:
        if self.minimum_valid_minutes < 1:
            raise ValueError("minimum valid minutes must be positive")
        if self.history_sessions < 5:
            raise ValueError("branch history must contain at least five sessions")
        if not 0.0 < self.minimum_member_fraction <= 1.0:
            raise ValueError("minimum member fraction must lie in (0, 1]")


class TwoBranchDailyStateBuilder:
    def __init__(
        self,
        *,
        minimum_valid_minutes: int = 30,
        history_sessions: int = 20,
        minimum_member_fraction: float = 0.8,
    ) -> None:
        self.config = TwoBranchDailyStateConfig(
            minimum_valid_minutes=minimum_valid_minutes,
            history_sessions=history_sessions,
            minimum_member_fraction=minimum_member_fraction,
        )

    def build(
        self,
        panel: NormalizedBoundaryPanel,
        topologies: tuple[MonthlyRiverTopology, ...],
    ) -> TwoBranchDataset:
        memberships = self._memberships(panel.symbols, topologies)
        membership_by_month = {item.month: item for item in memberships}
        sessions = np.unique(panel.timestamps.astype("datetime64[D]"))
        minute_sessions = panel.timestamps.astype("datetime64[D]")
        daily_returns, daily_valid = self._member_daily_returns(
            panel, sessions, minute_sessions
        )
        values = np.full(
            (len(sessions), 2, len(BRANCH_FEATURE_NAMES)), np.nan, dtype=np.float64
        )
        valid = np.zeros((len(sessions), 2), dtype=np.bool_)
        branch_returns = np.full((len(sessions), 2), np.nan, dtype=np.float64)
        branch_coherence = np.full_like(branch_returns, np.nan)
        path_efficiency = np.full((len(sessions), 2), np.nan, dtype=np.float64)
        convection_strength = np.full_like(path_efficiency, np.nan)
        convection_valid = np.zeros_like(path_efficiency, dtype=np.bool_)
        convection_total_flow = np.zeros_like(path_efficiency)
        convection_observations = np.zeros_like(path_efficiency, dtype=np.int64)
        symbol_index = {symbol: index for index, symbol in enumerate(panel.symbols)}

        for row, session in enumerate(sessions):
            membership = membership_by_month.get(session.astype("datetime64[M]"))
            if membership is None:
                continue
            minute_rows = np.flatnonzero(minute_sessions == session)
            for branch, symbols in enumerate(membership.branches):
                columns = np.asarray([symbol_index[symbol] for symbol in symbols])
                available = daily_valid[row, columns]
                if float(np.mean(available)) < self.config.minimum_member_fraction:
                    continue
                branch_returns[row, branch] = float(
                    np.mean(daily_returns[row, columns[available]])
                )
                branch_coherence[row, branch] = self._coherence(
                    panel, minute_rows, columns
                )
                path_efficiency[row, branch] = self._intraday_path_efficiency(
                    panel, minute_rows, columns
                )
                (
                    convection_strength[row, branch],
                    convection_valid[row, branch],
                    convection_total_flow[row, branch],
                    convection_observations[row, branch],
                ) = self._intraday_convection(panel, minute_rows, columns)

        history = self.config.history_sessions
        for row in range(history - 1, len(sessions)):
            for branch in range(2):
                trailing = branch_returns[row - history + 1 : row + 1, branch]
                coherence = branch_coherence[row, branch]
                intraday = path_efficiency[row, branch]
                if (
                    np.any(~np.isfinite(trailing))
                    or not np.isfinite(coherence)
                    or not np.isfinite(intraday)
                ):
                    continue
                values[row, branch] = (
                    trailing[-1],
                    float(np.mean(trailing[-5:])),
                    float(np.mean(trailing)),
                    max(float(np.std(trailing)), 1e-8),
                    coherence,
                    intraday,
                )
                valid[row, branch] = True

        topology_months = sessions.astype("datetime64[M]")
        states = TwoBranchDailyStates(
            sessions=sessions,
            topology_months=topology_months,
            feature_names=BRANCH_FEATURE_NAMES,
            values=values,
            valid=valid,
        )
        target_time = np.full(
            len(sessions), np.datetime64("NaT", "ns"), dtype="datetime64[ns]"
        )
        next_return = np.full((len(sessions), 2), np.nan, dtype=np.float64)
        historical_volatility = np.full_like(next_return, np.nan)
        label_valid = np.zeros_like(valid)
        member_next_returns = np.full_like(daily_returns, np.nan)
        member_next_valid = np.zeros_like(daily_valid)
        for row in range(len(sessions) - 1):
            target_time[row] = sessions[row + 1].astype("datetime64[ns]")
            member_next_returns[row] = np.where(
                daily_valid[row + 1], daily_returns[row + 1], 0.0
            )
            member_next_valid[row] = True
            membership = membership_by_month.get(sessions[row].astype("datetime64[M]"))
            if membership is None:
                continue
            for branch, symbols in enumerate(membership.branches):
                columns = np.asarray([symbol_index[symbol] for symbol in symbols])
                available = daily_valid[row + 1, columns]
                if (
                    states.valid[row, branch]
                    and float(np.mean(available)) >= self.config.minimum_member_fraction
                ):
                    next_return[row, branch] = float(
                        np.mean(daily_returns[row + 1, columns[available]])
                    )
                    historical_volatility[row, branch] = states.feature(
                        "volatility_20"
                    )[row, branch]
                    label_valid[row, branch] = True

        labels = TwoBranchLabels(
            signal_sessions=sessions.copy(),
            target_time=target_time,
            next_return=next_return,
            historical_volatility=historical_volatility,
            valid=label_valid,
        )
        return TwoBranchDataset(
            states=states,
            labels=labels,
            memberships=memberships,
            symbols=panel.symbols,
            member_next_returns=member_next_returns,
            member_next_valid=member_next_valid,
            convection_strength=convection_strength,
            convection_valid=convection_valid,
            convection_total_flow=convection_total_flow,
            convection_observations=convection_observations,
        )

    def _member_daily_returns(
        self,
        panel: NormalizedBoundaryPanel,
        sessions: np.ndarray,
        minute_sessions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        values = np.full((len(sessions), len(panel.symbols)), np.nan, dtype=np.float64)
        valid = np.zeros_like(values, dtype=np.bool_)
        for row, session in enumerate(sessions):
            minute_rows = np.flatnonzero(minute_sessions == session)
            for column in range(len(panel.symbols)):
                selected = minute_rows[panel.raw_velocity_valid[minute_rows, column]]
                if len(selected) < self.config.minimum_valid_minutes:
                    continue
                values[row, column] = float(
                    np.expm1(np.sum(panel.raw_velocity[selected, column]))
                )
                valid[row, column] = True
        return values, valid

    @staticmethod
    def _coherence(
        panel: NormalizedBoundaryPanel,
        rows: np.ndarray,
        columns: np.ndarray,
    ) -> float:
        ratios: list[float] = []
        for row in rows:
            available = panel.velocity_valid[row, columns]
            if not available.any():
                continue
            velocity = panel.standardized_velocity[row, columns[available]]
            denominator = float(np.mean(np.abs(velocity)))
            if denominator > 1e-12:
                ratios.append(abs(float(np.mean(velocity))) / denominator)
        return float(np.mean(ratios)) if ratios else np.nan

    def _intraday_path_efficiency(
        self,
        panel: NormalizedBoundaryPanel,
        rows: np.ndarray,
        columns: np.ndarray,
    ) -> float:
        branch_velocity: list[float] = []
        for row in rows:
            available = panel.raw_velocity_valid[row, columns]
            if float(np.mean(available)) < self.config.minimum_member_fraction:
                continue
            branch_velocity.append(
                float(np.mean(panel.raw_velocity[row, columns[available]]))
            )
        if len(branch_velocity) < self.config.minimum_valid_minutes:
            return np.nan

        velocity = np.asarray(branch_velocity, dtype=np.float64)
        total_variation = float(np.sum(np.abs(velocity)))
        if total_variation <= 1e-12:
            return 0.0
        displacement = float(np.sum(velocity))
        return float(np.clip(displacement / total_variation, -1.0, 1.0))

    def _intraday_convection(
        self,
        panel: NormalizedBoundaryPanel,
        rows: np.ndarray,
        columns: np.ndarray,
    ) -> tuple[float, bool, float, int]:
        branch_velocity: list[float] = []
        member_velocity: list[np.ndarray] = []
        for row in rows:
            available = panel.raw_velocity_valid[row, columns]
            if float(np.mean(available)) < self.config.minimum_member_fraction:
                continue
            values = panel.raw_velocity[row, columns[available]]
            branch_velocity.append(float(np.mean(values)))
            member_velocity.append(values)
        if len(branch_velocity) < self.config.minimum_valid_minutes:
            return np.nan, False, 0.0, 0

        direction_value = float(np.sum(branch_velocity))
        flattened = np.concatenate(member_velocity)
        observations = int(len(flattened))
        absolute_flow = np.abs(flattened)
        if (
            observations < 2
            or abs(direction_value) <= 1e-12
            or float(np.sum(absolute_flow)) <= 1e-12
        ):
            return np.nan, False, 0.0, observations
        direction = 1.0 if direction_value > 0.0 else -1.0
        opposing_flow = np.where(flattened * direction < 0.0, absolute_flow, 0.0)
        opposing_total = float(np.sum(opposing_flow))
        if opposing_total <= 1e-12:
            return np.nan, False, 0.0, observations

        shares = opposing_flow / opposing_total
        hhi = float(np.sum(np.square(shares)))
        concentration = (observations * hhi - 1.0) / (observations - 1.0)
        return float(np.clip(concentration, 0.0, 1.0)), True, opposing_total, observations

    @staticmethod
    def _memberships(
        panel_symbols: tuple[str, ...],
        topologies: tuple[MonthlyRiverTopology, ...],
    ) -> tuple[MonthlyBranchMembership, ...]:
        if not topologies:
            raise ValueError("two-branch states require frozen monthly topologies")
        available = set(panel_symbols)
        result: list[MonthlyBranchMembership] = []
        for topology in sorted(topologies, key=lambda item: item.month):
            labels = sorted(np.unique(topology.band_ids[topology.assigned]).tolist())
            if len(labels) != 2:
                continue
            branches = tuple(
                tuple(
                    symbol
                    for symbol, assigned, label in zip(
                        topology.symbols,
                        topology.assigned,
                        topology.band_ids,
                        strict=True,
                    )
                    if assigned and label == branch_label and symbol in available
                )
                for branch_label in labels
            )
            if not branches[0] or not branches[1]:
                continue
            result.append(
                MonthlyBranchMembership(
                    month=topology.month,
                    training_cutoff=topology.training_cutoff,
                    branches=branches,
                )
            )
        return tuple(result)
