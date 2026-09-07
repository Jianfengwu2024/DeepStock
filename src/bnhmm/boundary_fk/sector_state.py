"""Causal whole-sector daily states without clustering constituents."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .branch_state import BRANCH_FEATURE_NAMES
from .types import (
    CONVECTION_LOOKBACKS,
    NormalizedBoundaryPanel,
    TwoBranchDailyStates,
    TwoBranchDataset,
    TwoBranchLabels,
)


@dataclass(frozen=True, slots=True)
class SectorDailyStateConfig:
    minimum_valid_minutes: int = 30
    history_sessions: int = 20
    minimum_member_fraction: float = 0.8

    def __post_init__(self) -> None:
        if self.minimum_valid_minutes < 1:
            raise ValueError("minimum valid minutes must be positive")
        if self.history_sessions < 5:
            raise ValueError("sector history must contain at least five sessions")
        if not 0.0 < self.minimum_member_fraction <= 1.0:
            raise ValueError("minimum member fraction must lie in (0, 1]")


class SectorDailyStateBuilder:
    """Compress all constituents into one equal-weight sector state per day."""

    def __init__(
        self,
        *,
        minimum_valid_minutes: int = 30,
        history_sessions: int = 20,
        minimum_member_fraction: float = 0.8,
    ) -> None:
        self.config = SectorDailyStateConfig(
            minimum_valid_minutes=minimum_valid_minutes,
            history_sessions=history_sessions,
            minimum_member_fraction=minimum_member_fraction,
        )

    def build(self, panel: NormalizedBoundaryPanel) -> TwoBranchDataset:
        sessions = np.unique(panel.timestamps.astype("datetime64[D]"))
        minute_sessions = panel.timestamps.astype("datetime64[D]")
        minute_rows = tuple(
            np.flatnonzero(minute_sessions == session) for session in sessions
        )
        daily_returns, daily_valid = self._member_daily_returns(
            panel, sessions, minute_rows
        )
        sector_returns = np.full(len(sessions), np.nan, dtype=np.float64)
        coherence = np.full(len(sessions), np.nan, dtype=np.float64)
        path_efficiency = np.full(len(sessions), np.nan, dtype=np.float64)
        convection_shape = (len(sessions), 1, len(CONVECTION_LOOKBACKS))
        convection_strength = np.full(convection_shape, np.nan, dtype=np.float64)
        convection_valid = np.zeros(convection_shape, dtype=np.bool_)
        convection_total_flow = np.zeros(convection_shape, dtype=np.float64)
        convection_observations = np.zeros(convection_shape, dtype=np.int64)
        columns = np.arange(len(panel.symbols), dtype=np.int64)

        for row, rows in enumerate(minute_rows):
            available = daily_valid[row]
            if float(np.mean(available)) < self.config.minimum_member_fraction:
                continue
            sector_returns[row] = float(np.mean(daily_returns[row, available]))
            coherence[row] = self._coherence(panel, rows, columns)
            path_efficiency[row] = self._path_efficiency(panel, rows, columns)
            for window_index, lookback in enumerate(CONVECTION_LOOKBACKS):
                if row + 1 < lookback:
                    continue
                trailing_rows = np.concatenate(minute_rows[row - lookback + 1 : row + 1])
                (
                    convection_strength[row, 0, window_index],
                    convection_valid[row, 0, window_index],
                    convection_total_flow[row, 0, window_index],
                    convection_observations[row, 0, window_index],
                ) = self._reverse_flow(
                    panel,
                    trailing_rows,
                    columns,
                    minimum_valid_minutes=self.config.minimum_valid_minutes * lookback,
                )

        values = np.full(
            (len(sessions), 1, len(BRANCH_FEATURE_NAMES)), np.nan, dtype=np.float64
        )
        state_valid = np.zeros((len(sessions), 1), dtype=np.bool_)
        history = self.config.history_sessions
        for row in range(history - 1, len(sessions)):
            trailing = sector_returns[row - history + 1 : row + 1]
            recent_path = path_efficiency[row - 2 : row + 1]
            if (
                np.any(~np.isfinite(trailing))
                or np.any(~np.isfinite(recent_path))
                or not np.isfinite(coherence[row])
            ):
                continue
            direction = float(np.sign(path_efficiency[row]))
            consistency = (
                0.0
                if direction == 0.0
                else direction * float(np.mean(np.sign(recent_path) == direction))
            )
            available = daily_valid[row]
            synchronization = (
                0.0
                if direction == 0.0
                else direction
                * float(np.mean(np.sign(daily_returns[row, available]) == direction))
            )
            values[row, 0] = (
                trailing[-1],
                float(np.mean(trailing[-5:])),
                float(np.mean(trailing)),
                max(float(np.std(trailing)), 1e-8),
                coherence[row],
                path_efficiency[row],
                float(path_efficiency[row] - recent_path[-2]),
                consistency,
                synchronization,
            )
            state_valid[row, 0] = True

        target_time = np.full(
            len(sessions), np.datetime64("NaT", "ns"), dtype="datetime64[ns]"
        )
        next_return = np.full((len(sessions), 1), np.nan, dtype=np.float64)
        historical_volatility = np.full_like(next_return, np.nan)
        label_valid = np.zeros_like(state_valid)
        member_next_returns = np.full_like(daily_returns, np.nan)
        member_next_valid = np.zeros_like(daily_valid)
        for row in range(len(sessions) - 1):
            target_time[row] = sessions[row + 1].astype("datetime64[ns]")
            member_next_returns[row] = np.where(
                daily_valid[row + 1], daily_returns[row + 1], 0.0
            )
            member_next_valid[row] = True
            if state_valid[row, 0] and np.isfinite(sector_returns[row + 1]):
                next_return[row, 0] = sector_returns[row + 1]
                historical_volatility[row, 0] = values[
                    row, 0, BRANCH_FEATURE_NAMES.index("volatility_20")
                ]
                label_valid[row, 0] = True

        states = TwoBranchDailyStates(
            sessions=sessions,
            topology_months=sessions.astype("datetime64[M]"),
            feature_names=BRANCH_FEATURE_NAMES,
            values=values,
            valid=state_valid,
        )
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
            memberships=(),
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
        minute_rows: tuple[np.ndarray, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        values = np.full((len(sessions), len(panel.symbols)), np.nan, dtype=np.float64)
        valid = np.zeros_like(values, dtype=np.bool_)
        for row, rows in enumerate(minute_rows):
            for column in range(len(panel.symbols)):
                selected = rows[panel.raw_velocity_valid[rows, column]]
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

    def _path_efficiency(
        self,
        panel: NormalizedBoundaryPanel,
        rows: np.ndarray,
        columns: np.ndarray,
    ) -> float:
        velocity: list[float] = []
        for row in rows:
            available = panel.raw_velocity_valid[row, columns]
            if float(np.mean(available)) < self.config.minimum_member_fraction:
                continue
            velocity.append(float(np.mean(panel.raw_velocity[row, columns[available]])))
        if len(velocity) < self.config.minimum_valid_minutes:
            return np.nan
        vector = np.asarray(velocity, dtype=np.float64)
        variation = float(np.sum(np.abs(vector)))
        return 0.0 if variation <= 1e-12 else float(np.sum(vector) / variation)

    def _reverse_flow(
        self,
        panel: NormalizedBoundaryPanel,
        rows: np.ndarray,
        columns: np.ndarray,
        *,
        minimum_valid_minutes: int,
    ) -> tuple[float, bool, float, int]:
        sector_velocity: list[float] = []
        member_velocity: list[np.ndarray] = []
        for row in rows:
            available = panel.raw_velocity_valid[row, columns]
            if float(np.mean(available)) < self.config.minimum_member_fraction:
                continue
            values = panel.raw_velocity[row, columns[available]]
            sector_velocity.append(float(np.mean(values)))
            member_velocity.append(values)
        if len(sector_velocity) < minimum_valid_minutes:
            return np.nan, False, 0.0, 0
        flattened = np.concatenate(member_velocity)
        observations = int(len(flattened))
        direction_value = float(np.sum(sector_velocity))
        if observations < 2 or abs(direction_value) <= 1e-12:
            return np.nan, False, 0.0, observations
        direction = 1.0 if direction_value > 0.0 else -1.0
        opposing = np.where(
            flattened * direction < 0.0, np.abs(flattened), 0.0
        )
        total = float(np.sum(opposing))
        if total <= 1e-12:
            return np.nan, False, 0.0, observations
        shares = opposing / total
        hhi = float(np.sum(np.square(shares)))
        concentration = (observations * hhi - 1.0) / (observations - 1.0)
        return float(np.clip(concentration, 0.0, 1.0)), True, total, observations
