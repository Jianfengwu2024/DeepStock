"""Causal whole-sector states built from continuous rolling price paths."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import (
    CONVECTION_LOOKBACKS,
    NormalizedBoundaryPanel,
    TwoBranchDailyStates,
    TwoBranchDataset,
    TwoBranchLabels,
)


SECTOR_ROLLING_FEATURE_NAMES = (
    "rolling_log_position_60d",
    "rolling_drift_60d",
    "rolling_diffusion_60d",
    "rolling_path_efficiency_60d",
)


@dataclass(frozen=True, slots=True)
class SectorDailyStateConfig:
    minimum_valid_minutes: int = 30
    history_sessions: int = 60
    minimum_member_fraction: float = 0.8

    def __post_init__(self) -> None:
        if self.minimum_valid_minutes < 1:
            raise ValueError("minimum valid minutes must be positive")
        if self.history_sessions < 5:
            raise ValueError("sector history must contain at least five sessions")
        if not 0.0 < self.minimum_member_fraction <= 1.0:
            raise ValueError("minimum member fraction must lie in (0, 1]")


class SectorDailyStateBuilder:
    """Compress constituents into one equal-weight, continuous sector path."""

    def __init__(
        self,
        *,
        minimum_valid_minutes: int = 30,
        history_sessions: int = 60,
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
        daily_returns, daily_valid = self._member_close_to_close_returns(
            panel, minute_rows
        )
        sector_returns = np.full(len(sessions), np.nan, dtype=np.float64)
        for row in range(1, len(sessions)):
            available = daily_valid[row]
            if float(np.mean(available)) >= self.config.minimum_member_fraction:
                sector_returns[row] = float(np.mean(daily_returns[row, available]))

        values = np.full(
            (len(sessions), 1, len(SECTOR_ROLLING_FEATURE_NAMES)),
            np.nan,
            dtype=np.float64,
        )
        state_valid = np.zeros((len(sessions), 1), dtype=np.bool_)
        sector_volatility = np.full(len(sessions), np.nan, dtype=np.float64)
        history = self.config.history_sessions
        columns = np.arange(len(panel.symbols), dtype=np.int64)
        for row in range(history, len(sessions)):
            state = self._rolling_state(
                panel,
                minute_rows,
                columns,
                anchor_row=row - history,
                first_window_row=row - history + 1,
                signal_row=row,
            )
            if state is None:
                continue
            values[row, 0] = state
            state_valid[row, 0] = True
            trailing_returns = sector_returns[row - history + 1 : row + 1]
            finite_returns = trailing_returns[np.isfinite(trailing_returns)]
            if len(finite_returns) >= max(2, history // 2):
                sector_volatility[row] = max(float(np.std(finite_returns)), 1e-8)

        (
            convection_strength,
            convection_valid,
            convection_total_flow,
            convection_observations,
        ) = self._build_convection(panel, minute_rows, columns)

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
            member_next_valid[row] = daily_valid[row + 1]
            if (
                state_valid[row, 0]
                and np.isfinite(sector_returns[row + 1])
                and np.isfinite(sector_volatility[row])
            ):
                next_return[row, 0] = sector_returns[row + 1]
                historical_volatility[row, 0] = sector_volatility[row]
                label_valid[row, 0] = True

        states = TwoBranchDailyStates(
            sessions=sessions,
            topology_months=sessions.astype("datetime64[M]"),
            feature_names=SECTOR_ROLLING_FEATURE_NAMES,
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

    def _rolling_state(
        self,
        panel: NormalizedBoundaryPanel,
        minute_rows: tuple[np.ndarray, ...],
        columns: np.ndarray,
        *,
        anchor_row: int,
        first_window_row: int,
        signal_row: int,
    ) -> tuple[float, float, float, float] | None:
        anchors, anchor_valid = self._session_last_prices(
            panel, minute_rows[anchor_row], columns
        )
        if float(np.mean(anchor_valid)) < self.config.minimum_member_fraction:
            return None

        coordinates: list[float] = [0.0]
        for session_row in range(first_window_row, signal_row + 1):
            valid_minutes = 0
            for minute_row in minute_rows[session_row]:
                available = anchor_valid & panel.price_valid[minute_row, columns]
                if float(np.mean(available)) < self.config.minimum_member_fraction:
                    continue
                prices = panel.close[minute_row, columns[available]]
                bases = anchors[available]
                coordinates.append(float(np.mean(np.log(prices / bases))))
                valid_minutes += 1
            if valid_minutes < self.config.minimum_valid_minutes:
                return None

        path = np.asarray(coordinates, dtype=np.float64)
        increments = np.diff(path)
        if len(increments) == 0 or np.any(~np.isfinite(increments)):
            return None
        position = float(path[-1])
        drift = float(np.mean(increments))
        diffusion = float(np.mean(np.square(increments - drift)))
        variation = float(np.sum(np.abs(increments)))
        efficiency = 0.0 if variation <= 1e-12 else position / variation
        return position, drift, diffusion, float(np.clip(efficiency, -1.0, 1.0))

    @staticmethod
    def _session_last_prices(
        panel: NormalizedBoundaryPanel,
        rows: np.ndarray,
        columns: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        prices = np.full(len(columns), np.nan, dtype=np.float64)
        valid = np.zeros(len(columns), dtype=np.bool_)
        for local_column, column in enumerate(columns):
            observed = rows[panel.price_valid[rows, column]]
            if len(observed) == 0:
                continue
            prices[local_column] = panel.close[observed[-1], column]
            valid[local_column] = True
        return prices, valid

    def _member_close_to_close_returns(
        self,
        panel: NormalizedBoundaryPanel,
        minute_rows: tuple[np.ndarray, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        shape = (len(minute_rows), len(panel.symbols))
        closes = np.full(shape, np.nan, dtype=np.float64)
        close_valid = np.zeros(shape, dtype=np.bool_)
        columns = np.arange(len(panel.symbols), dtype=np.int64)
        for row, rows in enumerate(minute_rows):
            closes[row], close_valid[row] = self._session_last_prices(
                panel, rows, columns
            )
        values = np.full(shape, np.nan, dtype=np.float64)
        valid = np.zeros(shape, dtype=np.bool_)
        for row in range(1, len(minute_rows)):
            available = close_valid[row - 1] & close_valid[row]
            values[row, available] = closes[row, available] / closes[row - 1, available] - 1.0
            valid[row, available] = True
        return values, valid

    def _build_convection(
        self,
        panel: NormalizedBoundaryPanel,
        minute_rows: tuple[np.ndarray, ...],
        columns: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        shape = (len(minute_rows), 1, len(CONVECTION_LOOKBACKS))
        strength = np.full(shape, np.nan, dtype=np.float64)
        valid = np.zeros(shape, dtype=np.bool_)
        total_flow = np.zeros(shape, dtype=np.float64)
        observations = np.zeros(shape, dtype=np.int64)
        for row in range(len(minute_rows)):
            for window_index, lookback in enumerate(CONVECTION_LOOKBACKS):
                if row + 1 < lookback:
                    continue
                trailing_rows = np.concatenate(minute_rows[row - lookback + 1 : row + 1])
                (
                    strength[row, 0, window_index],
                    valid[row, 0, window_index],
                    total_flow[row, 0, window_index],
                    observations[row, 0, window_index],
                ) = self._reverse_flow(
                    panel,
                    trailing_rows,
                    columns,
                    minimum_valid_minutes=self.config.minimum_valid_minutes * lookback,
                )
        return strength, valid, total_flow, observations

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
        opposing = np.where(flattened * direction < 0.0, np.abs(flattened), 0.0)
        total = float(np.sum(opposing))
        if total <= 1e-12:
            return np.nan, False, 0.0, observations
        shares = opposing / total
        hhi = float(np.sum(np.square(shares)))
        concentration = (observations * hhi - 1.0) / (observations - 1.0)
        return float(np.clip(concentration, 0.0, 1.0)), True, total, observations
