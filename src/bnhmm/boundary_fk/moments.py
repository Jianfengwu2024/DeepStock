"""Named minute fluid moments for BoundaryFlowFK."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .types import (
    BoundaryMinuteMoments,
    MonthlyIndexWeights,
    NormalizedBoundaryPanel,
    SyntheticSectorIndex,
)


@dataclass(frozen=True, slots=True)
class BoundaryFieldConfig:
    minimum_symbol_fraction: float = 0.5
    minimum_active_weight: float = 0.5
    viscosity_window: int = 60
    minimum_viscosity_transitions: int = 60
    thickness_slope_window: int = 30
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if not 0.0 < self.minimum_symbol_fraction <= 1.0:
            raise ValueError("minimum symbol fraction must lie in (0, 1]")
        if not 0.0 < self.minimum_active_weight <= 1.0:
            raise ValueError("minimum active weight must lie in (0, 1]")
        if self.viscosity_window < 1 or self.minimum_viscosity_transitions < 0:
            raise ValueError("viscosity history configuration is invalid")
        if self.thickness_slope_window < 1:
            raise ValueError("thickness slope window must be positive")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")


class BoundaryFieldExtractor:
    def __init__(self, config: BoundaryFieldConfig | None = None) -> None:
        self.config = config or BoundaryFieldConfig()

    def extract(
        self,
        panel: NormalizedBoundaryPanel,
        index: SyntheticSectorIndex,
        *,
        index_velocity_scale: float | np.ndarray,
        thickness_stabilizer: float,
    ) -> BoundaryMinuteMoments:
        self._validate_inputs(panel, index, index_velocity_scale, thickness_stabilizer)
        rows = len(panel.timestamps)
        scale = self._scale_vector(index_velocity_scale, rows)
        outputs = {
            name: np.full(rows, np.nan, dtype=np.float64)
            for name in (
                "water",
                "momentum",
                "velocity",
                "mean_standardized_velocity",
                "coherence",
                "turbulence",
                "breadth",
                "concentration",
                "effective_constituents",
                "decoupling",
                "thickness",
                "thickness_slope",
                "viscosity",
            )
        }
        active_weight = np.zeros(rows, dtype=np.float64)
        active_fraction = np.zeros(rows, dtype=np.float64)
        valid = np.zeros(rows, dtype=np.bool_)
        columns = {symbol: position for position, symbol in enumerate(panel.symbols)}
        weights_by_month = {item.month: item for item in index.monthly_weights}
        residual_history = {
            symbol: deque(maxlen=self.config.viscosity_window) for symbol in panel.symbols
        }
        previous_residual: dict[str, tuple[np.datetime64, float]] = {}
        thickness_history: deque[tuple[np.datetime64, float]] = deque(
            maxlen=self.config.thickness_slope_window
        )
        previous_timestamp: np.datetime64 | None = None

        for row, timestamp in enumerate(panel.timestamps):
            continuous = (
                previous_timestamp is not None
                and timestamp.astype("datetime64[m]").astype(np.int64)
                - previous_timestamp.astype("datetime64[m]").astype(np.int64)
                == 1
            )
            if not continuous:
                previous_residual.clear()
                thickness_history.clear()
            previous_timestamp = timestamp
            frozen = weights_by_month.get(timestamp.astype("datetime64[M]"))
            if frozen is None or not index.valid[row]:
                continue
            frozen_columns = np.asarray([columns[symbol] for symbol in frozen.symbols])
            feature_valid = (
                panel.raw_velocity_valid[row, frozen_columns]
                & panel.velocity_valid[row, frozen_columns]
                & panel.density_valid[row, frozen_columns]
            )
            fraction = float(np.mean(feature_valid))
            weight_sum = float(frozen.weights[feature_valid].sum())
            active_fraction[row] = fraction
            active_weight[row] = weight_sum
            if (
                fraction + self.config.epsilon < self.config.minimum_symbol_fraction
                or weight_sum + self.config.epsilon < self.config.minimum_active_weight
            ):
                self._clear_missing_residuals(
                    previous_residual, frozen.symbols, feature_valid
                )
                continue

            active_symbols = tuple(
                symbol
                for symbol, available in zip(frozen.symbols, feature_valid, strict=True)
                if available
            )
            selected_columns = frozen_columns[feature_valid]
            weights = frozen.weights[feature_valid]
            rho = panel.relative_volume[row, selected_columns]
            raw = panel.raw_velocity[row, selected_columns]
            standardized = panel.standardized_velocity[row, selected_columns]
            mass = weights * rho
            water = float(mass.sum())
            if not np.isfinite(water) or water <= self.config.epsilon:
                continue
            participation = mass / water
            momentum = float(np.dot(mass, raw))
            flow_velocity = momentum / water
            mean_z = float(np.dot(participation, standardized))
            absolute_z = float(np.dot(participation, np.abs(standardized)))
            coherence = abs(mean_z) / (absolute_z + self.config.epsilon)
            turbulence = float(
                np.dot(participation, np.square(standardized - mean_z))
            )
            breadth = float(np.dot(participation, np.sign(standardized)))
            concentration = float(np.dot(participation, participation))
            effective = 1.0 / (concentration + self.config.epsilon)
            decoupling = (float(index.velocity[row]) - flow_velocity) / scale[row]
            thickness = np.sqrt(max(turbulence, 0.0)) / (
                abs(mean_z) + thickness_stabilizer + self.config.epsilon
            )

            residuals = standardized - mean_z
            self._append_residual_transitions(
                residual_history,
                previous_residual,
                active_symbols,
                timestamp,
                residuals,
            )
            viscosity = self._viscosity(residual_history)
            if viscosity is None:
                continue
            thickness_history.append((timestamp, thickness))
            thickness_slope = self._thickness_slope(thickness_history)

            values = {
                "water": water,
                "momentum": momentum,
                "velocity": flow_velocity,
                "mean_standardized_velocity": mean_z,
                "coherence": min(max(coherence, 0.0), 1.0),
                "turbulence": turbulence,
                "breadth": breadth,
                "concentration": concentration,
                "effective_constituents": effective,
                "decoupling": decoupling,
                "thickness": thickness,
                "thickness_slope": thickness_slope,
                "viscosity": viscosity,
            }
            if any(not np.isfinite(value) for value in values.values()):
                raise ValueError(f"boundary moments became non-finite at {timestamp}")
            for name, value in values.items():
                outputs[name][row] = value
            valid[row] = True

        return BoundaryMinuteMoments(
            timestamps=panel.timestamps.copy(),
            active_weight=active_weight,
            active_fraction=active_fraction,
            valid=valid,
            **outputs,
        )

    @staticmethod
    def _validate_inputs(
        panel: NormalizedBoundaryPanel,
        index: SyntheticSectorIndex,
        index_velocity_scale: float | np.ndarray,
        thickness_stabilizer: float,
    ) -> None:
        if not np.array_equal(panel.timestamps, index.timestamps):
            raise ValueError("normalized panel and synthetic index timestamps differ")
        if len(set(panel.symbols)) != len(panel.symbols):
            raise ValueError("normalized panel symbols must be unique")
        index_symbols = {
            symbol for frozen in index.monthly_weights for symbol in frozen.symbols
        }
        if not index_symbols.issubset(set(panel.symbols)):
            raise ValueError("synthetic index contains symbols absent from the panel")
        if not np.isfinite(thickness_stabilizer) or thickness_stabilizer <= 0.0:
            raise ValueError("thickness stabilizer must be finite and positive")
        values = np.asarray(index_velocity_scale, dtype=np.float64)
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("index velocity scale must be finite and positive")

    @staticmethod
    def _scale_vector(value: float | np.ndarray, rows: int) -> np.ndarray:
        scale = np.asarray(value, dtype=np.float64)
        if scale.ndim == 0:
            return np.full(rows, float(scale), dtype=np.float64)
        if scale.shape != (rows,):
            raise ValueError("index velocity scale vector must align with timestamps")
        return scale

    @staticmethod
    def _clear_missing_residuals(
        previous: dict[str, tuple[np.datetime64, float]],
        symbols: tuple[str, ...],
        available: np.ndarray,
    ) -> None:
        for symbol, present in zip(symbols, available, strict=True):
            if not present:
                previous.pop(symbol, None)

    @staticmethod
    def _append_residual_transitions(
        history: dict[str, deque[tuple[float, float]]],
        previous: dict[str, tuple[np.datetime64, float]],
        symbols: tuple[str, ...],
        timestamp: np.datetime64,
        residuals: np.ndarray,
    ) -> None:
        minute = timestamp.astype("datetime64[m]").astype(np.int64)
        active = set(symbols)
        for symbol in tuple(previous):
            if symbol not in active:
                previous.pop(symbol, None)
        for symbol, residual in zip(symbols, residuals, strict=True):
            prior = previous.get(symbol)
            if prior is not None:
                prior_minute = prior[0].astype("datetime64[m]").astype(np.int64)
                if minute - prior_minute == 1:
                    history[symbol].append((prior[1], float(residual)))
            previous[symbol] = (timestamp, float(residual))

    def _viscosity(
        self,
        history: dict[str, deque[tuple[float, float]]],
    ) -> float | None:
        pairs = [pair for symbol_history in history.values() for pair in symbol_history]
        if len(pairs) < self.config.minimum_viscosity_transitions:
            return None
        if not pairs:
            return 1.0
        values = np.asarray(pairs, dtype=np.float64)
        denominator = float(np.dot(values[:, 0], values[:, 0]))
        phi = 0.0 if denominator <= self.config.epsilon else float(
            np.dot(values[:, 0], values[:, 1]) / denominator
        )
        return float(np.clip(1.0 - phi, 0.0, 2.0))

    @staticmethod
    def _thickness_slope(
        history: deque[tuple[np.datetime64, float]],
    ) -> float:
        if len(history) < 2:
            return 0.0
        timestamps = np.asarray(
            [item[0].astype("datetime64[m]").astype(np.int64) for item in history],
            dtype=np.float64,
        )
        timestamps -= timestamps[0]
        values = np.asarray([item[1] for item in history], dtype=np.float64)
        centered = timestamps - float(timestamps.mean())
        denominator = float(np.dot(centered, centered))
        if denominator <= 0.0:
            return 0.0
        return float(np.dot(centered, values - float(values.mean())) / denominator)
