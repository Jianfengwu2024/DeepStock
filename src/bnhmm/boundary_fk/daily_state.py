"""Compress completed minute boundary fields into named daily states."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..sector_flow.topology import RiverTopologyBuilder, TopologyConfig
from ..sector_flow.types import (
    MonthlyRiverTopology,
    NormalizedSectorPanel,
)
from .types import (
    BoundaryDailyStates,
    BoundaryMinuteMoments,
    NormalizedBoundaryPanel,
    SyntheticSectorIndex,
)


_MAIN_SCHEMA = (
    ("velocity_close", "log_return_per_minute"),
    ("velocity_last30", "log_return_per_minute"),
    ("velocity_last60", "log_return_per_minute"),
    ("velocity_integral", "log_return"),
    ("velocity_quadratic_variation", "log_return_squared"),
    ("momentum_close", "density_log_return"),
    ("momentum_last60", "density_log_return"),
    ("momentum_integral", "density_log_return"),
    ("water_close", "relative_density"),
    ("water_last60", "relative_density"),
    ("water_integral", "relative_density_minutes"),
    ("coherence_close", "ratio"),
    ("coherence_last60", "ratio"),
    ("coherence_high_occupancy", "ratio"),
    ("turbulence_close", "standardized_velocity_squared"),
    ("turbulence_last60", "standardized_velocity_squared"),
    ("turbulence_high_occupancy", "ratio"),
    ("decoupling_close", "index_velocity_sigma"),
    ("decoupling_last60", "index_velocity_sigma"),
    ("decoupling_integral", "index_velocity_sigma_minutes"),
    ("thickness_close", "ratio"),
    ("thickness_last60", "ratio"),
    ("thickness_slope_close", "ratio_per_minute"),
    ("viscosity_close", "ratio"),
)


class BoundaryTopologyBuilder:
    """Adapt BoundaryFlowFK normalization to the frozen sector topology."""

    def __init__(self, config: TopologyConfig | None = None) -> None:
        requested = config or TopologyConfig()
        self.builder = RiverTopologyBuilder(
            replace(requested, maximum_bands=min(8, requested.maximum_bands))
        )

    def build(
        self,
        panel: NormalizedBoundaryPanel,
        months: list[np.datetime64] | tuple[np.datetime64, ...],
    ) -> tuple[MonthlyRiverTopology, ...]:
        requested = sorted({np.datetime64(month, "M") for month in months})
        if not requested:
            raise ValueError("at least one topology month is required")
        adapted = NormalizedSectorPanel(
            symbols=panel.symbols,
            timestamps=panel.timestamps,
            raw_velocity=panel.raw_velocity,
            velocity=panel.standardized_velocity,
            activity=panel.relative_volume,
            velocity_scale=panel.velocity_scale,
            volume_baseline=panel.minute_volume_baseline,
            valid=panel.velocity_valid,
            activity_valid=panel.density_valid,
        )
        result: list[MonthlyRiverTopology] = []
        previous: MonthlyRiverTopology | None = None
        for month in requested:
            month_start = month.astype("datetime64[ns]")
            eligible = np.flatnonzero(panel.timestamps < month_start)
            if not len(eligible):
                raise ValueError(f"{month} 之前没有可用于支流拓扑的数据")
            cutoff = panel.timestamps[int(eligible[-1])]
            topology = self.builder.build(
                adapted,
                cutoff=cutoff,
                month=month,
                previous=previous,
            )
            result.append(topology)
            previous = topology
        return tuple(result)

    def build_periodic(
        self,
        panel: NormalizedBoundaryPanel,
        *,
        every_sessions: int = 5,
    ) -> tuple[MonthlyRiverTopology, ...]:
        """Freeze a topology every N completed trading sessions without look-ahead."""
        if every_sessions < 1:
            raise ValueError("topology period must be positive")
        sessions = np.unique(panel.timestamps.astype("datetime64[D]"))
        if len(sessions) < 2:
            return ()
        adapted = NormalizedSectorPanel(
            symbols=panel.symbols,
            timestamps=panel.timestamps,
            raw_velocity=panel.raw_velocity,
            velocity=panel.standardized_velocity,
            activity=panel.relative_volume,
            velocity_scale=panel.velocity_scale,
            volume_baseline=panel.minute_volume_baseline,
            valid=panel.velocity_valid,
            activity_valid=panel.density_valid,
        )
        result: list[MonthlyRiverTopology] = []
        previous: MonthlyRiverTopology | None = None
        minimum = self.builder.config.minimum_sessions
        for position in range(minimum, len(sessions), every_sessions):
            effective = sessions[position]
            prior = sessions[position - 1]
            cutoff_rows = np.flatnonzero(panel.timestamps.astype("datetime64[D]") == prior)
            if not len(cutoff_rows):
                continue
            topology = self.builder.build(
                adapted,
                cutoff=panel.timestamps[int(cutoff_rows[-1])],
                # RiverTopology keeps this legacy field; the membership carries
                # the true five-session effective date.
                month=effective.astype("datetime64[M]"),
                previous=previous,
                effective_session=effective,
            )
            result.append(topology)
            previous = topology
        return tuple(result)


@dataclass(frozen=True, slots=True)
class BoundaryDailyStateConfig:
    branch_count: int = 2
    high_coherence_threshold: float = 0.8
    minimum_valid_minutes: int = 30

    def __post_init__(self) -> None:
        if not 2 <= self.branch_count <= 8:
            raise ValueError("branch_count must lie in [2, 8]")
        if not 0.0 < self.high_coherence_threshold < 1.0:
            raise ValueError("high coherence threshold must lie in (0, 1)")
        if self.minimum_valid_minutes < 1:
            raise ValueError("minimum valid minutes must be positive")


class BoundaryDailyStateBuilder:
    def __init__(
        self,
        branch_count: int = 2,
        *,
        high_coherence_threshold: float = 0.8,
        minimum_valid_minutes: int = 1,
    ) -> None:
        self.config = BoundaryDailyStateConfig(
            branch_count=branch_count,
            high_coherence_threshold=high_coherence_threshold,
            minimum_valid_minutes=minimum_valid_minutes,
        )

    def build(
        self,
        panel: NormalizedBoundaryPanel,
        moments: BoundaryMinuteMoments,
        index: SyntheticSectorIndex,
        topologies: tuple[MonthlyRiverTopology, ...],
        *,
        high_turbulence_threshold: float,
    ) -> BoundaryDailyStates:
        self._validate_inputs(panel, moments, index, topologies, high_turbulence_threshold)
        topology_by_month = {item.month: item for item in topologies}
        sessions = np.unique(moments.timestamps.astype("datetime64[D]"))
        topology_months = sessions.astype("datetime64[M]")
        degraded_months = {
            month
            for month, topology in topology_by_month.items()
            if len(np.unique(topology.band_ids[topology.assigned]))
            != self.config.branch_count
        }
        use_branches = not degraded_months and all(
            month in topology_by_month for month in np.unique(topology_months)
        )
        schema = list(_MAIN_SCHEMA)
        if use_branches:
            for branch in range(self.config.branch_count):
                schema.extend(
                    (
                        (f"branch_{branch + 1}_close_contribution", "ratio"),
                        (f"branch_{branch + 1}_cumulative_contribution", "ratio_minutes"),
                        (f"branch_{branch + 1}_last60_coherence", "ratio"),
                    )
                )
        values = np.full((len(sessions), len(schema)), np.nan, dtype=np.float64)
        valid = np.zeros(len(sessions), dtype=np.bool_)
        branch_degraded = np.asarray(
            [month in degraded_months or month not in topology_by_month for month in topology_months],
            dtype=np.bool_,
        )
        minute_sessions = moments.timestamps.astype("datetime64[D]")

        for destination, session in enumerate(sessions):
            month = session.astype("datetime64[M]")
            topology = topology_by_month.get(month)
            rows = np.flatnonzero((minute_sessions == session) & moments.valid)
            if topology is None or len(rows) < self.config.minimum_valid_minutes:
                continue
            main = self._main_features(
                moments,
                rows,
                high_turbulence_threshold,
            )
            state = list(main)
            if use_branches:
                state.extend(self._branch_features(panel, index, topology, rows))
            if len(state) != len(schema) or np.any(~np.isfinite(state)):
                continue
            values[destination] = state
            valid[destination] = True

        return BoundaryDailyStates(
            sessions=sessions,
            topology_months=topology_months,
            feature_names=tuple(name for name, _ in schema),
            units=tuple(unit for _, unit in schema),
            values=values,
            valid=valid,
            branch_degraded=branch_degraded,
        )

    def _main_features(
        self,
        moments: BoundaryMinuteMoments,
        rows: np.ndarray,
        high_turbulence_threshold: float,
    ) -> tuple[float, ...]:
        water = moments.water[rows]

        def weighted(values: np.ndarray, count: int) -> float:
            selected_values = values[-count:]
            selected_water = water[-count:]
            total = float(selected_water.sum())
            return float(np.dot(selected_water, selected_values) / max(total, 1e-12))

        velocity = moments.velocity[rows]
        momentum = moments.momentum[rows]
        coherence = moments.coherence[rows]
        turbulence = moments.turbulence[rows]
        decoupling = moments.decoupling[rows]
        thickness = moments.thickness[rows]
        return (
            float(velocity[-1]),
            weighted(velocity, 30),
            weighted(velocity, 60),
            float(velocity.sum()),
            float(np.square(np.diff(velocity)).sum()),
            float(momentum[-1]),
            weighted(momentum, 60),
            float(momentum.sum()),
            float(water[-1]),
            weighted(water, 60),
            float(water.sum()),
            float(coherence[-1]),
            weighted(coherence, 60),
            float(np.mean(coherence >= self.config.high_coherence_threshold)),
            float(turbulence[-1]),
            weighted(turbulence, 60),
            float(np.mean(turbulence >= high_turbulence_threshold)),
            float(decoupling[-1]),
            weighted(decoupling, 60),
            float(decoupling.sum()),
            float(thickness[-1]),
            weighted(thickness, 60),
            float(moments.thickness_slope[rows][-1]),
            float(moments.viscosity[rows][-1]),
        )

    def _branch_features(
        self,
        panel: NormalizedBoundaryPanel,
        index: SyntheticSectorIndex,
        topology: MonthlyRiverTopology,
        rows: np.ndarray,
    ) -> tuple[float, ...]:
        panel_index = {symbol: column for column, symbol in enumerate(panel.symbols)}
        topology_index = {symbol: column for column, symbol in enumerate(topology.symbols)}
        frozen = index.weights_for(topology.month)
        labels = sorted(np.unique(topology.band_ids[topology.assigned]).tolist())
        branch_momentum = np.zeros((len(rows), self.config.branch_count), dtype=np.float64)
        branch_coherence = np.zeros_like(branch_momentum)

        for branch, label in enumerate(labels):
            members = [
                symbol
                for symbol in frozen.symbols
                if symbol in topology_index
                and topology.band_ids[topology_index[symbol]] == label
            ]
            if not members:
                continue
            frozen_index = {symbol: position for position, symbol in enumerate(frozen.symbols)}
            columns = np.asarray([panel_index[symbol] for symbol in members])
            weights = np.asarray([frozen.weights[frozen_index[symbol]] for symbol in members])
            for local, row in enumerate(rows):
                available = (
                    panel.raw_velocity_valid[row, columns]
                    & panel.velocity_valid[row, columns]
                    & panel.density_valid[row, columns]
                )
                if not available.any():
                    continue
                mass = weights[available] * panel.relative_volume[row, columns][available]
                raw = panel.raw_velocity[row, columns][available]
                z = panel.standardized_velocity[row, columns][available]
                branch_momentum[local, branch] = float(np.dot(mass, raw))
                mass_sum = float(mass.sum())
                if mass_sum > 0.0:
                    participation = mass / mass_sum
                    numerator = abs(float(np.dot(participation, z)))
                    denominator = float(np.dot(participation, np.abs(z)))
                    branch_coherence[local, branch] = numerator / max(denominator, 1e-12)

        denominator = np.abs(branch_momentum).sum(axis=1)
        contribution = np.divide(
            branch_momentum,
            denominator[:, None],
            out=np.zeros_like(branch_momentum),
            where=denominator[:, None] > 1e-12,
        )
        result: list[float] = []
        for branch in range(self.config.branch_count):
            result.extend(
                (
                    float(contribution[-1, branch]),
                    float(contribution[:, branch].sum()),
                    float(np.mean(branch_coherence[-60:, branch])),
                )
            )
        return tuple(result)

    @staticmethod
    def _validate_inputs(
        panel: NormalizedBoundaryPanel,
        moments: BoundaryMinuteMoments,
        index: SyntheticSectorIndex,
        topologies: tuple[MonthlyRiverTopology, ...],
        high_turbulence_threshold: float,
    ) -> None:
        if not np.array_equal(panel.timestamps, moments.timestamps) or not np.array_equal(
            panel.timestamps, index.timestamps
        ):
            raise ValueError("daily boundary inputs must share timestamps")
        if not topologies:
            raise ValueError("at least one frozen monthly topology is required")
        months = [item.month for item in topologies]
        if len(set(months)) != len(months):
            raise ValueError("monthly topologies must be unique")
        if any(
            not item.training_cutoff < item.month.astype("datetime64[D]")
            for item in topologies
        ):
            raise ValueError("monthly topology must be trained before its month")
        if not np.isfinite(high_turbulence_threshold) or high_turbulence_threshold <= 0.0:
            raise ValueError("high turbulence threshold must be finite and positive")
