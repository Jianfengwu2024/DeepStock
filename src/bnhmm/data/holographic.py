"""Leakage-safe two-dimensional price/participation flow fields."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..config import HOLOGRAPHIC_FIELD_COUNT, HOLOGRAPHIC_FIELD_NAMES, ProfileConfig
from ..types import BarBatch, HolographicState
from .normalization import NormalizedSession, PhaseSpaceNormalization, RollingPhaseSpaceNormalizer


@dataclass(slots=True)
class _NormalizerState:
    previous_close: float | None


class HolographicProfileBuilder:
    """Build daily fields on log-price x standardized relative-volume coordinates."""

    def __init__(self, config: ProfileConfig | None = None) -> None:
        self.config = config or ProfileConfig()
        self.normalizer = RollingPhaseSpaceNormalizer(self.config)
        self._field_indices = tuple(
            HOLOGRAPHIC_FIELD_NAMES.index(name)
            for name in self.config.holographic_fields
        )
        self._normalizers: dict[str, _NormalizerState] = defaultdict(
            lambda: _NormalizerState(previous_close=None)
        )

    def build(self, bars: BarBatch) -> list[HolographicState]:
        bars = self._continuous_session_bars(bars)
        if not len(bars):
            return []
        sessions = bars.timestamp.astype("datetime64[D]")
        states: list[HolographicState] = []
        for session in np.unique(sessions):
            day = bars.take(np.flatnonzero(sessions == session))
            eligible = len(day) >= self.config.minimum_day_bars
            state = self._normalizers[bars.symbol]
            if state.previous_close is None:
                state.previous_close = float(day.close[-1])
                if eligible:
                    self.normalizer.update(day)
                continue
            normalization = self.normalizer.snapshot(bars.symbol)
            if normalization is not None and eligible:
                normalized = self.normalizer.transform(day, normalization)
                built = self._build_state(
                    day, normalized, normalization, state.previous_close
                )
                if built is not None:
                    states.append(built)
            state.previous_close = float(day.close[-1])
            if eligible:
                self.normalizer.update(day)
        return states

    @staticmethod
    def _continuous_session_bars(bars: BarBatch) -> BarBatch:
        minute = bars.timestamp.astype("datetime64[m]").astype(np.int64) % (24 * 60)
        morning = (minute > 9 * 60 + 30) & (minute <= 11 * 60 + 30)
        afternoon = (minute > 13 * 60) & (minute <= 15 * 60)
        return bars.take(np.flatnonzero(morning | afternoon))

    def _build_state(
        self,
        bars: BarBatch,
        normalized: NormalizedSession,
        normalization: PhaseSpaceNormalization,
        anchor_price: float,
    ) -> HolographicState | None:
        log_bin_width = float(np.log1p(self.config.bin_width_bps / 10_000.0))
        low = np.log(bars.low / anchor_price)
        high = np.log(bars.high / anchor_price)
        minimum_edge = int(np.floor(float(low.min()) / log_bin_width))
        maximum_edge = int(np.ceil(float(high.max()) / log_bin_width))
        minimum_edge -= self.config.profile_margin_bins
        maximum_edge += self.config.profile_margin_bins
        if maximum_edge - minimum_edge < 3:
            maximum_edge = minimum_edge + 3
        clipped = maximum_edge - minimum_edge > self.config.max_profile_bins
        if clipped:
            center = (minimum_edge + maximum_edge) // 2
            minimum_edge = center - self.config.max_profile_bins // 2
            maximum_edge = minimum_edge + self.config.max_profile_bins

        price_edge_indices = np.arange(
            minimum_edge, maximum_edge + 1, dtype=np.float64
        )
        price_edges = price_edge_indices * log_bin_width
        price_coordinates = price_edge_indices[:-1] + 0.5
        derivative_price_coordinates = (
            price_coordinates * log_bin_width / normalization.location_scale
        )
        normalized_price_coordinates = np.clip(
            derivative_price_coordinates,
            -self.config.feature_clip,
            self.config.feature_clip,
        ) / self.config.feature_clip
        volume_edges = np.linspace(
            -self.config.feature_clip,
            self.config.feature_clip,
            self.config.holographic_volume_bins + 1,
        )
        volume_coordinates = (
            (volume_edges[:-1] + volume_edges[1:])
            / 2.0
            / self.config.feature_clip
        )
        shape = (len(price_coordinates), len(volume_coordinates))

        occupancy = np.zeros(shape, dtype=np.float64)
        close_position = np.log(bars.close / anchor_price)
        for price, volume in zip(
            close_position, normalized.relative_volume_level, strict=True
        ):
            occupancy[
                self._bin_index(price_edges, float(price)),
                self._bin_index(volume_edges, float(volume)),
            ] += 1.0

        count = np.zeros(shape, dtype=np.float64)
        price_sum = np.zeros(shape, dtype=np.float64)
        volume_sum = np.zeros(shape, dtype=np.float64)
        price_square = np.zeros(shape, dtype=np.float64)
        volume_square = np.zeros(shape, dtype=np.float64)
        cross_sum = np.zeros(shape, dtype=np.float64)
        volume_anchors = self._volume_anchor_levels(
            bars.timestamp,
            normalized.relative_volume_level,
            normalized.velocity_edge_valid,
        )
        valid_indices = np.flatnonzero(
            normalized.velocity_edge_valid & np.isfinite(volume_anchors)
        )
        if not len(valid_indices):
            return None
        log_anchor = np.log(anchor_price)
        for index in valid_indices:
            price_index = self._bin_index(
                price_edges,
                float(normalized.velocity_anchor_log_price[index] - log_anchor),
            )
            volume_index = self._bin_index(
                volume_edges, float(volume_anchors[index])
            )
            u = float(normalized.price_velocity[index])
            w = float(normalized.volume_velocity[index])
            cell = (price_index, volume_index)
            count[cell] += 1.0
            price_sum[cell] += u
            volume_sum[cell] += w
            price_square[cell] += u * u
            volume_square[cell] += w * w
            cross_sum[cell] += u * w

        smooth = self.config.holographic_smoothing
        density = self._smooth(occupancy, smooth)
        density /= max(float(density.sum()), 1e-12)
        smooth_count = self._smooth(count, smooth)
        denominator = np.maximum(smooth_count, 1e-8)
        mean_price = self._smooth(price_sum, smooth) / denominator
        mean_volume = self._smooth(volume_sum, smooth) / denominator
        second_price = self._smooth(price_square, smooth) / denominator
        second_volume = self._smooth(volume_square, smooth) / denominator
        cross = self._smooth(cross_sum, smooth) / denominator
        support = smooth_count > 1e-8
        mean_price *= support
        mean_volume *= support
        variance_price = np.maximum(second_price - mean_price**2, 0.0) * support
        variance_volume = np.maximum(second_volume - mean_volume**2, 0.0) * support
        covariance = (cross - mean_price * mean_volume) * support

        price_current = density * mean_price
        volume_current = density * mean_volume
        price_temperature = density * np.sqrt(variance_price)
        volume_temperature = density * np.sqrt(variance_volume)
        covariance_flux = density * covariance / np.sqrt(
            np.maximum(second_price * second_volume, 1e-8)
        )
        divergence = self._gradient(
            price_current, derivative_price_coordinates, axis=0
        ) + self._gradient(volume_current, volume_coordinates, axis=1)
        vorticity = self._gradient(
            volume_current, derivative_price_coordinates, axis=0
        ) - self._gradient(price_current, volume_coordinates, axis=1)
        divergence = np.clip(
            divergence, -self.config.feature_clip, self.config.feature_clip
        ) / self.config.feature_clip
        vorticity = np.clip(
            vorticity, -self.config.feature_clip, self.config.feature_clip
        ) / self.config.feature_clip

        fields = np.stack(
            (
                density,
                price_current,
                volume_current,
                price_temperature,
                volume_temperature,
                covariance_flux,
                divergence,
                vorticity,
            ),
            axis=-1,
        )
        if fields.shape[-1] != HOLOGRAPHIC_FIELD_COUNT:
            raise AssertionError("holographic field protocol is inconsistent")
        fields = fields[..., self._field_indices]
        session = bars.timestamp[0].astype("datetime64[D]")
        return HolographicState(
            symbol=bars.symbol,
            timeframe="1d",
            period_start=(session + np.timedelta64(9 * 60 + 30, "m")).astype(
                "datetime64[ns]"
            ),
            period_end=(session + np.timedelta64(15 * 60, "m")).astype(
                "datetime64[ns]"
            ),
            first_bar_time=bars.timestamp[0].astype("datetime64[ns]"),
            start_price=float(bars.open[0]),
            end_price=float(bars.close[-1]),
            high_price=float(bars.high.max()),
            low_price=float(bars.low.min()),
            anchor_price=anchor_price,
            log_bin_width=log_bin_width,
            bin_coordinates=price_coordinates,
            normalized_bin_coordinates=normalized_price_coordinates,
            volume_coordinates=volume_coordinates,
            fields=fields,
            normalization_sessions=normalization.sessions,
            average_minute_volume=normalization.average_minute_volume,
            location_scale=normalization.location_scale,
            price_velocity_mean=normalization.price_velocity_mean,
            price_velocity_scale=normalization.price_velocity_scale,
            relative_volume_mean=normalization.relative_volume_mean,
            relative_volume_scale=normalization.relative_volume_scale,
            volume_velocity_mean=normalization.volume_velocity_mean,
            volume_velocity_scale=normalization.volume_velocity_scale,
            total_raw_volume=float(bars.volume.sum()),
            total_relative_volume=float(normalized.relative_volume.sum()),
            bar_count=len(bars),
            profile_clipped=clipped,
            flow_samples=len(valid_indices),
        )

    def _volume_anchor_levels(
        self,
        timestamp: NDArray[np.datetime64],
        relative_volume_level: NDArray[np.float64],
        valid: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        result = np.full(len(timestamp), np.nan, dtype=np.float64)
        minute = timestamp.astype("datetime64[m]").astype(np.int64)
        width = self.config.velocity_aggregation_minutes
        for endpoint in np.flatnonzero(valid):
            start = int(endpoint) - 2 * width + 1
            middle = start + width
            stop = int(endpoint) + 1
            if start < 0 or stop - start != 2 * width:
                continue
            if np.any(np.diff(minute[start:stop]) != 1):
                continue
            first = float(relative_volume_level[start:middle].mean())
            second = float(relative_volume_level[middle:stop].mean())
            result[endpoint] = (first + second) / 2.0
        return result

    @staticmethod
    def _smooth(values: NDArray[np.float64], strength: float) -> NDArray[np.float64]:
        if strength <= 0:
            return values.copy()
        kernel = np.asarray(
            (
                (0.25 * strength, 0.5 * strength, 0.25 * strength),
                (0.5 * strength, 1.0, 0.5 * strength),
                (0.25 * strength, 0.5 * strength, 0.25 * strength),
            ),
            dtype=np.float64,
        )
        kernel /= float(kernel.sum())
        padded = np.pad(values, 1, mode="constant")
        result = np.zeros_like(values)
        for row in range(3):
            for column in range(3):
                result += kernel[row, column] * padded[
                    row : row + values.shape[0],
                    column : column + values.shape[1],
                ]
        return result

    @staticmethod
    def _gradient(
        values: NDArray[np.float64], coordinates: NDArray[np.float64], *, axis: int
    ) -> NDArray[np.float64]:
        if values.shape[axis] < 2:
            return np.zeros_like(values)
        coordinates = np.asarray(coordinates, dtype=np.float64)
        if (
            coordinates.ndim != 1
            or len(coordinates) != values.shape[axis]
            or not np.isfinite(coordinates).all()
            or np.any(np.diff(coordinates) <= 0.0)
        ):
            coordinates = np.arange(values.shape[axis], dtype=np.float64)
        return np.gradient(values, coordinates, axis=axis, edge_order=1)

    @staticmethod
    def _bin_index(edges: NDArray[np.float64], value: float) -> int:
        return int(
            np.clip(np.searchsorted(edges, value, side="right") - 1, 0, len(edges) - 2)
        )


class HolographicPhaseProfileBuilder(HolographicProfileBuilder):
    """Build price-position by 10-minute price-velocity fields weighted by volume mass."""

    def _build_state(
        self,
        bars: BarBatch,
        normalized: NormalizedSession,
        normalization: PhaseSpaceNormalization,
        anchor_price: float,
    ) -> HolographicState | None:
        log_bin_width = float(np.log1p(self.config.bin_width_bps / 10_000.0))
        low = np.log(bars.low / anchor_price)
        high = np.log(bars.high / anchor_price)
        minimum_edge = int(np.floor(float(low.min()) / log_bin_width))
        maximum_edge = int(np.ceil(float(high.max()) / log_bin_width))
        minimum_edge -= self.config.profile_margin_bins
        maximum_edge += self.config.profile_margin_bins
        if maximum_edge - minimum_edge < 3:
            maximum_edge = minimum_edge + 3
        clipped = maximum_edge - minimum_edge > self.config.max_profile_bins
        if clipped:
            center = (minimum_edge + maximum_edge) // 2
            minimum_edge = center - self.config.max_profile_bins // 2
            maximum_edge = minimum_edge + self.config.max_profile_bins

        price_edge_indices = np.arange(
            minimum_edge, maximum_edge + 1, dtype=np.float64
        )
        price_edges = price_edge_indices * log_bin_width
        price_coordinates = price_edge_indices[:-1] + 0.5
        derivative_price_coordinates = (
            price_coordinates * log_bin_width / normalization.location_scale
        )
        normalized_price_coordinates = np.clip(
            derivative_price_coordinates,
            -self.config.feature_clip,
            self.config.feature_clip,
        ) / self.config.feature_clip
        velocity_edges = np.linspace(
            -self.config.feature_clip,
            self.config.feature_clip,
            self.config.holographic_volume_bins + 1,
        )
        velocity_coordinates = (
            (velocity_edges[:-1] + velocity_edges[1:])
            / 2.0
            / self.config.feature_clip
        )
        shape = (len(price_coordinates), len(velocity_coordinates))
        mass = np.zeros(shape, dtype=np.float64)
        count = np.zeros(shape, dtype=np.float64)
        activity_sum = np.zeros(shape, dtype=np.float64)
        valid_indices, sample_mass = self._phase_mass_samples(bars, normalized)
        if not len(valid_indices):
            return None
        log_anchor = np.log(anchor_price)
        for index, relative_mass in zip(valid_indices, sample_mass, strict=True):
            price_index = self._bin_index(
                price_edges,
                float(normalized.velocity_anchor_log_price[index] - log_anchor),
            )
            velocity_index = self._bin_index(
                velocity_edges, float(normalized.price_velocity[index])
            )
            cell = (price_index, velocity_index)
            count[cell] += 1.0
            mass[cell] += relative_mass
            activity_sum[cell] += np.log(max(relative_mass, 1e-8))

        smooth = self.config.holographic_smoothing
        smooth_mass = self._smooth(mass, smooth)
        density = smooth_mass / max(float(smooth_mass.sum()), 1e-12)
        smooth_count = self._smooth(count, smooth)
        occupancy = smooth_count / max(float(smooth_count.sum()), 1e-12)
        denominator = np.maximum(smooth_count, 1e-8)
        log_activity = self._smooth(activity_sum, smooth) / denominator
        support = smooth_count > 1e-8
        log_activity = np.tanh(log_activity) * support
        confidence = smooth_count / max(float(smooth_count.max()), 1e-12)
        velocity_grid = np.broadcast_to(velocity_coordinates, shape)
        signed_flux = density * velocity_grid
        kinetic_energy = density * np.square(velocity_grid)
        activity = occupancy * log_activity
        price_gradient = self._gradient(
            density, derivative_price_coordinates, axis=0
        )
        velocity_gradient = self._gradient(
            density, velocity_coordinates, axis=1
        )
        price_gradient = np.clip(
            price_gradient, -self.config.feature_clip, self.config.feature_clip
        ) / self.config.feature_clip
        velocity_gradient = np.clip(
            velocity_gradient, -self.config.feature_clip, self.config.feature_clip
        ) / self.config.feature_clip
        fields = np.stack(
            (
                density,
                signed_flux,
                activity,
                kinetic_energy,
                occupancy,
                density * confidence,
                price_gradient,
                velocity_gradient,
            ),
            axis=-1,
        )
        fields = fields[..., self._field_indices]
        session = bars.timestamp[0].astype("datetime64[D]")
        return HolographicState(
            symbol=bars.symbol,
            timeframe="1d",
            period_start=(session + np.timedelta64(9 * 60 + 30, "m")).astype(
                "datetime64[ns]"
            ),
            period_end=(session + np.timedelta64(15 * 60, "m")).astype(
                "datetime64[ns]"
            ),
            first_bar_time=bars.timestamp[0].astype("datetime64[ns]"),
            start_price=float(bars.open[0]),
            end_price=float(bars.close[-1]),
            high_price=float(bars.high.max()),
            low_price=float(bars.low.min()),
            anchor_price=anchor_price,
            log_bin_width=log_bin_width,
            bin_coordinates=price_coordinates,
            normalized_bin_coordinates=normalized_price_coordinates,
            volume_coordinates=velocity_coordinates,
            fields=fields,
            normalization_sessions=normalization.sessions,
            average_minute_volume=normalization.average_minute_volume,
            location_scale=normalization.location_scale,
            price_velocity_mean=normalization.price_velocity_mean,
            price_velocity_scale=normalization.price_velocity_scale,
            relative_volume_mean=normalization.relative_volume_mean,
            relative_volume_scale=normalization.relative_volume_scale,
            volume_velocity_mean=normalization.volume_velocity_mean,
            volume_velocity_scale=normalization.volume_velocity_scale,
            total_raw_volume=float(bars.volume.sum()),
            total_relative_volume=float(normalized.relative_volume.sum()),
            bar_count=len(bars),
            profile_clipped=clipped,
            flow_samples=len(valid_indices),
        )

    def _phase_mass_samples(
        self, bars: BarBatch, normalized: NormalizedSession
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        minute = bars.timestamp.astype("datetime64[m]").astype(np.int64)
        width = self.config.velocity_aggregation_minutes
        indices: list[int] = []
        masses: list[float] = []
        for endpoint in np.flatnonzero(normalized.velocity_edge_valid):
            start = int(endpoint) - 2 * width + 1
            stop = int(endpoint) + 1
            if start < 0 or stop - start != 2 * width:
                continue
            if np.any(np.diff(minute[start:stop]) != 1):
                continue
            relative_mass = float(normalized.relative_volume[start:stop].mean())
            if not np.isfinite(relative_mass) or relative_mass <= 0.0:
                continue
            indices.append(int(endpoint))
            masses.append(relative_mass)
        return np.asarray(indices, dtype=np.int64), np.asarray(masses, dtype=np.float64)
