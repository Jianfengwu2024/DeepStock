"""Transparent path metrics and joint source-proximity posterior ranking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import (
    FlowAttributionPosterior,
    NormalizedSectorPanel,
    PathWindowMetrics,
    SectorFlowPosterior,
    SourceProximityPosterior,
    StockFlowAttribution,
    StockPathMetrics,
)


@dataclass(frozen=True, slots=True)
class PathMetricSeries:
    symbols: tuple[str, ...]
    minutes: int
    valid_minutes: np.ndarray
    efficiency: np.ndarray
    directional_agreement: np.ndarray
    reversal_count: np.ndarray
    activity_confirmation: np.ndarray
    residual_fraction: np.ndarray

    def at(self, endpoint: int) -> tuple[StockPathMetrics, ...]:
        if not 0 <= endpoint < self.valid_minutes.shape[0]:
            raise ValueError("path endpoint is outside the metric series")
        return tuple(
            StockPathMetrics(
                symbol=symbol,
                windows=(
                    PathWindowMetrics(
                        minutes=self.minutes,
                        valid_minutes=int(self.valid_minutes[endpoint, column]),
                        efficiency=float(self.efficiency[endpoint, column]),
                        directional_agreement=float(
                            self.directional_agreement[endpoint, column]
                        ),
                        reversal_count=int(
                            self.reversal_count[endpoint, column]
                        ),
                        activity_confirmation=float(
                            self.activity_confirmation[endpoint, column]
                        ),
                        residual_fraction=float(
                            self.residual_fraction[endpoint, column]
                        ),
                    ),
                ),
            )
            for column, symbol in enumerate(self.symbols)
        )


@dataclass(frozen=True, slots=True)
class SourceProximitySeries:
    symbols: tuple[str, ...]
    source_proximity_probability: np.ndarray
    lead_or_response_probability: np.ndarray
    coupling_probability: np.ndarray
    efficiency_probability: np.ndarray
    low_residual_probability: np.ndarray
    activity_confirmation_probability: np.ndarray


@dataclass(frozen=True, slots=True)
class RankingConfig:
    windows: tuple[int, ...] = (5, 15, 30, 60)
    ranking_window: int = 60
    minimum_efficiency: float = 0.6
    maximum_residual_fraction: float = 0.5
    minimum_activity_confirmation: float = 0.5
    monte_carlo_samples: int = 10_000
    random_seed: int = 20260820

    def __post_init__(self) -> None:
        if not self.windows or any(item <= 0 for item in self.windows):
            raise ValueError("path windows must contain positive minute counts")
        if self.ranking_window not in self.windows:
            raise ValueError("ranking window must be one of the path windows")
        probabilities = (
            self.minimum_efficiency,
            self.maximum_residual_fraction,
            self.minimum_activity_confirmation,
        )
        if any(not 0.0 <= item <= 1.0 for item in probabilities):
            raise ValueError("ranking event thresholds must be probabilities")
        if self.monte_carlo_samples < 100:
            raise ValueError("joint posterior requires at least 100 draws")


class PathEfficiencyAnalyzer:
    def __init__(self, windows: tuple[int, ...] = (5, 15, 30, 60)) -> None:
        if not windows or any(item <= 0 for item in windows):
            raise ValueError("path windows must contain positive minute counts")
        self.windows = windows

    def compute(
        self,
        panel: NormalizedSectorPanel,
        flow: SectorFlowPosterior,
        attribution: FlowAttributionPosterior,
        *,
        endpoint: int | None = None,
    ) -> tuple[StockPathMetrics, ...]:
        if panel.symbols != flow.symbols or panel.symbols != attribution.symbols:
            raise ValueError("path inputs do not share the same symbols")
        if not np.array_equal(panel.timestamps, flow.timestamps):
            raise ValueError("path panel and flow timestamps do not align")
        if not np.array_equal(panel.timestamps, attribution.timestamps):
            raise ValueError("path panel and attribution timestamps do not align")
        resolved_endpoint = len(panel.timestamps) - 1 if endpoint is None else endpoint
        if not 0 <= resolved_endpoint < len(panel.timestamps):
            raise ValueError("path endpoint is outside the panel")
        output: list[StockPathMetrics] = []
        for symbol, name in enumerate(panel.symbols):
            metrics = tuple(
                self._window_metrics(
                    panel,
                    flow,
                    attribution,
                    symbol,
                    minutes,
                    resolved_endpoint,
                )
                for minutes in self.windows
            )
            output.append(StockPathMetrics(symbol=name, windows=metrics))
        return tuple(output)

    def compute_series(
        self,
        panel: NormalizedSectorPanel,
        flow: SectorFlowPosterior,
        attribution: FlowAttributionPosterior,
        *,
        minutes: int,
    ) -> PathMetricSeries:
        if minutes not in self.windows:
            raise ValueError("path window is not configured")
        if panel.symbols != flow.symbols or panel.symbols != attribution.symbols:
            raise ValueError("path inputs do not share the same symbols")
        if not np.array_equal(panel.timestamps, flow.timestamps):
            raise ValueError("path panel and flow timestamps do not align")
        if not np.array_equal(panel.timestamps, attribution.timestamps):
            raise ValueError("path panel and attribution timestamps do not align")

        valid = panel.valid & flow.valid[:, None]
        raw = np.where(valid, panel.raw_velocity, 0.0)
        observed = np.where(valid, panel.velocity, 0.0)
        sector = np.where(valid, flow.leave_one_out_mean, 0.0)
        direction = np.sign(observed)
        sector_direction = np.sign(sector)

        valid_minutes = self._rolling_sum(valid.astype(np.int64), minutes)
        path_length = self._rolling_sum(np.abs(raw), minutes)
        displacement = np.abs(self._rolling_sum(raw, minutes))
        efficiency = np.divide(
            displacement,
            path_length,
            out=np.zeros_like(path_length),
            where=path_length > 0.0,
        )
        directional_matches = valid & (direction == sector_direction)
        directional_agreement = np.divide(
            self._rolling_sum(directional_matches.astype(np.int64), minutes),
            valid_minutes,
            out=np.zeros_like(path_length),
            where=valid_minutes > 0,
        )

        reversals = np.zeros_like(valid, dtype=np.int64)
        reversals[1:] = (
            valid[1:]
            & valid[:-1]
            & (direction[1:] != 0.0)
            & (direction[:-1] != 0.0)
            & (direction[1:] != direction[:-1])
        )
        reversal_count = self._rolling_sum(reversals, minutes)
        starts = np.maximum(
            np.arange(len(panel.timestamps), dtype=np.int64) + 1 - minutes,
            0,
        )
        reversal_count -= reversals[starts]

        activity_valid = valid & panel.activity_valid
        activity_matches = (
            activity_valid
            & (panel.activity >= 1.0)
            & (direction == sector_direction)
        )
        activity_count = self._rolling_sum(
            activity_valid.astype(np.int64), minutes
        )
        activity_confirmation = np.divide(
            self._rolling_sum(activity_matches.astype(np.int64), minutes),
            activity_count,
            out=np.zeros_like(path_length),
            where=activity_count > 0,
        )

        residual_valid = valid & attribution.valid
        residual = np.where(
            residual_valid, observed - attribution.response_mean, 0.0
        )
        residual_energy = self._rolling_sum(np.square(residual), minutes)
        observed_energy = self._rolling_sum(
            np.where(residual_valid, np.square(observed), 0.0), minutes
        )
        residual_count = self._rolling_sum(
            residual_valid.astype(np.int64), minutes
        )
        residual_fraction = np.divide(
            residual_energy,
            np.maximum(observed_energy, 1e-12),
            out=np.ones_like(residual_energy),
            where=residual_count > 0,
        )
        residual_fraction = np.clip(residual_fraction, 0.0, 1.0)

        return PathMetricSeries(
            symbols=panel.symbols,
            minutes=minutes,
            valid_minutes=valid_minutes,
            efficiency=efficiency,
            directional_agreement=directional_agreement,
            reversal_count=reversal_count,
            activity_confirmation=activity_confirmation,
            residual_fraction=residual_fraction,
        )

    @staticmethod
    def _rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
        cumulative = np.concatenate(
            [
                np.zeros((1, values.shape[1]), dtype=values.dtype),
                np.cumsum(values, axis=0),
            ],
            axis=0,
        )
        ends = np.arange(1, len(values) + 1, dtype=np.int64)
        starts = np.maximum(ends - window, 0)
        return cumulative[ends] - cumulative[starts]

    @staticmethod
    def _window_metrics(
        panel: NormalizedSectorPanel,
        flow: SectorFlowPosterior,
        attribution: FlowAttributionPosterior,
        symbol: int,
        minutes: int,
        endpoint: int,
    ) -> PathWindowMetrics:
        end = endpoint + 1
        start = max(0, end - minutes)
        raw = panel.raw_velocity[start:end, symbol]
        observed = panel.velocity[start:end, symbol]
        sector = flow.leave_one_out_mean[start:end, symbol]
        response = attribution.response_mean[start:end, symbol]
        valid = panel.valid[start:end, symbol] & flow.valid[start:end]
        values = raw[valid]
        path_length = float(np.sum(np.abs(values)))
        efficiency = (
            float(abs(np.sum(values)) / path_length) if path_length > 0.0 else 0.0
        )
        direction = np.sign(observed)
        sector_direction = np.sign(sector)
        directional_agreement = (
            float(np.mean(direction[valid] == sector_direction[valid]))
            if np.any(valid)
            else 0.0
        )
        valid_indices = np.flatnonzero(valid)
        reversal_count = 0
        if len(valid_indices) > 1:
            prior = valid_indices[:-1]
            current = valid_indices[1:]
            contiguous = current == prior + 1
            prior_sign = direction[prior]
            current_sign = direction[current]
            reversal_count = int(
                np.sum(
                    contiguous
                    & (prior_sign != 0.0)
                    & (current_sign != 0.0)
                    & (prior_sign != current_sign)
                )
            )
        activity_valid = valid & panel.activity_valid[start:end, symbol]
        activity_confirmation = (
            float(
                np.mean(
                    (panel.activity[start:end, symbol][activity_valid] >= 1.0)
                    & (
                        direction[activity_valid]
                        == sector_direction[activity_valid]
                    )
                )
            )
            if np.any(activity_valid)
            else 0.0
        )
        residual_valid = valid & attribution.valid[start:end, symbol]
        if np.any(residual_valid):
            residual = observed[residual_valid] - response[residual_valid]
            total_energy = float(np.sum(np.square(observed[residual_valid])))
            residual_fraction = float(
                np.clip(
                    np.sum(np.square(residual)) / max(total_energy, 1e-12),
                    0.0,
                    1.0,
                )
            )
        else:
            residual_fraction = 1.0
        return PathWindowMetrics(
            minutes=minutes,
            valid_minutes=int(np.sum(valid)),
            efficiency=efficiency,
            directional_agreement=directional_agreement,
            reversal_count=reversal_count,
            activity_confirmation=activity_confirmation,
            residual_fraction=residual_fraction,
        )


class SourceProximityRanker:
    def __init__(self, config: RankingConfig | None = None) -> None:
        self.config = config or RankingConfig()

    def rank(
        self,
        paths: tuple[StockPathMetrics, ...],
        attributions: tuple[StockFlowAttribution, ...],
    ) -> tuple[SourceProximityPosterior, ...]:
        attribution_by_symbol = {item.symbol: item for item in attributions}
        results: list[SourceProximityPosterior] = []
        for offset, path in enumerate(paths):
            attribution = attribution_by_symbol.get(path.symbol)
            if attribution is None:
                raise ValueError(f"missing flow attribution for {path.symbol}")
            metric = path.window(self.config.ranking_window)
            rng = np.random.default_rng(self.config.random_seed + offset)
            samples = self.config.monte_carlo_samples
            leader = rng.random(samples) < attribution.leader_probability
            response = rng.random(samples) < attribution.response_probability
            lead_or_response = leader | response
            coupling = rng.normal(
                attribution.coupling_mean,
                max(attribution.coupling_std, 1e-12),
                samples,
            ) > 0.0
            observations = max(metric.valid_minutes, 1)
            efficiency_draw = rng.beta(
                1.0 + metric.efficiency * observations,
                1.0 + (1.0 - metric.efficiency) * observations,
                samples,
            )
            residual_draw = rng.beta(
                1.0 + metric.residual_fraction * observations,
                1.0 + (1.0 - metric.residual_fraction) * observations,
                samples,
            )
            activity_draw = rng.beta(
                1.0 + metric.activity_confirmation * observations,
                1.0 + (1.0 - metric.activity_confirmation) * observations,
                samples,
            )
            efficient = efficiency_draw >= self.config.minimum_efficiency
            low_residual = (
                residual_draw <= self.config.maximum_residual_fraction
            )
            activity_confirmed = (
                activity_draw >= self.config.minimum_activity_confirmation
            )
            joint = (
                lead_or_response
                & coupling
                & efficient
                & low_residual
                & activity_confirmed
            )
            results.append(
                SourceProximityPosterior(
                    symbol=path.symbol,
                    source_proximity_probability=float(np.mean(joint)),
                    lead_or_response_probability=float(
                        np.mean(lead_or_response)
                    ),
                    coupling_probability=float(np.mean(coupling)),
                    efficiency_probability=float(np.mean(efficient)),
                    low_residual_probability=float(np.mean(low_residual)),
                    activity_confirmation_probability=float(
                        np.mean(activity_confirmed)
                    ),
                    leader_probability=attribution.leader_probability,
                    response_probability=attribution.response_probability,
                    path_metrics=metric,
                )
            )
        return tuple(
            sorted(
                results,
                key=lambda item: item.source_proximity_probability,
                reverse=True,
            )
        )

    def rank_series(
        self,
        metrics: PathMetricSeries,
        attributions: tuple[StockFlowAttribution, ...],
    ) -> SourceProximitySeries:
        attribution_by_symbol = {item.symbol: item for item in attributions}
        try:
            ordered = tuple(attribution_by_symbol[symbol] for symbol in metrics.symbols)
        except KeyError as error:
            raise ValueError(f"missing flow attribution for {error.args[0]}") from error

        leader = np.asarray(
            [item.leader_probability for item in ordered], dtype=np.float64
        )
        response = np.asarray(
            [item.response_probability for item in ordered], dtype=np.float64
        )
        lead_or_response = 1.0 - (1.0 - leader) * (1.0 - response)
        coupling_mean = np.asarray(
            [item.coupling_mean for item in ordered], dtype=np.float64
        )
        coupling_std = np.maximum(
            np.asarray(
                [item.coupling_std for item in ordered], dtype=np.float64
            ),
            1e-12,
        )
        coupling = self._normal_cdf(coupling_mean / coupling_std)
        observations = np.maximum(metrics.valid_minutes, 1).astype(np.float64)
        efficient = self._beta_normal_tail(
            metrics.efficiency,
            observations,
            self.config.minimum_efficiency,
            upper=True,
        )
        low_residual = self._beta_normal_tail(
            metrics.residual_fraction,
            observations,
            self.config.maximum_residual_fraction,
            upper=False,
        )
        activity = self._beta_normal_tail(
            metrics.activity_confirmation,
            observations,
            self.config.minimum_activity_confirmation,
            upper=True,
        )
        source = (
            lead_or_response[None, :]
            * coupling[None, :]
            * efficient
            * low_residual
            * activity
        )
        return SourceProximitySeries(
            symbols=metrics.symbols,
            source_proximity_probability=source,
            lead_or_response_probability=lead_or_response,
            coupling_probability=coupling,
            efficiency_probability=efficient,
            low_residual_probability=low_residual,
            activity_confirmation_probability=activity,
        )

    @classmethod
    def _beta_normal_tail(
        cls,
        estimate: np.ndarray,
        observations: np.ndarray,
        threshold: float,
        *,
        upper: bool,
    ) -> np.ndarray:
        alpha = 1.0 + np.clip(estimate, 0.0, 1.0) * observations
        beta = 1.0 + (1.0 - np.clip(estimate, 0.0, 1.0)) * observations
        total = alpha + beta
        mean = alpha / total
        variance = alpha * beta / (np.square(total) * (total + 1.0))
        standardized = (mean - threshold) / np.sqrt(
            np.maximum(variance, 1e-12)
        )
        probability = cls._normal_cdf(standardized)
        return probability if upper else 1.0 - probability

    @staticmethod
    def _normal_cdf(values: np.ndarray) -> np.ndarray:
        absolute = np.abs(values)
        scale = 1.0 / (1.0 + 0.2316419 * absolute)
        density = np.exp(-0.5 * np.square(absolute)) / np.sqrt(2.0 * np.pi)
        polynomial = scale * (
            0.319381530
            + scale
            * (
                -0.356563782
                + scale
                * (1.781477937 + scale * (-1.821255978 + scale * 1.330274429))
            )
        )
        positive = 1.0 - density * polynomial
        return np.where(values >= 0.0, positive, 1.0 - positive)
