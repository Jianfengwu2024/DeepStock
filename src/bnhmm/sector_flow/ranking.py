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
    ) -> tuple[StockPathMetrics, ...]:
        if panel.symbols != flow.symbols or panel.symbols != attribution.symbols:
            raise ValueError("path inputs do not share the same symbols")
        if not np.array_equal(panel.timestamps, flow.timestamps):
            raise ValueError("path panel and flow timestamps do not align")
        if not np.array_equal(panel.timestamps, attribution.timestamps):
            raise ValueError("path panel and attribution timestamps do not align")
        output: list[StockPathMetrics] = []
        for symbol, name in enumerate(panel.symbols):
            metrics = tuple(
                self._window_metrics(panel, flow, attribution, symbol, minutes)
                for minutes in self.windows
            )
            output.append(StockPathMetrics(symbol=name, windows=metrics))
        return tuple(output)

    @staticmethod
    def _window_metrics(
        panel: NormalizedSectorPanel,
        flow: SectorFlowPosterior,
        attribution: FlowAttributionPosterior,
        symbol: int,
        minutes: int,
    ) -> PathWindowMetrics:
        start = max(0, len(panel.timestamps) - minutes)
        raw = panel.raw_velocity[start:, symbol]
        observed = panel.velocity[start:, symbol]
        sector = flow.leave_one_out_mean[start:, symbol]
        response = attribution.response_mean[start:, symbol]
        valid = panel.valid[start:, symbol] & flow.valid[start:]
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
        activity_valid = valid & panel.activity_valid[start:, symbol]
        activity_confirmation = (
            float(
                np.mean(
                    (panel.activity[start:, symbol][activity_valid] >= 1.0)
                    & (
                        direction[activity_valid]
                        == sector_direction[activity_valid]
                    )
                )
            )
            if np.any(activity_valid)
            else 0.0
        )
        residual_valid = valid & attribution.valid[start:, symbol]
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
