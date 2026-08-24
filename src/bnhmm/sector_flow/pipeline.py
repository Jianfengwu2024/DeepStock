"""End-to-end monthly walk-forward orchestration for SectorFlowBayes."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from math import erf, sqrt
from pathlib import Path
from typing import Protocol

import numpy as np

from ..types import BarBatch
from .attribution import AttributionConfig, FlowAttributionModel
from .evaluation import (
    GraphPredictionObservation,
    RankingObservation,
    SectorFlowFold,
    SectorFlowPrediction,
    build_monthly_folds,
    evaluate_graph_predictions,
    evaluate_predictions,
    evaluate_ranking_observations,
    predictions_as_json,
)
from .factor import BayesianSectorFactor, SectorFactorConfig
from .graph_state import SectorGraphStateModel
from .normalization import CausalSectorNormalizer, SectorNormalizationConfig
from .panel import SectorMinutePanelBuilder
from .ranking import PathEfficiencyAnalyzer, RankingConfig, SourceProximityRanker
from .river import DailyRiverSliceBuilder
from .topology import RiverTopologyBuilder, TopologyConfig
from .types import (
    DailyRiverSlice,
    MonthlyRiverTopology,
    NormalizedSectorPanel,
    SectorMinutePanel,
    SourceProximityPosterior,
)


MODEL_VERSION = "sector-flow-bayes-v1"


class MinuteBarSource(Protocol):
    def fetch(
        self, symbol: str, timeframe: str, start: str, end: str
    ) -> BarBatch: ...


@dataclass(frozen=True, slots=True)
class SectorFlowConfig:
    symbols: tuple[str, ...]
    start: str
    end: str
    protocol: str = "legacy-v1"
    topology_lookback_sessions: int = 60
    normalization_lookback_sessions: int = 20
    normalization_minimum_observations: int = 10
    minimum_train_sessions: int = 60
    maximum_lag_minutes: int = 30
    minimum_attribution_rows: int = 120
    factor_process_variance: float | None = None
    path_windows: tuple[int, ...] = (5, 15, 30, 60)
    ranking_window: int = 60
    evaluation_horizon_minutes: int = 30
    top_k: int = 10
    transaction_cost_bps: float = 10.0
    monte_carlo_samples: int = 1_000
    random_seed: int = 20260820
    max_folds: int | None = None

    def __post_init__(self) -> None:
        if len(self.symbols) < 3 or len(set(self.symbols)) != len(self.symbols):
            raise ValueError("sector flow requires at least three unique symbols")
        if not self.start or not self.end or self.start >= self.end:
            raise ValueError("sector flow date range is invalid")
        if self.protocol not in {"legacy-v1", "river-topology-v2"}:
            raise ValueError("unsupported sector flow protocol")
        if self.topology_lookback_sessions < 2:
            raise ValueError("topology lookback must contain at least two sessions")
        if self.minimum_train_sessions < self.normalization_minimum_observations:
            raise ValueError("training history cannot be shorter than normalization history")
        if self.maximum_lag_minutes < 0 or self.minimum_attribution_rows < 2:
            raise ValueError("attribution settings are invalid")
        if self.evaluation_horizon_minutes < 1 or self.top_k < 1:
            raise ValueError("evaluation settings are invalid")
        if self.transaction_cost_bps < 0.0:
            raise ValueError("transaction cost cannot be negative")
        if self.monte_carlo_samples < 100:
            raise ValueError("at least 100 posterior draws are required")


@dataclass(frozen=True, slots=True)
class SectorFlowRunResult:
    folds: tuple[dict[str, object], ...]
    predictions: tuple[SectorFlowPrediction, ...]
    metrics: dict[str, float | int]
    baselines: dict[str, dict[str, float]]
    latest_ranking: tuple[SourceProximityPosterior, ...]
    latest_water_state: dict[str, float]
    excluded_symbols: tuple[str, ...]
    graph_predictions: tuple[GraphPredictionObservation, ...] = ()
    topologies: tuple[MonthlyRiverTopology, ...] = ()


class SectorFlowPipeline:
    def __init__(self, config: SectorFlowConfig) -> None:
        self.config = config

    def run(
        self, source: MinuteBarSource, output_dir: str | Path
    ) -> SectorFlowRunResult:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        requested_batches = {
            symbol: source.fetch(symbol, "1m", self.config.start, self.config.end)
            for symbol in self.config.symbols
        }
        excluded_symbols = tuple(
            symbol for symbol, batch in requested_batches.items() if not len(batch)
        )
        batches = {
            symbol: batch
            for symbol, batch in requested_batches.items()
            if len(batch)
        }
        if len(batches) < 3:
            raise ValueError(
                "sector flow requires at least three symbols with one-minute bars "
                f"in the requested period; excluded: {list(excluded_symbols)}"
            )
        panel = SectorMinutePanelBuilder().build(batches)
        normalized = CausalSectorNormalizer(
            SectorNormalizationConfig(
                lookback_sessions=self.config.normalization_lookback_sessions,
                minimum_observations=self.config.normalization_minimum_observations,
            )
        ).transform(panel)
        folds = build_monthly_folds(
            panel.timestamps,
            minimum_train_sessions=self.config.minimum_train_sessions,
            max_folds=self.config.max_folds,
        )
        if not folds:
            raise ValueError("date range does not produce a monthly sample-out fold")
        if self.config.protocol == "river-topology-v2":
            return self._run_river(
                panel,
                normalized,
                folds,
                destination,
                excluded_symbols,
            )
        eligible_columns = self._first_fold_eligible_columns(
            normalized, folds[0].training_rows
        )
        if len(eligible_columns) < 3:
            raise ValueError(
                "sector flow requires at least three symbols with sufficient "
                "first-fold attribution history"
            )
        eligible_set = set(eligible_columns.tolist())
        history_excluded = tuple(
            symbol
            for column, symbol in enumerate(panel.symbols)
            if column not in eligible_set
        )
        excluded_symbols += history_excluded
        if len(eligible_columns) != len(panel.symbols):
            panel, normalized = self._select_symbols(
                panel, normalized, eligible_columns
            )
        predictions: list[SectorFlowPrediction] = []
        fold_summaries: list[dict[str, object]] = []
        latest_ranking: tuple[SourceProximityPosterior, ...] = ()
        all_factor_means: list[float] = []
        all_factor_times: list[np.datetime64] = []
        latest_water_state: dict[str, float] = {}
        baseline_observations: dict[str, list[RankingObservation]] = {
            name: []
            for name in (
                "random",
                "momentum",
                "relative_volume",
                "path_efficiency",
                "static_beta",
            )
        }

        for fold in folds:
            (
                fold_predictions,
                fold_ranking,
                fold_baselines,
                fold_water_state,
                factor_mean,
                factor_times,
                summary,
            ) = (
                self._run_fold(panel, normalized, fold)
            )
            predictions.extend(fold_predictions)
            latest_ranking = fold_ranking or latest_ranking
            latest_water_state = fold_water_state or latest_water_state
            all_factor_means.extend(factor_mean)
            all_factor_times.extend(factor_times)
            fold_summaries.append(summary)
            for name, observations in fold_baselines.items():
                baseline_observations[name].extend(observations)
        if not predictions:
            raise ValueError("walk-forward folds produced no evaluable predictions")
        ordered = tuple(
            sorted(predictions, key=lambda item: (item.signal_time, item.symbol))
        )
        metrics = evaluate_predictions(
            ordered,
            top_k=self.config.top_k,
            transaction_cost_bps=self.config.transaction_cost_bps,
        )
        baselines = {
            name: evaluate_ranking_observations(
                observations,
                top_k=self.config.top_k,
                transaction_cost_bps=self.config.transaction_cost_bps,
            )
            for name, observations in baseline_observations.items()
        }
        self._write_artifacts(
            destination,
            ordered,
            metrics,
            baselines,
            tuple(fold_summaries),
            latest_ranking,
            latest_water_state,
            excluded_symbols,
            np.asarray(all_factor_times, dtype="datetime64[ns]"),
            np.asarray(all_factor_means, dtype=np.float64),
        )
        return SectorFlowRunResult(
            folds=tuple(fold_summaries),
            predictions=ordered,
            metrics=metrics,
            baselines=baselines,
            latest_ranking=latest_ranking,
            latest_water_state=latest_water_state,
            excluded_symbols=excluded_symbols,
        )

    def _run_river(
        self,
        panel: SectorMinutePanel,
        normalized: NormalizedSectorPanel,
        folds: tuple[SectorFlowFold, ...],
        destination: Path,
        excluded_symbols: tuple[str, ...],
    ) -> SectorFlowRunResult:
        minimum_topology_sessions = max(
            2,
            min(
                40,
                self.config.topology_lookback_sessions
                - self.config.normalization_lookback_sessions,
            ),
        )
        topology_builder = RiverTopologyBuilder(
            TopologyConfig(
                lookback_sessions=self.config.topology_lookback_sessions,
                minimum_sessions=minimum_topology_sessions,
                minimum_pair_sessions=minimum_topology_sessions,
            )
        )
        factor_model = BayesianSectorFactor(
            SectorFactorConfig(
                process_variance=self.config.factor_process_variance,
                minimum_symbols=min(5, len(panel.symbols)),
            )
        )
        graph_model = SectorGraphStateModel()
        slice_builder = DailyRiverSliceBuilder()
        sessions = np.unique(panel.timestamps.astype("datetime64[D]"))
        session_rows = {
            session: np.flatnonzero(
                panel.timestamps.astype("datetime64[D]") == session
            ).astype(np.int64)
            for session in sessions
        }
        observations: list[GraphPredictionObservation] = []
        published_topologies: list[MonthlyRiverTopology] = []
        published_slices: list[DailyRiverSlice] = []
        fold_summaries: list[dict[str, object]] = []
        topology_by_month = self._river_topologies_through(
            normalized,
            topology_builder,
            folds[-1].test_start.astype("datetime64[M]"),
        )

        for fold in folds:
            training_cutoff = panel.timestamps[fold.training_rows[-1]]
            parameters = factor_model.fit(normalized, fold.training_rows)
            flow = factor_model.filter(normalized, parameters)
            test_month = fold.test_start.astype("datetime64[M]")
            if test_month not in topology_by_month:
                raise ValueError("insufficient completed history for sample-out topology")
            topology = topology_by_month[test_month]
            training_samples: list[
                tuple[MonthlyRiverTopology, DailyRiverSlice, np.ndarray]
            ] = []
            training_sessions = np.unique(
                panel.timestamps[fold.training_rows].astype("datetime64[D]")
            )
            for session in training_sessions[:-1]:
                month = session.astype("datetime64[M]")
                historical_topology = topology_by_month.get(month)
                next_session = self._next_session(sessions, session)
                if historical_topology is None or next_session is None:
                    continue
                if next_session.astype("datetime64[M]") == test_month:
                    continue
                river_slice = slice_builder.build(
                    normalized,
                    parameters,
                    flow,
                    historical_topology,
                    session,
                )
                target = self._next_session_relative_return(
                    panel,
                    session_rows,
                    session,
                    next_session,
                    river_slice.symbols,
                )
                training_samples.append(
                    (historical_topology, river_slice, target)
                )
            if not training_samples:
                raise ValueError("river walk-forward fold has no causal training slices")
            graph_parameters = graph_model.fit(
                training_samples,
                training_cutoff=training_cutoff,
            )

            fold_observation_count = 0
            test_sessions = np.unique(
                panel.timestamps[fold.test_rows].astype("datetime64[D]")
            )
            for session in test_sessions:
                next_session = self._next_session(sessions, session)
                if next_session is None:
                    continue
                river_slice = slice_builder.build(
                    normalized,
                    parameters,
                    flow,
                    topology,
                    session,
                )
                available_at = panel.timestamps[session_rows[next_session][0]]
                prediction = graph_model.predict(
                    graph_parameters,
                    topology,
                    river_slice,
                    available_at=available_at,
                )
                realized = self._next_session_relative_return(
                    panel,
                    session_rows,
                    session,
                    next_session,
                    prediction.symbols,
                )
                for row, symbol in enumerate(prediction.symbols):
                    if not np.isfinite(realized[row]):
                        continue
                    observations.append(
                        GraphPredictionObservation(
                            symbol=symbol,
                            signal_time=prediction.signal_time,
                            available_at=prediction.available_at,
                            training_cutoff=prediction.training_cutoff,
                            model_version=prediction.model_version,
                            fold=fold.index,
                            expected_relative_return=float(
                                prediction.expected_relative_return[row]
                            ),
                            predictive_variance=float(
                                prediction.predictive_variance[row]
                            ),
                            realized_relative_return=float(realized[row]),
                        )
                    )
                    fold_observation_count += 1
                published_slices.append(river_slice)
            published_topologies.append(topology)
            fold_summaries.append(
                {
                    "fold": fold.index,
                    "training_cutoff": str(training_cutoff),
                    "test_start": str(fold.test_start),
                    "test_end": str(fold.test_end),
                    "training_samples": len(training_samples),
                    "prediction_count": fold_observation_count,
                    "topology_month": str(topology.month),
                }
            )

        if not observations:
            raise ValueError("river walk-forward folds produced no evaluable predictions")
        ordered = tuple(
            sorted(observations, key=lambda item: (item.signal_time, item.symbol))
        )
        metrics = evaluate_graph_predictions(
            ordered,
            top_k=self.config.top_k,
            transaction_cost_bps=self.config.transaction_cost_bps,
        )
        self._write_river_artifacts(
            destination,
            ordered,
            metrics,
            tuple(fold_summaries),
            tuple(published_topologies),
            tuple(published_slices),
            excluded_symbols,
        )
        return SectorFlowRunResult(
            folds=tuple(fold_summaries),
            predictions=(),
            metrics=metrics,
            baselines={},
            latest_ranking=(),
            latest_water_state={},
            excluded_symbols=excluded_symbols,
            graph_predictions=ordered,
            topologies=tuple(published_topologies),
        )

    def _river_topologies_through(
        self,
        normalized: NormalizedSectorPanel,
        builder: RiverTopologyBuilder,
        final_month: np.datetime64,
    ) -> dict[np.datetime64, MonthlyRiverTopology]:
        months = np.unique(normalized.timestamps.astype("datetime64[M]"))
        result: dict[np.datetime64, MonthlyRiverTopology] = {}
        previous: MonthlyRiverTopology | None = None
        for month in months:
            if month > final_month:
                break
            prior_rows = np.flatnonzero(
                normalized.timestamps.astype("datetime64[M]") < month
            )
            if not len(prior_rows):
                continue
            try:
                topology = builder.build(
                    normalized,
                    cutoff=normalized.timestamps[prior_rows[-1]],
                    month=month,
                    previous=previous,
                )
            except ValueError as error:
                if "insufficient completed sessions" in str(error):
                    continue
                raise
            result[month] = topology
            previous = topology
        return result

    @staticmethod
    def _next_session(
        sessions: np.ndarray, session: np.datetime64
    ) -> np.datetime64 | None:
        location = int(np.searchsorted(sessions, session))
        if location + 1 >= len(sessions):
            return None
        return sessions[location + 1]

    @staticmethod
    def _next_session_relative_return(
        panel: SectorMinutePanel,
        session_rows: dict[np.datetime64, np.ndarray],
        session: np.datetime64,
        next_session: np.datetime64,
        symbols: tuple[str, ...],
    ) -> np.ndarray:
        current_rows = session_rows[session]
        future_rows = session_rows[next_session]
        panel_index = {symbol: index for index, symbol in enumerate(panel.symbols)}
        result = np.full(len(symbols), np.nan, dtype=np.float64)
        for row, symbol in enumerate(symbols):
            column = panel_index[symbol]
            current_valid = current_rows[panel.valid[current_rows, column]]
            future_valid = future_rows[panel.valid[future_rows, column]]
            if not len(current_valid) or not len(future_valid):
                continue
            result[row] = np.log(
                panel.close[future_valid[-1], column]
                / panel.close[current_valid[-1], column]
            )
        finite = np.isfinite(result)
        if finite.any():
            result[finite] -= float(np.mean(result[finite]))
        return result

    def _write_river_artifacts(
        self,
        destination: Path,
        predictions: tuple[GraphPredictionObservation, ...],
        metrics: dict[str, float | int],
        folds: tuple[dict[str, object], ...],
        topologies: tuple[MonthlyRiverTopology, ...],
        slices: tuple[DailyRiverSlice, ...],
        excluded_symbols: tuple[str, ...],
    ) -> None:
        river = destination / "river"
        topology_dir = river / "topologies"
        slice_dir = river / "slices"
        topology_dir.mkdir(parents=True, exist_ok=True)
        slice_dir.mkdir(parents=True, exist_ok=True)
        topology_manifest: list[dict[str, object]] = []
        for topology in topologies:
            key = np.datetime_as_string(topology.month, unit="M")
            filename = f"topologies/{key}.npz"
            np.savez_compressed(
                river / filename,
                month=topology.month,
                training_cutoff=topology.training_cutoff,
                symbols=np.asarray(topology.symbols, dtype=np.str_),
                assigned=topology.assigned,
                band_ids=topology.band_ids,
                row_order=topology.row_order,
                common_similarity=topology.common_similarity,
                residual_similarity=topology.residual_similarity,
                fused_similarity=topology.fused_similarity,
                graph=topology.graph,
                cohesion=topology.cohesion,
                centrality=topology.centrality,
            )
            topology_manifest.append(
                {
                    "month": key,
                    "file": filename,
                    "training_cutoff": str(topology.training_cutoff),
                }
            )
        slice_manifest: list[dict[str, object]] = []
        for item in slices:
            key = np.datetime_as_string(item.session, unit="D")
            filename = f"slices/{key}.npz"
            np.savez_compressed(
                river / filename,
                session=item.session,
                topology_month=item.topology_month,
                symbols=np.asarray(item.symbols, dtype=np.str_),
                timestamps=item.timestamps,
                velocity=item.velocity,
                residual_velocity=item.residual_velocity,
                activity=item.activity,
                valid=item.valid,
            )
            slice_manifest.append(
                {
                    "session": key,
                    "topology_month": np.datetime_as_string(
                        item.topology_month, unit="M"
                    ),
                    "file": filename,
                }
            )
        manifest = {
            "protocol": "river-topology-v2",
            "model_version": "sector-flow-river-v2",
            "topologies": topology_manifest,
            "slices": slice_manifest,
        }
        (river / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        prediction_payload = [
            {
                **asdict(item),
                "signal_time": str(item.signal_time),
                "available_at": str(item.available_at),
                "training_cutoff": str(item.training_cutoff),
            }
            for item in predictions
        ]
        evaluation = {
            "model_version": "sector-flow-river-v2",
            "config": asdict(self.config),
            "folds": list(folds),
            "metrics": metrics,
            "predictions": prediction_payload,
            "excluded_symbols": list(excluded_symbols),
        }
        (destination / "evaluation.json").write_text(
            json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary = {
            "model_scope": "sector-flow",
            "model_version": "sector-flow-river-v2",
            "protocol": "river-topology-v2",
            "oos_predictions": len(predictions),
            "metrics": metrics,
            "folds": list(folds),
        }
        (destination / "run-summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        report = [
            "# SectorFlowBayes v2 Walk-Forward Report",
            "",
            "- Protocol: `river-topology-v2`",
            f"- Monthly topologies: {len(topologies)}",
            f"- Daily slices: {len(slices)}",
            f"- Predictions: {len(predictions)}",
            f"- Rank IC: {metrics['rank_ic']:.6f}",
            f"- Net Top-K minus Bottom-K return: {metrics['net_top_bottom_return']:.6f}",
            f"- Brier score: {metrics['brier_score']:.6f}",
            f"- Posterior coverage: {metrics['posterior_coverage']:.6f}",
            "",
            "Each sample-out month uses a topology fitted only through the prior month and frozen for the full month.",
        ]
        (destination / "sector-flow-v2-report.md").write_text(
            "\n".join(report) + "\n", encoding="utf-8"
        )

    def _first_fold_eligible_columns(
        self,
        panel: NormalizedSectorPanel,
        training_rows: np.ndarray,
    ) -> np.ndarray:
        lag_count = self.config.maximum_lag_minutes + 1
        if len(panel.timestamps) < lag_count:
            return np.asarray([], dtype=np.int64)
        selected = np.zeros(len(panel.timestamps), dtype=np.bool_)
        selected[training_rows] = True
        selected_windows = np.all(
            np.lib.stride_tricks.sliding_window_view(selected, lag_count),
            axis=1,
        )
        eligible: list[int] = []
        for column in range(len(panel.symbols)):
            valid_windows = np.all(
                np.lib.stride_tricks.sliding_window_view(
                    panel.valid[:, column], lag_count
                ),
                axis=1,
            )
            if int(np.sum(selected_windows & valid_windows)) >= (
                self.config.minimum_attribution_rows
            ):
                eligible.append(column)
        return np.asarray(eligible, dtype=np.int64)

    @staticmethod
    def _select_symbols(
        panel: SectorMinutePanel,
        normalized: NormalizedSectorPanel,
        columns: np.ndarray,
    ) -> tuple[SectorMinutePanel, NormalizedSectorPanel]:
        symbols = tuple(panel.symbols[int(column)] for column in columns)
        selected_panel = SectorMinutePanel(
            symbols=symbols,
            timestamps=panel.timestamps,
            close=panel.close[:, columns],
            volume=panel.volume[:, columns],
            valid=panel.valid[:, columns],
        )
        selected_normalized = NormalizedSectorPanel(
            symbols=symbols,
            timestamps=normalized.timestamps,
            raw_velocity=normalized.raw_velocity[:, columns],
            velocity=normalized.velocity[:, columns],
            activity=normalized.activity[:, columns],
            velocity_scale=normalized.velocity_scale[:, columns],
            volume_baseline=normalized.volume_baseline[:, columns],
            valid=normalized.valid[:, columns],
            activity_valid=normalized.activity_valid[:, columns],
        )
        return selected_panel, selected_normalized

    def _run_fold(self, panel, normalized, fold: SectorFlowFold):
        usable_training_rows = fold.training_rows[
            np.sum(normalized.valid[fold.training_rows], axis=1) >= 3
        ]
        factor_model = BayesianSectorFactor(
            SectorFactorConfig(
                process_variance=self.config.factor_process_variance,
                minimum_symbols=min(5, len(panel.symbols)),
            )
        )
        factor_parameters = factor_model.fit(normalized, usable_training_rows)
        flow = factor_model.filter(normalized, factor_parameters)
        attribution_model = FlowAttributionModel(
            AttributionConfig(
                max_lag_minutes=self.config.maximum_lag_minutes,
                minimum_rows=self.config.minimum_attribution_rows,
            )
        )
        attribution_parameters = attribution_model.fit(
            normalized, flow, fold.training_rows
        )
        attribution_summary = attribution_model.summarize(attribution_parameters)
        attribution = attribution_model.score_batch(
            normalized, flow, attribution_parameters
        )
        analyzer = PathEfficiencyAnalyzer(self.config.path_windows)
        ranker = SourceProximityRanker(
            RankingConfig(
                windows=self.config.path_windows,
                ranking_window=self.config.ranking_window,
                monte_carlo_samples=self.config.monte_carlo_samples,
                random_seed=self.config.random_seed + fold.index * 10_000,
            )
        )
        path_series = analyzer.compute_series(
            normalized,
            flow,
            attribution,
            minutes=self.config.ranking_window,
        )
        ranking_series = ranker.rank_series(path_series, attribution_summary)
        predictions: list[SectorFlowPrediction] = []
        baseline_observations: dict[str, list[RankingObservation]] = {
            name: []
            for name in (
                "random",
                "momentum",
                "relative_volume",
                "path_efficiency",
                "static_beta",
            )
        }
        random = np.random.default_rng(self.config.random_seed + fold.index)
        latest_ranking: tuple[SourceProximityPosterior, ...] = ()
        latest_ranking_endpoint: int | None = None
        summary_by_symbol = {item.symbol: item for item in attribution_summary}
        horizon = self.config.evaluation_horizon_minutes
        for row in fold.test_rows:
            target_row = int(row) + horizon
            if target_row >= len(panel.timestamps) or int(row) + 1 >= len(panel.timestamps):
                continue
            expected_target = panel.timestamps[row] + np.timedelta64(horizon, "m")
            if panel.timestamps[target_row] != expected_target:
                continue
            if not flow.valid[row]:
                continue
            latest_ranking_endpoint = int(row)
            for column, symbol in enumerate(panel.symbols):
                if not (
                    panel.valid[row, column]
                    and panel.valid[target_row, column]
                    and attribution.valid[row, column]
                ):
                    continue
                predicted_scale = normalized.velocity_scale[row, column]
                if not np.isfinite(predicted_scale):
                    continue
                predicted_mean = attribution.response_mean[row, column] * predicted_scale
                predicted_variance = max(
                    attribution.response_variance[row, column]
                    * predicted_scale**2,
                    1e-12,
                )
                predicted_std = sqrt(predicted_variance)
                direction_probability = self._normal_cdf(
                    predicted_mean / predicted_std
                )
                realized_return = float(
                    np.log(
                        panel.close[target_row, column]
                        / panel.close[row, column]
                    )
                )
                detail = summary_by_symbol[symbol]
                predictions.append(
                    SectorFlowPrediction(
                        symbol=symbol,
                        signal_time=panel.timestamps[row],
                        available_at=panel.timestamps[int(row) + 1],
                        training_cutoff=fold.test_start,
                        model_version=MODEL_VERSION,
                        fold=fold.index,
                        source_probability=float(
                            ranking_series.source_proximity_probability[row, column]
                        ),
                        direction_probability=direction_probability,
                        realized_return=realized_return,
                        interval_lower=predicted_mean - 1.96 * predicted_std,
                        interval_upper=predicted_mean + 1.96 * predicted_std,
                        leader_horizon_minutes=detail.leader_horizon_minutes,
                        response_delay_minutes=detail.response_delay_minutes,
                    )
                )
                baseline_scores = {
                    "random": float(random.random()),
                    "momentum": float(normalized.velocity[row, column]),
                    "relative_volume": float(normalized.activity[row, column]),
                    "path_efficiency": float(
                        path_series.efficiency[row, column]
                    ),
                    "static_beta": float(
                        factor_parameters.loadings[column] * flow.factor_mean[row]
                    ),
                }
                for name, score in baseline_scores.items():
                    baseline_observations[name].append(
                        RankingObservation(
                            symbol=symbol,
                            signal_time=panel.timestamps[row],
                            score=score,
                            realized_return=realized_return,
                        )
                    )
        if latest_ranking_endpoint is not None:
            latest_paths = analyzer.compute(
                normalized,
                flow,
                attribution,
                endpoint=latest_ranking_endpoint,
            )
            latest_ranking = ranker.rank(latest_paths, attribution_summary)
        test_valid = fold.test_rows[flow.valid[fold.test_rows]]
        latest_water_state: dict[str, float] = {}
        if len(test_valid):
            row = int(test_valid[-1])
            latest_water_state = {
                "direction": float(flow.factor_mean[row]),
                "uncertainty": sqrt(float(flow.factor_variance[row])),
                "activity": float(flow.activity[row]),
                "breadth": float(flow.breadth[row]),
                "coherence": float(flow.coherence[row]),
                "dispersion": float(flow.dispersion[row]),
                "flow_strength": float(flow.flow_strength[row]),
            }
        summary = {
            "fold": fold.index,
            "training_cutoff": str(fold.test_start),
            "test_end": str(fold.test_end),
            "training_rows": int(len(fold.training_rows)),
            "usable_training_rows": int(len(usable_training_rows)),
            "test_rows": int(len(fold.test_rows)),
            "prediction_count": len(predictions),
            "factor_persistence": factor_parameters.persistence,
            "factor_process_variance": factor_parameters.process_variance,
        }
        return (
            predictions,
            latest_ranking,
            baseline_observations,
            latest_water_state,
            flow.factor_mean[test_valid].tolist(),
            panel.timestamps[test_valid].tolist(),
            summary,
        )

    def _write_artifacts(
        self,
        destination: Path,
        predictions: tuple[SectorFlowPrediction, ...],
        metrics: dict[str, float | int],
        baselines: dict[str, dict[str, float]],
        folds: tuple[dict[str, object], ...],
        ranking: tuple[SourceProximityPosterior, ...],
        latest_water_state: dict[str, float],
        excluded_symbols: tuple[str, ...],
        factor_times: np.ndarray,
        factor_mean: np.ndarray,
    ) -> None:
        payload = predictions_as_json(predictions)
        (destination / "predictions.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with (destination / "predictions.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(payload[0]))
            writer.writeheader()
            writer.writerows(payload)
        evaluation = {
            "model_version": MODEL_VERSION,
            "config": asdict(self.config),
            "folds": list(folds),
            "metrics": metrics,
            "baselines": baselines,
            "latest_water_state": latest_water_state,
            "latest_ranking": [asdict(item) for item in ranking],
            "excluded_symbols": list(excluded_symbols),
        }
        (destination / "evaluation.json").write_text(
            json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        run_summary = {
            "model_scope": "sector-flow",
            "model_version": MODEL_VERSION,
            "oos_predictions": len(predictions),
            "metrics": metrics,
            "baselines": baselines,
            "folds": list(folds),
            "latest_water_state": latest_water_state,
            "latest_ranking": [asdict(item) for item in ranking],
            "excluded_symbols": list(excluded_symbols),
        }
        (destination / "run-summary.json").write_text(
            json.dumps(run_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with (destination / "ranking.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            fields = (
                "symbol",
                "source_proximity_probability",
                "lead_or_response_probability",
                "coupling_probability",
                "efficiency_probability",
                "low_residual_probability",
                "activity_confirmation_probability",
                "leader_probability",
                "response_probability",
            )
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for item in ranking:
                row = asdict(item)
                row.pop("path_metrics", None)
                writer.writerow({field: row[field] for field in fields})
        report = [
            "# SectorFlowBayes Walk-Forward Report",
            "",
            f"- Model version: `{MODEL_VERSION}`",
            f"- Predictions: {len(predictions)}",
            f"- Folds: {len(folds)}",
            f"- Symbols excluded for missing period data: {len(excluded_symbols)}",
            f"- Brier score: {metrics['brier_score']:.6f}",
            f"- Log loss: {metrics['log_loss']:.6f}",
            f"- Rank IC: {metrics['rank_ic']:.6f}",
            f"- Net Top-K minus Bottom-K return: {metrics['net_top_bottom_return']:.6f}",
            f"- Posterior coverage: {metrics['posterior_coverage']:.6f}",
            "",
            "## Baseline comparison",
            "",
            "| Baseline | Rank IC | Net Top-Bottom |",
            "| --- | ---: | ---: |",
            *[
                f"| {name} | {value['rank_ic']:.6f} | {value['net_top_bottom_return']:.6f} |"
                for name, value in baselines.items()
            ],
            "",
            "All folds use expanding history, frozen monthly parameters, causal minute filtering, and next-minute availability.",
        ]
        (destination / "sector-flow-report.md").write_text(
            "\n".join(report) + "\n", encoding="utf-8"
        )
        self._plot(destination / "sector-flow.png", factor_times, factor_mean)

    @staticmethod
    def _plot(path: Path, timestamps: np.ndarray, factor: np.ndarray) -> None:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        figure, axis = plt.subplots(figsize=(12, 4))
        axis.plot(timestamps, factor, color="#176B87", linewidth=0.8)
        axis.axhline(0.0, color="#888888", linewidth=0.7)
        axis.set_title("Causal sector flow posterior mean")
        axis.set_ylabel("normalized flow")
        axis.grid(alpha=0.2)
        figure.autofmt_xdate()
        figure.tight_layout()
        figure.savefig(path, dpi=140)
        plt.close(figure)

    @staticmethod
    def _normal_cdf(value: float) -> float:
        return 0.5 * (1.0 + erf(value / sqrt(2.0)))
