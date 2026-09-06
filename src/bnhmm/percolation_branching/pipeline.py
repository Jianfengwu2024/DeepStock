"""Causal month-by-month PercolationBranching walk-forward pipeline."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ..boundary_fk.execution import build_member_execution_constraints
from ..sector_flow.panel import SectorMinutePanelBuilder
from ..types import BarBatch
from .backbone import SlowBackboneBuilder
from .branching import BayesianBranchingModel
from .evaluation import evaluate_percolation, fit_phase_baselines
from .events import ActivationEventBuilder
from .filter import CriticalStateFilter
from .labels import NullThresholdCalibrator, classify_phase, temporal_reachability
from .policy import PercolationPolicy, execute_percolation_portfolio
from .reporting import render_chinese_report
from .simulator import PercolationSimulator
from .types import (
    ActivationConfig,
    ActivationPanel,
    BranchingPosterior,
    NullThresholdTable,
    PHASE_NAMES,
    PercolationEvaluationObservation,
    PercolationMetrics,
    PercolationPrediction,
    PhaseName,
)


class MinuteBarSource(Protocol):
    def fetch(self, symbol: str, timeframe: str, start: str, end: str) -> BarBatch: ...


@dataclass(frozen=True, slots=True)
class PercolationPipelineConfig:
    symbols: tuple[str, ...]
    dataset_id: str
    start: str
    end: str
    activation_threshold: float = 1.5
    maximum_lag_minutes: int = 3
    topology_lookback_sessions: int = 60
    null_permutations: int = 500
    posterior_paths: int = 10_000
    phase_probability_threshold: float = 0.60
    transaction_cost_bps: float = 5.0
    random_seed: int = 7
    max_folds: int | None = None
    minimum_valid_symbols: int = 20
    candidate_neighbors: int = 10

    def __post_init__(self) -> None:
        if len(self.symbols) < self.minimum_valid_symbols or len(set(self.symbols)) != len(self.symbols):
            raise ValueError("percolation requires enough unique sector symbols")
        if not self.dataset_id or not self.start or not self.end or self.start >= self.end:
            raise ValueError("percolation dataset and date range are required")
        if self.maximum_lag_minutes != 3 or self.topology_lookback_sessions != 60:
            raise ValueError("percolation physical lag and topology window are fixed")
        if self.activation_threshold <= 0.0:
            raise ValueError("activation threshold must be positive")
        if self.null_permutations < 1 or self.posterior_paths < 1:
            raise ValueError("percolation simulation budgets must be positive")
        if not 0.5 <= self.phase_probability_threshold <= 1.0:
            raise ValueError("phase probability threshold must lie in [0.5, 1]")
        if self.transaction_cost_bps < 0.0:
            raise ValueError("transaction cost cannot be negative")
        if self.max_folds is not None and self.max_folds < 1:
            raise ValueError("maximum fold count must be positive")


@dataclass(frozen=True, slots=True)
class PercolationRunResult:
    predictions: tuple[PercolationPrediction, ...]
    observations: tuple[PercolationEvaluationObservation, ...]
    metrics: PercolationMetrics
    baselines: dict[str, dict[str, float | int]]
    folds: tuple[dict[str, object], ...]
    excluded_symbols: tuple[str, ...]


class PercolationPipeline:
    def __init__(self, config: PercolationPipelineConfig) -> None:
        self.config = config

    def run(
        self, source: MinuteBarSource, output_dir: str | Path
    ) -> PercolationRunResult:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        requested = {
            symbol: source.fetch(symbol, "1m", self.config.start, self.config.end)
            for symbol in self.config.symbols
        }
        excluded = tuple(symbol for symbol, batch in requested.items() if not len(batch))
        batches = {symbol: batch for symbol, batch in requested.items() if len(batch)}
        if len(batches) < self.config.minimum_valid_symbols:
            raise ValueError(
                "percolation requires at least "
                f"{self.config.minimum_valid_symbols} symbols with one-minute bars"
            )
        panel = SectorMinutePanelBuilder().build(batches)
        events = ActivationEventBuilder(
            ActivationConfig(activation_threshold=self.config.activation_threshold)
        ).build(panel)
        constraints = build_member_execution_constraints(panel)
        sessions = np.unique(panel.timestamps.astype("datetime64[D]"))
        months = self._eligible_months(sessions)
        if len(months) == 0:
            raise ValueError("date range does not produce a mature sample-out month")

        rng = np.random.default_rng(self.config.random_seed)
        observations: list[PercolationEvaluationObservation] = []
        posterior_payloads: list[dict[str, object]] = []
        network_payloads: list[dict[str, object]] = []
        fold_payloads: list[dict[str, object]] = []
        baseline_hits = {name: 0 for name in ("majority", "persistence", "transition", "simple_reproduction")}
        baseline_total = 0
        previous_weights = np.zeros(len(panel.symbols), dtype=np.float64)
        session_to_constraint = {
            np.datetime64(session, "D"): index
            for index, session in enumerate(constraints.sessions)
        }

        for month in months:
            support = SlowBackboneBuilder(
                lookback_sessions=self.config.topology_lookback_sessions,
                candidate_neighbors=min(self.config.candidate_neighbors, len(panel.symbols) - 1),
            ).build(events, month)
            training_sessions = sessions[
                (sessions < month.astype("datetime64[D]"))
            ][-self.config.topology_lookback_sessions :]
            bull_posterior = BayesianBranchingModel(
                maximum_lag=self.config.maximum_lag_minutes
            ).fit(events, support, training_sessions, "bull")
            bear_posterior = BayesianBranchingModel(
                maximum_lag=self.config.maximum_lag_minutes
            ).fit(events, support, training_sessions, "bear")
            calibrator = NullThresholdCalibrator(
                permutations=self.config.null_permutations,
                maximum_lag=self.config.maximum_lag_minutes,
            )
            bull_null = calibrator.fit(
                events.bull,
                events.timestamps,
                support.adjacency,
                training_sessions,
                direction="bull",
                month=month,
                rng=rng,
            )
            bear_null = calibrator.fit(
                events.bear,
                events.timestamps,
                support.adjacency,
                training_sessions,
                direction="bear",
                month=month,
                rng=rng,
            )
            bull_filter = CriticalStateFilter()
            bear_filter = CriticalStateFilter()
            bull_state = bull_filter.initialize(
                bull_posterior.observe(events, session) for session in training_sessions
            )
            bear_state = bear_filter.initialize(
                bear_posterior.observe(events, session) for session in training_sessions
            )
            phase_history = [
                self._actual_phase(events, session, support.adjacency, bull_null, bear_null)
                for session in training_sessions
            ]
            last_history_session = training_sessions[-1]
            month_sessions = sessions[sessions.astype("datetime64[M]") == month]
            month_predictions = 0
            for signal_session in month_sessions:
                target_position = int(np.searchsorted(sessions, signal_session)) + 1
                if target_position >= len(sessions):
                    continue
                target_session = sessions[target_position]
                bull_state = bull_filter.update(
                    bull_state, bull_posterior.observe(events, signal_session)
                )
                bear_state = bear_filter.update(
                    bear_state, bear_posterior.observe(events, signal_session)
                )
                current_phase = self._actual_phase(
                    events, signal_session, support.adjacency, bull_null, bear_null
                )
                if signal_session > last_history_session:
                    phase_history.append(current_phase)
                    last_history_session = signal_session
                signal_rows = np.flatnonzero(
                    events.timestamps.astype("datetime64[D]") == signal_session
                )
                signal_time = events.timestamps[signal_rows[-1]]
                prediction = PercolationSimulator(paths=self.config.posterior_paths).predict(
                    bull_posterior,
                    bear_posterior,
                    bull_filter.predict(bull_state),
                    bear_filter.predict(bear_state),
                    bull_threshold=bull_null.threshold_for(
                        float(events.bull[signal_rows].mean())
                    ),
                    bear_threshold=bear_null.threshold_for(
                        float(events.bear[signal_rows].mean())
                    ),
                    signal_time=signal_time,
                    available_at=signal_time + np.timedelta64(1, "m"),
                    target_session=target_session,
                    training_cutoff=signal_time,
                    rng=rng,
                )
                actual_phase = self._actual_phase(
                    events, target_session, support.adjacency, bull_null, bear_null
                )
                decision = PercolationPolicy(
                    bull_probability_threshold=self.config.phase_probability_threshold
                ).decide(prediction, panel.symbols)
                execution_row = session_to_constraint[np.datetime64(signal_session, "D")]
                execution = execute_percolation_portfolio(
                    symbols=panel.symbols,
                    previous_weights=previous_weights,
                    decision=decision,
                    hold_returns=constraints.hold_returns[execution_row],
                    buy_returns=constraints.buy_returns[execution_row],
                    buy_executable=constraints.buy_executable[execution_row],
                    sell_returns=constraints.sell_returns[execution_row],
                    sell_executable=constraints.sell_executable[execution_row],
                    transaction_cost_bps=self.config.transaction_cost_bps,
                )
                observations.append(
                    PercolationEvaluationObservation(
                        prediction, actual_phase, decision, execution
                    )
                )
                previous_weights = execution.actual_weights.copy()
                baselines = fit_phase_baselines(
                    tuple(phase_history),
                    current_phase=current_phase,
                    bull_reproduction=prediction.bull_reproduction_mean,
                    bear_reproduction=prediction.bear_reproduction_mean,
                )
                for name in baseline_hits:
                    baseline_hits[name] += int(getattr(baselines, name) == actual_phase)
                baseline_total += 1
                phase_history.append(actual_phase)
                last_history_session = target_session
                month_predictions += 1
            posterior_payloads.extend(
                self._posterior_payload(month, item)
                for item in (bull_posterior, bear_posterior)
            )
            network_payloads.append(self._network_payload(month, bull_posterior, bear_posterior))
            fold_payloads.append(
                {
                    "month": str(month),
                    "training_cutoff": str(support.training_cutoff),
                    "training_sessions": len(training_sessions),
                    "predictions": month_predictions,
                    "candidate_edges": int(support.adjacency.sum() // 2),
                }
            )

        if not observations:
            raise ValueError("percolation walk-forward produced no predictions")
        metrics = evaluate_percolation(tuple(observations))
        baseline_payload = {
            name: {
                "correct": hits,
                "observations": baseline_total,
                "accuracy": hits / baseline_total,
            }
            for name, hits in baseline_hits.items()
        }
        result = PercolationRunResult(
            predictions=tuple(item.prediction for item in observations),
            observations=tuple(observations),
            metrics=metrics,
            baselines=baseline_payload,
            folds=tuple(fold_payloads),
            excluded_symbols=excluded,
        )
        self._write_artifacts(root, result, posterior_payloads, network_payloads)
        return result

    def _eligible_months(self, sessions: np.ndarray) -> np.ndarray:
        eligible: list[np.datetime64] = []
        for month in np.unique(sessions.astype("datetime64[M]")):
            if int(np.sum(sessions < month.astype("datetime64[D]"))) >= self.config.topology_lookback_sessions:
                eligible.append(month)
                if self.config.max_folds is not None and len(eligible) >= self.config.max_folds:
                    break
        return np.asarray(eligible, dtype="datetime64[M]")

    def _actual_phase(
        self,
        events: ActivationPanel,
        session: np.datetime64,
        adjacency: np.ndarray,
        bull_null: NullThresholdTable,
        bear_null: NullThresholdTable,
    ) -> PhaseName:
        rows = np.flatnonzero(events.timestamps.astype("datetime64[D]") == session)
        bull = temporal_reachability(
            events.bull[rows], adjacency, maximum_lag=self.config.maximum_lag_minutes
        )
        bear = temporal_reachability(
            events.bear[rows], adjacency, maximum_lag=self.config.maximum_lag_minutes
        )
        return classify_phase(
            bull.giant_fraction,
            bear.giant_fraction,
            bull_null.threshold_for(bull.event_density),
            bear_null.threshold_for(bear.event_density),
        )

    @staticmethod
    def _posterior_payload(
        month: np.datetime64, posterior: BranchingPosterior
    ) -> dict[str, object]:
        return {
            "month": str(month),
            "direction": posterior.direction,
            "training_cutoff": str(posterior.training_cutoff),
            "symbols": list(posterior.symbols),
            "mean": posterior.mean.tolist(),
            "variance": posterior.variance.tolist(),
            "normalized_mean": posterior.normalized_mean.tolist(),
            "lag_weights": posterior.lag_weights.tolist(),
            "external_rate": posterior.external_rate.tolist(),
            "self_rate": posterior.self_rate.tolist(),
        }

    @staticmethod
    def _network_payload(
        month: np.datetime64,
        bull: BranchingPosterior,
        bear: BranchingPosterior,
    ) -> dict[str, object]:
        def sources(posterior: BranchingPosterior) -> list[dict[str, object]]:
            strength = posterior.normalized_mean.sum(axis=1)
            order = np.argsort(-strength, kind="stable")[:10]
            return [
                {"symbol": posterior.symbols[index], "outgoing_strength": float(strength[index])}
                for index in order
            ]
        return {
            "month": str(month),
            "bull_sources": sources(bull),
            "bear_sources": sources(bear),
        }

    def _write_artifacts(
        self,
        root: Path,
        result: PercolationRunResult,
        posterior_payloads: list[dict[str, object]],
        network_payloads: list[dict[str, object]],
    ) -> None:
        prediction_payloads = []
        gross_equity = 1.0
        net_equity = 1.0
        peak = 1.0
        equity_payload = []
        for item in result.observations:
            prediction = item.prediction
            prediction_payloads.append(
                {
                    "signal_time": str(prediction.signal_time),
                    "available_at": str(prediction.available_at),
                    "target_session": str(prediction.target_session),
                    "training_cutoff": str(prediction.training_cutoff),
                    "model_version": prediction.model_version,
                    "phase_probabilities": list(prediction.phase_probabilities),
                    "predicted_phase": prediction.predicted_phase,
                    "actual_phase": item.actual_phase,
                    "bull_reproduction_mean": prediction.bull_reproduction_mean,
                    "bear_reproduction_mean": prediction.bear_reproduction_mean,
                    "bull_supercritical_probability": prediction.bull_supercritical_probability,
                    "bear_supercritical_probability": prediction.bear_supercritical_probability,
                    "bull_giant_fraction_mean": prediction.bull_giant_fraction_mean,
                    "bear_giant_fraction_mean": prediction.bear_giant_fraction_mean,
                    "bull_membership_probability": prediction.bull_membership_probability.tolist(),
                    "bear_membership_probability": prediction.bear_membership_probability.tolist(),
                    "selected_symbols": list(item.decision.selected_symbols),
                    "target_weights": list(item.decision.target_weights),
                    "actual_weights": item.execution.actual_weights.tolist(),
                    "gross_return": item.execution.gross_return,
                    "net_return": item.execution.net_return,
                    "turnover": item.execution.turnover,
                    "failed_buy_orders": item.execution.failed_buy_orders,
                    "failed_sell_orders": item.execution.failed_sell_orders,
                }
            )
            gross_equity *= 1.0 + item.execution.gross_return
            net_equity *= 1.0 + item.execution.net_return
            peak = max(peak, net_equity)
            equity_payload.append(
                {
                    "session": str(prediction.target_session),
                    "gross_equity": gross_equity,
                    "net_equity": net_equity,
                    "drawdown": 1.0 - net_equity / peak,
                    "exposure": float(item.execution.actual_weights.sum()),
                    "turnover": item.execution.turnover,
                }
            )
        metric_payload = asdict(result.metrics)
        metric_payload["confusion_matrix"] = result.metrics.confusion_matrix.tolist()
        _write_json(root / "percolation-predictions.json", prediction_payloads)
        _write_json(root / "percolation-posteriors.json", posterior_payloads)
        _write_json(root / "percolation-network-summary.json", network_payloads)
        _write_json(root / "percolation-evaluation.json", metric_payload)
        _write_json(root / "percolation-baselines.json", result.baselines)
        _write_json(root / "percolation-equity.json", equity_payload)
        _write_json(
            root / "run-summary.json",
            {
                "strategy": "percolation-branching",
                "model_version": "percolation-branching-v1",
                "dataset_id": self.config.dataset_id,
                "symbols": len(self.config.symbols),
                "excluded_symbols": list(result.excluded_symbols),
                "folds": list(result.folds),
                "predictions": len(result.predictions),
                "configuration": asdict(self.config),
            },
        )
        baseline_accuracy = {
            name: float(values["accuracy"]) for name, values in result.baselines.items()
        }
        report = render_chinese_report(
            dataset_id=self.config.dataset_id,
            observations=len(result.predictions),
            metrics=result.metrics,
            baseline_accuracy=baseline_accuracy,
            smoke_only=(self.config.null_permutations < 500 or self.config.posterior_paths < 10_000),
        )
        temporary = root / ".percolation-report-zh.md.tmp"
        temporary.write_text(report, encoding="utf-8")
        temporary.replace(root / "percolation-report-zh.md")


def _write_json(path: Path, payload: Any) -> None:
    normalized = _json_value(payload)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.datetime64):
        return str(value)
    return value
