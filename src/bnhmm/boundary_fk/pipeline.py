"""Fixed five-stage monthly walk-forward orchestration for BoundaryFlowFK."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np

from .bayesian_state import (
    BayesianBoundaryStateConfig,
    BayesianBoundaryStateModel,
    BoundaryStateLabels,
)
from .evaluation import (
    BoundaryEvaluationConfig,
    BoundaryEvaluationMetrics,
    BoundaryForecastObservation,
    evaluate_boundary_forecasts,
)
from .feynman_kac import BoundaryFeynmanKacConfig, BoundaryFeynmanKacForecaster
from .policy import BoundaryPolicyDecision
from .reporting import write_ablation_report, write_json
from .types import BoundaryDailyStates


BOUNDARY_MODEL_NAMES = (
    "BFK-GBM",
    "BFK-Global",
    "BFK-Boundary",
    "BFK-Branches",
    "BoundaryFlowFK",
)
_PROFILES = ("gbm", "global", "boundary", "branches", "full")


@dataclass(frozen=True, slots=True)
class BoundaryWalkForwardData:
    states: BoundaryDailyStates
    labels: BoundaryStateLabels
    thickness_risk_quantile: np.ndarray
    gross_strategy_returns: np.ndarray

    def __post_init__(self) -> None:
        rows = len(self.states.sessions)
        arrays = (
            self.thickness_risk_quantile,
            self.gross_strategy_returns,
        )
        if any(item.shape != (rows,) for item in arrays):
            raise ValueError("walk-forward side data must align with daily states")
        if not np.array_equal(self.states.sessions, self.labels.signal_sessions):
            raise ValueError("walk-forward states and labels must align")
        if np.any(~np.isfinite(self.thickness_risk_quantile)) or np.any(
            (self.thickness_risk_quantile < 0.0)
            | (self.thickness_risk_quantile > 1.0)
        ):
            raise ValueError("thickness risk quantiles must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class BoundaryFlowPipelineConfig:
    minimum_train_sessions: int = 120
    max_folds: int | None = 1
    fk_paths: int = 10_000
    seed: int = 7
    neutral_sigma: float = 0.35
    minimum_neutral_threshold: float = 0.002
    transaction_cost_bps: float = 10.0
    metric_tolerance: float = 0.0
    minimum_raw_bullish_probability: float = 0.55
    minimum_survival_bullish_probability: float = 0.55
    minimum_trend_survival_probability: float = 0.60
    maximum_bullish_probability_std: float = 0.15
    maximum_thickness_risk_quantile: float = 0.90

    def __post_init__(self) -> None:
        if self.minimum_train_sessions < 30:
            raise ValueError("BoundaryFlowFK requires at least 30 training sessions")
        if self.max_folds is not None and self.max_folds < 1:
            raise ValueError("maximum folds must be positive")
        if self.neutral_sigma <= 0.0 or self.minimum_neutral_threshold <= 0.0:
            raise ValueError("neutral threshold settings must be positive")
        if self.metric_tolerance < 0.0:
            raise ValueError("metric tolerance cannot be negative")
        gates = (
            self.minimum_raw_bullish_probability,
            self.minimum_survival_bullish_probability,
            self.minimum_trend_survival_probability,
            self.maximum_bullish_probability_std,
            self.maximum_thickness_risk_quantile,
        )
        if any(not 0.0 <= value <= 1.0 for value in gates):
            raise ValueError("BoundaryFlowFK gates must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class BoundaryStageResult:
    model_name: str
    status: str
    passed: bool
    test_months: tuple[str, ...]
    drift_feature_names: tuple[str, ...] = ()
    diffusion_feature_names: tuple[str, ...] = ()
    minimum_fit_rows: int = 0
    metrics: BoundaryEvaluationMetrics | None = None
    failure_reason: str = ""


@dataclass(frozen=True, slots=True)
class BoundaryFlowRunResult:
    model_names: tuple[str, ...]
    stages: tuple[BoundaryStageResult, ...]


StageGate = Callable[
    [BoundaryEvaluationMetrics, BoundaryEvaluationMetrics, str], bool
]


class BoundaryFlowPipeline:
    def __init__(
        self,
        config: BoundaryFlowPipelineConfig | None = None,
        *,
        stage_gate: StageGate | None = None,
    ) -> None:
        self.config = config or BoundaryFlowPipelineConfig()
        self.stage_gate = stage_gate or self._default_stage_gate

    def run(
        self,
        data: BoundaryWalkForwardData,
        output_directory: str | Path,
    ) -> BoundaryFlowRunResult:
        root = Path(output_directory)
        root.mkdir(parents=True, exist_ok=True)
        folds = self._mature_monthly_folds(data)
        if not folds:
            raise ValueError("no monthly fold has enough mature BoundaryFlowFK history")
        test_months = tuple(str(test_start.astype("datetime64[M]")) for _, test_start, _, _ in folds)

        stages: list[BoundaryStageResult] = []
        all_predictions: list[dict[str, object]] = []
        posterior_artifacts: list[dict[str, object]] = []
        previous_metrics: BoundaryEvaluationMetrics | None = None
        stopped = False
        for stage_index, (model_name, profile) in enumerate(
            zip(BOUNDARY_MODEL_NAMES, _PROFILES, strict=True)
        ):
            if stopped:
                stages.append(
                    BoundaryStageResult(
                        model_name, "skipped", False, test_months, failure_reason="previous_stage_failed"
                    )
                )
                continue
            observations: list[BoundaryForecastObservation] = []
            stage_drift: tuple[str, ...] = ()
            stage_diffusion: tuple[str, ...] = ()
            fit_rows: list[int] = []
            for fold_index, test_start, training_rows, test_rows in folds:
                cutoff = test_start - np.timedelta64(1, "ns")
                training_selected = np.zeros(len(data.states.sessions), dtype=np.bool_)
                training_selected[training_rows] = True
                matured = data.labels.target_time <= cutoff
                bounded_labels = replace(
                    data.labels,
                    return_valid=data.labels.return_valid & training_selected & matured,
                    survival_valid=data.labels.survival_valid & training_selected & matured,
                )
                posterior = BayesianBoundaryStateModel(
                    BayesianBoundaryStateConfig(feature_profile=profile)
                ).fit(data.states, bounded_labels, training_cutoff=cutoff)
                stage_drift = posterior.drift_feature_names
                stage_diffusion = posterior.diffusion_feature_names
                fit_rows.append(posterior.fit_rows)
                posterior_artifacts.append(
                    {
                        "model_name": model_name,
                        "fold": fold_index,
                        "training_cutoff": str(cutoff),
                        "drift_feature_names": list(stage_drift),
                        "diffusion_feature_names": list(stage_diffusion),
                        "drift_mean": posterior.drift_mean.tolist(),
                        "diffusion_mean": posterior.diffusion_mean.tolist(),
                        "potential_mean": posterior.potential_mean.tolist(),
                    }
                )
                forecaster = BoundaryFeynmanKacForecaster(
                    BoundaryFeynmanKacConfig(
                        paths=self.config.fk_paths,
                        seed=self.config.seed + fold_index,
                        use_effective_potential=model_name == "BoundaryFlowFK",
                    )
                )
                for row in test_rows:
                    if not data.labels.return_valid[row]:
                        continue
                    threshold = max(
                        self.config.minimum_neutral_threshold,
                        self.config.neutral_sigma * data.labels.historical_volatility[row],
                    )
                    forecast = forecaster.predict(
                        posterior,
                        data.states,
                        int(row),
                        data.labels,
                        neutral_threshold=float(threshold),
                    )
                    decision = self._first_layer_decision(
                        forecast,
                        float(data.thickness_risk_quantile[row]),
                        use_effective_potential=model_name == "BoundaryFlowFK",
                    )
                    signal = data.states.sessions[row].astype("datetime64[D]") + np.timedelta64(15, "h")
                    available = signal + np.timedelta64(1, "ns")
                    observation = BoundaryForecastObservation(
                        signal_time=signal,
                        available_at=available,
                        target_time=data.labels.target_time[row],
                        training_cutoff=cutoff,
                        model_version=model_name,
                        forecast=forecast,
                        actual_return=float(data.labels.next_return[row]),
                        neutral_threshold=float(threshold),
                        decision=decision,
                        gross_strategy_return=float(data.gross_strategy_returns[row]),
                    )
                    observations.append(observation)
                    all_predictions.append(self._prediction_payload(observation, fold_index))
            if not observations:
                raise ValueError(f"{model_name} produced no sample-out predictions")
            metrics = evaluate_boundary_forecasts(
                tuple(observations),
                BoundaryEvaluationConfig(
                    transaction_cost_bps=self.config.transaction_cost_bps
                ),
            )
            passed = previous_metrics is None or self.stage_gate(
                previous_metrics, metrics, model_name
            )
            stage = BoundaryStageResult(
                model_name=model_name,
                status="passed" if passed else "failed",
                passed=passed,
                test_months=test_months,
                drift_feature_names=stage_drift,
                diffusion_feature_names=stage_diffusion,
                minimum_fit_rows=min(fit_rows),
                metrics=metrics,
                failure_reason="" if passed else "sample_out_metrics_did_not_improve",
            )
            stages.append(stage)
            previous_metrics = metrics
            stopped = not passed

        stage_payload = [self._stage_payload(stage) for stage in stages]
        write_json(root / "boundary-model-posterior.json", posterior_artifacts)
        write_json(root / "feynman-kac-predictions.json", all_predictions)
        write_json(root / "boundary-policy-evaluation.json", stage_payload)
        write_json(
            root / "boundary-residual-analysis.json",
            [
                {
                    "model_name": stage.model_name,
                    "probability_residual_autocorrelation": (
                        stage.metrics.probability_residual_autocorrelation
                        if stage.metrics is not None
                        else None
                    ),
                }
                for stage in stages
            ],
        )
        write_json(
            root / "boundary-daily-states.json",
            {
                "sessions": [str(item) for item in data.states.sessions],
                "feature_names": list(data.states.feature_names),
                "units": list(data.states.units),
                "valid": data.states.valid.tolist(),
            },
        )
        write_ablation_report(root / "boundary-ablation-report.md", stage_payload)
        return BoundaryFlowRunResult(BOUNDARY_MODEL_NAMES, tuple(stages))

    def _mature_monthly_folds(
        self, data: BoundaryWalkForwardData
    ) -> tuple[tuple[int, np.datetime64, np.ndarray, np.ndarray], ...]:
        sessions = data.states.sessions.astype("datetime64[ns]")
        eligible = data.states.valid & data.labels.return_valid
        months = np.unique(sessions[eligible].astype("datetime64[M]"))
        folds: list[tuple[int, np.datetime64, np.ndarray, np.ndarray]] = []
        for month in months:
            test_start = month.astype("datetime64[ns]")
            test_end = (month + np.timedelta64(1, "M")).astype("datetime64[ns]")
            cutoff = test_start - np.timedelta64(1, "ns")
            training_rows = np.flatnonzero(
                eligible & (sessions < test_start) & (data.labels.target_time <= cutoff)
            )
            test_rows = np.flatnonzero(
                eligible & (sessions >= test_start) & (sessions < test_end)
            )
            mature_sessions = np.unique(sessions[training_rows].astype("datetime64[D]"))
            if len(mature_sessions) < self.config.minimum_train_sessions or not len(test_rows):
                continue
            folds.append((len(folds), test_start, training_rows, test_rows))
            if self.config.max_folds is not None and len(folds) >= self.config.max_folds:
                break
        return tuple(folds)

    def _default_stage_gate(
        self,
        previous: BoundaryEvaluationMetrics,
        current: BoundaryEvaluationMetrics,
        model_name: str,
    ) -> bool:
        tolerance = self.config.metric_tolerance
        return (
            current.brier_score <= previous.brier_score + tolerance
            and current.log_loss <= previous.log_loss + tolerance
            and current.ece <= previous.ece + tolerance
        )

    def _first_layer_decision(
        self,
        forecast,
        thickness_risk_quantile: float,
        *,
        use_effective_potential: bool,
    ) -> BoundaryPolicyDecision:
        conditional = (
            forecast.survival_joint_probabilities[2]
            / forecast.trend_survival_probability
            if forecast.trend_survival_probability > 0.0
            else 0.0
        )
        opened = (
            forecast.raw_state_probabilities[2]
            >= self.config.minimum_raw_bullish_probability
        )
        reason = "raw_bullish_probability"
        if opened and use_effective_potential:
            gates = (
                (
                    conditional
                    >= self.config.minimum_survival_bullish_probability,
                    "survival_bullish_probability",
                ),
                (
                    forecast.trend_survival_probability
                    >= self.config.minimum_trend_survival_probability,
                    "trend_survival_probability",
                ),
                (
                    forecast.bullish_probability_std
                    <= self.config.maximum_bullish_probability_std,
                    "bullish_probability_uncertainty",
                ),
                (
                    thickness_risk_quantile
                    < self.config.maximum_thickness_risk_quantile,
                    "boundary_thickening_risk",
                ),
            )
            for passed, failed_reason in gates:
                if not passed:
                    opened = False
                    reason = failed_reason
                    break
        if opened:
            reason = "first_layer_passed"
        return BoundaryPolicyDecision(opened, reason, conditional)

    @staticmethod
    def _prediction_payload(
        observation: BoundaryForecastObservation, fold: int
    ) -> dict[str, object]:
        forecast = observation.forecast
        return {
            "signal_time": str(observation.signal_time),
            "available_at": str(observation.available_at),
            "target_time": str(observation.target_time),
            "training_cutoff": str(observation.training_cutoff),
            "model_version": observation.model_version,
            "fold": fold,
            "raw_state_probabilities": list(forecast.raw_state_probabilities),
            "survival_joint_probabilities": list(forecast.survival_joint_probabilities),
            "trend_survival_probability": forecast.trend_survival_probability,
            "expected_drift": forecast.expected_drift,
            "expected_diffusion_variance": forecast.expected_diffusion_variance,
            "expected_effective_potential": forecast.expected_effective_potential,
            "epistemic_return_variance": forecast.epistemic_return_variance,
            "aleatoric_return_variance": forecast.aleatoric_return_variance,
            "gate_reason": observation.decision.gate_reason,
            "actual_return": observation.actual_return,
        }

    @staticmethod
    def _stage_payload(stage: BoundaryStageResult) -> dict[str, object]:
        return {
            "model_name": stage.model_name,
            "status": stage.status,
            "passed": stage.passed,
            "test_months": list(stage.test_months),
            "drift_feature_names": list(stage.drift_feature_names),
            "diffusion_feature_names": list(stage.diffusion_feature_names),
            "minimum_fit_rows": stage.minimum_fit_rows,
            "metrics": asdict(stage.metrics) if stage.metrics is not None else None,
            "failure_reason": stage.failure_reason,
        }
