"""Monthly walk-forward orchestration for the two-branch FK strategy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .branch_model import BayesianBranchModel, BayesianBranchModelConfig
from .convection import background_potential_features
from .evaluation import (
    BoundaryEvaluationConfig,
    BoundaryEvaluationMetrics,
    TwoBranchForecastObservation,
    evaluate_two_branch_forecasts,
)
from .feynman_kac import BoundaryFeynmanKacConfig, BranchFeynmanKacForecaster
from .policy import BoundaryPolicy, BoundaryPolicyConfig, BranchPositionState
from .reporting import write_background_potential_report, write_baseline_report, write_json
from .types import MonthlyBranchMembership, TwoBranchDataset


BOUNDARY_MODEL_NAMES = (
    "PathEfficiencyFK-60D-Baseline",
    "PathEfficiencyFK-60D-BackgroundPotential",
)
PRIMARY_POLICY_NAME = "path_efficiency_fk"
BoundaryWalkForwardData = TwoBranchDataset


@dataclass(frozen=True, slots=True)
class BoundaryFlowPipelineConfig:
    minimum_train_sessions: int = 120
    max_folds: int | None = 1
    fk_paths: int = 10_000
    seed: int = 7
    neutral_sigma: float = 0.35
    minimum_neutral_threshold: float = 0.002
    transaction_cost_bps: float = 10.0
    drift_half_life_sessions: float | None = 60.0
    minimum_direction_probability: float = 0.60
    minimum_signal_to_noise: float = 0.10
    minimum_survival_probability: float = 0.35

    def __post_init__(self) -> None:
        if self.minimum_train_sessions < 30:
            raise ValueError("PathEfficiencyFK requires at least 30 training sessions")
        if self.max_folds is not None and self.max_folds < 1:
            raise ValueError("maximum folds must be positive")
        if self.neutral_sigma <= 0.0 or self.minimum_neutral_threshold <= 0.0:
            raise ValueError("neutral threshold settings must be positive")
        if self.fk_paths < 1_000:
            raise ValueError("PathEfficiencyFK requires at least 1000 FK paths")
        if self.transaction_cost_bps < 0.0:
            raise ValueError("transaction cost cannot be negative")
        BoundaryPolicyConfig(
            minimum_direction_probability=self.minimum_direction_probability,
            minimum_signal_to_noise=self.minimum_signal_to_noise,
            minimum_survival_probability=self.minimum_survival_probability,
        )
        if (
            self.drift_half_life_sessions is not None
            and self.drift_half_life_sessions <= 0.0
        ):
            raise ValueError("drift half-life must be positive when provided")


@dataclass(frozen=True, slots=True)
class BoundaryStageResult:
    model_name: str
    status: str
    passed: bool
    test_months: tuple[str, ...]
    promotion_eligible: bool = True
    drift_feature_names: tuple[str, ...] = ()
    diffusion_feature_names: tuple[str, ...] = ("constant_residual_variance",)
    minimum_fit_rows: int = 0
    metrics: BoundaryEvaluationMetrics | None = None
    failure_reason: str = ""


@dataclass(frozen=True, slots=True)
class BoundaryFlowRunResult:
    model_names: tuple[str, ...]
    stages: tuple[BoundaryStageResult, ...]


class BoundaryFlowPipeline:
    def __init__(self, config: BoundaryFlowPipelineConfig | None = None) -> None:
        self.config = config or BoundaryFlowPipelineConfig()

    def run(
        self,
        data: TwoBranchDataset,
        output_directory: str | Path,
    ) -> BoundaryFlowRunResult:
        root = Path(output_directory)
        root.mkdir(parents=True, exist_ok=True)
        folds = self._mature_monthly_folds(data)
        if not folds:
            raise ValueError("no monthly fold has enough mature PathEfficiencyFK history")

        membership_by_month = {item.month: item for item in data.memberships}
        observations_by_model: dict[str, list[TwoBranchForecastObservation]] = {
            name: [] for name in BOUNDARY_MODEL_NAMES
        }
        prediction_payloads: list[dict[str, object]] = []
        posterior_payloads: list[dict[str, object]] = []
        fit_rows: list[int] = []
        previous_weights = {name: {} for name in BOUNDARY_MODEL_NAMES}
        previous_branch_states = {
            name: (BranchPositionState(), BranchPositionState())
            for name in BOUNDARY_MODEL_NAMES
        }
        policy = BoundaryPolicy(
            BoundaryPolicyConfig(
                minimum_direction_probability=self.config.minimum_direction_probability,
                minimum_signal_to_noise=self.config.minimum_signal_to_noise,
                minimum_survival_probability=self.config.minimum_survival_probability,
            )
        )
        drift_names: tuple[str, ...] = ()
        potential_inputs = tuple(
            background_potential_features(data, branch) for branch in range(2)
        )

        for fold_index, test_start, training_rows, test_rows in folds:
            cutoff = test_start - np.timedelta64(1, "ns")
            posteriors = []
            for branch in range(2):
                _, potential_values, potential_valid = potential_inputs[branch]
                posterior = BayesianBranchModel(
                    BayesianBranchModelConfig(
                        minimum_rows=self.config.minimum_train_sessions,
                        neutral_sigma=self.config.neutral_sigma,
                        drift_half_life_sessions=self.config.drift_half_life_sessions,
                    )
                ).fit(
                    data.states,
                    data.labels,
                    potential_features=potential_values,
                    potential_valid=potential_valid,
                    branch=branch,
                    training_cutoff=cutoff,
                )
                posteriors.append(posterior)
                fit_rows.append(posterior.fit_rows)
                drift_names = posterior.drift_feature_names
                posterior_payloads.append(
                    {
                        "model_name": BOUNDARY_MODEL_NAMES[1],
                        "policy": PRIMARY_POLICY_NAME,
                        "fold": fold_index,
                        "branch": branch + 1,
                        "training_cutoff": str(cutoff),
                        "drift_feature_names": list(posterior.drift_feature_names),
                        "potential_feature_names": list(
                            posterior.potential_feature_names
                        ),
                        "drift_mean": posterior.drift_mean.tolist(),
                        "drift_half_life_sessions": self.config.drift_half_life_sessions,
                        "potential_target": "drift_disruption",
                        "diffusion_variance": posterior.diffusion_variance,
                        "potential_mean": posterior.potential_mean.tolist(),
                    }
                )

            for row in test_rows:
                membership = membership_by_month.get(data.states.topology_months[row])
                if (
                    membership is None
                    or not data.labels.valid[row].all()
                    or not all(item[2][row] for item in potential_inputs)
                ):
                    continue
                thresholds = tuple(
                    max(
                        self.config.minimum_neutral_threshold,
                        self.config.neutral_sigma
                        * data.labels.historical_volatility[row, branch],
                    )
                    for branch in range(2)
                )
                signal = data.states.sessions[row].astype("datetime64[D]") + np.timedelta64(15, "h")
                for model_index, model_name in enumerate(BOUNDARY_MODEL_NAMES):
                    use_potential = model_index == 1
                    forecasts = tuple(
                        BranchFeynmanKacForecaster(
                            BoundaryFeynmanKacConfig(
                                paths=self.config.fk_paths,
                                seed=self.config.seed + fold_index * 2 + branch,
                                use_effective_potential=use_potential,
                            )
                        ).predict(
                            posteriors[branch],
                            data.states,
                            int(row),
                            potential_features=potential_inputs[branch][1][row],
                            neutral_threshold=float(thresholds[branch]),
                        )
                        for branch in range(2)
                    )
                    decision = policy.decide(
                        (forecasts[0], forecasts[1]),
                        branches=membership.branches,
                        neutral_thresholds=(float(thresholds[0]), float(thresholds[1])),
                        previous_branch_states=previous_branch_states[model_name],
                    )
                    current_weights = dict(
                        zip(
                            decision.selected_symbols,
                            decision.target_weights,
                            strict=True,
                        )
                    )
                    turnover = self._turnover(
                        previous_weights[model_name], current_weights
                    )
                    gross_return = self._portfolio_return(
                        data, int(row), current_weights
                    )
                    observation = TwoBranchForecastObservation(
                        signal_time=signal,
                        available_at=signal + np.timedelta64(1, "ns"),
                        target_time=data.labels.target_time[row],
                        training_cutoff=cutoff,
                        model_version=model_name,
                        forecasts=(forecasts[0], forecasts[1]),
                        actual_returns=(
                            float(data.labels.next_return[row, 0]),
                            float(data.labels.next_return[row, 1]),
                        ),
                        neutral_thresholds=(
                            float(thresholds[0]),
                            float(thresholds[1]),
                        ),
                        decision=decision,
                        gross_strategy_return=gross_return,
                        turnover=turnover,
                    )
                    observations_by_model[model_name].append(observation)
                    previous_weights[model_name] = current_weights
                    previous_branch_states[model_name] = decision.branch_position_states
                    prediction_payloads.append(
                        self._prediction_payload(
                            observation,
                            fold_index,
                            int(row),
                            potential_inputs,
                            effective_potential_enabled=use_potential,
                        )
                    )

        if any(not items for items in observations_by_model.values()):
            raise ValueError("PathEfficiencyFK produced no sample-out predictions")
        evaluation_config = BoundaryEvaluationConfig(
            transaction_cost_bps=self.config.transaction_cost_bps
        )
        test_months = tuple(
            str(test_start.astype("datetime64[M]")) for _, test_start, _, _ in folds
        )
        stages = tuple(
            BoundaryStageResult(
                model_name=model_name,
                status="completed",
                passed=True,
                test_months=test_months,
                drift_feature_names=drift_names,
                minimum_fit_rows=min(fit_rows),
                metrics=evaluate_two_branch_forecasts(
                    tuple(observations_by_model[model_name]), evaluation_config
                ),
            )
            for model_name in BOUNDARY_MODEL_NAMES
        )
        stage_payload = [self._stage_payload(stage) for stage in stages]
        potential_observations = observations_by_model[BOUNDARY_MODEL_NAMES[1]]
        potentials = np.asarray(
            [
                forecast.expected_effective_potential
                for item in potential_observations
                for forecast in item.forecasts
            ],
            dtype=np.float64,
        )
        survivals = np.asarray(
            [
                forecast.trend_survival_probability
                for item in potential_observations
                for forecast in item.forecasts
            ],
            dtype=np.float64,
        )
        background_potential_payload = {
            "observations": int(len(potentials)),
            "mean_effective_potential": float(np.mean(potentials)),
            "median_effective_potential": float(np.median(potentials)),
            "mean_trend_survival_probability": float(np.mean(survivals)),
            "minimum_trend_survival_probability": float(np.min(survivals)),
        }
        write_json(root / "boundary-model-posterior.json", posterior_payloads)
        write_json(root / "feynman-kac-predictions.json", prediction_payloads)
        write_json(root / "boundary-policy-evaluation.json", stage_payload)
        write_json(
            root / "background-potential-evaluation.json",
            background_potential_payload,
        )
        write_json(
            root / "boundary-residual-analysis.json",
            [
                {
                    "model_name": stage.model_name,
                    "probability_residual_autocorrelation": (
                        stage.metrics.probability_residual_autocorrelation
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
                "valid": data.states.valid.tolist(),
            },
        )
        write_baseline_report(
            root / "path-efficiency-fk-report.md",
            stage_payload,
        )
        write_background_potential_report(
            root / "path-efficiency-background-potential-report.md",
            stage_payload,
            background_potential_payload,
        )
        return BoundaryFlowRunResult(BOUNDARY_MODEL_NAMES, stages)

    def _mature_monthly_folds(
        self, data: TwoBranchDataset
    ) -> tuple[tuple[int, np.datetime64, np.ndarray, np.ndarray], ...]:
        sessions = data.states.sessions.astype("datetime64[ns]")
        eligible = np.all(data.states.valid & data.labels.valid, axis=1)
        if data.evaluation_start is not None:
            eligible &= sessions >= np.datetime64(data.evaluation_start, "ns")
        if data.evaluation_end is not None:
            eligible &= sessions < (
                np.datetime64(data.evaluation_end, "D") + np.timedelta64(1, "D")
            ).astype("datetime64[ns]")
        months = np.unique(sessions[eligible].astype("datetime64[M]"))
        folds: list[tuple[int, np.datetime64, np.ndarray, np.ndarray]] = []
        for month in months:
            test_start = month.astype("datetime64[ns]")
            test_end = (month + np.timedelta64(1, "M")).astype("datetime64[ns]")
            cutoff = test_start - np.timedelta64(1, "ns")
            matured = np.all(data.states.valid & data.labels.valid, axis=1)
            training_rows = np.flatnonzero(
                matured & (sessions < test_start) & (data.labels.target_time <= cutoff)
            )
            test_rows = np.flatnonzero(
                eligible & (sessions >= test_start) & (sessions < test_end)
            )
            if len(training_rows) < self.config.minimum_train_sessions or not len(test_rows):
                continue
            folds.append((len(folds), test_start, training_rows, test_rows))
            if self.config.max_folds is not None and len(folds) >= self.config.max_folds:
                break
        return tuple(folds)

    @staticmethod
    def _turnover(previous: dict[str, float], current: dict[str, float]) -> float:
        return float(
            sum(
                abs(current.get(symbol, 0.0) - previous.get(symbol, 0.0))
                for symbol in set(previous) | set(current)
            )
        )

    @staticmethod
    def _portfolio_return(
        data: TwoBranchDataset, row: int, weights: dict[str, float]
    ) -> float:
        if not weights:
            return 0.0
        symbol_index = {symbol: index for index, symbol in enumerate(data.symbols)}
        result = 0.0
        for symbol, weight in weights.items():
            column = symbol_index[symbol]
            if not data.member_next_valid[row, column]:
                raise ValueError(f"missing next-session return for held symbol {symbol}")
            result += weight * float(data.member_next_returns[row, column])
        return float(result)

    @staticmethod
    def _prediction_payload(
        observation: TwoBranchForecastObservation,
        fold: int,
        row: int,
        potential_inputs: tuple[
            tuple[tuple[str, ...], np.ndarray, np.ndarray],
            tuple[tuple[str, ...], np.ndarray, np.ndarray],
        ],
        *,
        effective_potential_enabled: bool,
    ) -> dict[str, object]:
        branches = []
        for branch, (forecast, actual, threshold, direction) in enumerate(
            zip(
                observation.forecasts,
                observation.actual_returns,
                observation.neutral_thresholds,
                observation.decision.branch_directions,
                strict=True,
            )
        ):
            branches.append(
                {
                    "branch": branch + 1,
                    "direction": direction,
                    "state_probabilities": list(forecast.raw_state_probabilities),
                    "expected_drift": forecast.expected_drift,
                    "effective_drift": forecast.expected_attenuated_drift,
                    "expected_effective_potential": forecast.expected_effective_potential,
                    "survival_confidence": forecast.trend_survival_probability,
                    "bullish_drift_probability": forecast.bullish_drift_probability,
                    "drift_signal_to_noise": forecast.drift_signal_to_noise,
                    "expected_diffusion_variance": forecast.expected_diffusion_variance,
                    "actual_return": actual,
                    "neutral_threshold": threshold,
                    "effective_potential_enabled": effective_potential_enabled,
                }
            )
        return {
            "signal_time": str(observation.signal_time),
            "available_at": str(observation.available_at),
            "target_time": str(observation.target_time),
            "training_cutoff": str(observation.training_cutoff),
            "model_version": observation.model_version,
            "fold": fold,
            "branches": branches,
            "held_branches": list(observation.decision.held_branches),
            "branch_target_exposures": list(
                observation.decision.branch_target_exposures
            ),
            "branch_flat_signal_counts": [
                item.consecutive_flat_signals
                for item in observation.decision.branch_position_states
            ],
            "selected_symbols": list(observation.decision.selected_symbols),
            "target_weights": list(observation.decision.target_weights),
            "gross_strategy_return": observation.gross_strategy_return,
            "turnover": observation.turnover,
        }

    @staticmethod
    def _stage_payload(stage: BoundaryStageResult) -> dict[str, object]:
        return {
            "model_name": stage.model_name,
            "status": stage.status,
            "passed": stage.passed,
            "promotion_eligible": stage.promotion_eligible,
            "test_months": list(stage.test_months),
            "drift_feature_names": list(stage.drift_feature_names),
            "diffusion_feature_names": list(stage.diffusion_feature_names),
            "minimum_fit_rows": stage.minimum_fit_rows,
            "metrics": asdict(stage.metrics) if stage.metrics is not None else None,
            "failure_reason": stage.failure_reason,
        }
