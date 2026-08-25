"""Monthly walk-forward orchestration for the two-branch FK strategy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .branch_model import BayesianBranchModel, BayesianBranchModelConfig
from .evaluation import (
    BoundaryEvaluationConfig,
    BoundaryEvaluationMetrics,
    TwoBranchForecastObservation,
    evaluate_two_branch_forecasts,
)
from .feynman_kac import BoundaryFeynmanKacConfig, BranchFeynmanKacForecaster
from .policy import BoundaryPolicy, BoundaryPolicyConfig
from .reporting import write_ablation_report, write_json
from .types import MonthlyBranchMembership, TwoBranchDataset


BOUNDARY_MODEL_NAMES = ("TwoBranchFK",)
POLICY_ABLATION_NAMES = ("posterior_signal", "legacy_attenuated")
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
            raise ValueError("TwoBranchFK requires at least 30 training sessions")
        if self.max_folds is not None and self.max_folds < 1:
            raise ValueError("maximum folds must be positive")
        if self.neutral_sigma <= 0.0 or self.minimum_neutral_threshold <= 0.0:
            raise ValueError("neutral threshold settings must be positive")
        if self.fk_paths < 1_000:
            raise ValueError("TwoBranchFK requires at least 1000 FK paths")
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
            raise ValueError("no monthly fold has enough mature TwoBranchFK history")

        membership_by_month = {item.month: item for item in data.memberships}
        observations_by_policy: dict[str, list[TwoBranchForecastObservation]] = {
            name: [] for name in POLICY_ABLATION_NAMES
        }
        prediction_payloads: list[dict[str, object]] = []
        posterior_payloads: list[dict[str, object]] = []
        fit_rows: list[int] = []
        held_by_policy = {name: (False, False) for name in POLICY_ABLATION_NAMES}
        previous_weights_by_policy = {name: {} for name in POLICY_ABLATION_NAMES}
        policies = {
            "posterior_signal": BoundaryPolicy(
                BoundaryPolicyConfig(
                    minimum_direction_probability=self.config.minimum_direction_probability,
                    minimum_signal_to_noise=self.config.minimum_signal_to_noise,
                    minimum_survival_probability=self.config.minimum_survival_probability,
                )
            ),
            "legacy_attenuated": BoundaryPolicy(
                BoundaryPolicyConfig(mode="legacy_attenuated")
            ),
        }
        drift_names: tuple[str, ...] = ()

        for fold_index, test_start, training_rows, test_rows in folds:
            cutoff = test_start - np.timedelta64(1, "ns")
            posteriors = []
            for branch in range(2):
                posterior = BayesianBranchModel(
                    BayesianBranchModelConfig(
                        minimum_rows=self.config.minimum_train_sessions,
                        neutral_sigma=self.config.neutral_sigma,
                        drift_half_life_sessions=self.config.drift_half_life_sessions,
                    )
                ).fit(
                    data.states,
                    data.labels,
                    branch=branch,
                    training_cutoff=cutoff,
                )
                posteriors.append(posterior)
                fit_rows.append(posterior.fit_rows)
                drift_names = posterior.drift_feature_names
                posterior_payloads.append(
                    {
                        "model_name": BOUNDARY_MODEL_NAMES[0],
                        "fold": fold_index,
                        "branch": branch + 1,
                        "training_cutoff": str(cutoff),
                        "drift_feature_names": list(posterior.drift_feature_names),
                        "potential_feature_names": list(posterior.potential_feature_names),
                        "drift_mean": posterior.drift_mean.tolist(),
                        "drift_half_life_sessions": self.config.drift_half_life_sessions,
                        "diffusion_variance": posterior.diffusion_variance,
                        "potential_mean": posterior.potential_mean.tolist(),
                    }
                )

            for row in test_rows:
                membership = membership_by_month.get(data.states.topology_months[row])
                if membership is None or not data.labels.valid[row].all():
                    continue
                thresholds = tuple(
                    max(
                        self.config.minimum_neutral_threshold,
                        self.config.neutral_sigma
                        * data.labels.historical_volatility[row, branch],
                    )
                    for branch in range(2)
                )
                forecasts = tuple(
                    BranchFeynmanKacForecaster(
                        BoundaryFeynmanKacConfig(
                            paths=self.config.fk_paths,
                            seed=self.config.seed + fold_index * 2 + branch,
                        )
                    ).predict(
                        posteriors[branch],
                        data.states,
                        int(row),
                        neutral_threshold=float(thresholds[branch]),
                    )
                    for branch in range(2)
                )
                signal = data.states.sessions[row].astype("datetime64[D]") + np.timedelta64(15, "h")
                for policy_name, policy in policies.items():
                    decision = policy.decide(
                        (forecasts[0], forecasts[1]),
                        branches=membership.branches,
                        held_branches=held_by_policy[policy_name],
                        neutral_thresholds=(
                            float(thresholds[0]),
                            float(thresholds[1]),
                        ),
                    )
                    current_weights = dict(
                        zip(
                            decision.selected_symbols,
                            decision.target_weights,
                            strict=True,
                        )
                    )
                    turnover = self._turnover(
                        previous_weights_by_policy[policy_name], current_weights
                    )
                    gross_return = self._portfolio_return(
                        data, int(row), current_weights
                    )
                    observation = TwoBranchForecastObservation(
                        signal_time=signal,
                        available_at=signal + np.timedelta64(1, "ns"),
                        target_time=data.labels.target_time[row],
                        training_cutoff=cutoff,
                        model_version=BOUNDARY_MODEL_NAMES[0],
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
                    observations_by_policy[policy_name].append(observation)
                    held_by_policy[policy_name] = decision.held_branches
                    previous_weights_by_policy[policy_name] = current_weights
                    if policy_name == "posterior_signal":
                        prediction_payloads.append(
                            self._prediction_payload(observation, fold_index)
                        )

        observations = observations_by_policy["posterior_signal"]
        if not observations:
            raise ValueError("TwoBranchFK produced no sample-out predictions")
        evaluation_config = BoundaryEvaluationConfig(
            transaction_cost_bps=self.config.transaction_cost_bps
        )
        policy_metrics = {
            name: evaluate_two_branch_forecasts(tuple(items), evaluation_config)
            for name, items in observations_by_policy.items()
        }
        metrics = policy_metrics["posterior_signal"]
        test_months = tuple(
            str(test_start.astype("datetime64[M]")) for _, test_start, _, _ in folds
        )
        stage = BoundaryStageResult(
            model_name=BOUNDARY_MODEL_NAMES[0],
            status="completed",
            passed=True,
            test_months=test_months,
            drift_feature_names=drift_names,
            minimum_fit_rows=min(fit_rows),
            metrics=metrics,
        )
        stage_payload = [self._stage_payload(stage)]
        write_json(root / "boundary-model-posterior.json", posterior_payloads)
        write_json(root / "feynman-kac-predictions.json", prediction_payloads)
        write_json(root / "boundary-policy-evaluation.json", stage_payload)
        write_json(
            root / "boundary-policy-ablation.json",
            {name: asdict(value) for name, value in policy_metrics.items()},
        )
        write_json(
            root / "boundary-residual-analysis.json",
            {
                "model_name": stage.model_name,
                "probability_residual_autocorrelation": metrics.probability_residual_autocorrelation,
            },
        )
        write_json(
            root / "boundary-daily-states.json",
            {
                "sessions": [str(item) for item in data.states.sessions],
                "feature_names": list(data.states.feature_names),
                "valid": data.states.valid.tolist(),
            },
        )
        write_ablation_report(root / "boundary-ablation-report.md", stage_payload)
        return BoundaryFlowRunResult(BOUNDARY_MODEL_NAMES, (stage,))

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
        observation: TwoBranchForecastObservation, fold: int
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
                    "effective_drift": forecast.expected_return,
                    "legacy_attenuated_drift": forecast.expected_attenuated_drift,
                    "expected_effective_potential": forecast.expected_effective_potential,
                    "survival_confidence": forecast.trend_survival_probability,
                    "bullish_drift_probability": forecast.bullish_drift_probability,
                    "drift_signal_to_noise": forecast.drift_signal_to_noise,
                    "expected_diffusion_variance": forecast.expected_diffusion_variance,
                    "actual_return": actual,
                    "neutral_threshold": threshold,
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
