"""Monthly walk-forward orchestration for the whole-sector Bayesian FK strategy."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import ceil
from pathlib import Path

import numpy as np

from .branch_model import (
    BayesianBranchModel,
    BayesianBranchModelConfig,
    BayesianBranchPosterior,
    DRIFT_FEATURE_NAMES,
)
from .convection import convection_potential_features
from .evaluation import (
    BoundaryEvaluationConfig,
    BoundaryEvaluationMetrics,
    TwoBranchForecastObservation,
    evaluate_two_branch_forecasts,
)
from .feynman_kac import BoundaryFeynmanKacConfig, BranchFeynmanKacForecaster
from .hurdle_model import (
    BayesianHurdleModel,
    BayesianHurdleModelConfig,
    BayesianHurdlePosterior,
)
from .policy import (
    BoundaryPolicy,
    BoundaryPolicyConfig,
    BranchPositionState,
)
from .reporting import write_json
from .trailing_stop import BranchTrailingStop
from .types import TwoBranchDataset


SECTOR_FK_MODEL = "FK-Bayesian-Rolling60"
SECTOR_MODEL_FEATURES = {SECTOR_FK_MODEL: DRIFT_FEATURE_NAMES}
SECTOR_MODEL_NAMES = tuple(SECTOR_MODEL_FEATURES)
SECTOR_HURDLE_MODELS = frozenset({SECTOR_FK_MODEL})
SECTOR_FEATURE_EVIDENCE_MODELS = frozenset({SECTOR_FK_MODEL})
SECTOR_PRIMARY_MODEL = SECTOR_FK_MODEL
MINIMUM_TRADABLE_DIRECTION_SIGNALS = 30
MINIMUM_TRADABLE_SIGNALS_PER_SIDE = 15
BoundaryWalkForwardData = TwoBranchDataset


def _passes_sector_direction_gate(
    metrics: BoundaryEvaluationMetrics,
) -> bool:
    return (
        metrics.direction_signal_count >= MINIMUM_TRADABLE_DIRECTION_SIGNALS
        and metrics.direction_signal_bull_count >= MINIMUM_TRADABLE_SIGNALS_PER_SIDE
        and metrics.direction_signal_bear_count >= MINIMUM_TRADABLE_SIGNALS_PER_SIDE
        and metrics.direction_signal_accuracy
        > metrics.direction_signal_majority_baseline
        and metrics.direction_signal_balanced_accuracy > 0.5
    )


def _sector_direction_failure_reason(metrics: BoundaryEvaluationMetrics) -> str:
    if metrics.direction_signal_count < MINIMUM_TRADABLE_DIRECTION_SIGNALS:
        return "fewer than 30 tradable bull/bear signals"
    if (
        metrics.direction_signal_bull_count < MINIMUM_TRADABLE_SIGNALS_PER_SIDE
        or metrics.direction_signal_bear_count < MINIMUM_TRADABLE_SIGNALS_PER_SIDE
    ):
        return "fewer than 15 predicted signals on each bull/bear side"
    if metrics.direction_signal_accuracy <= metrics.direction_signal_majority_baseline:
        return "tradable direction accuracy did not beat its matched majority baseline"
    return "tradable bull/bear balanced accuracy did not beat 50 percent"


@dataclass(frozen=True, slots=True)
class BoundaryFlowPipelineConfig:
    minimum_train_sessions: int = 120
    max_folds: int | None = 1
    fk_paths: int = 10_000
    seed: int = 7
    neutral_sigma: float = 0.35
    minimum_neutral_threshold: float = 0.002
    transaction_cost_bps: float = 5.0
    drift_half_life_sessions: float | None = 60.0
    minimum_direction_probability: float = 0.51
    minimum_significant_move_probability: float = 0.50
    minimum_signal_to_noise: float = 0.10
    minimum_survival_probability: float = 0.35
    relative_strength_lookback_sessions: int = 20
    drift_feature_names: tuple[str, ...] = DRIFT_FEATURE_NAMES
    maximum_drawdown: float = 0.05

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
        if not 2 <= self.relative_strength_lookback_sessions <= 252:
            raise ValueError("relative strength lookback must lie in [2, 252]")
        BoundaryPolicyConfig(
            minimum_direction_probability=self.minimum_direction_probability,
            minimum_significant_move_probability=(
                self.minimum_significant_move_probability
            ),
            minimum_signal_to_noise=self.minimum_signal_to_noise,
            minimum_survival_probability=self.minimum_survival_probability,
        )
        if (
            self.drift_half_life_sessions is not None
            and self.drift_half_life_sessions <= 0.0
        ):
            raise ValueError("drift half-life must be positive when provided")
        if not 0.0 < self.maximum_drawdown < 1.0:
            raise ValueError("maximum drawdown must lie in (0, 1)")

    @property
    def direction_label_sigma(self) -> float:
        return self.neutral_sigma


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


@dataclass(frozen=True, slots=True)
class PortfolioExecution:
    weights: dict[str, float]
    gross_return: float
    turnover: float
    failed_buy_orders: int
    failed_sell_orders: int
    branch_returns: tuple[float, ...]
    symbol_branches: dict[str, int]


class BoundaryFlowPipeline:
    def __init__(self, config: BoundaryFlowPipelineConfig | None = None) -> None:
        self.config = config or BoundaryFlowPipelineConfig()

    def _strongest_half_branches(
        self,
        data: TwoBranchDataset,
        row: int,
        branches: tuple[tuple[str, ...], ...],
    ) -> tuple[tuple[tuple[str, ...], ...], dict[str, float]]:
        start = max(0, row - self.config.relative_strength_lookback_sessions)
        if row - start < 2:
            raise ValueError("insufficient completed returns for relative strength")
        symbol_index = {symbol: index for index, symbol in enumerate(data.symbols)}
        scores: dict[str, float] = {}
        selected_branches: list[tuple[str, ...]] = []
        for symbols in branches:
            columns = np.asarray([symbol_index[symbol] for symbol in symbols])
            returns = data.member_next_returns[start:row, columns]
            valid = data.member_next_valid[start:row, columns] & np.isfinite(returns)
            log_returns = np.where(valid, np.log1p(np.clip(returns, -0.999, None)), np.nan)
            cumulative = np.nansum(log_returns, axis=0)
            observations = np.sum(valid, axis=0)
            minimum_observations = max(2, int(np.ceil((row - start) * 0.8)))
            cumulative = np.where(observations >= minimum_observations, cumulative, -np.inf)
            finite = np.isfinite(cumulative)
            branch_center = float(np.mean(cumulative[finite])) if np.any(finite) else 0.0
            for symbol, value in zip(symbols, cumulative, strict=True):
                scores[symbol] = float(value - branch_center) if np.isfinite(value) else -np.inf
            keep = max(1, ceil(len(symbols) / 2))
            ranked = sorted(symbols, key=lambda symbol: (-scores[symbol], symbol))
            selected_branches.append(tuple(ranked[:keep]))
        return tuple(selected_branches), scores

    def run(
        self,
        data: TwoBranchDataset,
        output_directory: str | Path,
    ) -> BoundaryFlowRunResult:
        if data.memberships:
            raise ValueError("whole-sector FK data must not contain branch memberships")
        if data.states.values.shape[1] != 1:
            raise ValueError("whole-sector FK data must contain exactly one state stream")
        root = Path(output_directory)
        root.mkdir(parents=True, exist_ok=True)
        folds = self._mature_monthly_folds(data)
        if not folds:
            raise ValueError("no monthly fold has enough mature whole-sector history")
        observations_by_model: dict[str, list[TwoBranchForecastObservation]] = {
            name: [] for name in SECTOR_MODEL_NAMES
        }
        previous_weights: dict[str, dict[str, float]] = {
            name: {} for name in SECTOR_MODEL_NAMES
        }
        previous_states = {name: BranchPositionState() for name in SECTOR_MODEL_NAMES}
        trailing_stops = {
            name: BranchTrailingStop(1, self.config.maximum_drawdown)
            for name in SECTOR_MODEL_NAMES
        }
        prediction_payloads: list[dict[str, object]] = []
        posterior_payloads: list[dict[str, object]] = []
        fit_rows: dict[str, list[int]] = {name: [] for name in SECTOR_MODEL_NAMES}
        policies = {
            name: BoundaryPolicy(
                BoundaryPolicyConfig(
                    minimum_direction_probability=(
                        self.config.minimum_direction_probability
                    ),
                    minimum_significant_move_probability=(
                        self.config.minimum_significant_move_probability
                    ),
                    minimum_signal_to_noise=self.config.minimum_signal_to_noise,
                    minimum_survival_probability=(
                        self.config.minimum_survival_probability
                    ),
                    require_feature_evidence=(
                        name in SECTOR_FEATURE_EVIDENCE_MODELS
                    ),
                )
            )
            for name in SECTOR_MODEL_NAMES
        }
        potential_names, potential_values, potential_valid = (
            convection_potential_features(data, 0)
        )

        for fold_index, test_start, _, test_rows in folds:
            cutoff = test_start - np.timedelta64(1, "ns")
            posteriors: dict[str, BayesianBranchPosterior] = {}
            hurdle_posteriors: dict[str, BayesianHurdlePosterior] = {}
            for model_name, feature_names in SECTOR_MODEL_FEATURES.items():
                posterior = BayesianBranchModel(
                    BayesianBranchModelConfig(
                        minimum_rows=self.config.minimum_train_sessions,
                        neutral_sigma=self.config.neutral_sigma,
                        drift_half_life_sessions=self.config.drift_half_life_sessions,
                        drift_feature_names=feature_names,
                    )
                ).fit(
                    data.states,
                    data.labels,
                    potential_features=potential_values,
                    potential_valid=potential_valid,
                    branch=0,
                    training_cutoff=cutoff,
                )
                posteriors[model_name] = posterior
                hurdle_posterior = None
                if model_name in SECTOR_HURDLE_MODELS:
                    hurdle_posterior = BayesianHurdleModel(
                        BayesianHurdleModelConfig(
                            minimum_rows=self.config.minimum_train_sessions,
                            minimum_direction_rows=self.config.minimum_train_sessions,
                            neutral_sigma=self.config.neutral_sigma,
                            half_life_sessions=self.config.drift_half_life_sessions,
                            feature_names=feature_names,
                        )
                    ).fit(
                        data.states,
                        data.labels,
                        branch=0,
                        training_cutoff=cutoff,
                    )
                    hurdle_posteriors[model_name] = hurdle_posterior
                fit_rows[model_name].append(posterior.fit_rows)
                posterior_payload = {
                        "model_name": model_name,
                        "model_type": (
                            "bayesian_hurdle"
                            if hurdle_posterior is not None
                            else "bayesian_direction_logit"
                        ),
                        "scope": "whole_sector",
                        "fold": fold_index,
                        "training_cutoff": str(cutoff),
                        "drift_feature_names": list(posterior.drift_feature_names),
                        "potential_feature_names": list(posterior.potential_feature_names),
                        "drift_mean": posterior.drift_mean.tolist(),
                        "drift_covariance": posterior.drift_covariance.tolist(),
                        "drift_return_scale": posterior.drift_return_scale,
                        "direction_fit_rows": posterior.direction_fit_rows,
                        "direction_label_sigma": self.config.neutral_sigma,
                        "potential_mean": posterior.potential_mean.tolist(),
                        "potential_mode": posterior.potential_mode,
                    }
                if hurdle_posterior is not None:
                    posterior_payload.update(
                        {
                            "move_mean": hurdle_posterior.move_mean.tolist(),
                            "move_covariance": hurdle_posterior.move_covariance.tolist(),
                            "direction_mean": hurdle_posterior.direction_mean.tolist(),
                            "direction_covariance": (
                                hurdle_posterior.direction_covariance.tolist()
                            ),
                            "hurdle_fit_rows": hurdle_posterior.fit_rows,
                            "hurdle_direction_fit_rows": (
                                hurdle_posterior.direction_fit_rows
                            ),
                            "minimum_feature_evidence_probability": (
                                hurdle_posterior.minimum_feature_evidence_probability
                            ),
                            "calibration_rows": hurdle_posterior.calibration_rows,
                        }
                    )
                posterior_payloads.append(posterior_payload)

            for row in test_rows:
                if not data.labels.valid[row, 0] or not potential_valid[row]:
                    continue
                threshold = max(
                    self.config.minimum_neutral_threshold,
                    self.config.neutral_sigma
                    * float(data.labels.historical_volatility[row, 0]),
                )
                signal = (
                    data.states.sessions[row].astype("datetime64[D]")
                    + np.timedelta64(15, "h")
                )
                selected_groups, strength_scores = self._strongest_half_branches(
                    data, int(row), (data.symbols,)
                )
                selected_symbols = selected_groups[0]
                for model_index, model_name in enumerate(SECTOR_MODEL_NAMES):
                    forecast = BranchFeynmanKacForecaster(
                        BoundaryFeynmanKacConfig(
                            paths=self.config.fk_paths,
                            seed=self.config.seed + fold_index * 8 + model_index,
                            use_effective_potential=True,
                        )
                    ).predict(
                        posteriors[model_name],
                        data.states,
                        int(row),
                        potential_features=potential_values[row],
                        neutral_threshold=threshold,
                        hurdle_posterior=hurdle_posteriors.get(model_name),
                    )
                    decision = policies[model_name].decide(
                        (forecast,),
                        branches=(selected_symbols,),
                        neutral_thresholds=(threshold,),
                        previous_branch_states=(previous_states[model_name],),
                    )
                    forced_exit = trailing_stops[model_name].liquidation_pending
                    if forced_exit:
                        decision = replace(
                            decision,
                            open_position=False,
                            gate_reason="sector_trailing_stop",
                            selected_symbols=(),
                            target_weights=(),
                            held_branches=(False,),
                            branch_target_exposures=(0.0,),
                            branch_position_states=(BranchPositionState(),),
                        )
                    target = dict(
                        zip(decision.selected_symbols, decision.target_weights, strict=True)
                    )
                    execution = self._execute_portfolio(
                        data,
                        int(row),
                        previous=previous_weights[model_name],
                        target=target,
                        previous_symbol_branches={
                            symbol: 0 for symbol in previous_weights[model_name]
                        },
                        target_symbol_branches={symbol: 0 for symbol in target},
                    )
                    held = bool(execution.weights)
                    stop = trailing_stops[model_name].update(
                        execution.branch_returns, (held,)
                    )
                    observation = TwoBranchForecastObservation(
                        signal_time=signal,
                        available_at=signal + np.timedelta64(1, "ns"),
                        target_time=data.labels.target_time[row],
                        training_cutoff=cutoff,
                        model_version=model_name,
                        forecasts=(forecast,),
                        actual_returns=(float(data.labels.next_return[row, 0]),),
                        neutral_thresholds=(threshold,),
                        decision=decision,
                        gross_strategy_return=execution.gross_return,
                        turnover=execution.turnover,
                        branch_ids=(0,),
                        actual_symbols=tuple(execution.weights),
                        actual_weights=tuple(execution.weights.values()),
                        failed_buy_orders=execution.failed_buy_orders,
                        failed_sell_orders=execution.failed_sell_orders,
                        branch_equities=(stop.branch_equities[0],),
                        branch_peak_equities=(stop.branch_peaks[0],),
                        branch_trailing_drawdowns=(stop.branch_drawdowns[0],),
                        trailing_stop_pending=stop.liquidation_pending,
                        trailing_stop_triggered_branches=(stop.newly_triggered[0],),
                        trailing_stop_forced_exit=forced_exit,
                    )
                    observations_by_model[model_name].append(observation)
                    previous_weights[model_name] = execution.weights
                    previous_states[model_name] = (
                        BranchPositionState()
                        if forced_exit
                        else decision.branch_position_states[0]
                    )
                    prediction_payloads.append(
                        self._sector_prediction_payload(
                            observation,
                            fold=fold_index,
                            potential_names=potential_names,
                            potential_values=potential_values[row],
                            strength_scores=strength_scores,
                        )
                    )

        if any(not observations for observations in observations_by_model.values()):
            raise ValueError("whole-sector FK produced no sample-out predictions")
        evaluation_config = BoundaryEvaluationConfig(
            transaction_cost_bps=self.config.transaction_cost_bps
        )
        test_months = tuple(
            str(test_start.astype("datetime64[M]")) for _, test_start, _, _ in folds
        )
        metrics_by_model = {
            model_name: evaluate_two_branch_forecasts(
                tuple(observations_by_model[model_name]), evaluation_config
            )
            for model_name in SECTOR_MODEL_NAMES
        }
        stages = tuple(
            BoundaryStageResult(
                model_name=model_name,
                status="completed",
                passed=_passes_sector_direction_gate(metrics_by_model[model_name]),
                promotion_eligible=_passes_sector_direction_gate(
                    metrics_by_model[model_name]
                ),
                test_months=test_months,
                drift_feature_names=SECTOR_MODEL_FEATURES[model_name],
                minimum_fit_rows=min(fit_rows[model_name]),
                metrics=metrics_by_model[model_name],
                failure_reason=(
                    ""
                    if _passes_sector_direction_gate(metrics_by_model[model_name])
                    else _sector_direction_failure_reason(metrics_by_model[model_name])
                ),
            )
            for model_name in SECTOR_MODEL_NAMES
        )
        stage_payload = [self._stage_payload(stage) for stage in stages]
        direction_report = self._sector_direction_report(
            data, observations_by_model, stage_payload
        )
        write_json(root / "boundary-model-posterior.json", posterior_payloads)
        write_json(root / "feynman-kac-predictions.json", prediction_payloads)
        write_json(root / "boundary-policy-evaluation.json", stage_payload)
        write_json(root / "sector-direction-evaluation.json", direction_report)
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
                if stage.metrics is not None
            ],
        )
        write_json(
            root / "boundary-daily-states.json",
            {
                "scope": "whole_sector",
                "sessions": [str(item) for item in data.states.sessions],
                "feature_names": list(data.states.feature_names),
                "valid": data.states.valid[:, 0].tolist(),
            },
        )
        self._write_sector_report(
            root / "path-efficiency-fk-report.md", direction_report
        )
        return BoundaryFlowRunResult(SECTOR_MODEL_NAMES, stages)

    @staticmethod
    def _sector_prediction_payload(
        observation: TwoBranchForecastObservation,
        *,
        fold: int,
        potential_names: tuple[str, ...],
        potential_values: np.ndarray,
        strength_scores: dict[str, float],
    ) -> dict[str, object]:
        forecast = observation.forecasts[0]
        direction = observation.decision.branch_directions[0]
        return {
            "signal_time": str(observation.signal_time),
            "available_at": str(observation.available_at),
            "target_time": str(observation.target_time),
            "training_cutoff": str(observation.training_cutoff),
            "model_version": observation.model_version,
            "fold": fold,
            "scope": "whole_sector",
            "sector": {
                "direction": direction,
                "state_probabilities": list(forecast.raw_state_probabilities),
                "expected_drift": forecast.expected_drift,
                "effective_drift": forecast.expected_attenuated_drift,
                "expected_effective_potential": forecast.expected_effective_potential,
                "survival_confidence": forecast.trend_survival_probability,
                "bullish_drift_probability": forecast.bullish_drift_probability,
                "drift_signal_to_noise": forecast.drift_signal_to_noise,
                "significant_move_probability": (
                    forecast.significant_move_probability
                ),
                "conditional_bullish_probability": (
                    forecast.conditional_bullish_probability
                ),
                "prior_bullish_probability": forecast.prior_bullish_probability,
                "feature_log_odds_delta": forecast.feature_log_odds_delta,
                "feature_bullish_probability": (
                    forecast.feature_bullish_probability
                ),
                "minimum_feature_evidence_probability": (
                    forecast.minimum_feature_evidence_probability
                ),
                "expected_diffusion_variance": forecast.expected_diffusion_variance,
                "actual_return": observation.actual_returns[0],
                "neutral_threshold": observation.neutral_thresholds[0],
                "convection_potential_features": dict(
                    zip(
                        potential_names,
                        (float(item) for item in potential_values),
                        strict=True,
                    )
                ),
            },
            "target_exposure": sum(observation.decision.target_weights),
            "flat_signal_count": (
                observation.decision.branch_position_states[0].consecutive_flat_signals
            ),
            "selected_symbols": list(observation.decision.selected_symbols),
            "selected_relative_strength_scores": {
                symbol: strength_scores[symbol]
                for symbol in observation.decision.selected_symbols
            },
            "target_weights": list(observation.decision.target_weights),
            "actual_symbols": list(observation.actual_symbols),
            "actual_weights": list(observation.actual_weights),
            "gross_strategy_return": observation.gross_strategy_return,
            "turnover": observation.turnover,
            "failed_buy_orders": observation.failed_buy_orders,
            "failed_sell_orders": observation.failed_sell_orders,
            "sector_equity": observation.branch_equities[0],
            "sector_peak_equity": observation.branch_peak_equities[0],
            "sector_trailing_drawdown": observation.branch_trailing_drawdowns[0],
            "trailing_stop_pending": observation.trailing_stop_pending,
            "trailing_stop_forced_exit": observation.trailing_stop_forced_exit,
        }

    @staticmethod
    def _sector_direction_report(
        data: TwoBranchDataset,
        observations_by_model: dict[str, list[TwoBranchForecastObservation]],
        models: list[dict[str, object]],
    ) -> dict[str, object]:
        reference = observations_by_model[SECTOR_PRIMARY_MODEL]
        row_by_session = {
            session.astype("datetime64[D]"): row
            for row, session in enumerate(data.states.sessions)
        }
        actual = np.asarray([item.actual_returns[0] for item in reference])
        neutral_thresholds = np.asarray(
            [item.neutral_thresholds[0] for item in reference]
        )
        directional = np.abs(actual) > neutral_thresholds
        rows = np.asarray(
            [row_by_session[item.signal_time.astype("datetime64[D]")] for item in reference]
        )

        def accuracy(predicted: np.ndarray) -> float:
            if not np.any(directional):
                return 0.0
            return float(
                np.mean(
                    np.sign(predicted[directional]) == np.sign(actual[directional])
                )
            )

        previous_day = np.full(len(rows), np.nan, dtype=np.float64)
        has_previous = rows > 0
        previous_day[has_previous] = data.labels.next_return[
            rows[has_previous] - 1, 0
        ]
        baselines = {
            "previous_day_direction": accuracy(previous_day),
            "path_efficiency_direction": accuracy(
                data.states.feature("rolling_path_efficiency_60d")[rows, 0]
            ),
        }
        enriched_models = []
        for item in models:
            observations = observations_by_model[str(item["model_name"])]
            months = np.asarray(
                [entry.target_time.astype("datetime64[M]") for entry in observations]
            )
            monthly = []
            for month in np.unique(months):
                selected = np.flatnonzero(months == month)
                month_metrics = evaluate_two_branch_forecasts(
                    tuple(observations[index] for index in selected)
                )
                monthly.append(
                    {
                        "month": str(month),
                        "observations": len(selected),
                        "directional_observations": (
                            month_metrics.directional_observations
                        ),
                        "signal_count": month_metrics.direction_signal_count,
                        "coverage": month_metrics.direction_signal_coverage,
                        "accuracy": month_metrics.direction_signal_accuracy,
                        "majority_baseline": (
                            month_metrics.direction_signal_majority_baseline
                        ),
                        "balanced_accuracy": (
                            month_metrics.direction_signal_balanced_accuracy
                        ),
                    }
                )
            enriched_models.append({**item, "monthly_direction_accuracy": monthly})
        return {
            "scope": "whole_sector",
            "observations": len(reference),
            "baselines": baselines,
            "preferred_model": SECTOR_PRIMARY_MODEL,
            "selection_rule": "fixed_predeclared_baseline",
            "models": enriched_models,
        }

    @staticmethod
    def _write_sector_report(path: Path, report: dict[str, object]) -> None:
        baselines = report["baselines"]
        failure_labels = {
            "fewer than 30 tradable bull/bear signals": "多空信号少于 30 个",
            "fewer than 15 predicted signals on each bull/bear side": (
                "预测多或预测空少于 15 个"
            ),
            "tradable direction accuracy did not beat its matched majority baseline": (
                "命中率未超过同样本多数方向"
            ),
            "tradable bull/bear balanced accuracy did not beat 50 percent": (
                "多空平衡正确率未超过 50%"
            ),
        }
        lines = [
            "# 板块方向 FK 样本外报告",
            "",
            "本实验只预测下一交易日板块方向，不使用动态支流。",
            "真实平盘日不评分，模型看平视为弃权，只通过覆盖率体现。",
            "预测多和预测空必须各不少于 15 个，防止单边押注制造虚假的高准确率。",
            "当前只运行 `FK-Bayesian-Rolling60`，不再保留候选、正式或旧模型。",
            "",
            "## 全覆盖方向参考",
            "",
            f"- 昨日方向：{float(baselines['previous_day_direction']):.2%}",
            f"- 当日路径效率方向：{float(baselines['path_efficiency_direction']):.2%}",
            "",
            "## 贝叶斯漂移模型",
            "",
            "| 模型 | 多空信号数 | 预测多 | 预测空 | 多空命中率 | 同样本多数基线 | 多空平衡正确率 | 方向覆盖率 | 失败原因 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for item in report["models"]:
            metrics = item["metrics"]
            raw_failure_reason = str(item.get("failure_reason") or "")
            failure_reason = failure_labels.get(
                raw_failure_reason, raw_failure_reason or "—"
            )
            lines.append(
                f"| {item['model_name']} | {metrics['direction_signal_count']} | "
                f"{metrics['direction_signal_bull_count']} | "
                f"{metrics['direction_signal_bear_count']} | "
                f"{metrics['direction_signal_accuracy']:.2%} | "
                f"{metrics['direction_signal_majority_baseline']:.2%} | "
                f"{metrics['direction_signal_balanced_accuracy']:.2%} | "
                f"{metrics['direction_signal_coverage']:.2%} | {failure_reason} |"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _mature_monthly_folds(
        self, data: TwoBranchDataset
    ) -> tuple[tuple[int, np.datetime64, np.ndarray, np.ndarray], ...]:
        sessions = data.states.sessions.astype("datetime64[ns]")
        # A day is usable only after the whole-sector state and label mature.
        eligible = np.any(data.states.valid & data.labels.valid, axis=1)
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
            matured = np.any(data.states.valid & data.labels.valid, axis=1)
            training_rows = np.flatnonzero(
                matured & (sessions < test_start) & (data.labels.target_time <= cutoff)
            )
            test_rows = np.flatnonzero(
                eligible & (sessions >= test_start) & (sessions < test_end)
            )
            direction_ready = self.config.direction_label_sigma <= 0.0 or all(
                int(
                    np.sum(
                        data.labels.valid[training_rows, branch]
                        & (np.abs(data.labels.next_return[training_rows, branch])
                        > self.config.direction_label_sigma
                        * data.labels.historical_volatility[training_rows, branch])
                    )
                )
                >= self.config.minimum_train_sessions
                for branch in range(data.states.values.shape[1])
                if int(data.labels.valid[training_rows, branch].sum())
                >= self.config.minimum_train_sessions
            )
            if (
                len(training_rows) < self.config.minimum_train_sessions
                or not direction_ready
                or not len(test_rows)
            ):
                continue
            folds.append((len(folds), test_start, training_rows, test_rows))
            if self.config.max_folds is not None and len(folds) >= self.config.max_folds:
                break
        return tuple(folds)

    @staticmethod
    def _execute_portfolio(
        data: TwoBranchDataset,
        row: int,
        *,
        previous: dict[str, float],
        target: dict[str, float],
        previous_symbol_branches: dict[str, int] | None = None,
        target_symbol_branches: dict[str, int] | None = None,
    ) -> PortfolioExecution:
        previous_owners = previous_symbol_branches or {}
        target_owners = target_symbol_branches or {}
        for owner in (*previous_owners.values(), *target_owners.values()):
            if owner < 0 or owner >= data.states.values.shape[1]:
                raise ValueError("symbol branch ownership is outside the state tensor")
        symbol_index = {symbol: index for index, symbol in enumerate(data.symbols)}
        weights = dict(previous)
        failed_sell_orders = 0
        failed_buy_orders = 0

        for symbol, previous_weight in previous.items():
            column = symbol_index[symbol]
            desired = target.get(symbol, 0.0)
            if desired >= previous_weight:
                continue
            if data.member_next_sell_executable[row, column]:
                if desired > 1e-12:
                    weights[symbol] = desired
                else:
                    weights.pop(symbol, None)
            else:
                failed_sell_orders += 1

        requested_buys: dict[str, float] = {}
        for symbol, desired in target.items():
            current = weights.get(symbol, 0.0)
            if desired <= current:
                continue
            column = symbol_index[symbol]
            if data.member_next_buy_executable[row, column]:
                requested_buys[symbol] = desired - current
            else:
                failed_buy_orders += 1

        available_cash = max(0.0, 1.0 - sum(weights.values()))
        requested_cash = sum(requested_buys.values())
        fill_ratio = min(1.0, available_cash / requested_cash) if requested_cash else 0.0
        for symbol, increase in requested_buys.items():
            weights[symbol] = weights.get(symbol, 0.0) + increase * fill_ratio

        gross_return = 0.0
        branch_contributions = [0.0] * data.states.values.shape[1]

        def add_contribution(owner: int | None, value: float) -> None:
            nonlocal gross_return
            gross_return += value
            if owner is not None:
                branch_contributions[owner] += value

        for symbol in set(previous) | set(weights):
            column = symbol_index[symbol]
            before = previous.get(symbol, 0.0)
            after = weights.get(symbol, 0.0)
            retained = min(before, after)
            if retained:
                add_contribution(
                    target_owners.get(symbol, previous_owners.get(symbol)),
                    retained * float(data.member_next_hold_returns[row, column]),
                )
            if after > before:
                add_contribution(
                    target_owners.get(symbol),
                    (after - before) * float(data.member_next_buy_returns[row, column]),
                )
            elif before > after:
                add_contribution(
                    previous_owners.get(symbol),
                    (before - after) * float(data.member_next_sell_returns[row, column]),
                )

        cleaned = {
            symbol: float(weight)
            for symbol, weight in weights.items()
            if weight > 1e-12
        }
        turnover = float(
            sum(
                abs(cleaned.get(symbol, 0.0) - previous.get(symbol, 0.0))
                for symbol in set(previous) | set(cleaned)
            )
        )
        symbol_branches = {
            symbol: (
                target_owners[symbol]
                if symbol in target_owners
                else previous_owners[symbol]
            )
            for symbol in cleaned
            if symbol in target_owners or symbol in previous_owners
        }
        branch_exposures = [0.0] * data.states.values.shape[1]
        for symbol, weight in cleaned.items():
            owner = symbol_branches.get(symbol)
            if owner is not None:
                branch_exposures[owner] += weight
        return PortfolioExecution(
            weights=cleaned,
            gross_return=float(gross_return),
            turnover=turnover,
            failed_buy_orders=failed_buy_orders,
            failed_sell_orders=failed_sell_orders,
            branch_returns=tuple(
                contribution / exposure if exposure > 1e-12 else 0.0
                for contribution, exposure in zip(
                    branch_contributions, branch_exposures, strict=True
                )
            ),
            symbol_branches=symbol_branches,
        )

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
