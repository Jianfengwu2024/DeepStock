"""Monthly walk-forward orchestration for the two-branch FK strategy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
from .policy import BoundaryPolicy, BoundaryPolicyConfig, BranchPositionState
from .reporting import write_baseline_report, write_convection_report, write_json
from .types import MonthlyBranchMembership, TwoBranchDataset


BOUNDARY_MODEL_NAMES = (
    "PathEfficiencyFK-60D-Beta0PlusC1-Baseline",
    "PathEfficiencyFK-60D-StrongestThird",
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
    transaction_cost_bps: float = 5.0
    drift_half_life_sessions: float | None = 60.0
    minimum_direction_probability: float = 0.60
    minimum_signal_to_noise: float = 0.10
    minimum_survival_probability: float = 0.35
    relative_strength_lookback_sessions: int = 20
    drift_feature_names: tuple[str, ...] = DRIFT_FEATURE_NAMES
    direction_label_sigma: float = 0.1

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
            minimum_signal_to_noise=self.minimum_signal_to_noise,
            minimum_survival_probability=self.minimum_survival_probability,
        )
        if (
            self.drift_half_life_sessions is not None
            and self.drift_half_life_sessions <= 0.0
        ):
            raise ValueError("drift half-life must be positive when provided")
        if not 0.0 <= self.direction_label_sigma <= 5.0:
            raise ValueError("direction label sigma must lie in [0, 5]")


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


class BoundaryFlowPipeline:
    def __init__(self, config: BoundaryFlowPipelineConfig | None = None) -> None:
        self.config = config or BoundaryFlowPipelineConfig()

    def _strongest_third_branches(
        self,
        data: TwoBranchDataset,
        row: int,
        branches: tuple[tuple[str, ...], tuple[str, ...]],
    ) -> tuple[tuple[tuple[str, ...], tuple[str, ...]], dict[str, float]]:
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
            keep = max(1, ceil(len(symbols) / 3))
            ranked = sorted(symbols, key=lambda symbol: (-scores[symbol], symbol))
            selected_branches.append(tuple(ranked[:keep]))
        return (selected_branches[0], selected_branches[1]), scores

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
            convection_potential_features(data, branch) for branch in range(2)
        )
        for fold_index, test_start, training_rows, test_rows in folds:
            cutoff = test_start - np.timedelta64(1, "ns")
            posteriors: list[BayesianBranchPosterior] = []
            for branch in range(2):
                _, potential_values, potential_valid = potential_inputs[branch]
                model = BayesianBranchModel(
                    BayesianBranchModelConfig(
                        minimum_rows=self.config.minimum_train_sessions,
                        neutral_sigma=self.config.neutral_sigma,
                        drift_half_life_sessions=self.config.drift_half_life_sessions,
                        potential_mode="beta0_c1",
                        drift_feature_names=self.config.drift_feature_names,
                        direction_label_sigma=self.config.direction_label_sigma,
                    )
                )
                posterior = model.fit(
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
                        "model_name": BOUNDARY_MODEL_NAMES[0],
                        "policy": PRIMARY_POLICY_NAME,
                        "fold": fold_index,
                        "branch": branch + 1,
                        "training_cutoff": str(cutoff),
                        "drift_feature_names": list(posterior.drift_feature_names),
                        "potential_feature_names": list(posterior.potential_feature_names),
                        "drift_mean": posterior.drift_mean.tolist(),
                        "drift_model": posterior.drift_model,
                        "drift_return_scale": posterior.drift_return_scale,
                        "direction_label_sigma": posterior.direction_label_sigma,
                        "direction_fit_rows": posterior.direction_fit_rows,
                        "drift_half_life_sessions": self.config.drift_half_life_sessions,
                        "potential_target": "drift_disruption",
                        "diffusion_variance": posterior.diffusion_variance,
                        "potential_mean": posterior.potential_mean.tolist(),
                        "potential_mode": posterior.potential_mode,
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
                strongest_branches, strength_scores = self._strongest_third_branches(
                    data, int(row), membership.branches
                )
                for model_name in BOUNDARY_MODEL_NAMES:
                    forecasts = tuple(
                        BranchFeynmanKacForecaster(
                            BoundaryFeynmanKacConfig(
                                paths=self.config.fk_paths,
                                seed=self.config.seed + fold_index * 2 + branch,
                                use_effective_potential=True,
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
                        branches=(
                            strongest_branches
                            if model_name == BOUNDARY_MODEL_NAMES[1]
                            else membership.branches
                        ),
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
                    execution = self._execute_portfolio(
                        data,
                        int(row),
                        previous=previous_weights[model_name],
                        target=current_weights,
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
                        gross_strategy_return=execution.gross_return,
                        turnover=execution.turnover,
                        actual_symbols=tuple(execution.weights),
                        actual_weights=tuple(execution.weights.values()),
                        failed_buy_orders=execution.failed_buy_orders,
                        failed_sell_orders=execution.failed_sell_orders,
                    )
                    observations_by_model[model_name].append(observation)
                    previous_weights[model_name] = execution.weights
                    previous_branch_states[model_name] = decision.branch_position_states
                    prediction_payloads.append(
                        self._prediction_payload(
                            observation,
                            fold_index,
                            int(row),
                            potential_inputs,
                            relative_strength_scores=(
                                strength_scores
                                if model_name == BOUNDARY_MODEL_NAMES[1]
                                else None
                            ),
                            relative_strength_lookback_sessions=(
                                self.config.relative_strength_lookback_sessions
                                if model_name == BOUNDARY_MODEL_NAMES[1]
                                else None
                            ),
                            effective_potential_enabled=True,
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
        potential_diagnostics = {
            model_name: self._potential_diagnostics(observations_by_model[model_name])
            for model_name in BOUNDARY_MODEL_NAMES
        }
        primary_diagnostics = potential_diagnostics[BOUNDARY_MODEL_NAMES[-1]]
        convection_payload = {**primary_diagnostics, "models": potential_diagnostics}
        write_json(root / "boundary-model-posterior.json", posterior_payloads)
        write_json(root / "feynman-kac-predictions.json", prediction_payloads)
        write_json(root / "boundary-policy-evaluation.json", stage_payload)
        write_json(
            root / "convection-potential-evaluation.json",
            convection_payload,
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
        write_convection_report(
            root / "path-efficiency-convection-potential-report.md",
            stage_payload,
            convection_payload,
        )
        self._write_ablation_report(
            root / "boundary-ablation-report.md",
            stages,
            potential_diagnostics,
            self.config.relative_strength_lookback_sessions,
        )
        return BoundaryFlowRunResult(BOUNDARY_MODEL_NAMES, stages)

    @staticmethod
    def _potential_diagnostics(
        observations: list[TwoBranchForecastObservation],
    ) -> dict[str, float | int]:
        potentials = np.asarray(
            [
                forecast.expected_effective_potential
                for item in observations
                for forecast in item.forecasts
            ],
            dtype=np.float64,
        )
        survivals = np.asarray(
            [
                forecast.trend_survival_probability
                for item in observations
                for forecast in item.forecasts
            ],
            dtype=np.float64,
        )
        return {
            "observations": int(len(potentials)),
            "mean_effective_potential": float(np.mean(potentials)),
            "median_effective_potential": float(np.median(potentials)),
            "mean_trend_survival_probability": float(np.mean(survivals)),
            "minimum_trend_survival_probability": float(np.min(survivals)),
        }

    @staticmethod
    def _write_ablation_report(
        path: Path,
        stages: tuple[BoundaryStageResult, ...],
        diagnostics: dict[str, dict[str, float | int]],
        relative_strength_lookback_sessions: int,
    ) -> None:
        labels = {
            BOUNDARY_MODEL_NAMES[0]: "β0+C1 基线",
            BOUNDARY_MODEL_NAMES[1]: "β0+C1 + 支流内最强 1/3",
        }
        lines = [
            "# 路径效率 FK 相对强弱选股实验",
            "",
            "两组实验使用相同的 β0+C1 外势、样本外日期、交易成本和涨跌停成交约束。",
            "",
            "| 模型 | 净收益 | 最大回撤 | 平均仓位 | 平均换手 | 涨停未买入 | 跌停未卖出 | 平均势能 | 平均存活率 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for stage in stages:
            metrics = stage.metrics
            if metrics is None:
                continue
            model_diagnostics = diagnostics.get(stage.model_name, {})
            lines.append(
                "| {label} | {ret:.2%} | {drawdown:.2%} | {exposure:.2%} | {turnover:.2%} | "
                "{failed_buy} | {failed_sell} | {potential} | {survival} |".format(
                    label=labels[stage.model_name],
                    ret=metrics.net_total_return,
                    drawdown=metrics.maximum_drawdown,
                    exposure=metrics.average_exposure,
                    turnover=metrics.average_turnover,
                    failed_buy=metrics.failed_buy_orders,
                    failed_sell=metrics.failed_sell_orders,
                    potential=(
                        f"{float(model_diagnostics['mean_effective_potential']):.4f}"
                        if model_diagnostics else "-"
                    ),
                    survival=(
                        f"{float(model_diagnostics['mean_trend_survival_probability']):.2%}"
                        if model_diagnostics else "-"
                    ),
                )
            )
        lines.extend(
            [
                "",
                "## 判断方法",
                "",
                "- 两组使用完全相同的 β0+C1 FK 支流方向与仓位。",
                f"- 修正版仅交易每支流近 {relative_strength_lookback_sessions} 日相对强度最高的前 1/3 股票。",
                "- 收益改善但换手显著上升时，需要同时检查交易成本后的结果。",
            ]
        )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

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
    def _execute_portfolio(
        data: TwoBranchDataset,
        row: int,
        *,
        previous: dict[str, float],
        target: dict[str, float],
    ) -> PortfolioExecution:
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
        for symbol in set(previous) | set(weights):
            column = symbol_index[symbol]
            before = previous.get(symbol, 0.0)
            after = weights.get(symbol, 0.0)
            retained = min(before, after)
            if retained:
                if not data.member_next_valid[row, column]:
                    raise ValueError(f"missing next-session return for held symbol {symbol}")
                gross_return += retained * float(
                    data.member_next_hold_returns[row, column]
                )
            if after > before:
                gross_return += (after - before) * float(
                    data.member_next_buy_returns[row, column]
                )
            elif before > after:
                gross_return += (before - after) * float(
                    data.member_next_sell_returns[row, column]
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
        return PortfolioExecution(
            weights=cleaned,
            gross_return=float(gross_return),
            turnover=turnover,
            failed_buy_orders=failed_buy_orders,
            failed_sell_orders=failed_sell_orders,
        )

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
        relative_strength_scores: dict[str, float] | None,
        relative_strength_lookback_sessions: int | None,
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
            names, values, _ = potential_inputs[branch]
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
                    "convection_potential_features": dict(
                        zip(names, (float(item) for item in values[row]), strict=True)
                    ),
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
            "relative_strength_lookback_sessions": (
                relative_strength_lookback_sessions
            ),
            "selected_relative_strength_scores": {
                symbol: relative_strength_scores[symbol]
                for symbol in observation.decision.selected_symbols
                if relative_strength_scores is not None
            },
            "target_weights": list(observation.decision.target_weights),
            "actual_symbols": list(observation.actual_symbols),
            "actual_weights": list(observation.actual_weights),
            "gross_strategy_return": observation.gross_strategy_return,
            "turnover": observation.turnover,
            "failed_buy_orders": observation.failed_buy_orders,
            "failed_sell_orders": observation.failed_sell_orders,
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
