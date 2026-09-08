"""Monthly walk-forward orchestration for the two-branch FK strategy."""

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
from .policy import (
    BoundaryPolicy,
    BoundaryPolicyConfig,
    BranchPositionState,
)
from .reporting import write_baseline_report, write_convection_report, write_json
from .trailing_stop import BranchTrailingStop
from .types import MonthlyBranchMembership, TwoBranchDataset


BOUNDARY_MODEL_NAMES = (
    "PathEfficiencyFK-60D-Beta0PlusC1-Baseline",
                "PathEfficiencyFK-60D-StrongestHalf",
)
SECTOR_MODEL_FEATURES = {
    "SectorFK-PathOnly": ("signed_path_efficiency",),
    "SectorFK-PathSync": (
        "signed_path_efficiency",
        "signed_member_synchronization",
    ),
    "SectorFK-PathMemory": (
        "signed_path_efficiency",
        "path_efficiency_delta_1d",
        "path_direction_consistency_3d",
    ),
    "SectorFK-PathFull": (
        "signed_path_efficiency",
        "path_efficiency_delta_1d",
        "path_direction_consistency_3d",
        "signed_member_synchronization",
    ),
}
SECTOR_MODEL_NAMES = tuple(SECTOR_MODEL_FEATURES)
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
    minimum_direction_probability: float = 0.51
    minimum_signal_to_noise: float = 0.10
    minimum_survival_probability: float = 0.35
    relative_strength_lookback_sessions: int = 20
    drift_feature_names: tuple[str, ...] = DRIFT_FEATURE_NAMES
    direction_label_sigma: float = 0.1
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
        if not 0.0 < self.maximum_drawdown < 1.0:
            raise ValueError("maximum drawdown must lie in (0, 1)")


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

    # Compatibility shim for callers that still use the pre-multibranch name.
    def _strongest_third_branches(self, data, row, branches):
        return self._strongest_half_branches(data, row, branches)

    def run(
        self,
        data: TwoBranchDataset,
        output_directory: str | Path,
    ) -> BoundaryFlowRunResult:
        root = Path(output_directory)
        root.mkdir(parents=True, exist_ok=True)
        if not data.memberships:
            return self._run_sector_direct(data, root)
        folds = self._mature_monthly_folds(data)
        if not folds:
            raise ValueError("no monthly fold has enough mature PathEfficiencyFK history")

        membership_by_session = {item.effective_session: item for item in data.memberships}
        observations_by_model: dict[str, list[TwoBranchForecastObservation]] = {
            name: [] for name in BOUNDARY_MODEL_NAMES
        }
        prediction_payloads: list[dict[str, object]] = []
        posterior_payloads: list[dict[str, object]] = []
        fit_rows: list[int] = []
        previous_weights = {name: {} for name in BOUNDARY_MODEL_NAMES}
        previous_branch_states: dict[str, dict[int, BranchPositionState]] = {
            name: {} for name in BOUNDARY_MODEL_NAMES
        }
        previous_symbol_branches = {name: {} for name in BOUNDARY_MODEL_NAMES}
        trailing_stops: dict[str, BranchTrailingStop] = {}
        policy = BoundaryPolicy(
            BoundaryPolicyConfig(
                minimum_direction_probability=self.config.minimum_direction_probability,
                minimum_signal_to_noise=self.config.minimum_signal_to_noise,
                minimum_survival_probability=self.config.minimum_survival_probability,
            )
        )
        drift_names: tuple[str, ...] = ()
        potential_inputs = tuple(
            convection_potential_features(data, branch)
            for branch in range(data.states.values.shape[1])
        )
        for fold_index, test_start, training_rows, test_rows in folds:
            cutoff = test_start - np.timedelta64(1, "ns")
            posteriors: dict[int, BayesianBranchPosterior] = {}
            for branch in range(data.states.values.shape[1]):
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
                try:
                    posterior = model.fit(
                        data.states,
                        data.labels,
                        potential_features=potential_values,
                        potential_valid=potential_valid,
                        branch=branch,
                        training_cutoff=cutoff,
                    )
                except ValueError as error:
                    if not str(error).startswith("insufficient "):
                        raise
                    continue
                posteriors[branch] = posterior
                fit_rows.append(posterior.fit_rows)
                drift_names = posterior.drift_feature_names
                posterior_payloads.append(
                    {
                        "model_name": BOUNDARY_MODEL_NAMES[0],
                        "policy": PRIMARY_POLICY_NAME,
                        "fold": fold_index,
                        "branch": branch + 1,
                        "branch_id": branch,
                        "training_cutoff": str(cutoff),
                        "drift_feature_names": list(posterior.drift_feature_names),
                        "potential_feature_names": list(posterior.potential_feature_names),
                        "drift_mean": posterior.drift_mean.tolist(),
                        "drift_model": posterior.drift_model,
                        "drift_baseline_mode": posterior.drift_baseline_mode,
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
                membership = self._membership_at(membership_by_session, data.states.sessions[row])
                branch_ids = membership.branch_ids if membership is not None else ()
                if (
                    membership is None
                    or not all(branch_id in posteriors for branch_id in branch_ids)
                    or not data.labels.valid[row, list(branch_ids)].all()
                    or not all(potential_inputs[branch_id][2][row] for branch_id in branch_ids)
                ):
                    continue
                branch_count = len(membership.branches)
                for model_name in BOUNDARY_MODEL_NAMES:
                    if model_name not in trailing_stops:
                        trailing_stops[model_name] = BranchTrailingStop(
                            data.states.values.shape[1],
                            self.config.maximum_drawdown,
                        )
                    active_ids = set(branch_ids)
                    previous_branch_states[model_name] = {
                        branch_id: state
                        for branch_id, state in previous_branch_states[model_name].items()
                        if branch_id in active_ids
                    }
                thresholds = tuple(
                    max(
                        self.config.minimum_neutral_threshold,
                        self.config.neutral_sigma
                        * data.labels.historical_volatility[row, branch_id],
                    )
                    for branch_id in branch_ids
                )
                signal = data.states.sessions[row].astype("datetime64[D]") + np.timedelta64(15, "h")
                strongest_branches, strength_scores = self._strongest_half_branches(
                    data, int(row), membership.branches
                )
                for model_name in BOUNDARY_MODEL_NAMES:
                    forecasts = tuple(
                        BranchFeynmanKacForecaster(
                            BoundaryFeynmanKacConfig(
                                paths=self.config.fk_paths,
                                seed=self.config.seed + fold_index * 8 + branch_id,
                                use_effective_potential=True,
                            )
                        ).predict(
                            posteriors[branch_id],
                            data.states,
                            int(row),
                            potential_features=potential_inputs[branch_id][1][row],
                            neutral_threshold=float(threshold),
                        )
                        for branch_id, threshold in zip(
                            branch_ids, thresholds, strict=True
                        )
                    )
                    prior_states = tuple(
                        previous_branch_states[model_name].get(
                            branch_id, BranchPositionState()
                        )
                        for branch_id in branch_ids
                    )
                    decision = policy.decide(
                        forecasts,
                        branches=(
                            strongest_branches
                            if model_name == BOUNDARY_MODEL_NAMES[1]
                            else membership.branches
                        ),
                        neutral_thresholds=tuple(float(value) for value in thresholds),
                        previous_branch_states=prior_states,
                    )
                    trailing_stop_forced_exit = trailing_stops[
                        model_name
                    ].liquidation_pending
                    if trailing_stop_forced_exit:
                        decision = replace(
                            decision,
                            open_position=False,
                            gate_reason="branch_trailing_stop",
                            selected_symbols=(),
                            target_weights=(),
                            held_branches=(False,) * branch_count,
                            branch_target_exposures=(0.0,) * branch_count,
                            branch_position_states=(BranchPositionState(),) * branch_count,
                        )
                    current_weights = dict(
                        zip(
                            decision.selected_symbols,
                            decision.target_weights,
                            strict=True,
                        )
                    )
                    decision_branches = (
                        strongest_branches
                        if model_name == BOUNDARY_MODEL_NAMES[1]
                        else membership.branches
                    )
                    target_symbol_branches = {
                        symbol: branch_id
                        for branch_id, symbols in zip(
                            branch_ids, decision_branches, strict=True
                        )
                        for symbol in symbols
                        if symbol in current_weights
                    }
                    execution = self._execute_portfolio(
                        data,
                        int(row),
                        previous=previous_weights[model_name],
                        target=current_weights,
                        previous_symbol_branches=previous_symbol_branches[model_name],
                        target_symbol_branches=target_symbol_branches,
                    )
                    actual_held_branches = tuple(
                        any(branch == index for branch in execution.symbol_branches.values())
                        for index in range(data.states.values.shape[1])
                    )
                    stop_snapshot = trailing_stops[model_name].update(
                        execution.branch_returns,
                        actual_held_branches,
                    )
                    observation = TwoBranchForecastObservation(
                        signal_time=signal,
                        available_at=signal + np.timedelta64(1, "ns"),
                        target_time=data.labels.target_time[row],
                        training_cutoff=cutoff,
                        model_version=model_name,
                        forecasts=forecasts,
                        actual_returns=tuple(
                            float(data.labels.next_return[row, branch_id])
                            for branch_id in branch_ids
                        ),
                        neutral_thresholds=tuple(float(value) for value in thresholds),
                        decision=decision,
                        gross_strategy_return=execution.gross_return,
                        turnover=execution.turnover,
                        branch_ids=branch_ids,
                        actual_symbols=tuple(execution.weights),
                        actual_weights=tuple(execution.weights.values()),
                        failed_buy_orders=execution.failed_buy_orders,
                        failed_sell_orders=execution.failed_sell_orders,
                        branch_equities=tuple(
                            stop_snapshot.branch_equities[branch_id]
                            for branch_id in branch_ids
                        ),
                        branch_peak_equities=tuple(
                            stop_snapshot.branch_peaks[branch_id]
                            for branch_id in branch_ids
                        ),
                        branch_trailing_drawdowns=tuple(
                            stop_snapshot.branch_drawdowns[branch_id]
                            for branch_id in branch_ids
                        ),
                        trailing_stop_pending=stop_snapshot.liquidation_pending,
                        trailing_stop_triggered_branches=(
                            tuple(
                                stop_snapshot.newly_triggered[branch_id]
                                for branch_id in branch_ids
                            )
                        ),
                        trailing_stop_forced_exit=trailing_stop_forced_exit,
                    )
                    observations_by_model[model_name].append(observation)
                    previous_weights[model_name] = execution.weights
                    previous_symbol_branches[model_name] = execution.symbol_branches
                    if "bear" in decision.branch_directions or trailing_stop_forced_exit:
                        previous_branch_states[model_name].clear()
                    previous_branch_states[model_name].update(
                        zip(branch_ids, decision.branch_position_states, strict=True)
                    )
                    prediction_payloads.append(
                        self._prediction_payload(
                            observation,
                            fold_index,
                            int(row),
                            potential_inputs,
                            branch_ids=branch_ids,
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

    def _run_sector_direct(
        self,
        data: TwoBranchDataset,
        root: Path,
    ) -> BoundaryFlowRunResult:
        if data.states.values.shape[1] != 1:
            raise ValueError("whole-sector FK data must contain exactly one state stream")
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
        policy = BoundaryPolicy(
            BoundaryPolicyConfig(
                minimum_direction_probability=self.config.minimum_direction_probability,
                minimum_signal_to_noise=self.config.minimum_signal_to_noise,
                minimum_survival_probability=self.config.minimum_survival_probability,
            )
        )
        potential_names, potential_values, potential_valid = (
            convection_potential_features(data, 0)
        )

        for fold_index, test_start, _, test_rows in folds:
            cutoff = test_start - np.timedelta64(1, "ns")
            posteriors: dict[str, BayesianBranchPosterior] = {}
            for model_name, feature_names in SECTOR_MODEL_FEATURES.items():
                posterior = BayesianBranchModel(
                    BayesianBranchModelConfig(
                        minimum_rows=self.config.minimum_train_sessions,
                        neutral_sigma=self.config.neutral_sigma,
                        drift_half_life_sessions=self.config.drift_half_life_sessions,
                        potential_mode="beta0_c1",
                        drift_feature_names=feature_names,
                        direction_label_sigma=self.config.direction_label_sigma,
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
                fit_rows[model_name].append(posterior.fit_rows)
                posterior_payloads.append(
                    {
                        "model_name": model_name,
                        "scope": "whole_sector",
                        "fold": fold_index,
                        "training_cutoff": str(cutoff),
                        "drift_feature_names": list(posterior.drift_feature_names),
                        "potential_feature_names": list(posterior.potential_feature_names),
                        "drift_mean": posterior.drift_mean.tolist(),
                        "drift_covariance": posterior.drift_covariance.tolist(),
                        "drift_return_scale": posterior.drift_return_scale,
                        "direction_fit_rows": posterior.direction_fit_rows,
                        "potential_mean": posterior.potential_mean.tolist(),
                        "potential_mode": posterior.potential_mode,
                    }
                )

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
                    )
                    decision = policy.decide(
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
        stages = tuple(
            BoundaryStageResult(
                model_name=model_name,
                status="completed",
                passed=True,
                test_months=test_months,
                drift_feature_names=SECTOR_MODEL_FEATURES[model_name],
                minimum_fit_rows=min(fit_rows[model_name]),
                metrics=evaluate_two_branch_forecasts(
                    tuple(observations_by_model[model_name]), evaluation_config
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
        reference = observations_by_model[SECTOR_MODEL_NAMES[0]]
        row_by_session = {
            session.astype("datetime64[D]"): row
            for row, session in enumerate(data.states.sessions)
        }
        actual = np.asarray([item.actual_returns[0] for item in reference])
        rows = np.asarray(
            [row_by_session[item.signal_time.astype("datetime64[D]")] for item in reference]
        )

        def accuracy(predicted: np.ndarray) -> float:
            return float(np.mean(np.sign(predicted) == np.sign(actual)))

        positive = float(np.mean(actual > 0.0))
        baselines = {
            "majority_direction": max(positive, 1.0 - positive),
            "previous_day_direction": accuracy(
                data.states.feature("return_1d")[rows, 0]
            ),
            "path_efficiency_direction": accuracy(
                data.states.feature("signed_path_efficiency")[rows, 0]
            ),
        }
        return {
            "scope": "whole_sector",
            "observations": len(reference),
            "baselines": baselines,
            "models": models,
        }

    @staticmethod
    def _write_sector_report(path: Path, report: dict[str, object]) -> None:
        baselines = report["baselines"]
        lines = [
            "# 板块方向 FK 样本外报告",
            "",
            "本实验只预测下一交易日板块方向，不使用动态支流。",
            "",
            "## 基线",
            "",
            f"- 多数方向：{float(baselines['majority_direction']):.2%}",
            f"- 昨日方向：{float(baselines['previous_day_direction']):.2%}",
            f"- 当日路径效率方向：{float(baselines['path_efficiency_direction']):.2%}",
            "",
            "## 贝叶斯漂移模型",
            "",
            "| 模型 | 漂移方向正确率 | 平衡准确率 | 方向覆盖率 |",
            "|---|---:|---:|---:|",
        ]
        for item in report["models"]:
            metrics = item["metrics"]
            lines.append(
                f"| {item['model_name']} | {metrics['drift_direction_accuracy']:.2%} | "
                f"{metrics['balanced_accuracy']:.2%} | {metrics['direction_signal_coverage']:.2%} |"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

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
            BOUNDARY_MODEL_NAMES[1]: "β0+C1 + 支流内最强 1/2",
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
                f"- 修正版仅交易每支流近 {relative_strength_lookback_sessions} 日相对强度最高的前 1/2 股票。",
                "- 收益改善但换手显著上升时，需要同时检查交易成本后的结果。",
            ]
        )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _mature_monthly_folds(
        self, data: TwoBranchDataset
    ) -> tuple[tuple[int, np.datetime64, np.ndarray, np.ndarray], ...]:
        sessions = data.states.sessions.astype("datetime64[ns]")
        # Padding branches are intentionally invalid; a day is usable when at
        # least one currently active branch has a mature state and label.
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
    def _membership_at(
        memberships: dict[np.datetime64, MonthlyBranchMembership],
        session: np.datetime64,
    ) -> MonthlyBranchMembership | None:
        eligible = [effective for effective in memberships if effective <= session]
        return memberships[max(eligible)] if eligible else None

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
                if not data.member_next_valid[row, column]:
                    raise ValueError(f"missing next-session return for held symbol {symbol}")
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
    def _prediction_payload(
        observation: TwoBranchForecastObservation,
        fold: int,
        row: int,
        potential_inputs: tuple[tuple[tuple[str, ...], np.ndarray, np.ndarray], ...],
        *,
        branch_ids: tuple[int, ...],
        relative_strength_scores: dict[str, float] | None,
        relative_strength_lookback_sessions: int | None,
        effective_potential_enabled: bool,
    ) -> dict[str, object]:
        branches = []
        for branch_id, forecast, actual, threshold, direction in zip(
            branch_ids,
            observation.forecasts,
            observation.actual_returns,
            observation.neutral_thresholds,
            observation.decision.branch_directions,
            strict=True,
        ):
            names, values, _ = potential_inputs[branch_id]
            branches.append(
                {
                    "branch": branch_id + 1,
                    "branch_id": branch_id,
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
            "branch_ids": list(branch_ids),
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
            "branch_equities": list(observation.branch_equities),
            "branch_peak_equities": list(observation.branch_peak_equities),
            "branch_trailing_drawdowns": list(
                observation.branch_trailing_drawdowns
            ),
            "trailing_stop_pending": observation.trailing_stop_pending,
            "trailing_stop_triggered_branches": [
                branch_id
                for branch_id, triggered in zip(
                    branch_ids,
                    observation.trailing_stop_triggered_branches,
                    strict=True,
                )
                if triggered
            ],
            "trailing_stop_forced_exit": observation.trailing_stop_forced_exit,
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
