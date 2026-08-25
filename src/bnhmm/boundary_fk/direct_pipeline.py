"""Causal evaluation pipeline for direct signed path-efficiency trading."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path

import numpy as np

from .direct_policy import DirectPathPolicy
from .reporting import write_json
from .types import MonthlyBranchMembership, TwoBranchDataset


MODEL_NAME = "PathEfficiency"
MODEL_VERSION = "PathEfficiency-v1"


@dataclass(frozen=True, slots=True)
class DirectPathPipelineConfig:
    transaction_cost_bps: float = 10.0

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.transaction_cost_bps)
            or self.transaction_cost_bps < 0.0
        ):
            raise ValueError("transaction cost must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class DirectPathMetrics:
    observations: int
    gross_total_return: float
    net_total_return: float
    maximum_drawdown: float
    average_turnover: float
    average_exposure: float
    open_rate: float
    profit_concentration: float
    direction_counts: tuple[int, int, int]
    signal_rank_ic: float
    direction_accuracy: float
    positive_signal_mean_return: float
    negative_signal_mean_return: float


@dataclass(frozen=True, slots=True)
class DirectPathBenchmark:
    gross_total_return: float
    net_total_return: float
    maximum_drawdown: float
    average_turnover: float
    average_exposure: float = 1.0


@dataclass(frozen=True, slots=True)
class DirectPathRunResult:
    model_name: str
    metrics: DirectPathMetrics
    benchmark: DirectPathBenchmark


class DirectPathPipeline:
    def __init__(self, config: DirectPathPipelineConfig | None = None) -> None:
        self.config = config or DirectPathPipelineConfig()

    def run(
        self, data: TwoBranchDataset, output_directory: str | Path
    ) -> DirectPathRunResult:
        root = Path(output_directory)
        root.mkdir(parents=True, exist_ok=True)
        rows = self._eligible_rows(data)
        if not len(rows):
            raise ValueError("direct path strategy has no eligible sample-out rows")

        memberships = {item.month: item for item in data.memberships}
        policy = DirectPathPolicy()
        previous_exposures = (0.0, 0.0)
        previous_weights: dict[str, float] = {}
        previous_benchmark_weights: dict[str, float] = {}
        gross_returns: list[float] = []
        turnovers: list[float] = []
        benchmark_returns: list[float] = []
        benchmark_turnovers: list[float] = []
        exposures: list[float] = []
        signals: list[float] = []
        actual_returns: list[float] = []
        directions: list[str] = []
        predictions: list[dict[str, object]] = []

        path_efficiency = data.states.feature("signed_path_efficiency")
        for row in rows:
            membership = memberships[data.states.topology_months[row]]
            signal_values = (
                float(path_efficiency[row, 0]),
                float(path_efficiency[row, 1]),
            )
            decision = policy.decide(
                signal_values,
                branches=membership.branches,
                previous_exposures=previous_exposures,
            )
            current_weights = dict(
                zip(
                    decision.selected_symbols,
                    decision.target_weights,
                    strict=True,
                )
            )
            turnover = self._turnover(previous_weights, current_weights)
            gross_return = self._portfolio_return(data, int(row), current_weights)

            benchmark_weights = self._branch_weights(membership, (0.5, 0.5))
            benchmark_turnover = self._turnover(
                previous_benchmark_weights, benchmark_weights
            )
            benchmark_return = self._portfolio_return(
                data, int(row), benchmark_weights
            )

            signal_time = data.states.sessions[row].astype("datetime64[D]") + np.timedelta64(15, "h")
            branches = [
                {
                    "branch": branch + 1,
                    "path_efficiency": signal_values[branch],
                    "direction": decision.branch_directions[branch],
                    "actual_return": float(data.labels.next_return[row, branch]),
                    "target_exposure": decision.branch_target_exposures[branch],
                    "symbols": list(membership.branches[branch]),
                }
                for branch in range(2)
            ]
            predictions.append(
                {
                    "signal_time": str(signal_time),
                    "available_at": str(signal_time + np.timedelta64(1, "ns")),
                    "target_time": str(data.labels.target_time[row]),
                    "training_cutoff": str(membership.training_cutoff),
                    "model_version": MODEL_VERSION,
                    "branches": branches,
                    "branch_target_exposures": list(
                        decision.branch_target_exposures
                    ),
                    "selected_symbols": list(decision.selected_symbols),
                    "target_weights": list(decision.target_weights),
                    "gross_strategy_return": gross_return,
                    "turnover": turnover,
                }
            )

            gross_returns.append(gross_return)
            turnovers.append(turnover)
            benchmark_returns.append(benchmark_return)
            benchmark_turnovers.append(benchmark_turnover)
            exposures.append(sum(decision.branch_target_exposures))
            signals.extend(signal_values)
            actual_returns.extend(
                (
                    float(data.labels.next_return[row, 0]),
                    float(data.labels.next_return[row, 1]),
                )
            )
            directions.extend(decision.branch_directions)
            previous_exposures = decision.branch_target_exposures
            previous_weights = current_weights
            previous_benchmark_weights = benchmark_weights

        metrics = self._metrics(
            np.asarray(gross_returns),
            np.asarray(turnovers),
            np.asarray(exposures),
            np.asarray(signals),
            np.asarray(actual_returns),
            tuple(directions),
        )
        benchmark = self._benchmark(
            np.asarray(benchmark_returns), np.asarray(benchmark_turnovers)
        )
        result = DirectPathRunResult(MODEL_NAME, metrics, benchmark)
        evaluation = [
            {
                "model_name": MODEL_NAME,
                "status": "completed",
                "passed": True,
                "signal": "signed_path_efficiency_sign",
                "metrics": asdict(metrics),
                "benchmark": asdict(benchmark),
            }
        ]
        write_json(root / "path-efficiency-predictions.json", predictions)
        write_json(root / "path-efficiency-evaluation.json", evaluation)
        self._write_report(root / "path-efficiency-report.md", result)
        return result

    def _eligible_rows(self, data: TwoBranchDataset) -> np.ndarray:
        sessions = data.states.sessions.astype("datetime64[ns]")
        eligible = np.all(data.states.valid & data.labels.valid, axis=1)
        available_months = {item.month for item in data.memberships}
        eligible &= np.asarray(
            [month in available_months for month in data.states.topology_months]
        )
        if data.evaluation_start is not None:
            eligible &= sessions >= np.datetime64(data.evaluation_start, "ns")
        if data.evaluation_end is not None:
            end = (
                np.datetime64(data.evaluation_end, "D") + np.timedelta64(1, "D")
            ).astype("datetime64[ns]")
            eligible &= sessions < end
        return np.flatnonzero(eligible)

    def _metrics(
        self,
        gross_returns: np.ndarray,
        turnover: np.ndarray,
        exposure: np.ndarray,
        signals: np.ndarray,
        actual_returns: np.ndarray,
        directions: tuple[str, ...],
    ) -> DirectPathMetrics:
        net_returns = gross_returns - (
            turnover * self.config.transaction_cost_bps / 10_000.0
        )
        net_equity = np.cumprod(1.0 + net_returns)
        positive = signals > 0.0
        negative = signals < 0.0
        return DirectPathMetrics(
            observations=len(gross_returns),
            gross_total_return=float(np.prod(1.0 + gross_returns) - 1.0),
            net_total_return=float(net_equity[-1] - 1.0),
            maximum_drawdown=_maximum_drawdown(net_equity),
            average_turnover=float(np.mean(turnover)),
            average_exposure=float(np.mean(exposure)),
            open_rate=float(np.mean(exposure > 0.0)),
            profit_concentration=_profit_concentration(
                net_returns[exposure > 0.0]
            ),
            direction_counts=(
                directions.count("bear"),
                directions.count("hold"),
                directions.count("bull"),
            ),
            signal_rank_ic=_rank_correlation(signals, actual_returns),
            direction_accuracy=float(
                np.mean(np.sign(signals) == np.sign(actual_returns))
            ),
            positive_signal_mean_return=_selected_mean(actual_returns, positive),
            negative_signal_mean_return=_selected_mean(actual_returns, negative),
        )

    def _benchmark(
        self, gross_returns: np.ndarray, turnover: np.ndarray
    ) -> DirectPathBenchmark:
        net_returns = gross_returns - (
            turnover * self.config.transaction_cost_bps / 10_000.0
        )
        net_equity = np.cumprod(1.0 + net_returns)
        return DirectPathBenchmark(
            gross_total_return=float(np.prod(1.0 + gross_returns) - 1.0),
            net_total_return=float(net_equity[-1] - 1.0),
            maximum_drawdown=_maximum_drawdown(net_equity),
            average_turnover=float(np.mean(turnover)),
        )

    @staticmethod
    def _branch_weights(
        membership: MonthlyBranchMembership,
        exposures: tuple[float, float],
    ) -> dict[str, float]:
        result: dict[str, float] = {}
        for symbols, exposure in zip(
            membership.branches, exposures, strict=True
        ):
            result.update({symbol: exposure / len(symbols) for symbol in symbols})
        return result

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
        symbol_index = {symbol: index for index, symbol in enumerate(data.symbols)}
        result = 0.0
        for symbol, weight in weights.items():
            column = symbol_index[symbol]
            if not data.member_next_valid[row, column]:
                raise ValueError(f"missing next-session return for held symbol {symbol}")
            result += weight * float(data.member_next_returns[row, column])
        return float(result)

    @staticmethod
    def _write_report(path: Path, result: DirectPathRunResult) -> None:
        metrics = result.metrics
        benchmark = result.benchmark
        path.write_text(
            "\n".join(
                (
                    "# 路径效率直接交易报告",
                    "",
                    "规则：路径效率为正买入该支流50%，为负清仓，等于零保持原仓位。",
                    "",
                    f"- 样本外交易日：{metrics.observations}",
                    f"- 策略净收益：{metrics.net_total_return:.2%}",
                    f"- 策略最大回撤：{metrics.maximum_drawdown:.2%}",
                    f"- 策略平均仓位：{metrics.average_exposure:.2%}",
                    f"- 路径效率秩相关：{metrics.signal_rank_ic:.4f}",
                    f"- 正路径效率次日平均收益：{metrics.positive_signal_mean_return:.4%}",
                    f"- 负路径效率次日平均收益：{metrics.negative_signal_mean_return:.4%}",
                    f"- 始终满仓基准净收益：{benchmark.net_total_return:.2%}",
                    f"- 始终满仓基准最大回撤：{benchmark.maximum_drawdown:.2%}",
                    "",
                )
            ),
            encoding="utf-8",
        )


def _maximum_drawdown(equity: np.ndarray) -> float:
    with_initial = np.concatenate((np.ones(1), equity))
    peaks = np.maximum.accumulate(with_initial)
    return float(np.max(1.0 - with_initial / peaks))


def _profit_concentration(returns: np.ndarray) -> float:
    positive = np.sort(returns[returns > 0.0])[::-1]
    if not len(positive):
        return 0.0
    count = max(1, ceil(len(positive) * 0.1))
    return float(np.sum(positive[:count]) / np.sum(positive))


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2:
        return 0.0
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    if np.std(left_rank) <= 1e-15 or np.std(right_rank) <= 1e-15:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) * 0.5
        start = end
    return ranks


def _selected_mean(values: np.ndarray, selected: np.ndarray) -> float:
    return float(np.mean(values[selected])) if np.any(selected) else 0.0
