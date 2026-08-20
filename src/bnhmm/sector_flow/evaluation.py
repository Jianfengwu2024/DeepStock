"""Walk-forward splits, prediction contracts, and sample-out metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class SectorFlowFold:
    index: int
    test_start: np.datetime64
    test_end: np.datetime64
    training_rows: NDArray[np.int64]
    test_rows: NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class SectorFlowPrediction:
    symbol: str
    signal_time: np.datetime64
    available_at: np.datetime64
    training_cutoff: np.datetime64
    model_version: str
    fold: int
    source_probability: float
    direction_probability: float
    realized_return: float
    interval_lower: float
    interval_upper: float
    leader_horizon_minutes: int
    response_delay_minutes: int

    def __post_init__(self) -> None:
        if not self.symbol or not self.model_version:
            raise ValueError("prediction identity fields cannot be empty")
        if self.available_at <= self.signal_time:
            raise ValueError("prediction cannot be available at its signal time")
        if self.training_cutoff > self.signal_time:
            raise ValueError("training cutoff cannot follow the signal")
        probabilities = (self.source_probability, self.direction_probability)
        if any(not 0.0 <= item <= 1.0 for item in probabilities):
            raise ValueError("prediction probabilities must be in [0, 1]")
        if self.interval_lower > self.interval_upper:
            raise ValueError("prediction interval is reversed")

    def as_json(self) -> dict[str, object]:
        payload = asdict(self)
        for key in ("signal_time", "available_at", "training_cutoff"):
            payload[key] = str(payload[key])
        return payload


@dataclass(frozen=True, slots=True)
class RankingObservation:
    symbol: str
    signal_time: np.datetime64
    score: float
    realized_return: float


def build_monthly_folds(
    timestamps: NDArray[np.datetime64],
    *,
    minimum_train_sessions: int,
    max_folds: int | None = None,
) -> tuple[SectorFlowFold, ...]:
    if minimum_train_sessions < 1:
        raise ValueError("minimum training sessions must be positive")
    if max_folds is not None and max_folds < 1:
        raise ValueError("maximum fold count must be positive")
    values = np.asarray(timestamps, dtype="datetime64[ns]")
    if len(values) and np.any(values[1:] <= values[:-1]):
        raise ValueError("walk-forward timestamps must be strictly increasing")
    sessions = values.astype("datetime64[D]")
    months = values.astype("datetime64[M]")
    folds: list[SectorFlowFold] = []
    for month in np.unique(months):
        test_start = month.astype("datetime64[ns]")
        test_end = (month + np.timedelta64(1, "M")).astype("datetime64[ns]")
        training_rows = np.flatnonzero(values < test_start).astype(np.int64)
        training_sessions = np.unique(sessions[training_rows])
        test_rows = np.flatnonzero(
            (values >= test_start) & (values < test_end)
        ).astype(np.int64)
        if len(training_sessions) < minimum_train_sessions or not len(test_rows):
            continue
        folds.append(
            SectorFlowFold(
                index=len(folds),
                test_start=test_start,
                test_end=test_end,
                training_rows=training_rows,
                test_rows=test_rows,
            )
        )
        if max_folds is not None and len(folds) >= max_folds:
            break
    return tuple(folds)


def evaluate_predictions(
    predictions: Sequence[SectorFlowPrediction],
    *,
    top_k: int,
    transaction_cost_bps: float,
) -> dict[str, float | int]:
    if not predictions:
        raise ValueError("cannot evaluate an empty prediction set")
    if top_k < 1 or transaction_cost_bps < 0.0:
        raise ValueError("evaluation portfolio settings are invalid")
    probability = np.asarray(
        [item.direction_probability for item in predictions], dtype=np.float64
    )
    probability = np.clip(probability, 1e-9, 1.0 - 1e-9)
    realized = np.asarray(
        [item.realized_return for item in predictions], dtype=np.float64
    )
    target = (realized > 0.0).astype(np.float64)
    brier = float(np.mean(np.square(probability - target)))
    log_loss = float(
        -np.mean(
            target * np.log(probability)
            + (1.0 - target) * np.log(1.0 - probability)
        )
    )
    coverage = float(
        np.mean(
            [
                item.interval_lower <= item.realized_return <= item.interval_upper
                for item in predictions
            ]
        )
    )
    ranking = evaluate_ranking_observations(
        [
            RankingObservation(
                symbol=item.symbol,
                signal_time=item.signal_time,
                score=item.source_probability,
                realized_return=item.realized_return,
            )
            for item in predictions
        ],
        top_k=top_k,
        transaction_cost_bps=transaction_cost_bps,
    )
    leader_lags = np.asarray(
        [item.leader_horizon_minutes for item in predictions], dtype=np.float64
    )
    response_lags = np.asarray(
        [item.response_delay_minutes for item in predictions], dtype=np.float64
    )
    return {
        "prediction_count": len(predictions),
        "brier_score": brier,
        "log_loss": log_loss,
        **ranking,
        "posterior_coverage": coverage,
        "leader_lag_stability": exp(-float(np.std(leader_lags))),
        "response_lag_stability": exp(-float(np.std(response_lags))),
    }


def evaluate_ranking_observations(
    observations: Sequence[RankingObservation],
    *,
    top_k: int,
    transaction_cost_bps: float,
) -> dict[str, float]:
    if not observations:
        raise ValueError("cannot evaluate empty ranking observations")
    grouped: dict[int, list[RankingObservation]] = {}
    for item in observations:
        key = int(item.signal_time.astype("datetime64[ns]").astype(np.int64))
        grouped.setdefault(key, []).append(item)
    rank_correlations: list[float] = []
    gross_spreads: list[float] = []
    turnovers: list[float] = []
    prior_top: set[str] = set()
    for key in sorted(grouped):
        items = grouped[key]
        if len(items) < 2:
            continue
        scores = np.asarray([item.score for item in items], dtype=np.float64)
        returns = np.asarray(
            [item.realized_return for item in items], dtype=np.float64
        )
        score_rank = _ranks(scores)
        return_rank = _ranks(returns)
        if np.std(score_rank) > 0.0 and np.std(return_rank) > 0.0:
            rank_correlations.append(
                float(np.corrcoef(score_rank, return_rank)[0, 1])
            )
        count = min(top_k, len(items) // 2)
        order = np.argsort(scores)
        bottom = returns[order[:count]]
        top = returns[order[-count:]]
        gross_spreads.append(float(np.mean(top) - np.mean(bottom)))
        current_top = {items[index].symbol for index in order[-count:]}
        denominator = max(len(current_top), 1)
        turnovers.append(float(len(current_top.symmetric_difference(prior_top)) / denominator))
        prior_top = current_top
    gross = float(np.mean(gross_spreads)) if gross_spreads else 0.0
    turnover = float(np.mean(turnovers)) if turnovers else 0.0
    cost = turnover * transaction_cost_bps / 10_000.0
    return {
        "rank_ic": float(np.mean(rank_correlations)) if rank_correlations else 0.0,
        "top_bottom_return": gross,
        "turnover": turnover,
        "transaction_cost": cost,
        "net_top_bottom_return": gross - cost,
    }


def predictions_as_json(
    predictions: Iterable[SectorFlowPrediction],
) -> list[dict[str, object]]:
    return [item.as_json() for item in predictions]


def _ranks(values: NDArray[np.float64]) -> NDArray[np.float64]:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        for index in range(len(unique)):
            members = np.flatnonzero(inverse == index)
            ranks[members] = float(np.mean(ranks[members]))
    return ranks
