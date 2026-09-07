from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bnhmm.percolation_branching.pipeline import (
    PercolationPipeline,
    PercolationPipelineConfig,
)
from bnhmm.types import BarBatch


class MemorySource:
    def __init__(self, batches: dict[str, BarBatch]) -> None:
        self.batches = batches

    def fetch(self, symbol: str, timeframe: str, start: str, end: str) -> BarBatch:
        assert timeframe == "1m"
        batch = self.batches[symbol]
        selected = (batch.timestamp >= np.datetime64(start)) & (
            batch.timestamp < np.datetime64(end)
        )
        return batch.take(np.flatnonzero(selected))


def _sector() -> dict[str, BarBatch]:
    sessions = np.busday_offset(np.datetime64("2024-10-01"), np.arange(110))
    slots = np.concatenate((np.arange(571, 691), np.arange(781, 901)))
    timestamps = np.asarray(
        [session + np.timedelta64(int(slot), "m") for session in sessions for slot in slots],
        dtype="datetime64[ns]",
    )
    rng = np.random.default_rng(111)
    common = rng.normal(0.0, 0.00035, (len(sessions), len(slots)))
    result: dict[str, BarBatch] = {}
    for column in range(5):
        returns = common + rng.normal(0.0, 0.00018, common.shape)
        for day in range(len(sessions)):
            minute = 30 + column
            returns[day, minute] += 0.0015
        close = 30.0 * np.exp(np.cumsum(returns.reshape(-1)))
        volume = rng.lognormal(7.0 + column * 0.05, 0.2, len(close))
        result[f"S{column}"] = BarBatch(
            symbol=f"S{column}",
            timestamp=timestamps,
            open=close.copy(),
            high=close * 1.001,
            low=close * 0.999,
            close=close,
            volume=volume,
        )
    return result


def test_pipeline_writes_causal_finite_artifacts(tmp_path: Path) -> None:
    source = MemorySource(_sector())
    config = PercolationPipelineConfig(
        symbols=tuple(source.batches),
        dataset_id="synthetic-sector",
        start="2024-10-01",
        end="2025-04-01",
        activation_threshold=0.8,
        null_permutations=4,
        posterior_paths=40,
        minimum_valid_symbols=3,
        max_folds=2,
        random_seed=23,
    )

    result = PercolationPipeline(config).run(source, tmp_path)

    assert result.predictions
    assert len(result.folds) == 2
    required = {
        "run-summary.json",
        "percolation-evaluation.json",
        "percolation-baselines.json",
        "percolation-predictions.json",
        "percolation-posteriors.json",
        "percolation-equity.json",
        "percolation-network-summary.json",
        "percolation-report-zh.md",
    }
    assert required <= {path.name for path in tmp_path.iterdir()}
    payload = json.loads((tmp_path / "percolation-predictions.json").read_text(encoding="utf-8"))
    assert payload
    assert all(abs(sum(item["phase_probabilities"]) - 1.0) < 1e-9 for item in payload)
    assert all(item["signal_time"] < item["available_at"] for item in payload)
    assert "NaN" not in (tmp_path / "percolation-evaluation.json").read_text(encoding="utf-8")
    assert "渗流" in (tmp_path / "percolation-report-zh.md").read_text(encoding="utf-8")
