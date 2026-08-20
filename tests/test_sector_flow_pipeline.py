from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from bnhmm.sector_flow.pipeline import (
    SectorFlowConfig,
    SectorFlowPipeline,
)
from bnhmm.types import BarBatch


class MemoryBarSource:
    def __init__(self, batches: dict[str, BarBatch]) -> None:
        self.batches = batches

    def fetch(
        self, symbol: str, timeframe: str, start: str, end: str
    ) -> BarBatch:
        if timeframe != "1m":
            raise ValueError("test source only contains one-minute bars")
        batch = self.batches[symbol]
        mask = (batch.timestamp >= np.datetime64(start)) & (
            batch.timestamp < np.datetime64(end)
        )
        return batch.take(np.flatnonzero(mask))


def synthetic_sector() -> dict[str, BarBatch]:
    sessions = [
        np.datetime64(f"2025-01-{day:02d}") for day in range(2, 10)
    ] + [np.datetime64(f"2025-02-{day:02d}") for day in range(3, 8)]
    rng = np.random.default_rng(41)
    factors = rng.normal(0.0, 0.001, (len(sessions), 15))
    batches: dict[str, BarBatch] = {}
    for symbol_index in range(6):
        timestamps = np.asarray(
            [
                session + np.timedelta64(9 * 60 + 31 + minute, "m")
                for session in sessions
                for minute in range(15)
            ],
            dtype="datetime64[ns]",
        )
        returns = (
            factors * (0.8 + symbol_index * 0.08)
            + rng.normal(0.0, 0.0002, factors.shape)
        ).reshape(-1)
        close = 20.0 * np.exp(np.cumsum(returns))
        volume = np.full(len(close), 1000.0 + symbol_index * 100.0)
        batches[f"S{symbol_index}"] = BarBatch(
            symbol=f"S{symbol_index}",
            timestamp=timestamps,
            open=close.copy(),
            high=close * 1.001,
            low=close * 0.999,
            close=close,
            volume=volume,
        )
    return batches


class SectorFlowPipelineTests(unittest.TestCase):
    def test_small_walk_forward_writes_causal_report_artifacts(self) -> None:
        source = MemoryBarSource(synthetic_sector())
        config = SectorFlowConfig(
            symbols=tuple(source.batches),
            start="2025-01-01",
            end="2025-03-01",
            normalization_lookback_sessions=5,
            normalization_minimum_observations=2,
            minimum_train_sessions=5,
            maximum_lag_minutes=3,
            minimum_attribution_rows=20,
            path_windows=(5,),
            ranking_window=5,
            evaluation_horizon_minutes=3,
            top_k=1,
            monte_carlo_samples=200,
        )
        with tempfile.TemporaryDirectory() as directory:
            result = SectorFlowPipeline(config).run(source, directory)
            root = Path(directory)

            self.assertTrue(result.predictions)
            self.assertIn("direction", result.latest_water_state)
            self.assertIn("coherence", result.latest_water_state)
            self.assertEqual(
                set(result.baselines),
                {"random", "momentum", "relative_volume", "path_efficiency", "static_beta"},
            )
            self.assertTrue(all(item.available_at > item.signal_time for item in result.predictions))
            self.assertTrue(all(item.training_cutoff <= item.signal_time for item in result.predictions))
            for name in (
                "predictions.json",
                "predictions.csv",
                "evaluation.json",
                "ranking.csv",
                "sector-flow-report.md",
                "sector-flow.png",
                "run-summary.json",
            ):
                self.assertTrue((root / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
