from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bnhmm.sector_flow.pipeline import (
    SectorFlowConfig,
    SectorFlowPipeline,
)
from bnhmm.sector_flow.ranking import PathEfficiencyAnalyzer, SourceProximityRanker
from bnhmm.sector_flow.topology import RiverTopologyBuilder
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


def synthetic_river_sector(session_count: int = 55) -> dict[str, BarBatch]:
    sessions = np.busday_offset(
        np.datetime64("2025-01-02"), np.arange(session_count)
    )
    slots = np.concatenate((np.arange(571, 691), np.arange(781, 901)))
    rng = np.random.default_rng(412)
    rows = (len(sessions), len(slots))
    common = rng.normal(0.0, 0.00035, rows)
    branches = (
        rng.normal(0.0, 0.00045, rows),
        rng.normal(0.0, 0.00045, rows),
    )
    batches: dict[str, BarBatch] = {}
    timestamps = np.asarray(
        [session + np.timedelta64(int(slot), "m") for session in sessions for slot in slots],
        dtype="datetime64[ns]",
    )
    for symbol_index in range(6):
        branch = branches[0 if symbol_index < 3 else 1]
        returns = (
            common
            + branch
            + rng.normal(0.0, 0.00008, rows)
        ).reshape(-1)
        close = 20.0 * np.exp(np.cumsum(returns))
        volume = rng.lognormal(7.0 + symbol_index * 0.03, 0.2, len(close))
        batches[f"R{symbol_index}"] = BarBatch(
            symbol=f"R{symbol_index}",
            timestamp=timestamps,
            open=close.copy(),
            high=close * 1.001,
            low=close * 0.999,
            close=close,
            volume=volume,
        )
    return batches


class SectorFlowPipelineTests(unittest.TestCase):
    def test_river_protocol_builds_each_monthly_topology_only_once(self) -> None:
        source = MemoryBarSource(synthetic_river_sector(80))
        config = SectorFlowConfig(
            symbols=tuple(source.batches),
            start="2025-01-01",
            end="2025-05-01",
            protocol="river-topology-v2",
            topology_lookback_sessions=20,
            normalization_lookback_sessions=5,
            normalization_minimum_observations=2,
            minimum_train_sessions=35,
            top_k=1,
            max_folds=2,
        )
        months: list[np.datetime64] = []
        original_build = RiverTopologyBuilder.build

        def counted_build(instance, panel, cutoff, month, previous=None):
            months.append(np.datetime64(month, "M"))
            return original_build(instance, panel, cutoff, month, previous)

        with tempfile.TemporaryDirectory() as directory, patch.object(
            RiverTopologyBuilder, "build", new=counted_build
        ):
            result = SectorFlowPipeline(config).run(source, directory)

        self.assertEqual(len(result.folds), 2)
        self.assertEqual(len(months), len(set(months)))

    def test_river_protocol_freezes_monthly_topology_and_writes_artifacts(self) -> None:
        source = MemoryBarSource(synthetic_river_sector())
        config = SectorFlowConfig(
            symbols=tuple(source.batches),
            start="2025-01-01",
            end="2025-04-01",
            protocol="river-topology-v2",
            topology_lookback_sessions=20,
            normalization_lookback_sessions=5,
            normalization_minimum_observations=2,
            minimum_train_sessions=35,
            top_k=1,
            max_folds=1,
        )

        with tempfile.TemporaryDirectory() as directory:
            result = SectorFlowPipeline(config).run(source, directory)
            root = Path(directory)

            self.assertTrue(result.graph_predictions)
            self.assertEqual(len(result.topologies), 1)
            topology = result.topologies[0]
            self.assertTrue(
                all(item.signal_time.astype("datetime64[M]") == topology.month for item in result.graph_predictions)
            )
            self.assertTrue(
                all(item.available_at > item.signal_time for item in result.graph_predictions)
            )
            self.assertTrue((root / "river" / "manifest.json").is_file())
            self.assertEqual(len(list((root / "river" / "topologies").glob("*.npz"))), 1)
            self.assertTrue(list((root / "river" / "slices").glob("*.npz")))
            self.assertTrue((root / "sector-flow-v2-report.md").is_file())

    def test_symbols_without_bars_are_excluded_from_requested_period(self) -> None:
        batches = synthetic_sector()
        missing_symbol = tuple(batches)[-1]
        batches[missing_symbol] = BarBatch.empty(missing_symbol)
        source = MemoryBarSource(batches)
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
            evaluation = Path(directory, "evaluation.json").read_text(
                encoding="utf-8"
            )

        self.assertEqual(result.excluded_symbols, (missing_symbol,))
        self.assertNotIn(missing_symbol, {item.symbol for item in result.predictions})
        self.assertIn(missing_symbol, evaluation)

    def test_symbol_without_first_fold_training_history_is_excluded(self) -> None:
        batches = synthetic_sector()
        late_symbol = tuple(batches)[-1]
        late_batch = batches[late_symbol]
        late_rows = np.flatnonzero(
            late_batch.timestamp >= np.datetime64("2025-02-01")
        )
        batches[late_symbol] = late_batch.take(late_rows)
        source = MemoryBarSource(batches)
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

        self.assertEqual(result.excluded_symbols, (late_symbol,))
        self.assertNotIn(late_symbol, {item.symbol for item in result.predictions})

    def test_fewer_than_three_symbols_with_bars_is_rejected(self) -> None:
        batches = synthetic_sector()
        for symbol in tuple(batches)[2:]:
            batches[symbol] = BarBatch.empty(symbol)
        source = MemoryBarSource(batches)
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
            with self.assertRaisesRegex(ValueError, "at least three"):
                SectorFlowPipeline(config).run(source, directory)

    def test_detailed_ranking_runs_only_once_per_fold(self) -> None:
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
        rank_calls: list[int] = []
        compute_calls: list[int | None] = []
        original_rank = SourceProximityRanker.rank
        original_compute = PathEfficiencyAnalyzer.compute

        def counted_rank(instance, paths, attributions):
            rank_calls.append(len(paths))
            return original_rank(instance, paths, attributions)

        def counted_compute(instance, panel, flow, attribution, *, endpoint=None):
            compute_calls.append(endpoint)
            return original_compute(
                instance, panel, flow, attribution, endpoint=endpoint
            )

        with tempfile.TemporaryDirectory() as directory, patch.object(
            SourceProximityRanker, "rank", new=counted_rank
        ), patch.object(PathEfficiencyAnalyzer, "compute", new=counted_compute):
            result = SectorFlowPipeline(config).run(source, directory)

        self.assertEqual(len(rank_calls), len(result.folds))
        self.assertEqual(len(compute_calls), len(result.folds))

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
