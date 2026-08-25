from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from bnhmm.boundary_fk.cache import (
    CACHE_CONSTRUCTION_VERSION,
    CACHE_LOOKBACKS,
    BoundaryDataCache,
    BoundaryDataCacheKey,
    BoundaryDataCacheMiss,
)
from bnhmm.boundary_fk.precompute import build_boundary_cache_variants
from test_boundary_fk_pipeline import _walk_forward_data


def _key(tmp_path: Path, lookback: int = 60) -> BoundaryDataCacheKey:
    source = tmp_path / "market.duckdb"
    source.write_bytes(b"duckdb-fixture")
    return BoundaryDataCacheKey.from_source(
        dataset_id="semiconductor",
        source_path=source,
        symbols=("A", "B", "C", "D"),
        start="2024-01-01",
        end="2025-12-31",
        lookback_sessions=lookback,
    )


def test_boundary_cache_round_trips_dataset_without_pickle(tmp_path: Path) -> None:
    cache = BoundaryDataCache(tmp_path / "cache")
    key = _key(tmp_path)
    original = _walk_forward_data(180)

    cache.save(key, original)
    restored = cache.load(key, evaluation_start="2025-01-01", evaluation_end="2025-06-30")

    assert restored.symbols == original.symbols
    assert restored.memberships == original.memberships
    np.testing.assert_array_equal(restored.states.sessions, original.states.sessions)
    np.testing.assert_allclose(restored.states.values, original.states.values, equal_nan=True)
    np.testing.assert_allclose(restored.labels.next_return, original.labels.next_return, equal_nan=True)
    np.testing.assert_allclose(restored.member_next_returns, original.member_next_returns, equal_nan=True)
    assert restored.evaluation_start == np.datetime64("2025-01-01")
    assert restored.evaluation_end == np.datetime64("2025-06-30")
    assert not any(path.suffix in {".pkl", ".pickle"} for path in cache.root.rglob("*"))


def test_boundary_cache_round_trips_convection_diagnostics(tmp_path: Path) -> None:
    cache = BoundaryDataCache(tmp_path / "cache")
    key = _key(tmp_path)
    base = _walk_forward_data(180)
    shape = (len(base.states.sessions), 2)
    original = replace(
        base,
        convection_strength=np.full(shape, 0.25),
        convection_valid=np.ones(shape, dtype=np.bool_),
        convection_total_flow=np.full(shape, 0.08),
        convection_observations=np.full(shape, 6, dtype=np.int64),
    )
    cache.save(key, original)
    restored = cache.load(key)

    np.testing.assert_allclose(restored.convection_strength, original.convection_strength)
    np.testing.assert_array_equal(restored.convection_valid, original.convection_valid)
    np.testing.assert_allclose(restored.convection_total_flow, original.convection_total_flow)
    np.testing.assert_array_equal(
        restored.convection_observations, original.convection_observations
    )


def test_boundary_cache_uses_reverse_flow_concentration_version() -> None:
    assert (
        CACHE_CONSTRUCTION_VERSION
        == "path-efficiency-two-branch-v3-reverse-flow-concentration"
    )


def test_boundary_cache_rejects_changed_source_and_wrong_window(tmp_path: Path) -> None:
    cache = BoundaryDataCache(tmp_path / "cache")
    key = _key(tmp_path)
    cache.save(key, _walk_forward_data(180))

    key.source_path.write_bytes(b"source-changed")
    changed = BoundaryDataCacheKey.from_source(
        dataset_id=key.dataset_id,
        source_path=key.source_path,
        symbols=key.symbols,
        start=key.start,
        end=key.end,
        lookback_sessions=key.lookback_sessions,
    )
    with pytest.raises(BoundaryDataCacheMiss, match="source fingerprint"):
        cache.load(changed)
    with pytest.raises(ValueError, match="supported lookback"):
        replace(key, lookback_sessions=15)


def test_boundary_cache_manifest_lists_all_standard_windows(tmp_path: Path) -> None:
    cache = BoundaryDataCache(tmp_path / "cache")
    for lookback in CACHE_LOOKBACKS:
        cache.save(_key(tmp_path, lookback), _walk_forward_data(180))

    manifest = json.loads(
        (cache.root / "semiconductor" / "manifest.json").read_text(encoding="utf-8")
    )
    assert tuple(item["lookback_sessions"] for item in manifest["variants"]) == CACHE_LOOKBACKS
    assert all(item["status"] == "ready" for item in manifest["variants"])


def test_precompute_reads_and_normalizes_minutes_once_for_all_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, list[object]] = {"fetch": [], "normalize": [], "lookbacks": [], "minimums": [], "save": []}
    source = object()

    class FakeSource:
        def fetch(self, symbol, timeframe, start, end):
            calls["fetch"].append((symbol, timeframe, start, end))
            return symbol

    class FakePanelBuilder:
        def build(self, bars):
            assert tuple(bars) == ("A", "B", "C", "D")
            return "raw-panel"

    class FakeNormalizer:
        def transform(self, panel):
            calls["normalize"].append(panel)
            return "normalized-panel"

    class FakeTopologyConfig:
        def __init__(self, *, lookback_sessions, **kwargs):
            self.lookback_sessions = lookback_sessions
            calls["minimums"].append(
                (kwargs["minimum_sessions"], kwargs["minimum_pair_sessions"])
            )

    class FakeTopologyBuilder:
        def __init__(self, config):
            calls["lookbacks"].append(config.lookback_sessions)

        def build(self, panel, months):
            assert panel == "normalized-panel"
            return (f"topology-{calls['lookbacks'][-1]}",)

    class FakeStateBuilder:
        def build(self, panel, topologies):
            assert panel == "normalized-panel"
            return _walk_forward_data(180)

    class FakeCache:
        def save(self, key, data, *, topologies):
            calls["save"].append(key.lookback_sessions)
            assert topologies == (f"topology-{key.lookback_sessions}",)

    import bnhmm.boundary_fk.precompute as module

    monkeypatch.setattr(module, "SectorMinutePanelBuilder", FakePanelBuilder)
    monkeypatch.setattr(module, "CausalBoundaryNormalizer", FakeNormalizer)
    monkeypatch.setattr(module, "TopologyConfig", FakeTopologyConfig)
    monkeypatch.setattr(module, "BoundaryTopologyBuilder", FakeTopologyBuilder)
    monkeypatch.setattr(module, "TwoBranchDailyStateBuilder", FakeStateBuilder)
    monkeypatch.setattr(module, "BoundaryDataCacheKey", type("KeyFactory", (), {
        "from_source": staticmethod(lambda **kwargs: type("Key", (), kwargs)())
    }))

    build_boundary_cache_variants(
        FakeSource(),
        FakeCache(),
        dataset_id="semiconductor",
        source_path=tmp_path / "market.duckdb",
        symbols=("A", "B", "C", "D"),
        start="2024-01-01",
        end="2025-12-31",
    )

    assert len(calls["fetch"]) == 4
    assert calls["normalize"] == ["raw-panel"]
    assert tuple(calls["lookbacks"]) == CACHE_LOOKBACKS
    assert tuple(calls["minimums"]) == (
        (7, 5), (14, 10), (20, 15), (27, 20), (40, 30), (60, 45), (80, 60)
    )
    assert tuple(calls["save"]) == CACHE_LOOKBACKS
