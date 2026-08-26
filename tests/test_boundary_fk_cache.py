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
from bnhmm.boundary_fk.precompute import (
    build_boundary_cache_variants,
    topology_months_for_available_data,
    topology_months_with_minimum_history,
)
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
    np.testing.assert_allclose(
        restored.member_next_hold_returns,
        original.member_next_hold_returns,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        restored.member_next_buy_returns,
        original.member_next_buy_returns,
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        restored.member_next_buy_executable,
        original.member_next_buy_executable,
    )
    np.testing.assert_allclose(
        restored.member_next_sell_returns,
        original.member_next_sell_returns,
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        restored.member_next_sell_executable,
        original.member_next_sell_executable,
    )
    assert restored.evaluation_start == np.datetime64("2025-01-01")
    assert restored.evaluation_end == np.datetime64("2025-06-30")
    assert not any(path.suffix in {".pkl", ".pickle"} for path in cache.root.rglob("*"))


def test_boundary_cache_round_trips_convection_diagnostics(tmp_path: Path) -> None:
    cache = BoundaryDataCache(tmp_path / "cache")
    key = _key(tmp_path)
    base = _walk_forward_data(180)
    shape = (len(base.states.sessions), 2, 4)
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
    assert restored.convection_lookbacks == (1, 2, 3, 5)


def test_boundary_cache_uses_direction_feature_version() -> None:
    assert (
        CACHE_CONSTRUCTION_VERSION
        == "path-efficiency-two-branch-v7-direction-features"
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


def test_topology_months_never_precede_available_minute_history() -> None:
    months = topology_months_for_available_data(
        requested_start=np.datetime64("2023-03", "M"),
        requested_end=np.datetime64("2026-08", "M"),
        first_timestamp=np.datetime64("2024-01-02T09:30", "m"),
    )

    assert months[0] == np.datetime64("2024-02", "M")
    assert months[-1] == np.datetime64("2026-08", "M")
    assert np.all(months > np.datetime64("2024-01", "M"))


def test_topology_months_require_completed_session_history() -> None:
    sessions = np.busday_offset(
        np.datetime64("2024-01-02", "D"), np.arange(70), roll="forward"
    )
    months = np.arange(
        np.datetime64("2024-02", "M"),
        np.datetime64("2024-06", "M"),
        dtype="datetime64[M]",
    )

    eligible = topology_months_with_minimum_history(
        months=months,
        timestamps=sessions.astype("datetime64[ns]"),
        minimum_sessions=40,
    )

    assert eligible[0] == np.datetime64("2024-03", "M")
    assert all(np.sum(sessions < month.astype("datetime64[D]")) >= 40 for month in eligible)


def test_precompute_reads_and_normalizes_minutes_once_for_all_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, list[object]] = {"fetch": [], "normalize": [], "execution": [], "lookbacks": [], "minimums": [], "save": []}
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
            return type(
                "NormalizedPanel",
                (),
                {
                    "timestamps": np.busday_offset(
                        np.datetime64("2023-01-03", "D"),
                        np.arange(100),
                        roll="forward",
                    ).astype("datetime64[m]")
                },
            )()

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
            assert hasattr(panel, "timestamps")
            return (f"topology-{calls['lookbacks'][-1]}",)

    class FakeStateBuilder:
        def build(self, panel, topologies):
            assert hasattr(panel, "timestamps")
            return _walk_forward_data(180)

    class FakeCache:
        def save(self, key, data, *, topologies):
            calls["save"].append(key.lookback_sessions)
            assert topologies == (f"topology-{key.lookback_sessions}",)

    import bnhmm.boundary_fk.precompute as module

    monkeypatch.setattr(module, "SectorMinutePanelBuilder", FakePanelBuilder)
    monkeypatch.setattr(module, "CausalBoundaryNormalizer", FakeNormalizer)
    monkeypatch.setattr(
        module,
        "build_member_execution_constraints",
        lambda panel: calls["execution"].append(panel) or type(
            "Execution",
            (),
            {
                "sessions": _walk_forward_data(180).states.sessions,
                "hold_returns": _walk_forward_data(180).member_next_returns,
                "buy_returns": _walk_forward_data(180).member_next_returns,
                "buy_executable": _walk_forward_data(180).member_next_valid,
                "sell_returns": np.zeros_like(_walk_forward_data(180).member_next_returns),
                "sell_executable": _walk_forward_data(180).member_next_valid,
            },
        )(),
    )
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
    assert calls["execution"] == ["raw-panel"]
    assert tuple(calls["lookbacks"]) == CACHE_LOOKBACKS
    assert tuple(calls["minimums"]) == (
        (7, 5), (14, 10), (20, 15), (27, 20), (40, 30), (60, 45), (80, 60)
    )
    assert tuple(calls["save"]) == CACHE_LOOKBACKS
