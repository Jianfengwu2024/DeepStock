from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from bnhmm.boundary_fk.cache import (
    CACHE_FORMAT_VERSION,
    CACHE_CONSTRUCTION_VERSION,
    SECTOR_CACHE_SLOT,
    BoundaryDataCache,
    BoundaryDataCacheKey,
    BoundaryDataCacheMiss,
)
from bnhmm.boundary_fk.precompute import (
    build_sector_boundary_cache,
)
from bnhmm.boundary_fk.types import MonthlyBranchMembership
from boundary_fk_fixtures import walk_forward_sector_data


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
    original = walk_forward_sector_data(180)

    cache.save(key, original)
    restored = cache.load(key, evaluation_start="2025-01-01", evaluation_end="2025-06-30")

    assert restored.symbols == original.symbols
    assert restored.memberships == ()
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


def test_boundary_cache_rejects_removed_branch_memberships(tmp_path: Path) -> None:
    cache = BoundaryDataCache(tmp_path / "cache")
    key = _key(tmp_path)
    data = walk_forward_sector_data(180)
    first_session = data.states.sessions[0].astype("datetime64[D]")
    membership = MonthlyBranchMembership(
        effective_session=first_session,
        training_cutoff=first_session.astype("datetime64[ns]")
        - np.timedelta64(1, "ns"),
        branches=(("A", "B"), ("C", "D")),
    )

    with pytest.raises(ValueError, match="must not contain branch memberships"):
        cache.save(key, replace(data, memberships=(membership,)))


def test_boundary_cache_omits_removed_branch_and_topology_payloads(
    tmp_path: Path,
) -> None:
    cache = BoundaryDataCache(tmp_path / "cache")
    destination = cache.save(_key(tmp_path), walk_forward_sector_data(180))

    with np.load(destination, allow_pickle=False) as payload:
        assert "memberships_json" not in payload.files
        assert not any(name.startswith("topology_") for name in payload.files)


def test_boundary_cache_round_trips_convection_diagnostics(tmp_path: Path) -> None:
    cache = BoundaryDataCache(tmp_path / "cache")
    key = _key(tmp_path)
    base = walk_forward_sector_data(180)
    shape = (len(base.states.sessions), 1, 4)
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
    assert CACHE_FORMAT_VERSION == 3
    assert CACHE_CONSTRUCTION_VERSION == "sector-rolling-anchor-60d-v2"


def test_boundary_cache_rejects_daily_reset_v2_manifest(tmp_path: Path) -> None:
    cache = BoundaryDataCache(tmp_path / "cache")
    key = _key(tmp_path)
    cache.save(key, walk_forward_sector_data(180))
    manifest_path = cache.root / key.dataset_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["format_version"] = 2
    manifest["variants"][0]["construction_version"] = "sector-direct-direction-v1"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(BoundaryDataCacheMiss, match="format version"):
        cache.load(key)


def test_boundary_cache_rejects_changed_source_and_wrong_window(tmp_path: Path) -> None:
    cache = BoundaryDataCache(tmp_path / "cache")
    key = _key(tmp_path)
    cache.save(key, walk_forward_sector_data(180))

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


def test_boundary_cache_manifest_contains_one_sector_state_variant(tmp_path: Path) -> None:
    cache = BoundaryDataCache(tmp_path / "cache")
    cache.save(_key(tmp_path, SECTOR_CACHE_SLOT), walk_forward_sector_data(180))

    manifest = json.loads(
        (cache.root / "semiconductor" / "manifest.json").read_text(encoding="utf-8")
    )
    assert tuple(item["lookback_sessions"] for item in manifest["variants"]) == (
        SECTOR_CACHE_SLOT,
    )
    assert all(item["status"] == "ready" for item in manifest["variants"])
    assert manifest["anchor_protocol"] == "rolling-60-session-previous-close"


def test_sector_cache_manifest_discards_legacy_window_variants(tmp_path: Path) -> None:
    cache = BoundaryDataCache(tmp_path / "cache")
    dataset_root = cache.root / "semiconductor"
    dataset_root.mkdir(parents=True)
    (dataset_root / "manifest.json").write_text(
        json.dumps(
            {
                "format_version": 2,
                "dataset_id": "semiconductor",
                "variants": [
                    {
                        "lookback_sessions": 10,
                        "status": "ready",
                        "file": "lookback-10.npz",
                        "construction_version": "legacy-branch-cache",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    cache.save(_key(tmp_path, SECTOR_CACHE_SLOT), walk_forward_sector_data(180))

    manifest = cache.status("semiconductor")
    assert [item["lookback_sessions"] for item in manifest["variants"]] == [
        SECTOR_CACHE_SLOT
    ]
    assert manifest["variants"][0]["construction_version"] == CACHE_CONSTRUCTION_VERSION


def test_precompute_builds_one_whole_sector_state_without_topology(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, list[object]] = {
        "fetch": [],
        "normalize": [],
        "execution": [],
        "state": [],
        "save": [],
    }

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

    class FakeStateBuilder:
        def build(self, panel):
            assert hasattr(panel, "timestamps")
            calls["state"].append(panel)
            return walk_forward_sector_data(180)

    class FakeCache:
        def save(self, key, data):
            calls["save"].append(key.lookback_sessions)
            assert data.memberships == ()

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
                "sessions": walk_forward_sector_data(180).states.sessions,
                "hold_returns": walk_forward_sector_data(180).member_next_returns,
                "buy_returns": walk_forward_sector_data(180).member_next_returns,
                "buy_executable": walk_forward_sector_data(180).member_next_valid,
                "sell_returns": np.zeros_like(
                    walk_forward_sector_data(180).member_next_returns
                ),
                "sell_executable": walk_forward_sector_data(180).member_next_valid,
            },
        )(),
    )
    monkeypatch.setattr(module, "SectorDailyStateBuilder", FakeStateBuilder)
    monkeypatch.setattr(module, "BoundaryDataCacheKey", type("KeyFactory", (), {
        "from_source": staticmethod(lambda **kwargs: type("Key", (), kwargs)())
    }))

    result = build_sector_boundary_cache(
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
    assert len(calls["state"]) == 1
    assert calls["save"] == [SECTOR_CACHE_SLOT]
    assert result["rows"] == 180
    assert "lookbacks" not in result
