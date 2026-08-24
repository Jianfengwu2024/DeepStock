# Sector River Topology Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build SectorFlowBayes v2 with a causal monthly synchronization topology, frozen daily river slices, a small Bayesian graph-state predictor for next-session relative return, and an interactive DeepStock Lab river workspace.

**Architecture:** Extend the existing `bnhmm.sector_flow` package and monthly walk-forward pipeline. A target-independent topology builder creates stable monthly bands from the prior 60 sessions; a daily builder projects completed one-minute sessions onto that geometry; a low-dimensional Bayesian graph model predicts next-session cross-sectional relative return. Existing job and artifact boundaries serve a dedicated river visualization page.

**Tech Stack:** Python 3.12, NumPy, SciPy hierarchical clustering/assignment, pytest, FastAPI, vanilla JavaScript/CSS, DuckDB broker-backed data loading.

---

## File Structure

**Create**

- `src/bnhmm/sector_flow/topology.py`: robust pairwise synchronization, uncertainty shrinkage, stable bands, canonical monthly ordering.
- `src/bnhmm/sector_flow/river.py`: fixed monthly stock-by-minute daily slice projection and daily sufficient statistics.
- `src/bnhmm/sector_flow/graph_state.py`: graph-Laplacian daily state posterior and Bayesian relative-return head.
- `src/bnhmm/sector_flow/structure_evaluation.py`: structural baselines, stability metrics, and declared ablations.
- `src/bnhmm/web/static/river.html`: full-width DeepStock river workspace.
- `src/bnhmm/web/static/river.js`: bounded job-scoped data loading and interactions.
- `src/bnhmm/web/static/river.css`: workspace layout, heatmap, topology, migration, inspector, and responsive rules.
- `tests/test_sector_flow_topology.py`: causality, permutation invariance, planted bands, stable labels, and ordering.
- `tests/test_sector_flow_river.py`: slice shape, row freeze, market gaps, masks, and volume invariance.
- `tests/test_sector_flow_graph_state.py`: graph smoothing, uncertainty, fit/predict timing, and target definition.
- `tests/test_sector_flow_structure_evaluation.py`: random baselines, stability, and ablation contracts.
- `tests/test_river_web.py`: API traversal and artifact isolation.

**Modify**

- `pyproject.toml`: add SciPy to the core numerical runtime.
- `src/bnhmm/sector_flow/types.py`: immutable topology, slice, posterior, and prediction contracts.
- `src/bnhmm/sector_flow/pipeline.py`: v2 monthly orchestration, artifacts, reports, and next-session targets.
- `src/bnhmm/sector_flow/evaluation.py`: daily cross-sectional metrics and calibration.
- `src/bnhmm/sector_flow/__init__.py`: export the new public contracts.
- `src/bnhmm/cli.py`: expose the fixed v2 protocol through the existing SectorFlow command.
- `src/bnhmm/web/jobs.py`: request defaults and CLI command arguments for v2.
- `src/bnhmm/web/app.py`: river page and bounded job-scoped artifact endpoints.
- `src/bnhmm/web/static/index.html`: add the `Open river view` action to SectorFlow results.
- `src/bnhmm/web/static/app.js`: bind the selected job to the river workspace URL.
- `src/bnhmm/web/static/styles.css`: style the compact result-page action only.
- Existing SectorFlow and Web tests: update stable contracts without weakening v1 regression coverage.
- `docs/sector-flow-bayes-v2-report.md`: final implementation and sample-out experiment report.

## Task 1: Add Immutable River Contracts

**Files:**
- Modify: `src/bnhmm/sector_flow/types.py`
- Modify: `src/bnhmm/sector_flow/__init__.py`
- Create: `tests/test_sector_flow_topology.py`
- Create: `tests/test_sector_flow_river.py`

- [ ] **Step 1: Write failing shape and causality-contract tests**

```python
def test_monthly_topology_requires_one_value_per_symbol() -> None:
    with pytest.raises(ValueError, match="topology arrays"):
        MonthlyRiverTopology(
            month=np.datetime64("2025-02", "M"),
            training_cutoff=np.datetime64("2025-01-27T15:00"),
            symbols=("A", "B"),
            assigned=np.ones(2, dtype=np.bool_),
            band_ids=np.array([0], dtype=np.int64),
            row_order=np.array([0, 1], dtype=np.int64),
            common_similarity=np.eye(2),
            residual_similarity=np.eye(2),
            fused_similarity=np.eye(2),
            graph=np.eye(2),
            cohesion=np.ones(2),
            centrality=np.ones(2),
            stale=False,
        )


def test_daily_slice_is_symbols_by_minutes() -> None:
    item = DailyRiverSlice(
        session=np.datetime64("2025-02-05", "D"),
        topology_month=np.datetime64("2025-02", "M"),
        symbols=("B", "A"),
        timestamps=np.array(["2025-02-05T09:31", "2025-02-05T09:32"], dtype="datetime64[m]"),
        velocity=np.zeros((2, 2)),
        residual_velocity=np.zeros((2, 2)),
        activity=np.ones((2, 2)),
        valid=np.ones((2, 2), dtype=np.bool_),
    )
    assert item.velocity.shape == (len(item.symbols), len(item.timestamps))
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest tests/test_sector_flow_topology.py tests/test_sector_flow_river.py -q`

Expected: collection fails because `MonthlyRiverTopology` and `DailyRiverSlice` do not exist.

- [ ] **Step 3: Add minimal frozen dataclasses and validation**

```python
@dataclass(frozen=True, slots=True)
class MonthlyRiverTopology:
    month: np.datetime64
    training_cutoff: np.datetime64
    symbols: tuple[str, ...]
    assigned: NDArray[np.bool_]
    band_ids: NDArray[np.int64]
    row_order: NDArray[np.int64]
    common_similarity: NDArray[np.float64]
    residual_similarity: NDArray[np.float64]
    fused_similarity: NDArray[np.float64]
    graph: NDArray[np.float64]
    cohesion: NDArray[np.float64]
    centrality: NDArray[np.float64]
    stale: bool = False

    def __post_init__(self) -> None:
        count = len(self.symbols)
        if any(item.shape != (count,) for item in (self.assigned, self.band_ids, self.row_order, self.cohesion, self.centrality)):
            raise ValueError("topology arrays must contain one value per symbol")
        if any(item.shape != (count, count) for item in (self.common_similarity, self.residual_similarity, self.fused_similarity, self.graph)):
            raise ValueError("topology matrices must be square and symbol-aligned")
        if sorted(self.row_order.tolist()) != list(range(count)):
            raise ValueError("topology row order must be a symbol permutation")


@dataclass(frozen=True, slots=True)
class DailyRiverSlice:
    session: np.datetime64
    topology_month: np.datetime64
    symbols: tuple[str, ...]
    timestamps: NDArray[np.datetime64]
    velocity: NDArray[np.float64]
    residual_velocity: NDArray[np.float64]
    activity: NDArray[np.float64]
    valid: NDArray[np.bool_]

    def __post_init__(self) -> None:
        shape = (len(self.symbols), len(self.timestamps))
        if any(item.shape != shape for item in (self.velocity, self.residual_velocity, self.activity, self.valid)):
            raise ValueError("river slice channels must be symbols by minutes")
```

Add the remaining shared contracts with exact fields:

```python
@dataclass(frozen=True, slots=True)
class DailyGraphState:
    session: np.datetime64
    symbols: tuple[str, ...]
    sector_mean: float
    band_mean: NDArray[np.float64]
    stock_mean: NDArray[np.float64]
    stock_variance: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class GraphStateParameters:
    feature_center: NDArray[np.float64]
    feature_scale: NDArray[np.float64]
    coefficient_mean: NDArray[np.float64]
    coefficient_covariance: NDArray[np.float64]
    residual_variance: float
    graph_strength: float
    training_cutoff: np.datetime64
    model_version: str


@dataclass(frozen=True, slots=True)
class SectorGraphPrediction:
    topology_month: np.datetime64
    symbols: tuple[str, ...]
    expected_relative_return: NDArray[np.float64]
    predictive_variance: NDArray[np.float64]
    signal_time: np.datetime64
    available_at: np.datetime64
    training_cutoff: np.datetime64
    model_version: str
```

Validate one value per symbol, finite nonnegative variances, positive feature scales, and `training_cutoff < signal_time < available_at`.

- [ ] **Step 4: Run focused and existing type tests**

Run: `python -m pytest tests/test_sector_flow_topology.py tests/test_sector_flow_river.py tests/test_sector_flow_panel.py -q`

Expected: all pass.

- [ ] **Step 5: Commit the contracts**

```bash
git add src/bnhmm/sector_flow/types.py src/bnhmm/sector_flow/__init__.py tests/test_sector_flow_topology.py tests/test_sector_flow_river.py
git commit -m "feat: add sector river data contracts"
```

## Task 2: Build Causal Pairwise Synchronization

**Files:**
- Modify: `pyproject.toml`
- Create: `src/bnhmm/sector_flow/topology.py`
- Modify: `tests/test_sector_flow_topology.py`

- [ ] **Step 1: Add failing planted-synchronization, future-mutation, and volume-scale tests**

```python
def test_similarity_recovers_planted_common_and_residual_groups() -> None:
    panel = planted_normalized_panel(sessions=60, symbols=8, seed=7)
    result = RiverTopologyBuilder(TopologyConfig(bootstrap_samples=32)).build(
        panel,
        cutoff=np.datetime64("2025-03-31T15:00"),
        month=np.datetime64("2025-04", "M"),
    )
    assert result.fused_similarity[0, 1] > result.fused_similarity[0, 6]
    assert result.residual_similarity[4, 5] > result.residual_similarity[4, 1]


def test_topology_does_not_change_when_future_rows_change() -> None:
    panel = planted_normalized_panel(sessions=65, symbols=8, seed=11)
    cutoff = np.datetime64("2025-03-31T15:00")
    original = RiverTopologyBuilder().build(panel, cutoff=cutoff, month=np.datetime64("2025-04", "M"))
    changed = mutate_rows_after(panel, cutoff, amount=1000.0)
    repeated = RiverTopologyBuilder().build(changed, cutoff=cutoff, month=np.datetime64("2025-04", "M"))
    np.testing.assert_allclose(original.fused_similarity, repeated.fused_similarity)


def test_uniform_volume_rescaling_does_not_change_similarity() -> None:
    panel = planted_normalized_panel(sessions=60, symbols=8, seed=13)
    scaled = replace(panel, activity=panel.activity * 50.0)
    left = RiverTopologyBuilder().similarities(panel)
    right = RiverTopologyBuilder().similarities(scaled)
    np.testing.assert_allclose(left.fused, right.fused, atol=1e-10)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `python -m pytest tests/test_sector_flow_topology.py -q`

Expected: failures show `RiverTopologyBuilder` and `TopologyConfig` are missing.

- [ ] **Step 3: Implement robust rank synchronization and session-block shrinkage**

Add `scipy>=1.12` to project dependencies. Implement:

```python
@dataclass(frozen=True, slots=True)
class TopologyConfig:
    lookback_sessions: int = 60
    minimum_sessions: int = 40
    minimum_pair_observations: int = 2400
    activity_cap: float = 3.0
    shrinkage_observations: float = 2400.0
    bootstrap_samples: int = 128
    random_seed: int = 1729


def _weighted_spearman(x, y, weights, valid) -> tuple[float, float]:
    indices = np.flatnonzero(valid)
    if len(indices) < 3:
        return 0.0, 0.0
    rx = scipy.stats.rankdata(x[indices])
    ry = scipy.stats.rankdata(y[indices])
    w = weights[indices] / np.maximum(weights[indices].sum(), 1e-12)
    mx, my = np.sum(w * rx), np.sum(w * ry)
    covariance = np.sum(w * (rx - mx) * (ry - my))
    variance = np.sqrt(np.sum(w * np.square(rx - mx)) * np.sum(w * np.square(ry - my)))
    return float(covariance / max(variance, 1e-12)), float(1.0 / np.sum(np.square(w)))
```

Use the existing Bayesian sector factor to form residual velocity. Normalize capped activity weights to their pairwise median before correlation, so multiplying all raw volumes by a constant leaves topology unchanged. Shrink each off-diagonal correlation by `n_eff / (n_eff + shrinkage_observations)` and clip negative values to zero only when creating the graph, not in diagnostic similarity matrices.

- [ ] **Step 4: Run topology, normalizer, and factor tests**

Run: `python -m pytest tests/test_sector_flow_topology.py tests/test_sector_flow_normalization.py tests/test_sector_flow_factor.py -q`

Expected: all pass.

- [ ] **Step 5: Commit synchronization fitting**

```bash
git add pyproject.toml src/bnhmm/sector_flow/topology.py tests/test_sector_flow_topology.py
git commit -m "feat: estimate causal sector synchronization"
```

## Task 3: Form Stable Monthly Bands and Canonical Ordering

**Files:**
- Modify: `src/bnhmm/sector_flow/topology.py`
- Modify: `tests/test_sector_flow_topology.py`

- [ ] **Step 1: Add failing permutation, label-alignment, month-freeze, and planted-band tests**

```python
def test_input_symbol_permutation_preserves_canonical_topology() -> None:
    panel = planted_normalized_panel(sessions=60, symbols=12, seed=19)
    order = np.array([7, 1, 10, 0, 4, 2, 8, 3, 11, 5, 9, 6])
    left = RiverTopologyBuilder().build(panel, cutoff=CUTOFF, month=MONTH)
    right = RiverTopologyBuilder().build(permute_panel(panel, order), cutoff=CUTOFF, month=MONTH)
    assert canonical_band_sets(left) == canonical_band_sets(right)
    assert tuple(left.symbols[index] for index in left.row_order) == tuple(right.symbols[index] for index in right.row_order)


def test_previous_band_ids_and_orientation_remain_stable() -> None:
    first, second = adjacent_month_panels(seed=23)
    april = RiverTopologyBuilder().build(first, cutoff=APRIL_CUTOFF, month=MAY)
    may = RiverTopologyBuilder().build(second, cutoff=MAY_CUTOFF, month=JUNE, previous=april)
    assert overlap_band_ids(april, may) >= 0.8
    assert mean_row_displacement(april, may) < len(april.symbols) / 3
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest tests/test_sector_flow_topology.py -q`

Expected: permutation/order tests fail because no stable clustering or canonical order exists.

- [ ] **Step 3: Implement stable communities, migration evidence, and leaf ordering**

Use `scipy.cluster.hierarchy.linkage`, `fcluster`, and `optimal_leaf_ordering` on `distance = 1 - fused_similarity`. Evaluate candidate band counts from 2 through `min(8, assigned_count // 5)` with session-block bootstrap silhouette. Apply the one-standard-error rule: retain the previous count when statistically tied, otherwise choose the smaller tied count. Match band IDs with `scipy.optimize.linear_sum_assignment` on negative member overlap.

```python
def _canonical_order(distance, band_ids, symbols, previous):
    rows: list[int] = []
    for band in stable_band_sequence(band_ids, symbols, previous):
        members = np.flatnonzero(band_ids == band)
        if len(members) == 1:
            ordered = members
        else:
            condensed = scipy.spatial.distance.squareform(distance[np.ix_(members, members)], checks=False)
            tree = scipy.cluster.hierarchy.linkage(condensed, method="average")
            tree = scipy.cluster.hierarchy.optimal_leaf_ordering(tree, condensed)
            ordered = members[scipy.cluster.hierarchy.leaves_list(tree)]
        rows.extend(orient_against_previous(ordered, symbols, previous))
    rows.extend(sorted_unassigned_indices(band_ids, symbols))
    return np.asarray(rows, dtype=np.int64)
```

Require the block-bootstrap confidence interval for synchronization gain to be positive before moving an overlapping stock to another band. Keep insufficient-history symbols at band `-1` and at the bottom of the row order.

- [ ] **Step 4: Run the full topology test file twice to verify determinism**

Run: `python -m pytest tests/test_sector_flow_topology.py -q && python -m pytest tests/test_sector_flow_topology.py -q`

Expected: both runs pass with identical canonical output.

- [ ] **Step 5: Commit monthly topology**

```bash
git add src/bnhmm/sector_flow/topology.py tests/test_sector_flow_topology.py
git commit -m "feat: build stable monthly sector flow bands"
```

## Task 4: Build Frozen Daily River Slices

**Files:**
- Create: `src/bnhmm/sector_flow/river.py`
- Modify: `tests/test_sector_flow_river.py`

- [ ] **Step 1: Add failing row-order, mask, gap, and residual tests**

```python
def test_slice_uses_topology_order_and_real_minute_grid() -> None:
    panel, flow, topology = river_fixture()
    result = DailyRiverSliceBuilder().build(panel, factor_parameters, flow, topology, np.datetime64("2025-04-08", "D"))
    assert result.symbols == tuple(topology.symbols[index] for index in topology.row_order)
    assert result.velocity.shape == (len(topology.symbols), 240)
    assert np.datetime64("2025-04-08T11:31") not in result.timestamps
    assert np.datetime64("2025-04-08T13:00") not in result.timestamps


def test_missing_bar_stays_masked_and_does_not_bridge_gap() -> None:
    panel, flow, topology = river_fixture(missing=("A", "2025-04-08T10:15"))
    result = DailyRiverSliceBuilder().build(panel, factor_parameters, flow, topology, np.datetime64("2025-04-08", "D"))
    row = result.symbols.index("A")
    minute = np.flatnonzero(result.timestamps == np.datetime64("2025-04-08T10:15"))[0]
    assert not result.valid[row, minute]
    assert np.isnan(result.velocity[row, minute])
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest tests/test_sector_flow_river.py -q`

Expected: failures show `DailyRiverSliceBuilder` is missing.

- [ ] **Step 3: Implement slice projection and fixed daily summaries**

```python
class DailyRiverSliceBuilder:
    def build(self, panel, factor_parameters, flow, topology, session) -> DailyRiverSlice:
        if session.astype("datetime64[M]") != topology.month:
            raise ValueError("session does not belong to the frozen topology month")
        rows = np.flatnonzero(panel.timestamps.astype("datetime64[D]") == session)
        index = topology.row_order
        velocity = panel.velocity[rows][:, index].T
        valid = panel.valid[rows][:, index].T
        activity = panel.activity[rows][:, index].T
        loadings = factor_parameters.loadings[index]
        residual = velocity - loadings[:, None] * flow.factor_mean[rows][None, :]
        residual[~valid] = np.nan
        activity[~panel.activity_valid[rows][:, index].T] = np.nan
        return DailyRiverSlice(session, topology.month, ordered_symbols(topology), panel.timestamps[rows], velocity, residual, activity, valid)
```

Add `daily_statistics(slice, topology)` returning a fixed six-column matrix per stock: normalized displacement, band displacement, stock-minus-band displacement, path efficiency, residual energy, and median relative-volume density. It must consume the full 240-minute slice without ten-minute aggregation.

- [ ] **Step 4: Run slice, panel, and normalization tests**

Run: `python -m pytest tests/test_sector_flow_river.py tests/test_sector_flow_panel.py tests/test_sector_flow_normalization.py -q`

Expected: all pass.

- [ ] **Step 5: Commit daily slices**

```bash
git add src/bnhmm/sector_flow/river.py tests/test_sector_flow_river.py
git commit -m "feat: project daily sector river slices"
```

## Task 5: Add the Small Bayesian Graph-State Predictor

**Files:**
- Create: `src/bnhmm/sector_flow/graph_state.py`
- Create: `tests/test_sector_flow_graph_state.py`

- [ ] **Step 1: Add failing graph-smoothing, uncertainty, and next-session tests**

```python
def test_graph_prior_smooths_connected_stocks_more_than_disconnected_stocks() -> None:
    topology = chain_topology(weights=(0.9, 0.0))
    state = SectorGraphStateModel().infer_state(topology, daily_features(values=(1.0, 0.0, -1.0)))
    assert abs(state.stock_mean[0] - state.stock_mean[1]) < abs(state.stock_mean[1] - state.stock_mean[2])


def test_activity_changes_uncertainty_not_direction() -> None:
    topology = chain_topology(weights=(0.8, 0.8))
    low = SectorGraphStateModel().infer_state(topology, daily_features(activity=0.5))
    high = SectorGraphStateModel().infer_state(topology, daily_features(activity=2.0))
    np.testing.assert_array_equal(np.sign(low.stock_mean), np.sign(high.stock_mean))
    assert np.all(high.stock_variance <= low.stock_variance)


def test_prediction_is_available_only_next_session() -> None:
    prediction = fitted_graph_model().predict(topology_fixture(), slice_fixture("2025-04-08"))
    assert prediction.signal_time == np.datetime64("2025-04-08T15:00")
    assert prediction.available_at == np.datetime64("2025-04-09T09:31")
    assert prediction.training_cutoff < prediction.signal_time
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest tests/test_sector_flow_graph_state.py -q`

Expected: failures show `SectorGraphStateModel` is missing.

- [ ] **Step 3: Implement graph posterior and Bayesian ridge head**

For each completed day, compute one graph posterior rather than solving an 84-stock system at every minute:

```python
laplacian = np.diag(graph.sum(axis=1)) - graph
observation_precision = np.diag(np.clip(activity_reliability, 0.25, 4.0))
posterior_precision = observation_precision + graph_strength * laplacian + ridge * np.eye(symbols)
posterior_covariance = np.linalg.inv(posterior_precision)
posterior_mean = posterior_covariance @ observation_precision @ observed_departure
```

Fit a conjugate Bayesian ridge head on training stock-days only. The design matrix has six fixed daily statistics plus graph posterior mean, posterior standard deviation, cohesion, and centrality. Standardize columns using training data only. Store coefficient mean/covariance, residual variance, feature center/scale, graph strength, model version, and training cutoff in `GraphStateParameters`.

The target is:

```python
stock_next_return = next_close / current_close - 1.0
next_relative_return = stock_next_return - np.nanmean(stock_next_return)
```

Prediction variance includes coefficient uncertainty, graph-state uncertainty, and residual variance. Do not tune feature count or graph strength on sample-out folds.

- [ ] **Step 4: Run graph-state and numerical regression tests**

Run: `python -m pytest tests/test_sector_flow_graph_state.py tests/test_sector_flow_factor.py -q`

Expected: all pass and no linear-algebra warning is emitted.

- [ ] **Step 5: Commit the graph model**

```bash
git add src/bnhmm/sector_flow/graph_state.py tests/test_sector_flow_graph_state.py
git commit -m "feat: infer Bayesian sector graph states"
```

## Task 6: Integrate Monthly Walk-forward and Structural Evaluation

**Files:**
- Create: `src/bnhmm/sector_flow/structure_evaluation.py`
- Modify: `src/bnhmm/sector_flow/evaluation.py`
- Modify: `src/bnhmm/sector_flow/pipeline.py`
- Modify: `src/bnhmm/cli.py`
- Modify: `tests/test_sector_flow_pipeline.py`
- Modify: `tests/test_sector_flow_evaluation.py`
- Create: `tests/test_sector_flow_structure_evaluation.py`

- [ ] **Step 1: Add failing monthly-freeze, target-alignment, artifact, and ablation tests**

```python
def test_walk_forward_builds_one_topology_per_sample_out_month(tmp_path) -> None:
    result = run_synthetic_v2(tmp_path, months=4)
    assert len(result.topologies) == len(result.folds)
    for fold, topology in zip(result.folds, result.topologies, strict=True):
        assert topology.month == fold.test_start.astype("datetime64[M]")
        assert topology.training_cutoff < fold.test_start
        assert all(item.topology_month == topology.month for item in fold.predictions)


def test_pipeline_writes_compact_river_artifacts(tmp_path) -> None:
    run_synthetic_v2(tmp_path, months=3)
    assert (tmp_path / "river" / "manifest.json").is_file()
    assert list((tmp_path / "river" / "topologies").glob("*.npz"))
    assert list((tmp_path / "river" / "slices").glob("*.npz"))
    assert not (tmp_path / "predictions.csv").is_file()


def test_declared_ablations_share_identical_oos_dates() -> None:
    report = evaluate_structure(ablation_fixture())
    assert set(report.variants) == {"v1", "random", "common", "residual", "unstable", "full"}
    assert one_oos_date_set(report)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `python -m pytest tests/test_sector_flow_pipeline.py tests/test_sector_flow_evaluation.py tests/test_sector_flow_structure_evaluation.py -q`

Expected: new v2 tests fail because topologies, daily targets, and river artifacts are absent.

- [ ] **Step 3: Extend the existing pipeline without replacing v1 contracts**

Add a versioned `SectorFlowConfig(protocol="river-topology-v2", topology_lookback_sessions=60)` branch. In every fold:

```python
topology = topology_builder.build(
    normalized,
    cutoff=fold.training_cutoff,
    month=fold.test_start.astype("datetime64[M]"),
    previous=previous_topology,
)
parameters = graph_model.fit(training_slices, training_targets, training_topologies, cutoff=fold.training_cutoff)
for session in sample_out_sessions(fold):
    river_slice = slice_builder.build(normalized, flow, topology, session)
    prediction = graph_model.predict(parameters, topology, river_slice, next_session(session))
```

Write one compressed `.npz` topology per month and one compressed `.npz` slice per day. Keep `manifest.json`, `evaluation.json`, and `sector-flow-v2-report.md` compact. Calculate daily Rank IC and equal-count Top-Bottom portfolios, with policy and costs in evaluation only.

Implement structural evaluation for within-band lift over matched random groups, silhouette/modularity, adjusted Rand stability, migration rate, unassigned rate, and coverage. Use daily-block effective sample size and held-out calibration.

- [ ] **Step 4: Run SectorFlow regression and full local tests**

Run: `python -m pytest tests/test_sector_flow_*.py -q`

Expected: all SectorFlow v1 and v2 tests pass.

- [ ] **Step 5: Commit walk-forward integration**

```bash
git add src/bnhmm/sector_flow src/bnhmm/cli.py tests/test_sector_flow_pipeline.py tests/test_sector_flow_evaluation.py tests/test_sector_flow_structure_evaluation.py
git commit -m "feat: run sector river walk-forward evaluation"
```

## Task 7: Expose v2 Configuration Through the Existing Job System

**Files:**
- Modify: `src/bnhmm/web/jobs.py`
- Modify: `src/bnhmm/web/app.py`
- Modify: `tests/test_web_jobs.py`
- Create: `tests/test_river_web.py`

- [ ] **Step 1: Add failing request and bounded-artifact API tests**

```python
def test_sector_flow_v2_command_has_fixed_topology_protocol() -> None:
    request = WalkForwardJobRequest(strategy="sector-flow-bayes", sector_flow_protocol="river-topology-v2")
    command = build_command(request, Path("/tmp/run"))
    assert command[command.index("--protocol") + 1] == "river-topology-v2"
    assert command[command.index("--topology-lookback-sessions") + 1] == "60"


def test_river_artifact_cannot_escape_job_directory(client, completed_river_job) -> None:
    ok = client.get(f"/api/jobs/{completed_river_job.id}/river/topology/2025-04")
    escaped = client.get(f"/api/jobs/{completed_river_job.id}/river/topology/../job.json")
    assert ok.status_code == 200
    assert escaped.status_code in {404, 422}


def test_missing_day_slice_returns_not_found(client, completed_river_job) -> None:
    response = client.get(f"/api/jobs/{completed_river_job.id}/river/slice/1999-01-01")
    assert response.status_code == 404
```

- [ ] **Step 2: Run the Web tests and verify RED**

Run: `python -m pytest tests/test_web_jobs.py tests/test_river_web.py -q`

Expected: request field and river endpoints are missing.

- [ ] **Step 3: Add fixed defaults and safe artifact lookup**

Add request fields with narrow validation:

```python
sector_flow_protocol: Literal["legacy-v1", "river-topology-v2"] = "river-topology-v2"
topology_lookback_sessions: Literal[60] = 60
```

Add `JobManager.river_artifact_path(identifier, kind, key)` that permits only manifest-declared filenames resolved under `<job>/river`. Add routes:

```python
@app.get("/jobs/{identifier}/river", include_in_schema=False)
def river_page(identifier: str, request: Request) -> FileResponse: ...

@app.get("/api/jobs/{identifier}/river/topology/{month}")
def river_topology(identifier: str, month: str, request: Request) -> JSONResponse: ...

@app.get("/api/jobs/{identifier}/river/slice/{session}")
def river_slice(identifier: str, session: str, request: Request) -> JSONResponse: ...

@app.get("/api/jobs/{identifier}/river/migration")
def river_migration(identifier: str, request: Request) -> JSONResponse: ...

@app.get("/api/jobs/{identifier}/river/symbol/{symbol}")
def river_symbol(identifier: str, symbol: str, request: Request) -> JSONResponse: ...
```

The routes resolve only manifest-listed `.npz` files, load the requested one, and convert its arrays to bounded JSON for the browser. They never return all daily slices in one response.

- [ ] **Step 4: Run all job/API tests**

Run: `python -m pytest tests/test_web_jobs.py tests/test_web_config.py tests/test_river_web.py -q`

Expected: all pass.

- [ ] **Step 5: Commit job and API integration**

```bash
git add src/bnhmm/web/jobs.py src/bnhmm/web/app.py tests/test_web_jobs.py tests/test_river_web.py
git commit -m "feat: serve job-scoped sector river artifacts"
```

## Task 8: Build the DeepStock River Workspace

**Files:**
- Create: `src/bnhmm/web/static/river.html`
- Create: `src/bnhmm/web/static/river.js`
- Create: `src/bnhmm/web/static/river.css`
- Modify: `src/bnhmm/web/static/index.html`
- Modify: `src/bnhmm/web/static/app.js`
- Modify: `src/bnhmm/web/static/styles.css`
- Modify: `tests/test_web_static.py`

- [ ] **Step 1: Add failing static-contract tests**

```python
def test_river_workspace_has_four_views_and_stock_inspector() -> None:
    html = (ROOT / "river.html").read_text(encoding="utf-8")
    script = (ROOT / "river.js").read_text(encoding="utf-8")
    for identifier in ("river-heatmap", "topology-view", "migration-view", "structure-view", "stock-inspector"):
        assert f'id="{identifier}"' in html
    assert "loadTopology" in script
    assert "loadSlice" in script
    assert "renderStockInspector" in script


def test_sector_flow_result_links_selected_job_to_river_workspace() -> None:
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "app.js").read_text(encoding="utf-8")
    assert 'id="open-river-view"' in html
    assert "/jobs/${encodeURIComponent(job.id)}/river" in script
```

- [ ] **Step 2: Run static tests and verify RED**

Run: `python -m pytest tests/test_web_static.py -q`

Expected: failures show the workspace files and stable hooks are absent.

- [ ] **Step 3: Implement the approved full-width workspace**

Build the four-tab page approved in the visual prototype. Use Canvas for the stock-by-minute heatmap, SVG for the bounded monthly graph and migration view, and semantic HTML for diagnostics and the inspector. Keep one day and one month in memory.

```javascript
const riverState = { jobId: jobIdFromPath(), month: null, session: null, topology: null, slice: null, selectedSymbol: null };

async function loadTopology(month) {
  riverState.topology = await api(`/api/jobs/${encodeURIComponent(riverState.jobId)}/river/topology/${encodeURIComponent(month)}`);
  renderTopology(riverState.topology);
  renderBandBoundaries(riverState.topology);
}

async function loadSlice(session) {
  riverState.slice = await api(`/api/jobs/${encodeURIComponent(riverState.jobId)}/river/slice/${encodeURIComponent(session)}`);
  renderHeatmap(riverState.slice, selectedChannel());
}
```

Render signed velocity with a diverging green/neutral/red scale, activity density as optional opacity, missing cells in neutral gray, and crisp band separators. Hover displays raw normalized value, band mean, and mask state. Click locks the row and fills the stock inspector. Show topology cutoff and stale state at all times.

- [ ] **Step 4: Run static and API tests**

Run: `python -m pytest tests/test_web_static.py tests/test_river_web.py -q`

Expected: all pass.

- [ ] **Step 5: Commit the river UI**

```bash
git add src/bnhmm/web/static tests/test_web_static.py
git commit -m "feat: add DeepStock sector river workspace"
```

## Task 9: Browser QA and Responsive Verification

**Files:**
- Inspect and, only after a reproduced failure, modify: `src/bnhmm/web/static/river.html`, `river.js`, or `river.css`
- Store desktop and mobile screenshot evidence in the experiment artifact directory, not Git.

- [ ] **Step 1: Start DeepStock Lab with a deterministic fixture job**

Run: `python -m bnhmm.web.app`

Expected: `/health` returns `{"status":"ok"}` and the fixture job exposes one topology and one daily slice.

- [ ] **Step 2: Verify desktop behavior at 1440x900**

Open `/jobs/<fixture-job-id>/river`. Verify the heatmap is nonblank; all 240 minute columns fit without horizontal page overflow; flow-band lines align to rows; tooltips stay inside the viewport; selecting a row updates the inspector; all four tabs render.

- [ ] **Step 3: Verify mobile behavior at 390x844**

Verify the heatmap remains scrollable inside its own region, controls wrap without text overlap, the inspector moves below the main view, and no page-level horizontal overflow appears.

- [ ] **Step 4: Verify artifact failures visibly**

Delete one fixture day artifact, reload that date, and confirm the page shows an inline `该交易日切片不可用` state while month navigation and topology remain usable.

- [ ] **Step 5: Commit only evidence-backed UI fixes**

```bash
git add src/bnhmm/web/static
git commit -m "fix: harden sector river workspace rendering"
```

If no defect is found, do not create an empty commit.

## Task 10: Full Verification, Server Experiment, and Report

**Files:**
- Create: `docs/sector-flow-bayes-v2-report.md`
- Modify: `README.md` only to link the finished v2 report and river workspace.

- [ ] **Step 1: Run the complete local suite**

Run: `python -m pytest -q`

Expected: every test passes.

- [ ] **Step 2: Run topology determinism and causality checks separately**

Run: `python -m pytest tests/test_sector_flow_topology.py tests/test_sector_flow_river.py tests/test_sector_flow_graph_state.py -q --maxfail=1`

Expected: all pass on two consecutive runs.

- [ ] **Step 3: Deploy the committed branch to server 145 and run the semiconductor experiment**

Use the existing server repository `/home/wujianfeng/private/bnhmm-engine`, Python 3.12 environment, and `/home/share/data/sectors/semiconductor.duckdb`. Run 2024 training history with monthly sample-out predictions through 2025 using `river-topology-v2`, fixed 60-session topology windows, and the declared six ablations.

Expected artifacts: monthly topology files, daily slices, migration data, structural report, evaluation JSON, and `sector-flow-v2-report.md`.

- [ ] **Step 4: Compare v2 against every declared baseline**

Record Rank IC, gross and net Top-Bottom return, turnover, Brier score, log loss, calibration error, posterior coverage, residual autocorrelation, within-band synchronization lift, stability, migration, unassigned rate, runtime, and storage size. Report every fold; do not select favorable months.

- [ ] **Step 5: Write the final report and apply rejection criteria**

The report must contain: implementation summary, causal timeline, topology examples, structural metrics, all six ablations, predictive metrics, calibration plots, browser screenshots, runtime/storage, limitations, and a clear `research-only` or `eligible-for-publishing` verdict. A failure stays documented as a negative result.

- [ ] **Step 6: Run final verification and commit documentation**

Run: `python -m pytest -q && git diff --check`

Expected: tests pass and no whitespace errors are reported.

```bash
git add docs/sector-flow-bayes-v2-report.md README.md
git commit -m "docs: report sector river topology experiment"
```

## Completion Gate

Implementation is complete only when:

- all test-first cycles have been observed RED then GREEN;
- the monthly topology is causal, deterministic, permutation-invariant, and frozen;
- DeepStock displays real topology and daily data at desktop and mobile sizes;
- server 145 completes the full walk-forward and all ablations;
- the final report applies the rejection criteria honestly;
- TriadMind sync, verify, and governance checks pass or every remaining warning is documented.
