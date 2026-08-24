# BoundaryFlowFK Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. The user explicitly prohibited subagents and multi-agent development.

**Goal:** Build a causal, low-parameter BoundaryFlowFK strategy that compresses semiconductor constituent minute flow into named boundary-layer states, predicts the next-session real sector-index return distribution, and exposes five fixed ablations in DeepStock Lab.

**Architecture:** Keep all new model code in `bnhmm.boundary_fk`. Reuse sector panel alignment, monthly topology, ranking, job persistence, and web infrastructure, but do not insert BoundaryFlowFK branches into `sector_flow/pipeline.py`. Fit every projection, normalization, branch topology, threshold, and posterior inside each walk-forward fold; the UI and CLI only orchestrate the independent pipeline.

**Tech Stack:** Python 3.12, NumPy, SciPy, DuckDB/broker, FastAPI/Pydantic, vanilla JavaScript/CSS, pytest. No PyTorch, CNN, Transformer, PyMC, or new model runtime dependency.

---

### Task 1: Real 881121 minute-index data gate

**Files:**
- Create: `scripts/sync_sector_index_1m.py`
- Modify: `src/bnhmm/data/source.py`
- Test: `tests/test_sector_index_source.py`

- [ ] **Step 1: Write failing minute-index source tests**

```python
def test_duckdb_source_fetches_sector_index_minutes(tmp_path):
    database = make_sector_database(tmp_path, table="sector_index_1m")
    bars = DuckDBBarSource(str(database)).fetch_sector_index_minutes(
        "semiconductor", "881121", "2024-01-02", "2024-01-03"
    )
    assert bars.symbol == "881121"
    assert len(bars) == 240

def test_missing_minute_index_is_not_replaced_by_daily_data(tmp_path):
    database = make_sector_database(tmp_path, table="sector_index_1d")
    with pytest.raises(ValueError, match="一分钟指数数据"):
        DuckDBBarSource(str(database)).fetch_sector_index_minutes(
            "semiconductor", "881121", "2024-01-02", "2024-01-03"
        )
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_sector_index_source.py -q`

Expected: FAIL because `fetch_sector_index_minutes` and the `sector_index_1m` import path do not exist.

- [ ] **Step 3: Implement the explicit minute-index contract and idempotent transfer tool**

```python
def fetch_sector_index_minutes(
    self, dataset_id: str, symbol: str, start: DateLike, end: DateLike
) -> BarBatch:
    return self._fetch_sector_index_table(
        "sector_index_1m", dataset_id, symbol, start, end,
        missing_message="缺少真实板块一分钟指数数据",
    )
```

The sync script must copy only `dataset_id=semiconductor, symbol=881121`, preserve timestamps and OHLCV, upsert on `(dataset_id, symbol, timestamp)`, and print source/target row counts, first timestamp, last timestamp, duplicate count, and missing trading-minute count. It must refuse to open a broker-owned DuckDB while the broker is running.

- [ ] **Step 4: Transfer and verify data before model work**

Run on 144 under a maintenance window, transfer the exported Parquet to 145, then import to `/home/share/data/market.duckdb`. Verify with:

```sql
SELECT count(*), min(timestamp), max(timestamp), count(DISTINCT timestamp)
FROM sector_index_1m
WHERE dataset_id = 'semiconductor' AND symbol = '881121';
```

Expected: non-zero minute rows, `count(*) = count(DISTINCT timestamp)`, and coverage spanning the intended 2024-2025 experiment range. Stop here if this gate fails.

- [ ] **Step 5: Re-run tests and commit**

Run: `PYTHONPATH=src python -m pytest tests/test_sector_index_source.py -q`

Commit: `git commit -m "feat: add real sector minute index source"`

### Task 2: Boundary contracts, causal normalization, and frozen index projection

**Files:**
- Create: `src/bnhmm/boundary_fk/__init__.py`
- Create: `src/bnhmm/boundary_fk/types.py`
- Create: `src/bnhmm/boundary_fk/normalization.py`
- Create: `src/bnhmm/boundary_fk/index_projection.py`
- Test: `tests/test_boundary_fk_normalization.py`
- Test: `tests/test_boundary_fk_index_projection.py`

- [ ] **Step 1: Write failing causality and projection tests**

```python
def test_normalization_uses_only_prior_complete_sessions():
    base = normalize(panel_through("2024-03-01"))
    changed = normalize(panel_with_future_volume_shock("2024-03-04"))
    np.testing.assert_allclose(base.activity[:-240], changed.activity[:-240], equal_nan=True)

def test_projection_is_nonnegative_normalized_and_monthly_frozen():
    projection = IndexProjectionBuilder(lookback_sessions=60).fit(panel, index, cutoff)
    assert np.all(projection.weights >= 0.0)
    assert projection.weights.sum() == pytest.approx(1.0)
    assert projection.training_cutoff < projection.effective_month
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `PYTHONPATH=src python -m pytest tests/test_boundary_fk_normalization.py tests/test_boundary_fk_index_projection.py -q`

- [ ] **Step 3: Implement immutable public contracts**

```python
@dataclass(frozen=True, slots=True)
class IndexProjection:
    symbols: tuple[str, ...]
    weights: FloatArray
    effective_month: np.datetime64
    training_cutoff: np.datetime64
    source: Literal["published", "minute-fit"]
    normalized_rmse: float

@dataclass(frozen=True, slots=True)
class BoundaryMinuteMoments:
    timestamps: TimeArray
    water: FloatArray
    momentum: FloatArray
    velocity: FloatArray
    coherence: FloatArray
    turbulence: FloatArray
    breadth: FloatArray
    concentration: FloatArray
    decoupling: FloatArray
    thickness: FloatArray
    viscosity: FloatArray
    valid: BoolArray
```

Implement the design formulas exactly: price velocity is one-minute log return; its RMS uses the prior 10 complete sessions; relative volume is current minute volume divided by prior 10-session mean daily volume divided by 240. Current-session values never update their own baseline. Fit minute projection weights with non-negative constrained least squares plus previous-month shrinkage; freeze them for the whole month and fail when normalized RMSE exceeds 0.50.

- [ ] **Step 4: Run focused tests and commit**

Run: `PYTHONPATH=src python -m pytest tests/test_boundary_fk_normalization.py tests/test_boundary_fk_index_projection.py -q`

Commit: `git commit -m "feat: add causal boundary normalization and projection"`

### Task 3: Minute fluid moments and physical invariants

**Files:**
- Create: `src/bnhmm/boundary_fk/moments.py`
- Test: `tests/test_boundary_fk_moments.py`

- [ ] **Step 1: Write failing formula and synthetic-field tests**

```python
def test_high_volume_same_direction_flow_has_high_coherence():
    moments = extract(synthetic_flow(direction=1, activity=4.0, dissent=0.0))
    assert moments.momentum[-1] > 0.0
    assert moments.coherence[-1] > 0.9
    assert moments.turbulence[-1] < 0.05

def test_index_move_without_density_support_increases_decoupling():
    weak = extract(synthetic_index_jump(activity=0.1))
    strong = extract(synthetic_index_jump(activity=4.0))
    assert abs(weak.decoupling[-1]) > abs(strong.decoupling[-1])
```

- [ ] **Step 2: Run tests and verify failure**

Run: `PYTHONPATH=src python -m pytest tests/test_boundary_fk_moments.py -q`

- [ ] **Step 3: Implement `BoundaryFieldExtractor.extract`**

Compute `M, J, u, C, T, B, H, N_eff, D, delta, delta_slope, nu` using the formulas in the approved design. Preserve masks, never bridge lunch/cross-day gaps, require a configurable minimum valid-symbol fraction, use training-fold `q0.25` for the thickness stabilizer, and reject every NaN or infinity before returning.

- [ ] **Step 4: Run tests and commit**

Run: `PYTHONPATH=src python -m pytest tests/test_boundary_fk_moments.py -q`

Commit: `git commit -m "feat: extract boundary flow moments"`

### Task 4: Monthly frozen branches and daily state compression

**Files:**
- Create: `src/bnhmm/boundary_fk/daily_state.py`
- Modify: `src/bnhmm/boundary_fk/types.py`
- Test: `tests/test_boundary_fk_daily_state.py`

- [ ] **Step 1: Write failing tests for monthly freezing, permutation invariance, and feature limit**

```python
def test_branch_membership_is_frozen_inside_month():
    states = build_states(panel, cutoff=np.datetime64("2024-03-31"))
    assert states.branch_ids("2024-04-02") == states.branch_ids("2024-04-29")

def test_daily_schema_is_named_and_bounded():
    state = build_states(panel).daily[-1]
    assert len(state.feature_names) <= 36
    assert len(state.feature_names) == len(set(state.feature_names))
    assert all(name and unit for name, unit in zip(state.feature_names, state.units))
```

- [ ] **Step 2: Run tests and verify failure**

Run: `PYTHONPATH=src python -m pytest tests/test_boundary_fk_daily_state.py -q`

- [ ] **Step 3: Implement branch summaries and daily compression**

Reuse `RiverTopologyBuilder` with a 60-session lookback and canonical monthly ordering. Default to four branches, permit three to five, and mark `branch_degraded` rather than duplicating unstable branches. Compress the main flow to close, full-day integral/variance, last-30/last-60 density-weighted means, and high-coherence/high-turbulence occupancy; retain only branch close contribution, cumulative contribution, and last-60 coherence. Assert the final matrix width is at most 36.

- [ ] **Step 4: Run tests and commit**

Run: `PYTHONPATH=src python -m pytest tests/test_boundary_fk_daily_state.py -q`

Commit: `git commit -m "feat: build frozen branch boundary states"`

### Task 5: Bayesian drift, diffusion, and effective potential

**Files:**
- Create: `src/bnhmm/boundary_fk/bayesian_state.py`
- Test: `tests/test_boundary_fk_bayesian_state.py`

- [ ] **Step 1: Write failing posterior tests**

```python
def test_fit_rejects_unmatured_targets():
    with pytest.raises(ValueError, match="target_time"):
        BayesianBoundaryStateModel().fit(states, labels_after(training_cutoff))

def test_posterior_covariances_are_positive_semidefinite():
    posterior = BayesianBoundaryStateModel().fit(states, labels)
    assert np.linalg.eigvalsh(posterior.drift_covariance).min() >= -1e-10
    assert posterior.diffusion_variance > 0.0
```

- [ ] **Step 2: Run tests and verify failure**

Run: `PYTHONPATH=src python -m pytest tests/test_boundary_fk_bayesian_state.py -q`

- [ ] **Step 3: Implement low-parameter posterior fitting**

Use conjugate Gaussian shrinkage for drift; a low-dimensional log-variance regression with a Gaussian prior and Laplace covariance for diffusion; and Bayesian binomial regression with Laplace covariance for survival/effective potential. Standardize features only on each training fold. Add bounded Cholesky jitter once, then fail loudly. Do not add a hidden neural layer or unnamed latent embedding.

- [ ] **Step 4: Run tests and commit**

Run: `PYTHONPATH=src python -m pytest tests/test_boundary_fk_bayesian_state.py -q`

Commit: `git commit -m "feat: fit Bayesian boundary dynamics"`

### Task 6: Feynman-Kac forecasts, policy, and evaluation

**Files:**
- Create: `src/bnhmm/boundary_fk/feynman_kac.py`
- Create: `src/bnhmm/boundary_fk/policy.py`
- Create: `src/bnhmm/boundary_fk/evaluation.py`
- Test: `tests/test_boundary_fk_feynman_kac.py`
- Test: `tests/test_boundary_fk_policy.py`
- Test: `tests/test_boundary_fk_evaluation.py`

- [ ] **Step 1: Write failing distribution, uncertainty, and gate tests**

```python
def test_raw_probabilities_are_normalized_and_kept_separate_from_survival():
    forecast = forecaster.predict(posterior, state, seed=7)
    assert sum(forecast.raw_state_probabilities) == pytest.approx(1.0)
    assert sum(forecast.survival_joint_probabilities) == pytest.approx(
        forecast.trend_survival_probability
    )

def test_policy_calls_ranker_only_after_all_bullish_gates_pass(mocker):
    ranker = mocker.Mock()
    decision = BoundaryPolicy(ranker=ranker).decide(confident_bullish_forecast())
    assert decision.open_position
    ranker.rank.assert_called_once()
```

- [ ] **Step 2: Run tests and verify failure**

Run: `PYTHONPATH=src python -m pytest tests/test_boundary_fk_feynman_kac.py tests/test_boundary_fk_policy.py tests/test_boundary_fk_evaluation.py -q`

- [ ] **Step 3: Implement forecasts and metrics**

Sample posterior parameters and return noise separately so epistemic and aleatoric variances remain identifiable. Store raw three-state probabilities, survival probability, and joint direction-and-survival mass as different fields. Apply the approved five gates before calling `SourceProximityRanker`; trade next-session open to close, long-only, Top-K equal weight, 10 BP default cost. Evaluate Log Loss, Brier, ECE, CRPS, coverage, balanced accuracy, Macro F1, residual autocorrelation, return, drawdown, turnover, open rate, and concentration of profits across trade days.

- [ ] **Step 4: Run tests and commit**

Run: `PYTHONPATH=src python -m pytest tests/test_boundary_fk_feynman_kac.py tests/test_boundary_fk_policy.py tests/test_boundary_fk_evaluation.py -q`

Commit: `git commit -m "feat: forecast and evaluate BoundaryFlowFK"`

### Task 7: Fixed five-stage walk-forward pipeline and Chinese artifacts

**Files:**
- Create: `src/bnhmm/boundary_fk/pipeline.py`
- Create: `src/bnhmm/boundary_fk/reporting.py`
- Modify: `src/bnhmm/boundary_fk/__init__.py`
- Test: `tests/test_boundary_fk_pipeline.py`

- [ ] **Step 1: Write failing end-to-end artifact and stop-gate tests**

```python
def test_pipeline_writes_all_named_ablation_reports(tmp_path):
    result = BoundaryFlowPipeline(config).run(source, tmp_path)
    assert result.model_names == (
        "BFK-GBM", "BFK-Global", "BFK-Boundary", "BFK-Branches", "BoundaryFlowFK"
    )
    assert (tmp_path / "boundary-ablation-report.md").is_file()

def test_failed_stage_prevents_later_stage_promotion(tmp_path):
    result = BoundaryFlowPipeline(strict_failure_fixture()).run(source, tmp_path)
    assert result.stages[1].passed is False
    assert all(stage.status == "skipped" for stage in result.stages[2:])
```

- [ ] **Step 2: Run tests and verify failure**

Run: `PYTHONPATH=src python -m pytest tests/test_boundary_fk_pipeline.py -q`

- [ ] **Step 3: Implement causal monthly walk-forward orchestration**

Run `BFK-GBM -> BFK-Global -> BFK-Boundary -> BFK-Branches -> BoundaryFlowFK` with identical folds and seeds. Require at least 120 mature training sessions. Freeze projection, topology, scales, thresholds, and feature schema inside every fold. Stop progression when Brier/Log Loss or calibration gates fail. Write the nine named artifacts from the design plus PNGs for distribution, calibration, residuals, equity, and drawdown. Every prediction includes `signal_time`, `available_at`, `target_time`, `training_cutoff`, and `model_version`.

- [ ] **Step 4: Run tests and commit**

Run: `PYTHONPATH=src python -m pytest tests/test_boundary_fk_pipeline.py -q`

Commit: `git commit -m "feat: add BoundaryFlowFK walk-forward ablations"`

### Task 8: CLI and persistent job integration

**Files:**
- Modify: `src/bnhmm/cli.py`
- Modify: `src/bnhmm/web/jobs.py`
- Modify: `src/bnhmm/web/app.py`
- Test: `tests/test_boundary_fk_cli.py`
- Test: `tests/test_web_jobs.py`
- Test: `tests/test_web_config.py`

- [ ] **Step 1: Write failing request, command, and artifact tests**

```python
def test_boundary_flow_job_builds_independent_cli_command():
    request = WalkForwardJobRequest(strategy="boundary-flow-fk", dataset_id="semiconductor")
    command = build_command(request, Path("/tmp/bfk"))
    assert "boundary-flow-fk-walk-forward" in command
    assert "--sector-index-symbol" in command

def test_config_exposes_only_boundary_flow_parameters():
    defaults = config(fake_request())["strategy_defaults"]["boundary-flow-fk"]
    assert defaults["fk_paths"] == 10_000
    assert "layers" not in defaults
```

- [ ] **Step 2: Run tests and verify failure**

Run: `PYTHONPATH=src python -m pytest tests/test_boundary_fk_cli.py tests/test_web_jobs.py tests/test_web_config.py -q`

- [ ] **Step 3: Add the independent dispatch path**

Add strategy literal `boundary-flow-fk`, CLI command `boundary-flow-fk-walk-forward`, validated fields for projection history, branch count, topology history, viscosity window, minimum training days, neutral sigma, direction sigma, FK paths, gate thresholds, Top-K, costs, seed, and max folds. Keep GPU/model-layer controls out of this strategy. Extend `_job_payload` with BoundaryFlowFK artifacts while retaining cancel/delete semantics and unique experiment names.

- [ ] **Step 4: Run tests and commit**

Run: `PYTHONPATH=src python -m pytest tests/test_boundary_fk_cli.py tests/test_web_jobs.py tests/test_web_config.py -q`

Commit: `git commit -m "feat: expose BoundaryFlowFK job workflow"`

### Task 9: DeepStock Lab UI, server validation, and research report

**Files:**
- Modify: `src/bnhmm/web/static/index.html`
- Modify: `src/bnhmm/web/static/app.js`
- Modify: `src/bnhmm/web/static/styles.css`
- Test: `tests/test_web_static.py`
- Create: `docs/boundary-flow-fk-report.md`

- [ ] **Step 1: Write failing static UI tests**

Assert that the strategy selector contains `boundary-flow-fk`, the configuration panel exposes only its current parameters, and the result panel contains posterior distribution, raw/survival probabilities, drift/diffusion/potential, boundary diagnostics, branch attribution, gate reason, and five-stage comparison containers.

- [ ] **Step 2: Implement the restrained result surface**

Use one strategy selector item named `BoundaryFlowFK`, compact numeric controls, and an unframed result layout. Do not display source code or a decorative heatmap. Keep probability charts and diagnostics stable at desktop/mobile widths, and retain existing job edit/delete behavior.

- [ ] **Step 3: Run local verification**

Run:

```bash
PYTHONPATH=src python -m pytest -q
node --check src/bnhmm/web/static/app.js
git diff --check
```

Expected: all tests pass, JavaScript parses, and no whitespace errors.

- [ ] **Step 4: Deploy serially to 145 and run one-fold diagnostics**

Sync the committed branch to `/home/wujianfeng/private/bnhmm-engine`, install without a deep-learning extra, restart with `start.sh`, verify `/health`, run the semiconductor `BFK-GBM` and `BFK-Global` one-fold comparison first, then continue only when the stage gate passes. Record row coverage, fold cutoffs, metrics, runtime, CPU/GPU memory, and failure reasons in Chinese.

- [ ] **Step 5: Run full ablation only after stage gates pass**

Use identical sample-out months and seed for all five names. If a stage fails, stop and write the failure honestly; do not tune extra parameters around it. Verify the UI with browser screenshots at desktop and mobile sizes.

- [ ] **Step 6: Final topology and commit**

Run TriadMind sync and verify. Generate missing runtime artifacts before governance; if the runtime extractor is unavailable, report that limitation separately. Commit:

`git commit -m "feat: add BoundaryFlowFK to DeepStock Lab"`

## Completion Gate

Implementation is complete only when the real minute-index data gate passes, all causal/physical tests pass locally and on 145, the UI can start/cancel/delete/view experiments, every prediction is traceable, and the Chinese report records whether BoundaryFlowFK actually beat `BFK-GBM` on both sample-out Brier and Log Loss. Engineering completion does not imply trading value.
