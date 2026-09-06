# PercolationBranching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This repository must be implemented inline by one worker; do not dispatch subagents.

**Goal:** Build a causal dual-network branching-process model that predicts the next trading session's sector percolation phase, backtests a simple long-only policy, and exposes configuration and diagnostics in DeepStock Lab.

**Architecture:** One-minute bars become causally normalized bull/bear activation events with relative-volume marks. A 60-session monthly backbone fixes candidate propagation edges, a sparse Bayesian branching model estimates directional propagation shape, and a low-dimensional local-trend filter forecasts next-session reproduction numbers. Posterior branching simulations produce four phase probabilities and per-stock giant-cluster membership probabilities; a separate policy and evaluator turn only confident bull phases into trades.

**Tech Stack:** Python 3.12, NumPy, SciPy, DuckDB adapter already in the repository, pytest, FastAPI/Pydantic, vanilla HTML/CSS/JavaScript.

**Spec:** `docs/superpowers/specs/2026-09-06-percolation-branching-strategy-design.md`

## Global Constraints

- Work in a single isolated Git worktree and do not use subagents.
- Keep all model code in `bnhmm-engine`; do not copy it into Quant Platform.
- Use only one-minute price and volume data; do not add news, policy, financial, PyTorch, or graph-neural-network dependencies.
- Use separate bull and bear directed propagation networks.
- Use lags 1, 2, and 3 minutes; same-minute motion belongs to the external baseline.
- Use a 60-session monthly backbone; month membership and candidate edges remain frozen within the month.
- Preserve `signal_time`, `available_at`, `training_cutoff`, and `model_version` for every prediction.
- Default activation threshold is 1.5, null permutations 500, posterior paths 10,000, bull entry probability 0.60, and transaction cost 5BP.
- The first policy is binary: confident bull phase means full allocation to the highest net-membership third; every other phase means cash.
- Respect A-share limit-up buys and limit-down sells; failed sells remain actual holdings.
- Do not tune policy parameters when phase prediction fails to beat majority and persistence baselines.

---

### Task 1: Activation Events and Typed Contracts

**Files:**
- Create: `src/bnhmm/percolation_branching/__init__.py`
- Create: `src/bnhmm/percolation_branching/types.py`
- Create: `src/bnhmm/percolation_branching/events.py`
- Test: `tests/test_percolation_events.py`

**Interfaces:**
- Consumes: `bnhmm.sector_flow.types.SectorMinutePanel`.
- Produces: `ActivationConfig`, `ActivationPanel`, `ActivationEventBuilder.build(panel) -> ActivationPanel`.
- `ActivationPanel.direction(name: Literal["bull", "bear"]) -> NDArray[np.bool_]` returns one directional event matrix.

- [ ] **Step 1: Write failing activation tests**

```python
def test_builder_uses_only_completed_history_for_current_session():
    panel = synthetic_panel(sessions=12, symbols=3)
    baseline = ActivationEventBuilder().build(panel)
    changed = replace_last_session(panel, close_multiplier=10.0, volume_multiplier=50.0)
    rebuilt = ActivationEventBuilder().build(changed)
    np.testing.assert_allclose(
        baseline.velocity_scale[-240:], rebuilt.velocity_scale[-240:]
    )

def test_price_sets_direction_and_volume_only_sets_mark():
    events = ActivationEventBuilder().build(event_fixture())
    assert events.bull[EVENT_ROW, 0]
    assert not events.bear[EVENT_ROW, 0]
    assert events.marks[EVENT_ROW, 0] > 1.0
```

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_percolation_events.py -q`

Expected: import failure for `bnhmm.percolation_branching.events`.

- [ ] **Step 3: Implement immutable contracts and causal event construction**

```python
@dataclass(frozen=True, slots=True)
class ActivationConfig:
    velocity_lookback_sessions: int = 10
    volume_lookback_sessions: int = 20
    activation_threshold: float = 1.5
    volume_mark_cap: float = 3.0
    minimum_minutes_per_session: int = 200

@dataclass(frozen=True, slots=True)
class ActivationPanel:
    symbols: tuple[str, ...]
    timestamps: NDArray[np.datetime64]
    standardized_velocity: NDArray[np.float64]
    velocity_scale: NDArray[np.float64]
    relative_volume: NDArray[np.float64]
    marks: NDArray[np.float64]
    bull: NDArray[np.bool_]
    bear: NDArray[np.bool_]
    valid: NDArray[np.bool_]
```

For each session, freeze velocity and minute-of-day volume baselines before transforming current rows, then append current observations to history only after the session has been transformed.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_percolation_events.py -q`

Expected: all activation tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/bnhmm/percolation_branching tests/test_percolation_events.py
git commit -m "feat(percolation): build causal activation events"
```

### Task 2: Temporal Reachability and Causal Phase Labels

**Files:**
- Create: `src/bnhmm/percolation_branching/labels.py`
- Extend: `src/bnhmm/percolation_branching/types.py`
- Test: `tests/test_percolation_labels.py`

**Interfaces:**
- Consumes: a directional event matrix for one session and a frozen candidate adjacency matrix.
- Produces: `temporal_reachability(events, adjacency, maximum_lag=3) -> ReachabilityResult`.
- Produces: `NullThresholdCalibrator.fit(panel, support, training_sessions, rng) -> NullThresholdTable`.
- Produces: `classify_phase(bull_fraction, bear_fraction, bull_threshold, bear_threshold) -> PhaseName`.

- [ ] **Step 1: Write failing reachability and label tests**

```python
def test_temporal_chain_reaches_three_distinct_stocks():
    events = chain_events([(0, 0), (1, 1), (3, 2)])
    result = temporal_reachability(events, full_support(3), maximum_lag=3)
    assert result.giant_fraction == 1.0
    assert result.source_membership[0].tolist() == [True, True, True]

def test_same_minute_common_jump_does_not_form_a_chain():
    events = chain_events([(5, 0), (5, 1), (5, 2)])
    result = temporal_reachability(events, full_support(3), maximum_lag=3)
    assert result.giant_fraction == pytest.approx(1 / 3)

def test_phase_labels_cover_all_four_states():
    assert classify_phase(0.8, 0.2, 0.6, 0.6) == "bull"
    assert classify_phase(0.2, 0.8, 0.6, 0.6) == "bear"
    assert classify_phase(0.2, 0.2, 0.6, 0.6) == "none"
    assert classify_phase(0.8, 0.8, 0.6, 0.6) == "conflict"
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `.venv/bin/python -m pytest tests/test_percolation_labels.py -q`

Expected: missing label interfaces.

- [ ] **Step 3: Implement event-DAG reachability**

Track source-stock bitsets through events in timestamp order. Only edges from another stock at lags 1 to 3 may propagate a source. Return maximum reachable stock fraction, per-source membership, event density, and edge counts.

- [ ] **Step 4: Implement monthly density-conditioned null thresholds**

Shuffle event times within four fixed intraday buckets while preserving each stock's daily directional event count. Aggregate null giant fractions into four density bins and store the 95th percentile per bin. The threshold table carries `month`, `training_cutoff`, `direction`, and `permutations`.

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/test_percolation_labels.py -q`

Expected: all label tests pass, including deterministic seeded permutations.

- [ ] **Step 6: Commit**

```bash
git add src/bnhmm/percolation_branching tests/test_percolation_labels.py
git commit -m "feat(percolation): define causal phase labels"
```

### Task 3: Monthly Backbone and Sparse Bayesian Branching Shape

**Files:**
- Create: `src/bnhmm/percolation_branching/backbone.py`
- Create: `src/bnhmm/percolation_branching/branching.py`
- Extend: `src/bnhmm/percolation_branching/types.py`
- Test: `tests/test_percolation_branching.py`

**Interfaces:**
- Produces: `SlowBackboneBuilder.build(events, month) -> MonthlyBackboneSupport`.
- Produces: `BayesianBranchingModel.fit(events, support, training_sessions, direction) -> BranchingPosterior`.
- Produces: `BranchingPosterior.observe(events, session) -> DailyBranchingObservation`.

- [ ] **Step 1: Write failing backbone and branching tests**

```python
def test_backbone_uses_only_sessions_before_month():
    support = SlowBackboneBuilder().build(events, np.datetime64("2025-02", "M"))
    assert support.training_cutoff < np.datetime64("2025-02-01")
    assert not np.any(np.diag(support.adjacency))
    assert np.all(support.adjacency == support.adjacency.T)

def test_branching_recovers_directed_chain_and_unit_spectral_shape():
    posterior = BayesianBranchingModel().fit(
        directed_chain_events(), support=full_support_contract(4),
        training_sessions=training_sessions(), direction="bull"
    )
    assert posterior.mean[0, 1] > posterior.mean[1, 0]
    assert spectral_radius(posterior.normalized_mean) == pytest.approx(1.0)
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `.venv/bin/python -m pytest tests/test_percolation_branching.py -q`

- [ ] **Step 3: Implement the frozen 60-session candidate backbone**

Use standardized velocity and cross-sectional-mean residual velocity. Pairwise similarity is `0.5 * raw_correlation + 0.5 * residual_correlation`, shrunk by common observations. Retain the union of each node's ten strongest positive neighbors and require at least 80% complete training sessions.

- [ ] **Step 4: Implement nonnegative Gamma-Poisson variational EM**

```python
@dataclass(frozen=True, slots=True)
class BranchingPosterior:
    direction: Direction
    symbols: tuple[str, ...]
    adjacency: NDArray[np.bool_]
    mean: NDArray[np.float64]
    variance: NDArray[np.float64]
    normalized_mean: NDArray[np.float64]
    lag_weights: NDArray[np.float64]
    external_rate: NDArray[np.float64]
    self_rate: NDArray[np.float64]
    training_cutoff: np.datetime64
```

Initialize edge, external, and self intensities from smoothed counts. In each EM iteration, assign every child event responsibility across external, self, and eligible cross-stock parents, then update Gamma posterior shape/rate. Normalize the cross-stock posterior mean to spectral radius one after fitting; external and self rates stay outside that matrix.

- [ ] **Step 5: Add explicit common-shock and self-excitation tests**

Assert a same-minute all-stock jump raises external responsibility without creating cross-stock edge mass. Assert repeated events in one stock raise self rate without changing cross-stock spectral radius.

- [ ] **Step 6: Run tests**

Run: `.venv/bin/python -m pytest tests/test_percolation_branching.py -q`

- [ ] **Step 7: Commit**

```bash
git add src/bnhmm/percolation_branching tests/test_percolation_branching.py
git commit -m "feat(percolation): fit monthly branching networks"
```

### Task 4: Daily Critical Filter and Posterior Simulator

**Files:**
- Create: `src/bnhmm/percolation_branching/filter.py`
- Create: `src/bnhmm/percolation_branching/simulator.py`
- Extend: `src/bnhmm/percolation_branching/types.py`
- Test: `tests/test_percolation_forecast.py`

**Interfaces:**
- Produces: `CriticalStateFilter.initialize(observations) -> CriticalFilterState`.
- Produces: `CriticalStateFilter.update(state, observation) -> CriticalFilterState`.
- Produces: `CriticalStateFilter.predict(state) -> CriticalStateForecast`.
- Produces: `PercolationSimulator.predict(...) -> PercolationPrediction`.

- [ ] **Step 1: Write failing filter tests**

```python
def test_supercritical_observations_raise_crossing_probability():
    state = seeded_filter_state()
    for _ in range(8):
        state = CriticalStateFilter().update(
            state, DailyBranchingObservation(parent_exposure=20, offspring=30, external=2)
        )
    forecast = CriticalStateFilter().predict(state)
    assert forecast.probability_supercritical > 0.8

def test_external_rain_does_not_raise_reproduction_number():
    quiet = update_with(parent_exposure=20, offspring=10, external=2)
    rainy = update_with(parent_exposure=20, offspring=10, external=50)
    assert rainy.log_reproduction_mean == pytest.approx(quiet.log_reproduction_mean)
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `.venv/bin/python -m pytest tests/test_percolation_forecast.py -q`

- [ ] **Step 3: Implement the two-state Bayesian local-trend filter**

Use state `[log_R, slope]`, transition matrix `[[1, 1], [0, kappa]]`, and Gamma-Poisson daily observations converted to log-rate measurements using `digamma` and `polygamma`. Track external seed intensity with a separate Gamma posterior.

- [ ] **Step 4: Implement posterior branching simulation**

For each posterior path, sample next-session `R+`, `R-`, external seed counts, seed symbols, and multitype offspring on the normalized monthly matrices. Stop at 240 generations/minute slots or a bounded event cap. Classify both directions against the monthly density-conditioned null threshold. Aggregate:

```python
PercolationPrediction(
    phase_probabilities=(p_bear, p_none, p_conflict, p_bull),
    bull_reproduction_mean=...,
    bear_reproduction_mean=...,
    bull_supercritical_probability=...,
    bear_supercritical_probability=...,
    bull_giant_fraction_mean=...,
    bear_giant_fraction_mean=...,
    bull_membership_probability=...,
    bear_membership_probability=...,
    signal_time=...,
    available_at=...,
    target_session=...,
    training_cutoff=...,
    model_version="percolation-branching-v1",
)
```

- [ ] **Step 5: Test subcritical, supercritical, deterministic-seed, and timing behavior**

Run: `.venv/bin/python -m pytest tests/test_percolation_forecast.py -q`

Expected: subcritical giant probability is below supercritical probability; fixed seeds reproduce identical forecasts; invalid causal timestamps fail.

- [ ] **Step 6: Commit**

```bash
git add src/bnhmm/percolation_branching tests/test_percolation_forecast.py
git commit -m "feat(percolation): forecast critical phase transitions"
```

### Task 5: Policy, Execution, and Metrics

**Files:**
- Create: `src/bnhmm/percolation_branching/policy.py`
- Create: `src/bnhmm/percolation_branching/evaluation.py`
- Test: `tests/test_percolation_policy.py`
- Test: `tests/test_percolation_evaluation.py`

**Interfaces:**
- Produces: `PercolationPolicy.decide(prediction, symbols) -> PercolationDecision`.
- Produces: `execute_percolation_portfolio(...) -> PercolationExecution`.
- Produces: `evaluate_percolation(observations, transaction_cost_bps) -> PercolationMetrics`.

- [ ] **Step 1: Write failing policy tests**

```python
def test_only_confident_bull_phase_opens_position():
    decision = PercolationPolicy().decide(prediction(p_bull=0.61), symbols())
    assert decision.open_position
    assert len(decision.selected_symbols) == math.ceil(len(symbols()) / 3)
    assert sum(decision.target_weights) == pytest.approx(1.0)

@pytest.mark.parametrize("p_bull", [0.0, 0.40, 0.5999])
def test_every_non_confident_phase_is_cash(p_bull):
    assert not PercolationPolicy().decide(prediction(p_bull=p_bull), symbols()).open_position
```

- [ ] **Step 2: Implement net-membership ranking and equal weighting**

Rank by `bull_membership_probability - bear_membership_probability`, break ties by symbol, select `ceil(N / 3)`, and assign equal weights. Failed limit-up buys remain cash and do not select replacements.

- [ ] **Step 3: Reuse limit-up/limit-down arrays from `boundary_fk.execution`**

Execute sells before buys, preserve failed-sell weights, cap buys by available cash, and calculate realized return and turnover from actual weights.

- [ ] **Step 4: Write and implement prediction/economic metrics**

Metrics include confusion matrix, balanced accuracy, macro F1, per-class precision/recall, multiclass Brier, log loss, expected calibration error, total return, maximum drawdown, average exposure, turnover, open rate, failed buys, and failed sells.

- [ ] **Step 5: Implement majority, persistence, transition-matrix, and simple-`R` baselines**

Each baseline must be fitted only on labels mature at its signal date. No baseline may inspect the target session.

- [ ] **Step 6: Run tests**

Run: `.venv/bin/python -m pytest tests/test_percolation_policy.py tests/test_percolation_evaluation.py -q`

- [ ] **Step 7: Commit**

```bash
git add src/bnhmm/percolation_branching tests/test_percolation_policy.py tests/test_percolation_evaluation.py
git commit -m "feat(percolation): add phase policy and evaluation"
```

### Task 6: Walk-Forward Pipeline, CLI, Artifacts, and Chinese Report

**Files:**
- Create: `src/bnhmm/percolation_branching/pipeline.py`
- Create: `src/bnhmm/percolation_branching/experiment.py`
- Create: `src/bnhmm/percolation_branching/reporting.py`
- Modify: `src/bnhmm/cli.py`
- Test: `tests/test_percolation_pipeline.py`
- Test: `tests/test_percolation_cli.py`

**Interfaces:**
- Produces: `PercolationPipelineConfig` and `PercolationPipeline.run(source, output_dir) -> PercolationRunResult`.
- Adds CLI command: `percolation-branching-walk-forward`.

- [ ] **Step 1: Write failing CLI and synthetic pipeline tests**

```python
def test_cli_exposes_fixed_physical_defaults():
    args = build_parser().parse_args([
        "percolation-branching-walk-forward", "--duckdb-path", "market.duckdb",
        "--symbols", "A,B,C,D", "--dataset-id", "synthetic",
        "--start", "2024-01-01", "--end", "2025-01-01",
        "--output-dir", "/tmp/percolation",
    ])
    assert args.activation_threshold == 1.5
    assert args.maximum_lag_minutes == 3
    assert args.topology_lookback_sessions == 60
    assert args.phase_probability_threshold == 0.60
```

- [ ] **Step 2: Implement causal month-by-month orchestration**

Fetch warmup bars, build the canonical minute panel and execution constraints once, then for each test month build/freeze supports, fit both directional models, calibrate null thresholds, initialize filters from mature training days, and emit one next-session forecast after every valid signal day.

- [ ] **Step 3: Write artifacts atomically**

Required files:

```text
run-summary.json
percolation-evaluation.json
percolation-baselines.json
percolation-predictions.json
percolation-posteriors.json
percolation-equity.json
percolation-network-summary.json
percolation-report-zh.md
```

All predictions include causal timestamps and model version. JSON must contain finite numbers or explicit `null`, never non-standard `NaN`.

- [ ] **Step 4: Run focused tests**

Run: `.venv/bin/python -m pytest tests/test_percolation_pipeline.py tests/test_percolation_cli.py -q`

- [ ] **Step 5: Commit**

```bash
git add src/bnhmm/percolation_branching src/bnhmm/cli.py tests/test_percolation_pipeline.py tests/test_percolation_cli.py
git commit -m "feat(percolation): run causal walk-forward experiments"
```

### Task 7: Web Job Contract and Artifact Delivery

**Files:**
- Modify: `src/bnhmm/web/jobs.py`
- Modify: `src/bnhmm/web/app.py`
- Test: `tests/test_web_jobs.py`
- Test: `tests/test_web_config.py`

**Interfaces:**
- Adds job strategy literal `percolation-branching`.
- Adds request fields `percolation_activation_threshold`, `percolation_null_permutations`, `percolation_posterior_paths`, and `percolation_phase_probability_threshold`.
- `build_command()` emits only physical percolation parameters and CPU execution.

- [ ] **Step 1: Write failing request and command tests**

```python
def test_percolation_job_requires_sector_dataset_and_builds_cpu_command():
    request = WalkForwardJobRequest(
        strategy="percolation-branching", training_scope="sector-pooled",
        dataset_id="semiconductor"
    )
    command = build_command(request, Path("/tmp/percolation"))
    assert "percolation-branching-walk-forward" in command
    assert command[command.index("--maximum-lag-minutes") + 1] == "3"
    assert "--device" not in command
```

- [ ] **Step 2: Extend Pydantic validation and command construction**

Reject non-sector scope, lag values other than 3, topology values other than 60, probabilities outside `[0.5, 1]`, permutations below 20, and posterior paths below 100.

- [ ] **Step 3: Expose result artifacts through the existing safe artifact endpoint**

No new arbitrary-path endpoint is allowed. Add percolation artifact names to frontend payload loading and completed-run discovery without weakening path validation.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_web_jobs.py tests/test_web_config.py -q`

- [ ] **Step 5: Commit**

```bash
git add src/bnhmm/web/jobs.py src/bnhmm/web/app.py tests/test_web_jobs.py tests/test_web_config.py
git commit -m "feat(web): run percolation branching jobs"
```

### Task 8: DeepStock Lab Configuration and Result Views

**Files:**
- Modify: `src/bnhmm/web/static/index.html`
- Modify: `src/bnhmm/web/static/app.js`
- Modify: `src/bnhmm/web/static/styles.css`
- Modify: `tests/test_web_static.py`

**Interfaces:**
- Adds strategy radio `value="percolation-branching"` with display name `渗流分枝`.
- Adds a dedicated result container with seven tabs: evaluation, phases, posterior, network, returns, calibration, summary.

- [ ] **Step 1: Write failing static contract tests**

Assert the strategy radio, four editable fields, fixed 60-day/1-to-3-minute labels, seven result tabs, phase chart, critical `R=1` chart, network source table, equity chart, calibration panel, and artifact names exist.

- [ ] **Step 2: Add strategy-specific configuration controls**

Only show:

```text
板块数据集和日期
激活阈值（默认1.5）
传播延迟（固定1至3分钟）
随机置换次数（默认500）
后验生成路径（默认10000）
多头开仓概率（默认0.60）
交易成本（默认5BP）
随机种子
```

Disable all hidden controls before form serialization.

- [ ] **Step 3: Add result rendering**

Render four phase probabilities, predicted/actual phase timeline, bull/bear reproduction numbers with a fixed critical line at 1, giant-fraction forecasts, top propagation sources, strategy/baseline metrics, equity/drawdown, confusion matrix, and run summary. Technical labels receive short Chinese explanations; do not show source code.

- [ ] **Step 4: Preserve responsive layout**

Use existing compact result patterns, no nested cards, fixed canvas sizes with responsive CSS, and horizontal scrolling only for dense tables. The history column must not widen.

- [ ] **Step 5: Run static and backend tests**

Run: `.venv/bin/python -m pytest tests/test_web_static.py tests/test_web_jobs.py tests/test_web_config.py -q`

- [ ] **Step 6: Commit**

```bash
git add src/bnhmm/web/static tests/test_web_static.py
git commit -m "feat(web): visualize percolation phase forecasts"
```

### Task 9: Full Verification and Real Semiconductor Smoke Test

**Files:**
- Create: `docs/percolation-branching-v1-test-report-zh.md`
- Modify only if verification finds a defect: files introduced in Tasks 1 through 8.

**Interfaces:**
- Produces a reproducible real-data run and a Chinese test report for human-machine evaluation.

- [ ] **Step 1: Run the complete automated suite**

Run: `.venv/bin/python -m pytest -q`

Expected: zero failures.

- [ ] **Step 2: Run synthetic physical verification**

Run the focused synthetic suite with verbose output:

```bash
.venv/bin/python -m pytest \
  tests/test_percolation_branching.py \
  tests/test_percolation_forecast.py -vv
```

Record evidence that subcritical cascades decay, supercritical cascades grow, common shocks stay external, and future mutations do not alter historical predictions.

- [ ] **Step 3: Run a bounded real-data semiconductor walk-forward**

Use the server's semiconductor dataset, a date range containing enough warmup history, `max_folds=1`, 20 null permutations, and 500 posterior paths for the first operational smoke test. These reduced numerical budgets are labeled smoke-only and cannot support a strategy-performance conclusion.

- [ ] **Step 4: Validate all artifacts**

Check required JSON/Markdown files, finite probability sums, causal timestamp ordering, phase labels, selected top-third count, actual trade weights, and absence of `NaN` JSON tokens.

- [ ] **Step 5: Start the web service and perform browser QA**

Verify on desktop and mobile widths that the strategy can be selected, only relevant fields remain enabled, the job submits, history status updates, completed results open, all seven tabs render, canvases are nonblank, text does not overlap, and the old strategies still render.

- [ ] **Step 6: Write the Chinese test report**

The report must separate:

- automated correctness evidence;
- synthetic physical evidence;
- real-data smoke evidence;
- unresolved statistical questions;
- items reserved for human-machine evaluation.

- [ ] **Step 7: Run final repository checks**

```bash
git diff --check
.venv/bin/python -m pytest -q
git status --short
```

- [ ] **Step 8: Commit verification artifacts**

```bash
git add docs/percolation-branching-v1-test-report-zh.md
git commit -m "docs: report percolation branching verification"
```

## Human-Machine Evaluation Gate

Do not claim the strategy predicts markets successfully after automated and smoke tests. When Tasks 1 through 9 pass, stop with these materials ready for joint evaluation:

- the completed DeepStock Lab run;
- phase forecast and actual-phase timeline;
- `R+` and `R-` critical plots;
- phase-baseline comparison;
- strategy and buy-and-hold equity curves;
- source-stock attribution;
- calibration diagnostics;
- the Chinese verification report.

The next step is human review of physical plausibility and a full-budget out-of-sample experiment, not automatic parameter tuning.
