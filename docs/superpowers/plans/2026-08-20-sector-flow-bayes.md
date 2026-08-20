# SectorFlowBayes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minute-synchronous Bayesian sector-flow model that estimates leave-one-out sector water state and ranks leading, fast-responding, directionally efficient stocks.

**Architecture:** A causal sector panel feeds slot-aware normalization, a robust one-factor state-space filter, Bayesian distributed-lag attribution, and walk-forward evaluation. The implementation remains isolated in `bnhmm.sector_flow`; the existing web job boundary only dispatches the new pipeline and renders its artifacts.

**Tech Stack:** Python 3.12, NumPy, DuckDB, PyTorch only where GPU tensor operations materially help, FastAPI, vanilla JavaScript, pytest.

---

### Task 1: Sector panel contracts and synchronization

**Files:**
- Create: `src/bnhmm/sector_flow/__init__.py`
- Create: `src/bnhmm/sector_flow/types.py`
- Create: `src/bnhmm/sector_flow/panel.py`
- Test: `tests/test_sector_flow_panel.py`

- [ ] Write a failing test that builds two symbols with one missing minute and asserts a `[minutes, symbols]` close/volume panel, canonical A-share minute index, and a false validity mask rather than a zero-filled return.
- [ ] Run `python -m pytest tests/test_sector_flow_panel.py -q` and verify failure because `SectorMinutePanelBuilder` does not exist.
- [ ] Implement immutable `SectorMinutePanel` validation and `SectorMinutePanelBuilder.build()` using sorted symbol batches and canonical continuous-session minutes.
- [ ] Add a failing test proving a genuine unchanged close remains valid while a missing minute remains invalid.
- [ ] Implement return construction without crossing lunch, session, or missing-minute boundaries.
- [ ] Run the panel tests and commit `feat: add synchronized sector minute panel`.

### Task 2: Causal slot-aware normalization

**Files:**
- Create: `src/bnhmm/sector_flow/normalization.py`
- Test: `tests/test_sector_flow_normalization.py`

- [ ] Write a failing test with 20 historical sessions and one current session, then modify only the current session and assert unchanged velocity and volume baselines.
- [ ] Run the test and verify it fails because `CausalSectorNormalizer` is absent.
- [ ] Implement per-symbol, per-minute-slot zero-centered RMS velocity scale and median volume baseline fitted only on completed prior sessions.
- [ ] Add failing tests for no mean subtraction, minimum scale, same-slot opening-volume normalization, and clipped relative activity.
- [ ] Implement `fit`, `transform_session`, and rolling history update with the update occurring after transformation.
- [ ] Run normalization and panel tests and commit `feat: add causal sector normalization`.

### Task 3: Robust dynamic sector factor

**Files:**
- Create: `src/bnhmm/sector_flow/factor.py`
- Test: `tests/test_sector_flow_factor.py`

- [ ] Write a failing synthetic test where 12 symbols share a known latent factor and assert posterior factor correlation above `0.9`.
- [ ] Run the test and verify failure because `BayesianSectorFactor` is absent.
- [ ] Implement a one-factor linear Gaussian state model with AR(1) transition, iterative robust Student-t observation weights, posterior mean/variance, loadings, residual scales, breadth, coherence, dispersion, activity, and flow strength.
- [ ] Write a failing test that changes only target symbol `i` and asserts `leave_one_out(i)` is unchanged.
- [ ] Implement analytic leave-one-out observation updates from sufficient statistics rather than fitting one model per symbol.
- [ ] Add tests for missing-symbol masks, all-invalid minutes, and causal filtering that does not change earlier outputs when future observations change.
- [ ] Run factor tests and commit `feat: infer leave-one-out sector flow`.

### Task 4: Bayesian lead and response attribution

**Files:**
- Create: `src/bnhmm/sector_flow/attribution.py`
- Test: `tests/test_sector_flow_attribution.py`

- [ ] Write a failing synthetic test with one stock leading the factor by three minutes and another responding after two minutes.
- [ ] Implement Bayesian ridge distributed-lag regression with hierarchical pooled prior, posterior coefficient means/covariances, leader probability, response probability, and posterior peak lag.
- [ ] Add a test that unrelated noise receives a wider posterior and lower directional probability than the planted leader.
- [ ] Add a test proving only past fitted parameters and observations available at minute `t` participate in online scores.
- [ ] Run attribution tests and commit `feat: add Bayesian flow attribution`.

### Task 5: Path efficiency and source-proximity ranking

**Files:**
- Create: `src/bnhmm/sector_flow/ranking.py`
- Test: `tests/test_sector_flow_ranking.py`

- [ ] Write failing tests for a monotone path, an oscillating zero-net path, a path with missing minutes, and 5/15/30/60-minute windows.
- [ ] Implement net-displacement/path-length efficiency, directional agreement, reversal count, activity confirmation, and residual fraction.
- [ ] Write a failing test for the joint posterior event and assert all component probabilities remain visible in the result contract.
- [ ] Implement a deterministic Monte Carlo composition of posterior draws using a seeded generator; do not introduce a hand-tuned opaque weighted score.
- [ ] Run ranking tests and commit `feat: rank source-proximate stocks`.

### Task 6: Walk-forward pipeline and reports

**Files:**
- Create: `src/bnhmm/sector_flow/evaluation.py`
- Create: `src/bnhmm/sector_flow/pipeline.py`
- Modify: `src/bnhmm/cli.py`
- Test: `tests/test_sector_flow_evaluation.py`
- Test: `tests/test_sector_flow_pipeline.py`

- [ ] Write a failing walk-forward split test proving every model cutoff precedes the first test minute and every signal becomes available on the following minute.
- [ ] Implement monthly expanding folds, frozen test-month parameters, minute-causal filtering, and prediction records.
- [ ] Write failing metric tests for probabilistic direction, Brier/log loss, Rank IC, Top-K minus Bottom-K return, turnover costs, posterior coverage, and lag stability.
- [ ] Implement JSON, CSV, Markdown, and PNG report artifacts with explicit baseline comparisons.
- [ ] Add a `sector-flow-walk-forward` CLI command and a smoke test against a small temporary DuckDB sector dataset.
- [ ] Run pipeline tests and commit `feat: evaluate sector flow walk forward`.

### Task 7: DeepStock Lab job integration

**Files:**
- Modify: `src/bnhmm/web/jobs.py`
- Modify: `src/bnhmm/web/app.py`
- Test: `tests/test_web_jobs.py`
- Test: `tests/test_web_app.py`

- [ ] Add a failing request-validation test for `strategy="sector-flow-bayes"`, required sector dataset, one-minute cadence, normalization lookback, maximum lag, Top-K, and process-noise parameters.
- [ ] Extend `WalkForwardJobRequest` and `build_command` to dispatch only the new CLI options; keep old strategy commands unchanged.
- [ ] Add a failing `/api/config` test for strategy-specific defaults and result artifact discovery.
- [ ] Implement defaults and artifact URLs, then run web tests.
- [ ] Commit `feat: expose sector flow jobs`.

### Task 8: DeepStock Lab configuration and results UI

**Files:**
- Modify: `src/bnhmm/web/static/index.html`
- Modify: `src/bnhmm/web/static/app.js`
- Modify: `src/bnhmm/web/static/styles.css`
- Test: `tests/test_web_static.py`

- [ ] Add failing static-contract tests for the new strategy option, sector-flow-only controls, water-state summary, ranking table, attribution detail, and accessible loading/error states.
- [ ] Implement the compact configuration controls and hide unrelated BNHMM, Transformer, FKStock, and position-policy controls when this strategy is selected.
- [ ] Render sector direction/activity/breadth/coherence uncertainty, source-proximity components, monthly stability, and report downloads from generated artifacts.
- [ ] Verify text containment at desktop and mobile widths with browser screenshots.
- [ ] Commit `feat: add sector flow lab interface`.

### Task 9: Full verification and semiconductor experiment

**Files:**
- Create: `docs/sector-flow-bayes-v1-report.md`
- Modify only if verification finds defects: files owned by Tasks 1-8

- [ ] Run `python -m pytest -q` locally and record the exact pass count.
- [ ] Run TriadMind sync, verify, and govern; resolve any new topology violations.
- [ ] Deploy the reviewed commit to `/home/wujianfeng/private/bnhmm-engine` on server 145 without copying model artifacts into Git.
- [ ] Restart DeepStock Lab and verify `/health`, `/api/config`, job start, cancellation, result retrieval, and deletion.
- [ ] Run the semiconductor 2024-2025 monthly walk-forward experiment using `/home/share/data/sectors/semiconductor.duckdb`.
- [ ] Compare against random, momentum, relative-volume, path-efficiency, and static-beta baselines; do not claim predictive value unless sample-out metrics support it.
- [ ] Write the model definition, implementation details, diagnostics, limitations, and exact experimental results to `docs/sector-flow-bayes-v1-report.md`.
- [ ] Run the full local and server verification suites again, request code review, fix all critical and important findings, and commit `docs: report sector flow experiment`.

