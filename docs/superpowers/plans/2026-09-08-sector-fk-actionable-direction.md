# FK Actionable Direction Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. This project must be executed inline as one task; do not dispatch subagents.

**Goal:** Improve causal out-of-sample bull/bear skill by aligning economic labels, separating move significance from direction, and forbidding prior-only trades.

**Architecture:** Extend the existing `BoundaryFlowPipeline` and low-parameter Bayesian branch model. Keep data construction, FK potential and execution intact; add bounded posterior diagnostics and honest two-sided evaluation at the existing artifact boundary.

**Tech Stack:** Python 3.13, NumPy, SciPy, pytest, FastAPI static DeepStock Lab.

**Spec:** `docs/superpowers/specs/2026-09-08-sector-fk-actionable-direction-design.md`

## Global Constraints

- Strict monthly causal walk-forward; no sample-out threshold tuning.
- Preserve beta0+C1 potential and all execution constraints.
- No neural network, feature pile-up, subagent, or unrelated refactor.
- Each stage receives a distinct model name and report row.

---

### Task 1: Align Training And Evaluation Labels

**Files:**
- Modify: `src/bnhmm/boundary_fk/branch_model.py`
- Modify: `src/bnhmm/boundary_fk/pipeline.py`
- Modify: `src/bnhmm/boundary_fk/experiment.py`
- Modify: `src/bnhmm/cli.py`
- Test: `tests/test_two_branch_fk_model.py`
- Test: `tests/test_boundary_fk_pipeline.py`
- Test: `tests/test_boundary_fk_cli.py`

**Interfaces:**
- Consumes: `BoundaryFlowPipelineConfig.neutral_sigma: float`
- Produces: one effective direction label threshold and serialized `direction_label_sigma == neutral_sigma`

- [ ] Add a failing test proving a non-default `neutral_sigma` is passed to every Bayesian direction fit and fold readiness check.
- [ ] Run the focused test and confirm it fails because `direction_label_sigma` still defaults independently to `0.1`.
- [ ] Remove the independent runtime parameter and derive the direction threshold from `neutral_sigma` in model construction, fold selection and artifacts.
- [ ] Run branch-model, pipeline, CLI and experiment tests.
- [ ] Deploy this isolated stage to 145 and run the three existing sector datasets; save the metrics before proceeding.

### Task 2: Add The Bayesian Hurdle Posterior

**Files:**
- Create: `src/bnhmm/boundary_fk/hurdle_model.py`
- Modify: `src/bnhmm/boundary_fk/branch_model.py`
- Modify: `src/bnhmm/boundary_fk/feynman_kac.py`
- Modify: `src/bnhmm/boundary_fk/types.py`
- Test: `tests/test_boundary_fk_hurdle_model.py`
- Test: `tests/test_two_branch_fk_model.py`
- Test: `tests/test_boundary_fk_feynman_kac.py`

**Interfaces:**
- Produces: `BayesianHurdlePosterior` containing movement and conditional-direction means/covariances, base-rate probability and fit row counts.
- Produces: forecast diagnostics `significant_move_probability` and `conditional_bullish_probability`.

- [ ] Add failing tests for significance labels, conditional direction labels, posterior validity and causal cutoff exclusion.
- [ ] Confirm tests fail because no hurdle posterior exists.
- [ ] Implement two regularized Bayesian logistic fits using only matured fold rows and the existing recent-session weights.
- [ ] Add failing forecast tests proving low movement probability abstains and conditional direction is independent of flat prevalence.
- [ ] Integrate posterior sampling with the existing FK forecast while leaving beta0+C1 potential unchanged.
- [ ] Run focused model and forecast tests.

### Task 3: Require Feature Evidence Beyond The Prior

**Files:**
- Modify: `src/bnhmm/boundary_fk/hurdle_model.py`
- Modify: `src/bnhmm/boundary_fk/policy.py`
- Modify: `src/bnhmm/boundary_fk/pipeline.py`
- Test: `tests/test_boundary_fk_hurdle_model.py`
- Test: `tests/test_boundary_fk_policy.py`
- Test: `tests/test_sector_fk_pipeline.py`

**Interfaces:**
- Produces: `prior_bullish_probability`, `feature_log_odds_delta`, and fold-calibrated evidence thresholds.

- [ ] Add a failing test where a large bullish intercept and zero feature coefficients must output flat.
- [ ] Confirm the failure is prior-only bullish trading.
- [ ] Fit calibration thresholds using only a chronological tail inside each training fold.
- [ ] Require movement, direction and incremental-feature evidence before bull/bear; otherwise return flat.
- [ ] Add tests proving calibration rows precede `training_cutoff` and both bull/bear paths work.
- [ ] Run focused pipeline and policy tests.

### Task 4: Enforce Honest Two-Sided Evaluation

**Files:**
- Modify: `src/bnhmm/boundary_fk/types.py`
- Modify: `src/bnhmm/boundary_fk/evaluation.py`
- Modify: `src/bnhmm/boundary_fk/pipeline.py`
- Test: `tests/test_boundary_fk_evaluation.py`
- Test: `tests/test_sector_fk_pipeline.py`

**Interfaces:**
- Produces: predicted bull count, predicted bear count, side-specific precision/recall and a stable failure reason.

- [ ] Add failing tests showing 63 bull plus 1 bear cannot pass despite headline accuracy.
- [ ] Extend evaluation metrics with prediction-side counts and side-specific diagnostics.
- [ ] Require at least 15 predicted bull and 15 predicted bear signals in the promotion gate.
- [ ] Preserve the matched-date majority baseline and balanced accuracy requirements.
- [ ] Run all boundary FK tests.

### Task 5: Artifacts, UI, Real-Data Ablation And Conditional Regime Decision

**Files:**
- Modify: `src/bnhmm/boundary_fk/pipeline.py`
- Modify: `src/bnhmm/web/static/app.js`
- Modify: `src/bnhmm/web/static/index.html`
- Modify: `docs/sector-direct-direction-report-20260908.md`
- Test: `tests/test_web_static.py`
- Test: `tests/test_web_payload_js.py`

**Interfaces:**
- Produces: independently named stage rows and Chinese diagnostics in DeepStock Lab.

- [ ] Add failing UI/artifact tests for effective threshold, hurdle posteriors, prior-adjusted evidence and two-sided counts.
- [ ] Serialize and render the diagnostics without exposing irrelevant controls.
- [ ] Run all local tests.
- [ ] Deploy to the existing 145 worktree and run semiconductor, AI and new-energy walk-forward experiments with identical settings.
- [ ] Record each stage's signal count, coverage, accuracy, matched baseline, balanced accuracy and bull/bear counts in the Chinese report.
- [ ] Implement `SectorFK-Regime` only if the first three stages still lack usable two-sided accuracy; otherwise document why added complexity was rejected.
- [ ] Sync TriadMind, rerun topology verification, commit, push and restart DeepStock Lab.

