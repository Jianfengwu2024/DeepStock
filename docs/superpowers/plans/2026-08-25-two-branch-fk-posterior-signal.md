# TwoBranchFK Posterior Signal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the all-flat mean-versus-label-threshold decision with a causal recent-weighted Bayesian posterior signal while preserving a legacy ablation.

**Architecture:** Extend the existing branch posterior with recent-weighted regression statistics and explicit direction confidence. Keep Feynman-Kac forecast generation separate from policy decisions, and evaluate legacy and posterior policies on identical monthly folds.

**Tech Stack:** Python 3.12, NumPy, SciPy, pytest, existing static JavaScript UI.

---

### Task 1: Recent-weighted Bayesian drift

**Files:**
- Modify: `src/bnhmm/boundary_fk/branch_model.py`
- Test: `tests/test_two_branch_fk_model.py`

- [ ] Write a failing test proving a 60-session half-life gives recent observations more influence than equally weighted history.
- [ ] Run `python -m pytest tests/test_two_branch_fk_model.py -q` and confirm the new test fails because the half-life API is absent.
- [ ] Add validated `drift_half_life_sessions` configuration and weighted Gaussian posterior fitting.
- [ ] Re-run the focused test and confirm it passes.

### Task 2: Posterior direction forecast

**Files:**
- Modify: `src/bnhmm/boundary_fk/feynman_kac.py`
- Test: `tests/test_boundary_fk_feynman_kac.py`
- Test: `tests/test_two_branch_fk_model.py`

- [ ] Write failing tests requiring unattenuated drift mean, direction probability, drift signal-to-noise ratio, and potential survival confidence.
- [ ] Run the focused tests and confirm they fail on missing forecast fields and old attenuation.
- [ ] Compute these quantities from the Monte Carlo posterior without multiplying drift by survival.
- [ ] Re-run the focused tests and confirm they pass.

### Task 3: Confidence policy and legacy ablation

**Files:**
- Modify: `src/bnhmm/boundary_fk/policy.py`
- Modify: `src/bnhmm/boundary_fk/pipeline.py`
- Modify: `src/bnhmm/boundary_fk/evaluation.py`
- Test: `tests/test_boundary_fk_policy.py`
- Test: `tests/test_boundary_fk_pipeline.py`

- [ ] Write failing tests proving label thresholds do not control posterior trading directions and weak posterior confidence abstains.
- [ ] Run the focused tests and confirm the intended failures.
- [ ] Add validated probability, signal-to-noise and survival-confidence gates; retain the old direction function for ablation only.
- [ ] Evaluate both policies on the same observations and serialize their metrics and direction counts.
- [ ] Re-run focused tests and confirm they pass.

### Task 4: UI and report

**Files:**
- Modify: `src/bnhmm/web/static/index.html`
- Modify: `src/bnhmm/web/static/app.js`
- Modify: `src/bnhmm/boundary_fk/reporting.py`
- Test: `tests/test_web_static.py`

- [ ] Write failing static tests for the half-life and posterior confidence controls and the legacy comparison panel.
- [ ] Add compact controls and display signal-to-noise, direction probability and survival confidence.
- [ ] Re-run static and boundary tests.

### Task 5: Verification and deployment

**Files:**
- Modify: `docs/boundary-flow-fk-synthetic-index-report.md`

- [ ] Run `python -m pytest` and require zero failures.
- [ ] Run at least six consecutive sample-out months on `/home/share/data/sectors/semiconductor.duckdb` on server 145.
- [ ] Record both ablation results in the Chinese report without selecting thresholds from sample-out returns.
- [ ] Restart server 145, verify `/health`, verify the browser console, and push the resulting commit to GitLab `main`.
