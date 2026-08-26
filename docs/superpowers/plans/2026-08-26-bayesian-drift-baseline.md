# Bayesian Drift Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the forced zero drift intercept with a causally fitted, prior-shrunk Bayesian drift baseline while leaving features, FK potential, gates, and trading rules unchanged.

**Architecture:** The existing recency-weighted Bayesian logistic direction model will fit an intercept together with `signed_path_efficiency` and `signed_member_synchronization`. The same intercept-aware fit will be used inside causal calibration so the `beta0 + C1` potential is trained against predictions produced by the same model family. Walk-forward folds continue to use only labels whose `target_time` is no later than the fold cutoff.

**Tech Stack:** Python 3.11, NumPy, SciPy L-BFGS-B, pytest, existing DeepStock walk-forward pipeline.

---

### Task 1: Lock the Bayesian baseline behavior with tests

**Files:**
- Modify: `tests/test_two_branch_fk_model.py`
- Modify: `tests/test_boundary_fk_pipeline.py`

- [ ] Add a synthetic imbalanced-label test asserting that zero-valued direction features produce a posterior drift intercept above zero, while the feature coefficients remain finite.
- [ ] Add a balanced-label test asserting that the prior shrinks the fitted intercept toward zero.
- [ ] Retain and run the existing future-label mutation test to prove that rows after `training_cutoff` cannot change the intercept.
- [ ] Update pipeline metadata assertions to require the new Bayesian baseline marker instead of a zero intercept.
- [ ] Run `python -m pytest tests/test_two_branch_fk_model.py tests/test_boundary_fk_pipeline.py -q` and verify the new tests fail because the current implementation fixes `drift_mean[0]` to zero.

### Task 2: Fit the prior-shrunk drift intercept

**Files:**
- Modify: `src/bnhmm/boundary_fk/branch_model.py`
- Modify: `src/bnhmm/boundary_fk/pipeline.py`

- [ ] Fit the full logistic design matrix, including its constant column, with the same finite Gaussian prior precision used for drift coefficients.
- [ ] Preserve the weak-intercept behavior used by the separate convection-potential logistic fit; do not change `beta0 + C1`.
- [ ] Apply the intercept-aware fit in `_causal_drift_predictions` so potential calibration remains causally aligned with production inference.
- [ ] Serialize `drift_baseline_mode: bayesian_intercept` in `boundary-model-posterior.json`.
- [ ] Run the focused tests and then the complete test suite.

### Task 3: Run strict out-of-sample comparisons on server 145

**Files:**
- Create on server: `/home/wujianfeng/private/bnhmm-storage/analysis/drift-baseline-20260826/`

- [ ] Deploy the tested commit to `/home/wujianfeng/private/bnhmm-engine`.
- [ ] Run semiconductor, artificial-intelligence, and new-energy datasets with 60-day topology, direction gate `0.60`, SNR `0.10`, survival `0.35`, `beta0 + C1`, strongest one-third selection, 5BP cost, and unchanged execution rules.
- [ ] Compare drift direction accuracy, effective signal hit rate, coverage, net return, maximum drawdown, turnover, exposure, and failed limit orders against the existing three baseline runs.
- [ ] Reject the model if improvement comes only from higher exposure or if risk-adjusted behavior deteriorates materially.

### Task 4: Report and publish

**Files:**
- Create: `docs/fk-bayesian-drift-baseline-20260826.md`

- [ ] Write a direct Chinese report explaining the zero-intercept defect, the Bayesian baseline implementation, and all three out-of-sample comparisons.
- [ ] Run `git diff --check`, focused tests, and the full test suite.
- [ ] Commit and push `main`, then fast-forward server 145 to the verified commit.
