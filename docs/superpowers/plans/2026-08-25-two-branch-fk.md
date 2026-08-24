# Two-Branch FK Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace BoundaryFlowFK's boundary-layer and Top-K workflow with two frozen monthly branches, independent Bayesian FK forecasts, and an equal-weight long-only branch state machine.

**Architecture:** Keep the `boundary-flow-fk` external job key for compatibility, but route it through new branch-focused state, model, policy, and portfolio contracts. Reuse causal 60-session synchronization topology with exactly two clusters. Train and forecast each branch independently, then combine held branch members into one equal-weight portfolio.

**Tech Stack:** Python 3.13, NumPy, SciPy, pytest, FastAPI/Pydantic, vanilla JavaScript.

---

### Task 1: Fixed Two-Branch Topology

**Files:**
- Modify: `src/bnhmm/sector_flow/topology.py`
- Modify: `src/bnhmm/boundary_fk/daily_state.py`
- Test: `tests/test_sector_flow_topology.py`
- Test: `tests/test_boundary_fk_daily_state.py`

- [ ] Add failing tests proving `maximum_bands=2` yields exactly two non-empty clusters and that monthly topology ignores future rows.
- [ ] Run `pytest tests/test_sector_flow_topology.py tests/test_boundary_fk_daily_state.py -q` and confirm the new fixed-two test fails.
- [ ] Make BoundaryFlow's topology configuration fixed at two clusters, reject fewer than four usable symbols, and mark an unstable month unavailable rather than inventing groups.
- [ ] Re-run the focused tests and commit the passing topology change.

### Task 2: Branch Daily State and Labels

**Files:**
- Create: `src/bnhmm/boundary_fk/branch_state.py`
- Modify: `src/bnhmm/boundary_fk/types.py`
- Test: `tests/test_two_branch_fk_state.py`

- [ ] Add failing tests for a `(session, branch, feature)` state tensor with features `return_1d`, `return_mean_5`, `return_mean_20`, `volatility_20`, and `coherence_1d`.
- [ ] Add failing tests proving each branch next-return label and member return use only the next completed session.
- [ ] Run `pytest tests/test_two_branch_fk_state.py -q` and confirm failures are caused by missing branch contracts.
- [ ] Implement immutable two-branch state, label, and member-return contracts plus a causal builder from the minute panel and frozen topologies.
- [ ] Re-run focused tests and commit.

### Task 3: Independent Bayesian FK

**Files:**
- Create: `src/bnhmm/boundary_fk/branch_model.py`
- Modify: `src/bnhmm/boundary_fk/feynman_kac.py`
- Test: `tests/test_two_branch_fk_model.py`

- [ ] Add failing tests that branch 0 and branch 1 fit separate posteriors, diffusion is intercept-only, and future labels cannot enter a fit.
- [ ] Add a failing test that larger effective potential attenuates drift and moves FK probabilities toward flat/noise rather than multiplying all classes by the same scalar.
- [ ] Run `pytest tests/test_two_branch_fk_model.py -q` and observe the expected failures.
- [ ] Implement Bayesian ridge drift, intercept-only residual diffusion, logistic trend-survival potential, and FK Monte Carlo using `effective_drift = drift * exp(-potential)`.
- [ ] Re-run focused tests and commit.

### Task 4: Branch State Machine and Real Portfolio Returns

**Files:**
- Replace: `src/bnhmm/boundary_fk/policy.py`
- Replace: `src/bnhmm/boundary_fk/pipeline.py`
- Modify: `src/bnhmm/boundary_fk/evaluation.py`
- Test: `tests/test_boundary_fk_policy.py`
- Test: `tests/test_boundary_fk_pipeline.py`

- [ ] Add failing tests for `多 -> 建仓`, `平 -> 保持`, `空 -> 清仓`, and `平且无仓 -> 空仓` independently for both branches.
- [ ] Add a failing test proving all currently held symbols share one equal portfolio weight and strategy return comes from those symbols' next-day returns.
- [ ] Run focused policy/pipeline tests and confirm expected failures.
- [ ] Implement one `TwoBranchFK` walk-forward stage, independent branch forecasts, persistent branch holding state, actual constituent returns, turnover, and next-session signal timing.
- [ ] Re-run focused tests and commit.

### Task 5: Experiment, UI, Reports, and End-to-End Verification

**Files:**
- Modify: `src/bnhmm/boundary_fk/experiment.py`
- Modify: `src/bnhmm/boundary_fk/reporting.py`
- Modify: `src/bnhmm/web/jobs.py`
- Modify: `src/bnhmm/web/app.py`
- Modify: `src/bnhmm/web/static/index.html`
- Modify: `src/bnhmm/web/static/app.js`
- Modify: `tests/test_boundary_fk_experiment.py`
- Modify: `tests/test_boundary_fk_cli.py`
- Modify: `tests/test_web_api.py`
- Modify: `docs/boundary-flow-fk-synthetic-index-report.md`

- [ ] Add failing API/UI tests proving branch count is fixed at two and obsolete viscosity, boundary-risk, five-gate, and Top-K controls are absent.
- [ ] Remove obsolete job arguments and display the strategy as `两支流 FK`; report both branch memberships, drift, effective potential, state, holdings, turnover, and real portfolio returns.
- [ ] Run all BoundaryFlow/TwoBranch tests, then the full test suite.
- [ ] Run a real semiconductor walk-forward experiment on server 145, update the Chinese report with commands and measured results, and verify artifacts contain no boundary-layer feature names.
- [ ] Run TriadMind sync/verify/govern, commit the completed implementation, deploy to 145, and push GitLab `main`.
