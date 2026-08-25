# Direct Path Efficiency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a selectable long-only `PathEfficiency` strategy that maps each branch's signed path-efficiency sign directly to 50% or 0% capital without FK or Bayesian inference.

**Architecture:** Reuse the causal two-branch dataset and monthly frozen memberships, but introduce separate direct policy, evaluation pipeline, CLI experiment, web job route, and artifacts. Keep `TwoBranchFK` unchanged so direct and FK runs remain comparable.

**Tech Stack:** Python 3.12, NumPy, argparse, FastAPI/Pydantic job API, vanilla JavaScript/HTML, pytest.

---

### Task 1: Direct sign policy

**Files:**
- Create: `src/bnhmm/boundary_fk/direct_policy.py`
- Create: `tests/test_direct_path_policy.py`

- [ ] Write tests for positive-to-50%, negative-to-zero, zero-holds-previous, independent branches, and equal constituent weights.
- [ ] Run `PYTHONPATH=src python -m pytest tests/test_direct_path_policy.py -q` and confirm import/behavior failures.
- [ ] Implement immutable `DirectPathDecision` and `DirectPathPolicy.decide(path_efficiencies, branches, previous_exposures)` with only sign-based transitions.
- [ ] Run the policy tests and confirm all pass.

### Task 2: Causal direct pipeline and artifacts

**Files:**
- Create: `src/bnhmm/boundary_fk/direct_pipeline.py`
- Create: `tests/test_direct_path_pipeline.py`

- [ ] Write a synthetic two-branch pipeline test that verifies next-session alignment, target weights, turnover, transaction costs, and artifact fields.
- [ ] Verify the new test fails because the pipeline is absent.
- [ ] Implement `DirectPathPipeline` over `TwoBranchDataset` using `signed_path_efficiency`, monthly memberships, and evaluation date limits.
- [ ] Write `path-efficiency-predictions.json`, `path-efficiency-evaluation.json`, and a direct Chinese report containing direct and always-invested benchmark metrics.
- [ ] Run policy and pipeline tests until green.

### Task 3: CLI and real-data experiment

**Files:**
- Create: `src/bnhmm/boundary_fk/direct_experiment.py`
- Modify: `src/bnhmm/cli.py`
- Create: `tests/test_direct_path_cli.py`
- Modify: `tests/test_boundary_fk_experiment.py`

- [ ] Write failing parser and experiment tests for `path-efficiency-walk-forward` and shared causal dataset construction.
- [ ] Add the CLI handler and parser arguments: database, symbols, dataset, dates, output, fixed 60-day topology, and transaction cost.
- [ ] Reuse `build_boundary_experiment_data` and upstream topology artifacts; do not instantiate an FK model.
- [ ] Write `run-summary.json` with strategy `path-efficiency-direct` and display name `路径效率基准`.
- [ ] Run the CLI and experiment tests.

### Task 4: Web job API and strategy defaults

**Files:**
- Modify: `src/bnhmm/web/jobs.py`
- Modify: `src/bnhmm/web/app.py`
- Modify: `tests/test_web_jobs.py`
- Modify: `tests/test_web_config.py`

- [ ] Add failing tests accepting `path-efficiency-direct`, requiring a sector dataset, and building a command without FK/model parameters.
- [ ] Add the strategy literal, validation set membership, `_build_direct_path_command`, and config defaults.
- [ ] Expose the three direct artifacts through `_job_payload`.
- [ ] Run the web job/config tests.

### Task 5: DeepStock UI

**Files:**
- Modify: `src/bnhmm/web/static/index.html`
- Modify: `src/bnhmm/web/static/app.js`
- Modify: `tests/test_web_static.py`

- [ ] Write failing static tests for the `路径效率` strategy option, direct-only configuration, hidden FK parameters, CPU behavior, and direct artifact rendering.
- [ ] Add the strategy selector and a compact direct specification showing `正值买入50% / 负值清仓 / 零值不动`.
- [ ] Update strategy state, automatic experiment names, job history labels, progress text, result routing, and direct metrics rendering.
- [ ] Run `node --check src/bnhmm/web/static/app.js` and static tests.

### Task 6: Verification, deployment, and real experiment

**Files:**
- Create: `docs/path-efficiency-direct-report-20260825.md`

- [ ] Run focused tests, then `PYTHONPATH=src python -m pytest -q`, `git diff --check`, JavaScript syntax checks, and TriadMind sync/verify.
- [ ] Deploy changed files to `/home/wujianfeng/private/bnhmm-engine` on 145, normalize line endings, run focused server tests, and restart `bnhmm-web.service`.
- [ ] Submit `PathEfficiency-001` with the exact semiconductor symbols/date range and 10 BP cost used by `TwoBranchFK-004`.
- [ ] Compare direct results with `TwoBranchFK-004` and the always-invested benchmark; write the Chinese report with no inflated claims.
- [ ] Commit on server `main`, push GitLab `main`, and verify `/health`, `/api/config`, the new UI text, and completed experiment artifacts.
