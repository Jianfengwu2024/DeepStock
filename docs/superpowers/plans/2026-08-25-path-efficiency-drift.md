# Path Efficiency Drift Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pure signed-path-efficiency drift model to TwoBranchFK and compare it fairly with legacy inputs and legacy-plus-path inputs.

**Architecture:** Keep daily state construction and FK forecasting unchanged. Add one explicit drift feature set containing only `signed_path_efficiency`, wire it into the existing walk-forward policy ablation, and expose unambiguous Chinese labels in reports and the web UI. The current official closing-flow policy remains unchanged until the new candidate passes the documented promotion gate.

**Tech Stack:** Python 3.12, NumPy, pytest, FastAPI static UI, existing TwoBranchFK Bayesian regression and FK pipeline.

---

### Task 1: Pure Path Drift Feature Set

**Files:**
- Modify: `src/bnhmm/boundary_fk/branch_model.py`
- Test: `tests/test_two_branch_fk_model.py`

- [ ] **Step 1: Write the failing feature-set test**

Add an assertion that `DRIFT_FEATURE_SETS["path_efficiency_only"]` equals exactly `("signed_path_efficiency",)` and that a fitted posterior using this set exposes no legacy feature names.

- [ ] **Step 2: Run the focused test and verify failure**

Run: `PYTHONPATH=src pytest tests/test_two_branch_fk_model.py -q`

Expected: failure because `path_efficiency_only` is not defined.

- [ ] **Step 3: Add the minimal model configuration**

Add:

```python
"path_efficiency_only": ("signed_path_efficiency",),
```

to `DRIFT_FEATURE_SETS` and its `Literal` configuration. Do not change priors, half-life, target, state construction, or potential inputs.

- [ ] **Step 4: Run the focused model tests**

Run: `PYTHONPATH=src pytest tests/test_two_branch_fk_model.py -q`

Expected: all tests pass.

### Task 2: Fair Walk-Forward Ablation

**Files:**
- Modify: `src/bnhmm/boundary_fk/pipeline.py`
- Modify: `src/bnhmm/boundary_fk/reporting.py`
- Modify: `src/bnhmm/web/static/app.js`
- Test: `tests/test_boundary_fk_pipeline.py`
- Test: `tests/test_boundary_fk_reporting.py`
- Test: `tests/test_web_static.py`

- [ ] **Step 1: Write failing ablation contract tests**

Require a new policy key `path_efficiency_only`, mapped to the pure feature set. Require report and UI labels:

```text
纯路径效率
旧四项 + 路径效率
```

The existing `path_efficiency` key remains the legacy-plus-path calculation so historical result files keep their meaning.

- [ ] **Step 2: Run focused tests and verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_boundary_fk_pipeline.py tests/test_boundary_fk_reporting.py tests/test_web_static.py -q
```

Expected: failures for the missing policy and labels.

- [ ] **Step 3: Wire the candidate through the existing pipeline**

Add `path_efficiency_only` to `POLICY_ABLATION_NAMES` and map it to the new drift feature set. Keep `posterior_signal` mapped to `closing_flow`. Update report and UI labels without changing evaluation formulas.

- [ ] **Step 4: Run focused tests and syntax checks**

Run:

```bash
PYTHONPATH=src pytest tests/test_two_branch_fk_model.py tests/test_boundary_fk_pipeline.py tests/test_boundary_fk_reporting.py tests/test_web_static.py -q
node --check src/bnhmm/web/static/app.js
git diff --check
```

Expected: all commands pass.

### Task 3: Verification and Same-Protocol Experiment

**Files:**
- Modify only if required by a failing regression test.

- [ ] **Step 1: Run the complete local test suite**

Run: `PYTHONPATH=src pytest -q`

Expected: all tests pass; existing PyTorch transformer warnings are acceptable.

- [ ] **Step 2: Deploy to server 145 and verify service health**

Synchronize changed files to `/home/wujianfeng/private/bnhmm-engine`, run focused tests with the `rheos` environment, restart `bnhmm-web`, and require `/api/config` to return HTTP 200.

- [ ] **Step 3: Run the semiconductor walk-forward experiment**

Use the same semiconductor dataset, dates, costs, FK paths, posterior gates, half-life, and fold count as the comparison result. Confirm the generated ablation JSON includes both `path_efficiency_only` and `path_efficiency`.

- [ ] **Step 4: Evaluate without overstating the result**

Compare net return, maximum drawdown, drift rank IC, drift direction accuracy, opening rate, and turnover. Report whether PathOnly passes each promotion condition; do not claim stability from one fold.

- [ ] **Step 5: Commit and push**

Commit source, tests, specification, plan, and resulting report changes to GitLab `main`, then verify the server worktree is clean and the service remains active.
