# FKStock v2: Nested Ablation and Minimal Model

## Decision

FKStock v2 keeps one active input: `rolling_diffusion`, the rolling standard
deviation of fully matured standardized returns. The production path removes
the conservation field, density geometry, drift, latest return, and effective
potential. The fitted posterior therefore has only two coefficients: an
intercept and one feature coefficient.

This is a model reduction, not a claim that FKStock has passed the release
gate. Predictive performance remains weak and the strategy stays
research-only.

## Protocol

- Dataset: `semiconductor`, 96 available constituents.
- Source range: 2024-01-01 through 2025-09-30.
- Fit observations: 14,681.
- Tuning observations: 4,866, from 2024-11-19 through 2025-02-19.
- Frozen outer observations: 2,497, from 2025-02-20 through 2025-04-02.
- Feature selection used the tuning period only.
- A larger model was allowed only when both Log Loss improved by more than
  0.001 and Brier score improved by more than 0.0005.
- Models within tolerance were resolved in favor of fewer active features.
- The outer period was opened only after feature and potential selection.

The full artifact remains outside Git at
`/home/wujianfeng/private/bnhmm-storage/ablation/fkstock-v2-20260815` on the
145 server.

## Feature Ablation

| Candidate | Inputs | Tuning Log Loss | Tuning Brier | Outer Log Loss | Outer Brier |
|---|---:|---:|---:|---:|---:|
| Intercept only | 0 | 1.387262 | 0.750549 | 1.383219 | 0.748576 |
| Rolling drift | 1 | 1.387972 | 0.751007 | 1.382235 | 0.748104 |
| Latest matured return | 1 | 1.387362 | 0.750753 | 1.385186 | 0.749649 |
| **Rolling diffusion** | **1** | **1.385231** | **0.749653** | **1.382149** | **0.748209** |
| Signed conservation flow | 1 | 1.387473 | 0.750664 | 1.383556 | 0.748749 |
| Current coherence | 1 | 1.388529 | 0.751346 | 1.385265 | 0.749683 |
| Density center change | 1 | 1.388023 | 0.750962 | 1.383490 | 0.748729 |
| Density spread change | 1 | 1.387518 | 0.750712 | 1.380949 | 0.747481 |
| Stochastic current fraction | 1 | 1.387283 | 0.750560 | 1.383187 | 0.748561 |
| Density gradient energy | 1 | 1.386970 | 0.750405 | 1.382674 | 0.748311 |
| Return memory | 2 | 1.388150 | 0.751344 | 1.384432 | 0.749227 |
| Return dynamics | 3 | 1.385006 | 0.749823 | 1.382209 | 0.748176 |
| Density geometry | 3 | 1.387856 | 0.750875 | 1.380471 | 0.747249 |
| Return plus density | 4 | 1.388834 | 0.751595 | 1.384089 | 0.749231 |
| Conservation flow | 3 | 1.388751 | 0.751475 | 1.385698 | 0.749904 |
| Return plus flow | 5 | 1.389031 | 0.751866 | 1.387950 | 0.751084 |
| All v1 inputs | 9 | 1.385510 | 0.750094 | 1.383050 | 0.748858 |

`density-geometry` happened to score better on the unopened outer period, but
it did not improve the tuning period and therefore cannot be selected without
leaking the test result into model design.

## Effective-Potential Ablation

All potential candidates used the selected single feature.

| Candidate | Tuning Log Loss | Tuning Brier | Outer Log Loss | Outer Brier |
|---|---:|---:|---:|---:|
| **No potential** | **1.385231** | **0.749653** | **1.382149** | **0.748209** |
| Epistemic only | 1.385231 | 0.749653 | 1.382149 | 0.748209 |
| Flow only | 1.384929 | 0.749403 | 1.381632 | 0.747804 |
| Full v1 potential | 1.384929 | 0.749403 | 1.381632 | 0.747804 |

The apparent tuning gain from the flow potential was 0.000303 Log Loss and
0.000250 Brier. Both are below the precommitted materiality thresholds, so the
potential was removed.

## Production Changes

- Default active features changed from 9 to 1.
- The production row builder skips density projection and conservation-law
  fitting entirely.
- Effective-potential controls are absent from the production solver config.
- Monte Carlo and potential controls are absent from the FKStock UI and job
  request. Research-only potential settings live only in the ablation path.
- The UI states exactly which terms were removed instead of presenting unused
  tuning knobs.

## Interpretation

The ablation rejects the original physical narrative for this dataset and
window. FKStock v2 is currently a small Bayesian volatility-regime baseline,
not evidence that the conservation field or Feynman-Kac potential predicts the
next trading day. Any rejected term must earn its way back through a new,
predeclared nested sample-out experiment.
