# SectorFlowBayes v1 Implementation and Experiment Report

## Status

SectorFlowBayes v1 is implemented and deployed in DeepStock Lab. It is a
research model and must not be published as a trading model in its current
form. The full 2024-2025 walk-forward experiment finds a small positive gross
cross-sectional signal, but it does not survive transaction costs and its
posterior probabilities are severely miscalibrated.

## Research question

The model tests a sector-first view of stock dynamics:

1. Observe every stock in one sector at the same one-minute timestamps.
2. Infer the latent direction, breadth, activity, and coherence of the sector
   flow.
3. Estimate which stocks lead the flow or respond to it quickly.
4. Prefer stocks that travel efficiently with the flow instead of oscillating
   locally.

No Transformer, CNN, or deep neural network is used. The model is deliberately
small and interpretable.

## Model design

### Synchronized minute panel

All sector members are aligned on a common one-minute grid. Missing cells are
masked and never filled with future values. Symbols with no bars in the
requested period are excluded. The first walk-forward fold then freezes a
trainable universe: a symbol must have enough contiguous lagged observations
before the first sample-out month.

### Causal normalization

For each symbol and minute slot, log-price velocity is normalized only with
historical sessions. Relative activity is normalized against historical volume
at the same slot. The current session never contributes to its own scale.

### Bayesian sector flow

A one-factor observation model extracts the common sector velocity. Principal
loadings are obtained from the symbol covariance matrix, then a robust AR(1)
Bayesian filter estimates the minute-level factor posterior. Each stock also
receives a leave-one-out factor, so its own return cannot mechanically create
the sector signal used to explain it.

The reported water state contains posterior direction, uncertainty, activity,
breadth, coherence, dispersion, and flow strength.

### Lead and response attribution

For every stock, Bayesian ridge distributed-lag regressions estimate:

- whether the stock leads the leave-one-out sector flow;
- whether it responds quickly to prior sector flow;
- the preferred lead and response lag;
- coefficient uncertainty and residual scale.

The default lag range is 0-30 minutes.

### Path and source-proximity posterior

Path statistics are evaluated at 5, 15, 30, and 60 minutes: displacement over
path length, directional agreement, reversals, activity confirmation, and
residual energy. Windows use real timestamps and cannot cross lunch breaks or
overnight gaps.

The source-proximity score is the joint posterior of positive lead or response,
positive coupling, efficient travel, low residual energy, and activity
confirmation. A stock with no current path observations receives zero source
probability.

### Walk-forward protocol

Parameters are fitted on expanding history and frozen for each calendar-month
sample-out fold. Signals become available at the next minute. Every prediction
stores `signal_time`, `available_at`, `training_cutoff`, and `model_version`.
Historical ranking uses a deterministic Beta-normal posterior approximation;
the current detailed ranking retains Monte Carlo component estimates.

## Runtime engineering

The initial implementation repeated rolling-window calculations and 1,000
Monte Carlo draws for every minute and stock. That design was not executable at
full scale. The production implementation now uses:

- vectorized rolling path statistics;
- batched lag construction and attribution scoring;
- a covariance eigensolver instead of tall-matrix SVD;
- deterministic historical posterior composition;
- one detailed Monte Carlo ranking per fold.

A synthetic 120,000-minute by 100-stock path and posterior benchmark completes
in about 14 seconds locally. The full server experiment completes in about
6.5 minutes.

## Experiments

### One-month benchmark

- Job: `SectorFlow-Benchmark-003`
- Period loaded: 2024-01-01 to 2025-03-01
- Sample-out month: December 2024
- Trainable symbols: 82
- Sample-out predictions: 197,112
- Runtime: about 95 seconds including data loading
- Rank IC: 0.02905
- Gross Top-Bottom return: 0.07261%
- Net Top-Bottom return after 10 BP: 0.02977%
- Posterior coverage: 1.33%

This fold was encouraging but not representative enough for a conclusion.

### Full 2024-2025 experiment

- Job: `SectorFlow-002`
- Input period: 2024-01-01 to 2026-01-01
- Sample-out months: February-December 2025
- Walk-forward folds: 11
- Requested symbols: 96
- Trainable symbols: 84
- Excluded symbols: 12 without sufficient period or first-fold history
- Sample-out predictions: 2,036,067
- Runtime: 6 minutes 29 seconds

| Metric | SectorFlowBayes |
| --- | ---: |
| Brier score | 0.477824 |
| Log loss | 8.837628 |
| Rank IC | 0.004257 |
| Gross Top-Bottom return | 0.012424% |
| Turnover | 43.01% |
| Transaction cost | 0.043013% |
| Net Top-Bottom return | -0.030590% |
| Posterior interval coverage | 1.07% |
| Leader-lag stability | 100.00% |
| Response-lag stability | 90.45% |

| Baseline | Rank IC | Net Top-Bottom return |
| --- | ---: | ---: |
| Random | 0.000231 | -0.173071% |
| One-minute momentum | -0.078801 | -0.276914% |
| Relative volume | -0.003994 | -0.098400% |
| Path efficiency | -0.002479 | -0.037198% |
| Static beta | -0.011993 | -0.108602% |

The model has a small positive rank IC and beats the listed baselines, but the
gross edge is too small for the configured execution cost.

## Failure analysis

### Posterior overconfidence

Leader, response, and coupling probabilities frequently saturate near 100%,
while posterior coverage is only 1.07% and log loss is 8.84. The distributed
lag regression treats highly correlated minute observations too much like
independent evidence. Its coefficient covariance therefore shrinks far too
aggressively. This is the primary statistical defect.

### Collapsed source score

At the final endpoint, no stock passes the complete joint source event. The
fixed 0.60 path-efficiency threshold is too strict for noisy one-minute equity
paths, and multiplying several hard posterior events collapses almost every
score to zero. The score is interpretable but poorly calibrated for ranking.

### Weak and expensive edge

The sector factor persistence falls from about 0.245 to 0.226 across folds,
while process variance stays near 0.94. This confirms short memory and high
innovation noise. Re-ranking every minute creates enough turnover for costs to
dominate the weak gross edge.

### Universe and storage limitations

The experiment uses a current semiconductor membership list because historical
membership snapshots are unavailable. This creates survivorship bias. The raw
full-run JSON and CSV predictions also occupy about 1.5 GB; future runs should
write compressed columnar data or the signal database rather than duplicate
uncompressed artifacts.

## Required v2 changes

1. Estimate effective sample size with session blocks or HAC covariance before
   converting lag coefficients into posterior probabilities.
2. Add held-out probability calibration and reject models whose interval
   coverage is materially below target.
3. Replace hard absolute source thresholds with calibrated cross-sectional
   posterior ranks or a soft utility score.
4. Evaluate slower rebalance intervals and minimum holding periods before any
   trading interpretation.
5. Use point-in-time sector membership when a historical constituent source is
   available.
6. Store detailed predictions in Parquet or the independent signal database;
   keep JSON only for compact run summaries.

## Conclusion

The route-1 architecture is computationally viable and the synchronized sector
flow contains a small amount of cross-sectional information. The current
Bayesian attribution layer, however, mistakes serially correlated minute data
for independent evidence and becomes overconfident. The next experiment should
repair effective sample size and calibration before adding any model capacity.

