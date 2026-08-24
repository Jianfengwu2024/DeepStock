# SectorFlowBayes v2: Causal Sector River Topology

## 1. Objective

SectorFlowBayes v2 models a sector as one synchronized flow rather than treating
each stock as an isolated path. A completed trading day is a stock-by-minute
slice:

- the horizontal axis is the real 240-minute exchange clock;
- the vertical axis is a monthly frozen stock topology;
- signed normalized price velocity is the primary flow channel;
- relative volume is density and observation reliability, never volume speed;
- missing bars remain masked and never bridge the lunch break or another gap.

The prediction target is each stock's next-session close-to-close log return
minus the sector cross-sectional mean. The model therefore asks which stocks
will advance or lag relative to the same sector flow, not whether the whole
market will rise.

## 2. Causal Monthly Topology

For month `M`, the topology uses only completed bars through the last session of
`M-1`. It reads the preceding 60 sessions and remains frozen for every day in
`M`.

Two synchronization matrices are estimated with activity-weighted Spearman
correlation:

1. common synchronization from normalized one-minute price velocity;
2. residual synchronization after removing the Bayesian sector factor.

Effective-session shrinkage pulls weakly observed pairs toward zero. Uniformly
rescaling volume does not change the graph because activity is normalized to a
per-stock reliability scale. The fused similarity matrix is clustered by
average linkage. Candidate band counts are selected by silhouette score, stable
band identifiers are aligned to the prior month by maximum member overlap, and
optimal leaf ordering produces the visual row projection. The full graph, not
the row order, enters inference.

## 3. Daily Graph State and Prediction

Each completed daily slice produces six stock-level statistics:

- normalized displacement;
- band displacement;
- stock-minus-band displacement;
- path efficiency;
- residual energy;
- median relative-volume density.

A graph-Laplacian Gaussian posterior smooths stock state only along positive
monthly graph edges. Relative volume changes observation precision, so active
evidence reduces uncertainty without becoming a directional feature. A small
Bayesian ridge head then combines the six statistics, graph posterior mean and
variance, cohesion, and centrality. It outputs next-session relative-return mean
and predictive variance. This is deliberately a small graph model: no
Transformer, CNN, PyTorch runtime, or loading-time training is involved.

## 4. Walk-Forward and Leakage Controls

- Monthly folds use expanding history.
- Factor parameters and the return head are fitted only on fold training rows.
- A training daily slice is paired only with its already matured next-session
  target.
- The sample-out monthly topology is fitted through the prior month and frozen.
- `training_cutoff < signal_time < available_at` is enforced by data contracts.
- A signal formed at the completed session close becomes available at the next
  session's first observed minute.
- Each monthly topology is built once per run and reused across folds.
- Backtests evaluate policy and transaction costs separately from model fitting.

Artifacts are compact and job-scoped: one compressed topology per month, one
compressed slice per predicted day, a manifest, evaluation JSON, structural
diagnostics, and Markdown reports. The browser API serves only manifest-listed
files and never returns every daily slice in one response.

## 5. DeepStock Lab Workspace

The dedicated river workspace contains four views:

- **Daily river:** Canvas heatmap with price velocity, residual velocity, and
  density channels; monthly band separators remain fixed.
- **Synchronization topology:** bounded SVG graph with band color, centrality,
  and selectable stock nodes.
- **Monthly migration:** stable row and band movement between monthly freezes.
- **Structure diagnostics:** coverage, band count, cohesion, centrality, and the
  ordered stock table.

Selecting a heatmap row or topology node opens a stock inspector with daily
displacement, residual energy, graph statistics, and bounded recent predictions.
Desktop `1440x900` and mobile `390x844` were verified. The mobile heatmap scrolls
inside its own region, the inspector moves below it, and there is no page-level
horizontal overflow.

## 6. Semiconductor Experiment

The server experiment used:

| Setting | Value |
| --- | ---: |
| Dataset | `/home/share/data/sectors/semiconductor.duckdb` |
| Stocks | 96 |
| Input interval | 2024-01-01 to 2025-12-31 |
| Initial training history | 120 sessions |
| Topology lookback | 60 sessions |
| Normalization lookback | 20 sessions |
| Sample-out months | 17 |
| Sample-out predictions | 29,095 |
| Daily river slices | 344 |
| Transaction cost | 10 BP |

### Prediction Results

| Metric | Result | Reference |
| --- | ---: | ---: |
| Rank IC | 0.066223 | 0 for random ranks |
| Gross Top-K minus Bottom-K | 0.332636% | 0 |
| Turnover | 158.488372% | - |
| Transaction cost | 0.158488% | 10 BP policy |
| Net Top-K minus Bottom-K | 0.174148% | 0 |
| Brier score | 0.249875 | 0.25 binary uninformed |
| Log loss | 0.692947 | 0.693147 binary uninformed |
| 95% posterior coverage | 94.755113% | 95% nominal |

The positive Rank IC and net cross-sectional spread indicate that the river
state contains some ranking information. Calibration coverage is close to its
nominal level. Probability discrimination is nevertheless only marginally
better than an uninformed binary forecast.

### Structural Results

| Metric | Result |
| --- | ---: |
| Mean synchronization lift over random equal-size groups | 0.001798 |
| Minimum monthly synchronization lift | 0.001197 |
| Mean unassigned rate | 8.209438% |
| Mean band count | 2.0 |
| Mean monthly migration rate | 0.828835% |

Every tested month has positive synchronization lift, and monthly membership is
very stable. However, the lift is small and the model consistently selects only
two bands. The topology is therefore stable but weakly separated. Stability
alone is not evidence of a strong economic partition.

## 7. Decision and Next Experiments

SectorFlowBayes v2 passes the engineering and causality gates but remains a
research candidate, not a published trading model. Promotion should require:

1. positive Rank IC and net Top-Bottom spread across multiple non-overlapping
   sectors and market regimes;
2. materially positive topology lift over matched random groups;
3. probability scores that beat the uninformed Brier and log-loss baselines by
   more than the current marginal difference;
4. stable results under common-only, residual-only, random-topology, unstable
   daily-order, and no-graph ablations on identical sample-out dates;
5. turnover-aware portfolio tests with realistic limit-up, suspension, and
   liquidity eligibility rules.

The most important next change is not to enlarge the model. It is to test
whether stronger target-independent sector geometry can be extracted without
using future returns. If topology lift remains near zero, the graph prior should
be removed even if the ranking head remains useful.

## 8. Verification

- Local complete suite: 182 tests passed.
- Server complete suite in the production `rheos` environment: 182 tests passed.
- SectorFlow-focused suite after monthly topology caching: 57 tests passed.
- JavaScript syntax checks passed for the main Lab and river workspace.
- TriadMind topology verification passed with 54 nodes, no ghost nodes, no
  completeness violations, and no high-fanout cycles. The separate governance
  command could not evaluate rendered-edge consistency because this repository
  has no `runtime-map.json` or `runtime-diagnostics.json` instrumentation output.

