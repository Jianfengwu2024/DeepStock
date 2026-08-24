# Sector River Topology and SectorFlowBayes v2 Design

## Status

Approved in conversation on 2026-08-24. This document defines the design only;
it does not authorize implementation until the user has reviewed this file.

## Problem

The original sector river slice placed stocks on the vertical axis using a
fixed or incidental symbol order. Symbol order has no geometric meaning, so
vertical neighborhoods did not represent sector structure. The image looked
two-dimensional, but the second dimension was not a valid coordinate.

The revised model treats each completed trading day as one sector-state slice.
The horizontal axis is synchronized one-minute time. The vertical axis is a
causal, monthly frozen ordering derived from stock synchronization. Nearby rows
therefore represent stocks that tend to move together, and contiguous row bands
represent persistent subflows inside the sector.

## Goals

1. Build a causal monthly sector topology from the previous 60 trading days.
2. Freeze stock bands and row order for the entire following calendar month.
3. Separate sector-wide co-flow from residual subflow synchronization.
4. Use relative volume only as observation density and reliability, not as a
   directional volume velocity.
5. Produce an interactive DeepStock Lab workspace for daily slices, monthly
   topology, month-to-month migration, and structural diagnostics.
6. Feed the full synchronization graph, not only its one-dimensional visual
   projection, into a small Bayesian graph state model.
7. Predict next-trading-day cross-sectional relative return from a completed
   daily slice. The signal is first available at the next tradable session.
8. Evaluate the design with strict walk-forward, calibration, ablation, and
   structural tests.

## Non-goals

- No policy or news features; sector selection is already complete.
- No Transformer, large CNN, or high-capacity deep model in v2.
- No training or inference inside Quant Platform backtests.
- No use of future bars, future constituents, or within-month reordering.
- No volume-speed feature.
- No claim that visually adjacent rows contain every relevant relationship;
  the full graph remains the mathematical source of neighborhood structure.
- No publication as a trading model unless strict sample-out results pass the
  rejection criteria in this document.

## Causal Timeline

For prediction month `M`:

1. The topology cutoff is the final completed trading session of `M-1`.
2. The topology builder reads at most the 60 completed trading sessions ending
   at that cutoff.
3. Bands, band identifiers, and row ordering are frozen for all days in `M`.
4. A daily slice for date `D` is complete only after the close of `D`.
5. A prediction created from that slice has `signal_time = close(D)` and
   `available_at = next_tradable_time_after(close(D))`.
6. The target is each stock's next-session return relative to the sector's
   next-session return. Training labels therefore never affect topology or
   features for the same prediction.

Every prediction retains `signal_time`, `available_at`, `training_cutoff`, and
`model_version`.

## Architecture

The design has four isolated layers.

### 1. Monthly River Topology

`RiverTopologyBuilder` consumes a normalized synchronized sector panel and the
previous month's topology. It returns an immutable `MonthlyRiverTopology`:

- topology month and training cutoff;
- 60-session observation window;
- symbol universe and coverage;
- common-flow and residual-flow similarity matrices;
- the fused, uncertainty-shrunk synchronization graph;
- band assignments and stable band identifiers;
- within-band row order;
- cohesion, centrality, and migration diagnostics for every symbol;
- unassigned symbols and exclusion reasons.

The topology object is independent of predictive targets and can be validated
without training SectorFlowBayes v2.

### 2. Daily River Slice

`DailyRiverSliceBuilder` uses the frozen topology for its calendar month. It
produces a fixed-shape stock-by-minute tensor with channels:

1. normalized signed log-price velocity;
2. velocity residual after removing the causal sector common factor;
3. relative-volume density;
4. observation-validity mask.

Rows follow the frozen monthly order and columns follow the real A-share minute
grid. Lunch, overnight, suspensions, and absent bars remain explicitly masked.
The builder never forward-fills across a market gap.

### 3. Bayesian Graph State Model

`SectorGraphStateModel` consumes the full monthly synchronization graph and the
daily slice. Its decomposition is deliberately small:

`observed stock flow = sector flow + band flow + stock deviation + noise`

The stock-deviation prior uses the monthly graph Laplacian. Strongly connected
stocks are encouraged, but not forced, to share a latent state. Relative volume
changes observation uncertainty only: dense observations receive more weight;
volume never determines direction.

The model produces posterior summaries for:

- sector and band direction/strength;
- stock coupling to its band;
- stock departure from its band;
- stock lead/lag evidence;
- next-session relative-return score and uncertainty.

The visual row coordinate is not used as a substitute for graph adjacency.

### 4. DeepStock Lab River Workspace

The existing SectorFlow result panel keeps compact metrics and gains an
`Open river view` action. The action opens a dedicated full-width workspace
with four tabs:

1. **Daily river slice**: date selector, channel selector, band boundaries,
   minute hover details, and stock row selection.
2. **Monthly topology**: synchronization graph, band membership, cohesion,
   centrality, cutoff, and 60-session window.
3. **Monthly migration**: band-to-band stock movement and stable band matching.
4. **Structural diagnostics**: within-band synchronization, random-order
   comparison, topology stability, migration rate, and coverage.

Selecting a stock shows its graph neighbors, band rationale, current departure,
relative-volume density, and recent monthly positions. The UI always displays
the topology cutoff and marks stale or incomplete data explicitly.

## Synchronization Definition

The topology builder calculates two independent pairwise views.

### Common-flow synchronization

For each stock pair, estimate robust rank correlation between causally
normalized one-minute price velocities. Pair observations are weighted by a
capped function of the two stocks' relative-volume density and validity. This
measures whether the stocks join the same sector flow at the same time.

### Residual-subflow synchronization

First remove the robust causal sector common factor from every stock velocity.
Then estimate the same weighted robust correlation on residual velocities.
This prevents a broad market or sector move from collapsing all stocks into one
undifferentiated band and reveals persistent subflows.

Negative association is treated as distance, not similarity. Relative volume
is capped before weighting so a few high-volume minutes cannot dominate the
topology. Correlation uncertainty uses trading-session blocks rather than
treating serially correlated minutes as independent. Weak estimates shrink
toward zero according to effective sample size.

The two shrunk similarity matrices are fused with equal fixed weight. This is
the only v2 default; it must remain visible in ablation results rather than be
optimized against the final sample-out target.

## Stable Bands and Row Ordering

1. Build a positive synchronization graph from the fused similarities.
2. Form candidate stable communities from historical data only.
3. Choose band count by bootstrap stability and within-band separation; use a
   one-standard-error preference for the previous month's count, then for the
   simpler count. Predictive labels are not used.
4. Match new bands to previous stable identifiers by maximum member overlap.
5. Permit a stock to migrate only when block-bootstrap evidence favors the new
   band's synchronization over the old band's synchronization.
6. Apply optimal leaf ordering inside each band using fused correlation
   distance.
7. Orient each ordered band to minimize movement of overlapping symbols versus
   the previous month. This removes arbitrary left-right tree flips.

The output must be invariant, within deterministic tie-breaking, to the input
symbol list order.

## Universe and Missing-data Rules

- The symbol universe is frozen at the topology cutoff for the following month.
- A symbol lacking sufficient 60-session coverage is placed in an unassigned
  area and excluded from graph inference.
- A newly listed symbol can enter a band only at a later monthly rebuild after
  sufficient causal history exists.
- A symbol that becomes suspended remains in its frozen row with masked cells
  for that month; it is reconsidered at the next rebuild.
- If an entire topology month lacks sufficient data, the pipeline does not
  silently fit on future or partial current-month data. It either reuses the
  previous topology with a visible `stale` marker for visualization-only use or
  rejects that fold for predictive evaluation.

## Storage and API

Large prediction JSON files are not extended. Compact summaries remain JSON;
matrix and daily data are stored in compressed columnar or array artifacts.
The API loads one month or one day on demand.

Required artifact families:

- `river-topology-YYYY-MM` with graph, bands, order, cutoff, coverage, and
  diagnostics;
- `river-slice-YYYY-MM-DD` with the four channels and mask;
- `river-migration` with stable band mapping and stock moves;
- `river-structure-report` with structural baselines and ablations;
- existing evaluation and Markdown report artifacts extended for v2.

The web API should extend the existing job/artifact boundary rather than create
a second job system. It exposes job-scoped topology, day-slice, migration, and
stock-detail reads. Training remains offline in this repository; Quant Platform
continues to consume only published precomputed signals.

## Walk-forward Evaluation

Evaluation is monthly expanding walk-forward:

1. Fit normalization, topology, graph-state parameters, and any calibration on
   training data ending before the sample-out month.
2. Freeze the topology and model for the entire sample-out month.
3. Build each completed day's slice and predict the next trading session.
4. Rank stocks cross-sectionally by posterior expected relative return adjusted
   for uncertainty.
5. Apply the trading policy only in a separate evaluation layer.

Primary predictive metrics are Rank IC, Top-Bottom return, turnover, net return
after configured costs, Brier score, log loss, calibration error, posterior
coverage, and residual autocorrelation. Structural metrics are within-band
synchronization lift over random grouping, modularity/separation, adjusted Rand
stability, migration rate, unassigned rate, and coverage.

## Required Ablations

All variants use the same folds, targets, and cost assumptions:

1. SectorFlowBayes v1.
2. Random row order / no topology.
3. Common-flow synchronization only.
4. Residual-subflow synchronization only.
5. Full similarity without month-to-month stability constraints.
6. Full SectorFlowBayes v2.

No ablation may select features or parameters on the final sample-out folds.

## Tests

### Causality and contracts

- Changing data after a topology cutoff cannot change that topology.
- Changing data after a daily close cannot change that day's slice.
- Row order and band assignment stay fixed throughout a month.
- Predictions contain all four required timing/version fields.
- Prediction availability is the next tradable time, not the same close.

### Geometry

- Permuting input symbol order preserves bands and canonical ordering.
- Within-band synchronization exceeds a matched random grouping on synthetic
  planted communities.
- Stable identifiers survive harmless cluster label permutations.
- Band orientation does not flip arbitrarily across adjacent months.
- Graph matrices are symmetric, finite, and compatible with their symbol map.

### Normalization and missing data

- Multiplying raw volume by a constant does not change relative-density-driven
  topology.
- Missing bars do not create synthetic velocity edges.
- Lunch and overnight gaps never become one-minute observations.
- Sparse/new symbols remain unassigned rather than contaminating a band.

### Web and artifacts

- Topology, slice, migration, and stock-detail endpoints return job-scoped data.
- Missing artifacts return explicit not-found states.
- The river workspace remains usable at desktop and mobile widths.
- Heatmap, graph, migration, and inspector views render with real artifacts.

## Rejection Criteria

The model remains research-only if any of the following holds:

- causal or timing tests fail;
- topology is not materially more coherent than random grouping;
- posterior calibration remains severely overconfident;
- v2 improvement disappears across sample-out folds;
- cost-adjusted performance is not better than the declared baselines;
- results depend strongly on incidental symbol order or one unstable month.

Negative results are retained in the final Markdown report. They are not hidden
by selecting a favorable fold, band count, or visualization.

## Expected Implementation Boundaries

Prefer extending the existing sector-flow modules and web job/artifact system.
The expected new responsibilities are:

- a topology module under `src/bnhmm/sector_flow/`;
- a daily river-slice module under `src/bnhmm/sector_flow/`;
- a graph-state module under `src/bnhmm/sector_flow/`;
- pipeline/report integration in the existing SectorFlow pipeline;
- job-scoped read endpoints in the existing FastAPI app;
- a dedicated static river workspace reusing the current DeepStock visual
  patterns;
- focused unit, integration, causality, and browser rendering tests.

The implementation plan must use TriadMind impact navigation before code edits
and preserve the independent DeepStock repository boundary.
