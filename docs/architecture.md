# Architecture Baseline

The `HoloBNHMM` variant is documented in
[`holographic-bnhmm.md`](holographic-bnhmm.md). It shares labels, walk-forward
splits, Bayesian heads, and evaluation with BNHMM while replacing only the
daily profile builder and tokenizer with a two-dimensional price/participation
flow field.

## Runtime boundaries

The local web console separates data-service status, strategy configuration,
training results, local manual stock pools, AI provider settings, and Quant
Platform entitlement verification. The first five pages manage local runtime
state; authenticated cloud reads are limited to dataset metadata and the
signed-in user's saved AI stock-pool metadata and symbols.
Passwords and provider API keys are not stored in the cloud catalog, and
generated samples, checkpoints, and reports remain on the local workstation.

1. The data layer reads one symbol and a bounded date range at a time.
2. The profile layer creates one variable-length daily conditional phase-space
   state from each one-minute trading session. The previous close is zero and
   fixed 10-basis-point log-price bins preserve the real daily amplitude. Each
   price bin carries relative volume plus sample-weighted, smoothed price-velocity
   and volume-velocity distributions. Small source gaps do not discard the full day; hourly states
   are opt-in only.
3. The model layer encodes price motifs as the backbone and uses relative volume
   as its selected auxiliary branch. Ten-minute joint velocity remains available
   only for controlled ablation. It then applies a causal Transformer and a
   conditional 20-state transition model.
   The hidden-state count is configurable and may not be smaller than the profile
   channel count.
   Every symbol owns a complete model instance; Transformer weights, hidden-state
   transitions, evidence heads, and return moments are never shared across symbols.
4. The filtered latent state feeds a Bayesian standardized Student-t return
   posterior and a direct four-state evidence head. The evidence updates a
   training-only class prior. Brier, ordinal, and cross-head consistency losses
   are disabled after strict ablation; both aleatoric and epistemic uncertainty
   remain explicit.
5. The long-only policy targets half size on weak-bull, full size on strong-bull,
   and closes on either bearish state independently of the model.
6. Walk-forward evaluation publishes only point-in-time-correct predictions.
   Each model is trained on one symbol's past and tested only on that same
   symbol's future. Multi-symbol runs are orchestration and portfolio aggregation.
7. Quant Platform bulk-loads published predictions for a complete backtest.

## Leakage boundary

At prediction time `t`, feature normalization uses only the symbol's ten completed
sessions strictly before the current session. The current session is appended to
the rolling history only after its state is built. A signal becomes executable at
`available_at`, which must be
strictly later than the final input bar. Historical backtests select a model
whose `training_cutoff` is no later than its signal time.

Labels use the first bar strictly after the completed profile as their entry
price. A daily state completed at 15:00 therefore enters on the first continuous
bar of the next trading session. The target is scaled by the volatility of the
preceding 20 known sessions, while zero remains actual zero return. The rolling
mean is diagnostic only; no outer-fold observation is
used to set labels, class priors, or class weights. Walk-forward folds contain an inner tuning
window; after choosing the epoch count, the model is rebuilt on all information
whose labels mature before the outer-test cutoff. The outer window never
participates in model selection or fitting.

## Deployment boundary

The engine source, Python environment, weights, reports, and PostgreSQL database
remain outside the Quant Platform repository. Quant Platform owns only a generic
read-only adapter for external precomputed signals.

## Local market datasets

The browser submits a local stock-pool selection to the DeepStock service. A
background worker reads bounded symbol/month ranges through the local market
broker and materializes the requested `1m`, `5m`, `15m`, `30m`, `60m`, `2h`,
and `1d` bars in a separate DuckDB file under `local-state/market-datasets`.
Strategy jobs read that local file. Each dataset keeps its requested universe,
interval set, per-symbol coverage, row counts, and update timestamps in a local
SQLite catalog.

The service catches up the latest completed weekday after market close and
records failed attempts for manual retry or the following day's run. Incremental updates
overlap the last five calendar days so late revisions are replaced by the
primary key. Backfill, daily refresh, walk-forward training, and reusable FK
precomputation share a process lock around the DuckDB file. This prevents a
writer from changing a database while a strategy is reading it. Dataset and
model files remain on the local machine; the broker remains read-only.

## Persistent strategy inputs and progress

`BoundaryDataBuildManager` owns explicit FK preparation and shares the training
resource lock. `GET /api/strategy-datasets` validates saved manifests against the
source fingerprint, construction version, ordered symbols, date coverage, and
file existence without rereading minute bars. Local end dates are exclusive.
The cloud adapter converts its inclusive end date once before local validation.
Local and cloud FK jobs require prepared inputs and fail early when validation
fails; they do not invoke preparation. Repeated explicit preparation reuses valid
inputs, while changed variants retain the previous NPZ and manifest entry in
the dataset's version history. Build status persists across service restarts;
an unfinished build is exposed as interrupted.

Calculation processes atomically save `progress.json` beside their artifacts.
Counts advance after actual units complete: FK prediction days, BNHMM folds,
and preparation units for the current data stage. Epoch and batch updates expose
long-running work inside a fold. Phase ETA is measured phase elapsed time times
remaining units divided by completed units, not a timer-based percentage. No
completed units means no ETA. Backtest totals survive the evaluation phase,
so report generation cannot be mistaken for more backtesting. JobManager
overlays actual job elapsed time and terminal status. Failed, cancelled, and
interrupted jobs retain partial completion instead of being marked 100%.

The Training Board polls every three seconds and displays an indeterminate bar
before counts exist. After 30 seconds without a progress update it shows a
freshness warning and suppresses the old ETA. Completed progress is retained
and redisplayed after a reload. This telemetry changes no model calculations.

## Current source normalization

Historical one-minute rows include a 09:30 boundary/auction row while recent
rows begin at 09:31. Profile construction therefore admits only 09:31-11:30 and
13:01-15:00, yielding the same 240 continuous-trading minutes in every era. The
shared `kline_60m` table currently contains mixed six-row and four-row daily
conventions. Default profile states are built directly from complete `kline_1m`
sessions; `kline_60m` is retained only for validation and optional context.

Daily profiles are cropped to their observed low/high coordinate range and are
never resized to a common width. Batches pad only to their local maximum width
and carry a bin mask. A 512-bin cap keeps the 10-basis-point scale unchanged;
extreme overflow is deposited into edge bins and marked in state metadata.

Price coordinates use log returns relative to previous close. Relative minute
volume is divided by prior ten-day mean daily volume divided by 240. Contiguous
minutes are grouped into non-overlapping 10-minute blocks separately on each
side of the lunch break. Mean log price and mean relative volume are calculated
per block; price and `log1p(relative_volume)` velocities use the same adjacent
block pair, the same midpoint price anchor, and per-minute units. Both are
standardized from identically constructed blocks in the preceding ten sessions.
Clock time, acceleration, and hand-built direction-times-volume features are
intentionally absent.
