# Architecture Baseline

## Runtime boundaries

1. The data layer reads one symbol and a bounded date range at a time.
2. The profile layer creates one variable-length daily conditional phase-space
   state from each one-minute trading session. The previous close is zero and
   fixed 10-basis-point log-price bins preserve the real daily amplitude. Each
   price bin carries relative volume plus price-velocity and volume-velocity
   distributions. Small source gaps do not discard the full day; hourly states
   are opt-in only.
3. The model layer encodes adjacent two-bin and three-bin motifs, then
   applies a Bayesian Transformer and a conditional 50-state transition model.
   The hidden-state count is configurable and may not be smaller than the profile
   channel count.
   Every symbol owns a complete model instance; Transformer weights, hidden-state
   transitions, evidence heads, and return moments are never shared across symbols.
4. The output layer emits bearish, flat, and bullish posterior means and variances,
   through a fixed nine-node Bayesian hidden layer after the conditional latent
   state. It includes aleatoric and epistemic uncertainty. Flat covers forward
   returns from -0.5% through +0.5%.
5. The long-only policy layer opens on bullish, keeps the previous position on flat,
   and closes on bearish independently of the model.
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
bar of the next trading session. Walk-forward folds contain an inner tuning
window; after choosing the epoch count, the model is rebuilt on all information
whose labels mature before the outer-test cutoff. The outer window never
participates in model selection or fitting.

## Deployment boundary

The engine source, Python environment, weights, reports, and PostgreSQL database
remain outside the Quant Platform repository. Quant Platform owns only a generic
read-only adapter for external precomputed signals.

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
volume is divided by prior ten-day mean daily volume divided by 240. Price
velocity is the one-minute log-price difference; volume velocity is the
difference of `log1p(relative_volume)`. Both are standardized from the same ten
historical sessions and discretized into conditional histograms at each price
bin. Clock time, acceleration, and hand-built direction-times-volume features
are intentionally absent.
