# Architecture Baseline

## Runtime boundaries

1. The data layer reads one symbol and a bounded date range at a time.
2. The profile layer creates one variable-length daily price/relative-volume
   state from each one-minute trading session. The previous close is zero and
   fixed 10-basis-point log-price bins preserve the real daily amplitude. Small
   source gaps do not discard the full day; hourly states are opt-in only.
3. The model layer encodes adjacent two-bin and three-bin motifs, then
   applies a Bayesian Transformer and a conditional ten-state transition model.
4. The output layer emits five-state posterior means and variances, including
   aleatoric and epistemic uncertainty.
5. The policy layer maps posterior output to target positions independently of
   the model.
6. Walk-forward evaluation publishes only point-in-time-correct predictions.
7. Quant Platform bulk-loads published predictions for a complete backtest.

## Leakage boundary

At prediction time `t`, feature normalization may use only bars with timestamps
at or before `t`. A signal becomes executable at `available_at`, which must be
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
