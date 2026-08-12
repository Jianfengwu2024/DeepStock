# Architecture Baseline

## Runtime boundaries

1. The data layer reads one symbol and a bounded date range at a time.
2. The profile layer creates hourly and daily price/relative-volume states.
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

## Deployment boundary

The engine source, Python environment, weights, reports, and PostgreSQL database
remain outside the Quant Platform repository. Quant Platform owns only a generic
read-only adapter for external precomputed signals.

## Current source normalization

Historical one-minute rows include a 09:30 boundary/auction row while recent
rows begin at 09:31. Profile construction therefore admits only 09:31-11:30 and
13:01-15:00, yielding the same 240 continuous-trading minutes in every era. The
shared `kline_60m` table currently contains mixed six-row and four-row daily
conventions; canonical hourly profile states are derived from `kline_1m`, while
`kline_60m` is retained only for validation and optional context after auditing.
