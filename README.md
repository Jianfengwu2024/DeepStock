# BNHMM Engine

BNHMM Engine is the offline training and batch-inference boundary for the
Bayesian neural hidden-state market model. It is intentionally independent from
Quant Platform.

The first implemented vertical slice reads bounded OHLCV batches from the shared
DuckDB broker and converts each trading session's available one-minute bars into
one variable-length daily log-price phase-space profile. The previous close is
coordinate zero and every log-price bin is 10 basis points wide. Each price bin
contains price occupancy, relative-volume mass, confidence-weighted relative-volume
level, and a 3-by-3 joint price/volume velocity distribution. Price and relative
volume are first averaged inside contiguous 10-minute blocks. Each joint sample
then uses the same pair of adjacent blocks for both average velocities and is
anchored at the midpoint of their mean log prices. Lunch breaks, missing-minute
blocks, and cross-gap transitions are excluded. Sparse joint rows are smoothed
and weighted by their real sample share instead of being normalized to equal
confidence.
All scalers use only the symbol's preceding ten completed sessions. Relative
volume divides minute volume by prior ten-day mean daily volume divided by 240.
Hourly states remain an explicit comparison mode only.

The model core uses deterministic price motifs and a deterministic causal
Transformer backbone. Its default memory is a fixed rolling window of the five
most recent completed daily states. Each window predicts only the next-session
return attached to its final state; training never supervises truncated prefixes
or selects a longer context at inference. The production input uses price and
relative volume. Paired 10-minute joint velocity remains an independently gated
research branch, but strict three-seed ablation did not justify enabling it by
default. Four feature modes support controlled ablation. Bayesian layers begin at
the emission model and feed a configurable
latent-state filter. Its context-conditioned transition correction is low rank,
while its base transition matrix remains Bayesian. The filtered state posterior
  feeds a Bayesian standardized-return head and a Bayesian four-state evidence
  head. The hidden-state count defaults to 20 and cannot be smaller than the
  input channel count.

The target return is scaled by volatility from the preceding 20 known sessions,
while zero remains an actual flat return. The noisy rolling mean is retained only
as an evaluation baseline and is never subtracted from the label. Production
training combines a robust Student-t likelihood with direct four-state
classification, latent-state information, Z2 mirror consistency, Bayesian KL,
and causal self-supervised pretraining. Brier, ordinal, and return/classification
head-consistency training terms were removed by strict ablation. The evidence
head starts from the symbol/fold training
prior, then learns only the information contributed by the current state.
Both supervised heads use a small nonzero centered initialization so predictive
gradients reach the latent emissions and transition filter from the first update.
Strong and weak states are separated at 0.75 rolling standard deviations by
default; zero separates bearish from bullish.
`predict_posterior()` uses Monte Carlo weight sampling to report expected return,
predictive variance, and separate aleatoric and epistemic uncertainty.

## StockTransformer

StockTransformer is an independent conservation-first strategy alongside BNHMM.
It discovers a stochastic volume-density continuity law on each walk-forward
training slice, derives variable-width words and sentence boundaries from density
gradients, current reversals, and stochastic-current residuals, and only then
trains a token-only Transformer. The conservation, tokenization, and model layers
have separate configuration and data contracts; none can fit or reach backward
through another layer. See `docs/stock-transformer-v1-report.md` for the equations,
implementation boundaries, tests, and the first semiconductor-sector experiment.

## FKStock

FKStock is a compact Bayesian stochastic-process baseline without tokenization
or neural-network training. A nested ablation over the semiconductor dataset
retained only the rolling diffusion of fully matured standardized returns. The
production Bayesian ridge posterior therefore contains an intercept and one
feature coefficient, then maps its predictive return distribution to
strong-bear, weak-bear, weak-bull, and strong-bull probabilities.

FKStock always uses sector-pooled training, but transitions and rolling return
history never cross symbol boundaries. Every walk-forward fold fits only labels
mature before its cutoff; the frozen outer model records its cutoff and complete
coefficient posterior. FKStock runs on CPU and has no epochs, network depth,
hidden states, attention heads, conservation field, effective potential, or
token parameters. See `docs/fk-stock-v2-ablation-report.md` for the nested
selection protocol and complete ablation table. The rejected v1 design remains
documented in `docs/fk-stock-v1-report.md`.

## SectorFlowBayes v2

SectorFlowBayes v2 builds a causal monthly synchronization topology from the
prior 60 completed sessions, freezes the stock ordering for the full month, and
projects each completed trading day into a stock-by-minute river slice. A small
Bayesian graph-Laplacian model predicts next-session cross-sectional relative
return. DeepStock Lab provides daily river, topology, migration, structure, and
stock-inspector views through job-scoped compressed artifacts. See
`docs/sector-flow-bayes-v2-report.md` for the implementation, leakage controls,
browser verification, and the 96-stock semiconductor walk-forward experiment.

Self-supervised pretraining uses only states available before each walk-forward
cutoff. It reconstructs randomly masked daily metadata and fixed-width
mass/location/spread summaries, then predicts their next-session values from
causal context before the noisy return target is introduced. Pretraining and
supervised fine-tuning use separate deterministic random streams so controlled
ablations are reproducible. Effective batches are automatically capped at 256
daily-state tokens to preserve multiple optimizer updates on small per-symbol
datasets and bound memory use on long sequences.

The offline trainer reads bounded symbol/date slices from a static DuckDB
replica, constructs labels from future trading-session returns, and purges any
training observation whose label matures on or after the validation cutoff.
Model checkpoints and metrics belong in the external `bnhmm-storage` tree.

## Data contract

Required input columns are `symbol`, `timestamp`, `open`, `high`, `low`, `close`,
and `volume`. The production source currently contains:

- `kline_1m`: core profile input.
- `kline_60m`: optional validation context; not used to build default states.
- `kline_1d`: label validation context; not used to build default states.

Only completed bars are admitted. Financial statement fields are not used.

## Local API

```python
from bnhmm import BNHMMConfig, BrokerBarSource, PriceVolumeProfileBuilder

source = BrokerBarSource("http://127.0.0.1:9000/api/v1/db")
bars = source.fetch("000001", "1m", "2026-07-01", "2026-08-01")
states = PriceVolumeProfileBuilder(BNHMMConfig().profile).build(bars)
```

```python
from bnhmm.model import BNHMMNetwork

model = BNHMMNetwork()
posterior = model.predict_posterior(profiles, timeframe_ids, mc_samples=16)
```

## Server smoke check

```bash
python -m bnhmm.cli profile-smoke \
  --broker-url http://127.0.0.1:9000/api/v1/db \
  --symbol 000001 --start 2026-07-01 --end 2026-08-01
```

## Real-data training

Install the `training` extra in the engine environment, then run a bounded
single-symbol experiment:

```bash
python -m bnhmm.cli train \
  --duckdb-path /home/share/data/market.duckdb \
  --symbol 000090 --start 2025-01-01 --end 2026-08-05 \
  --output-dir /home/wujianfeng/private/bnhmm-storage/artifacts/000090 \
  --device cuda:0 --epochs 10 --state-timeframe daily
```

The output directory contains `model.pt` and `metrics.json`. The checkpoint
records the data range, temporal cutoff, architecture, training configuration,
and best validation epoch. It is an unpublished training artifact and is never
read directly by Quant Platform.

## Walk-forward evaluation

`walk-forward` can orchestrate several symbols in one command. Every fold uses an inner tuning
period to select the epoch count, retrains on all data available before the
outer test, and then predicts the untouched outer period. A multi-symbol command
creates one complete BNHMM per symbol; no model or transition parameters are
shared between symbols:

```bash
python -m bnhmm.cli walk-forward \
  --duckdb-path /home/share/data/market.duckdb \
  --symbols 000090,000099,000488 \
  --start 2024-01-01 --end 2026-08-05 \
  --output-dir /home/wujianfeng/private/bnhmm-storage/artifacts/walk-forward \
  --device cuda:0 --state-timeframe daily
```

Reports include out-of-sample four-directional predictions, Brier score, log loss,
calibration, probability/return/standardized residuals, residual
autocorrelation, posterior predictive checks, and policy equity/drawdown. The
report also compares the model with majority-class, prior-state persistence,
state reversal, zero-return, and prior-return baselines. A model remains marked
`research-only` unless it beats the majority and zero-return baselines, has
positive return correlation, exceeds random balanced accuracy, and contains at
least 200 out-of-sample observations. The
policy is a separate consumer of the model posterior. It targets half size on
weak-bull and full size on strong-bull signals, and closes on either bearish
signal; it never opens short positions. Only epistemic model
uncertainty reduces position confidence; irreducible aleatoric class overlap
remains a reported risk diagnostic. Completed daily signals
execute at the first continuous-trading bar of the next session.

## DeepStock Lab

The no-auth console can switch between BNHMM and StockTransformer without
requiring terminal commands. It keeps one GPU worker active at a time and
persists each job, log, checkpoint, and report under the external storage tree.

```bash
cp .env.example .env
# Edit .env when the host, port, or storage path differs.
python -m bnhmm.web.app
```

The Python entry point loads `.env` from the repository root automatically. Open
`http://SERVER_IP:8765` when using the example port. The console is intended for
the trusted local network and must not be exposed directly to the public internet.

The server installation includes convenience scripts in the repository root:

```bash
./start.sh
./stop.sh
```

Both scripts load the same root `.env`. The systemd template also loads it through
`EnvironmentFile`; if the repository is installed elsewhere, update
`WorkingDirectory`, `EnvironmentFile`, `PYTHONPATH`, and `ExecStart` in
`deploy/bnhmm-web.service` together.

`stop.sh` refuses to interrupt active training. Use `./stop.sh --force` only
when the active jobs should be cancelled before the service stops.

## Storage

The initial PostgreSQL migration stores immutable model versions, compressed
weights, symbol bindings, point-in-time predictions, training jobs, metrics, and
report artifacts. Quant Platform must query only published model versions and
must never load model weights.

The server deployment uses a dedicated PostgreSQL 15 container bound to
`127.0.0.1:15454`. Its credentials and persistent data live in the sibling
`bnhmm-storage` directory and are never committed to Git.

Start it with the external environment file so the host data path is resolved:

```bash
cd deploy
docker compose --env-file ../../bnhmm-storage/postgres.env up -d
```
