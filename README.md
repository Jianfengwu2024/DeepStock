# BNHMM Engine

BNHMM Engine is the offline training and batch-inference boundary for the
Bayesian neural hidden-state market model. It is intentionally independent from
Quant Platform.

The first implemented vertical slice reads bounded OHLCV batches from the shared
DuckDB broker and converts each trading session's available one-minute bars into
one variable-length daily log-price phase-space profile. The previous close is
coordinate zero and every log-price bin is 10 basis points wide. Each price bin
contains price occupancy, relative-volume mass, confidence-weighted relative-volume
level, and a 3-by-3 joint price/volume velocity distribution. Each joint sample
comes from one strictly continuous minute edge: both velocities share the same
`(t-1, t)` pair and are anchored at the midpoint of the two log close prices.
Lunch breaks and missing-minute edges are excluded. Sparse joint rows are smoothed
and weighted by their real sample share instead of being normalized to equal
confidence.
All scalers use only the symbol's preceding ten completed sessions. Relative
volume divides minute volume by prior ten-day mean daily volume divided by 240.
Hourly states remain an explicit comparison mode only.

The model core uses price motifs as its backbone. Relative volume and paired joint
velocity use gated residual motif branches initialized near zero, so
adding noisy features cannot immediately overwhelm the price baseline. Four
feature modes support controlled ablation. The temporal core includes causal
Bayesian multi-head attention, a conditional 50-state latent transition filter,
a fixed nine-node Bayesian output hidden layer, and a three-regime Dirichlet
evidence head. The first hidden-state count is configurable
but cannot be smaller than the input channel count; the second hidden layer is
always nine nodes. `predict_posterior()` separates
aleatoric and epistemic variance through Monte Carlo weight sampling.

Optional self-supervised pretraining uses only states available before each
walk-forward cutoff. It reconstructs randomly masked daily state features and
predicts the next daily state features from causal context before the noisy
three-regime target is introduced. Pretraining and supervised fine-tuning use
separate deterministic random streams so controlled ablations are reproducible.

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

Reports include out-of-sample bearish/flat/bullish predictions, Brier score, log loss,
calibration, probability/return/standardized residuals, residual
autocorrelation, posterior predictive checks, and policy equity/drawdown. The
policy is a separate consumer of the model posterior. It opens or holds a long
position on bullish signals, preserves the prior position on flat signals, and
closes on bearish signals; it never opens short positions. Completed daily signals
execute at the first continuous-trading bar of the next session.

## Test console

The no-auth console exposes the same walk-forward parameters without requiring
terminal commands. It keeps one GPU worker active at a time and persists each
job, log, checkpoint, and report under the external storage tree.

```bash
BNHMM_WEB_JOB_ROOT=/home/wujianfeng/private/bnhmm-storage/ui-runs \
python -m bnhmm.web.app
```

Open `http://SERVER_IP:8765`. The console is intended for the trusted local
network and must not be exposed directly to the public internet.

The server installation includes convenience scripts in the repository root:

```bash
./start.sh
./stop.sh
```

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
