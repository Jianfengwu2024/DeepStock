# BNHMM Engine

BNHMM Engine is the offline training and batch-inference boundary for the
Bayesian neural hidden-state market model. It is intentionally independent from
Quant Platform.

The first implemented vertical slice reads bounded OHLCV batches from the shared
DuckDB broker, computes a historical-only relative-volume baseline, and converts
one-minute bars into hourly and daily price/relative-volume profile states.

The model core now includes adjacent two/three-bin motif tokenization, causal
Bayesian multi-head attention, a conditional ten-state latent transition filter,
and a five-regime Dirichlet evidence head. `predict_posterior()` separates
aleatoric and epistemic variance through Monte Carlo weight sampling.

The offline trainer reads bounded symbol/date slices from a static DuckDB
replica, constructs labels from future trading-session returns, and purges any
training observation whose label matures on or after the validation cutoff.
Model checkpoints and metrics belong in the external `bnhmm-storage` tree.

## Data contract

Required input columns are `symbol`, `timestamp`, `open`, `high`, `low`, `close`,
and `volume`. The production source currently contains:

- `kline_1m`: core profile input.
- `kline_60m`: validation and multi-scale context.
- `kline_1d`: labels and multi-scale context.

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

model = BNHMMNetwork(profile_bins=48)
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
  --device cuda:0 --epochs 10
```

The output directory contains `model.pt` and `metrics.json`. The checkpoint
records the data range, temporal cutoff, architecture, training configuration,
and best validation epoch. It is an unpublished training artifact and is never
read directly by Quant Platform.

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
