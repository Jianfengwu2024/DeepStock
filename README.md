# BNHMM Engine

BNHMM Engine is the offline training and batch-inference boundary for the
Bayesian neural hidden-state market model. It is intentionally independent from
Quant Platform.

The first implemented vertical slice reads bounded OHLCV batches from the shared
DuckDB broker, computes a historical-only relative-volume baseline, and converts
one-minute bars into hourly and daily price/relative-volume profile states.

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

## Server smoke check

```bash
python -m bnhmm.cli profile-smoke \
  --broker-url http://127.0.0.1:9000/api/v1/db \
  --symbol 000001 --start 2026-07-01 --end 2026-08-01
```

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
