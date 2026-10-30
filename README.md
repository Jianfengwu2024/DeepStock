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

## Bayesian FK

The second supported strategy predicts the next trading session for one whole
sector. Every sample uses one continuous 60-session price path. Each symbol is
anchored to the close immediately before that 60-session window, so overnight
movement and diffusion across daily boundaries are preserved. The coordinate is
never reset at each close.

The Bayesian drift uses only rolling log position and rolling path efficiency.
One-day reverse-flow concentration enters only as the C1 external potential; it
cannot directly reverse the drift. The current model is always
`FK-Bayesian-Rolling60`. Monthly walk-forward fitting admits only labels mature
before the cutoff, and cached inputs are versioned with the
`rolling-60-session-previous-close` anchor protocol.

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

## Quant Platform integration

DeepStock production and training run on the local GPU workstation. Configure
`BNHMM_MARKET_SOURCE` to the local read-only Quant DuckDB broker; the engine
queries bounded symbol/date ranges and never opens the broker-owned database
file. The local web console binds to `127.0.0.1` and keeps datasets, checkpoints,
and reports under the ignored `local-state` directory.

The Quant Platform API at `/api/v1/deepstock/catalog` stores only administrator-
maintained dataset categories, names, descriptions, date ranges, and feature
schemas. It does not receive generated samples, source databases, checkpoints,
or trained models. Catalog reads require a logged-in account with the existing
`fast_screener` entitlement; administrators can always read and maintain the
catalog directly.

On Windows with an NVIDIA RTX 50-series GPU, run `setup-local.ps1` to create an
isolated Python 3.12 environment with CUDA 12.8 PyTorch, then start the local
console with `start-local.ps1`. The console is intentionally not exposed to the
public network.

The local console has six sections: Data Service, Deep Strategy, Training
Dashboard, Stock Pools, AI Configuration, and User Verification. Stock pools
are stored in the ignored local state directory and can load symbols into a
strategy run. The AI configuration accepts a free-form model name and can query
an OpenAI-compatible `/models` endpoint. Provider settings are saved in this
browser; the API key stays in the current browser session and is sent only
when checking the provider.

User Verification checks a Quant Platform account and the existing
`fast_screener` entitlement before displaying the remote dataset catalog. The
password is forwarded over HTTPS and discarded after login. Its access token
stays in page memory and can be cleared from the verification page.

After account verification, the Stock Pools page lists the signed-in user's
saved AI screener pools from Quant Platform. Syncing a pool copies its symbols,
description, run ID, and source pool ID into local state; syncing it again
updates the existing local copy instead of creating duplicates.

List the permission-filtered cloud catalog with `bnhmm-catalog list`. After a
local visual dataset exporter completes successfully, an administrator can
publish only its description and feature schema:

```powershell
bnhmm-catalog publish-visual .\local-state\visual-datasets\my-dataset\manifest.json `
  --category "图像训练集" --name "示例数据集" --description "日级价格与成交参与图像样本"
```

The command prompts for an administrator access token without echoing it. The
token is used for that process only and is not written to disk. Generated data,
market-source URLs, images, checkpoints, and reports remain on this computer.

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

The no-auth console can switch between BNHMM and Bayesian FK without
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

On the Windows research workstation, `start-local.ps1` points DeepStock at the
local read-only market broker and the CUDA GPU. Create a stock pool, then use
the Data Service page to materialize its selected `1m` through `1d` intervals
into a separate local DuckDB dataset. Initial history backfill and the daily
after-close refresh run in the background. Deep strategies select a ready local
dataset and train from that file; the shared broker is never used as the model's
training database.

Completed FK runs open as an interactive research dashboard in Training Board.
The evaluation overview contains gate status, return/drawdown metrics, monthly
direction accuracy against its matched majority baseline, and signal counts.
Returns are reconstructed from daily strategy returns minus turnover costs;
the resettable internal trailing-stop equity is not used as cumulative NAV.
Months without scored signals remain gaps. Charts open in a larger dialog;
prediction tables support direction filtering, pagination, and point-in-time
signal/holding details. Posterior intervals use a labelled normal approximation.
Raw JSON stays available in collapsed details for audit. These views read only
saved local artifacts and do not fetch market data or restart the market guardian.

Training Board also displays a persisted task monitor: current phase, completed
and remaining work, backtest percentage, elapsed time, and estimated remaining
time for the current phase. FK reports completed prediction days; BNHMM reports
completed walk-forward folds with the current stock, epoch, and batch. Estimates
use measured completed work, remain unknown until the first unit finishes, and
are hidden when progress has not updated for 30 seconds. The page polls every
three seconds; final status and counts remain visible after completion or a
page reload. A 100% backtest can still be followed by the report phase.

The Data Service page keeps reusable FK inputs under **策略数据集**. Prepare and
save these inputs explicitly before running a strategy. A repeated preparation
request validates and reuses an unchanged dataset. Jobs never rebuild missing
inputs: missing date coverage, changed source fingerprints, member universes,
or construction versions require explicit preparation. Rebuilt variants archive
their previous files and metadata. Market bars, strategy inputs, and run results
remain separate local artifacts; a service restart rediscovers saved inputs.

Completed FK runs can be published from Training Board with **发布到云端策略**
after signing in on the User Verification page. Quant Platform stores the result
under that account and displays it at Cloud Strategies → FK-贝叶斯. Publication
is idempotent by local run ID. Cloud-created jobs return the same result format
automatically, including measured progress, daily NAV/drawdown, direction
probabilities, prediction history, and fold sample counts. The cloud receives
only research results and run metadata; minute bars, model coefficients,
checkpoints, dataset files, and local filesystem paths stay on the workstation.

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
