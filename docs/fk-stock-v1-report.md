# FKStock v1: Conservation-Conditioned Feynman-Kac Experiment

## Objective

FKStock tests whether a stochastic-process model is better suited to market
profiles than semantic tokenization. It keeps StockTransformer and BNHMM intact
and adds a third strategy with no tokenizer, Transformer, neural-network
training, epochs, or GPU dependency.

The implementation separates three layers:

1. Discover a stochastic conservation law for volume density.
2. Estimate a Bayesian local generator for standardized next-session returns.
3. Solve a short-horizon Feynman-Kac posterior under an effective potential.

Volume density is not treated as a price probability density. Conservation
fields only condition the price-return generator.

## Mathematical Model

### Volume-density conservation

For projected relative-volume density `rho(x, t)` on normalized log-price
coordinate `x`, the first layer enforces

```text
delta rho + div J = 0
J = -D grad(rho) + v rho + xi
E[xi | training information] = 0
```

`D`, `v`, and the residual noise scale are fitted only on same-symbol adjacent
density states. No transition is constructed between two stocks.

### Bayesian local generator

Each row uses only information available at signal time. Its features are:

```text
rolling standardized-return drift
latest matured standardized return
rolling return diffusion
signed conservation flow
current coherence
density-center change
density-spread change
stochastic-current fraction
density-gradient energy
```

The target is the next-session standardized return `y`. With standardized
feature vector `z`, the local generator is

```text
y = beta_0 + beta' z + epsilon
epsilon ~ Normal(0, sigma_a^2)
beta | training data ~ Normal(mu_beta, Sigma_beta)
```

Bayesian ridge is solved in closed form. `sigma_a^2` is aleatoric diffusion and
`z' Sigma_beta z` is epistemic drift variance.

### Effective potential and Feynman-Kac map

The effective potential is a bounded non-negative combination of stochastic
flow, posterior uncertainty, and current incoherence:

```text
V = clip(
  V_floor
  + w_noise * stochastic_fraction
  + w_epistemic * epistemic_fraction
  + w_incoherence * (1 - coherence),
  V_floor,
  V_max
)

mu_effective = mu_raw * exp(-V)
```

Low potential does not imply a bullish direction. Direction comes only from the
sign of `mu_raw`; the potential suppresses unsupported drift. Monte Carlo draws
from the coefficient posterior produce class-probability means and epistemic
variance. Conditional normal overlap supplies aleatoric probability variance.

The four output intervals are:

```text
(-inf, -q), [-q, 0), [0, q), [q, +inf)
```

where `q` defaults to `0.75` rolling return standard deviations.

## Walk-Forward Controls

Every fold follows this sequence:

1. Fit the conservation law and generator using labels whose `target_time` is
   strictly earlier than the tuning boundary.
2. Evaluate the fixed model on the tuning interval without updating it.
3. Refit both layers using labels mature before the outer-test boundary.
4. Freeze the model and predict the untouched outer interval.

Feature histories include only returns with `target_time < signal_time`.
Newly listed constituents without symbol-specific training moments use pooled
sector moments calculated before the first outer-test cutoff. Predictions still
retain their own symbol, execution time, target time, and return scale.

Each fold saves `model.json` with the conservation law, coefficient mean and
covariance, feature normalization, aleatoric variance, class prior, configs,
and training cutoff. `fk-diagnostics.json` stores point-in-time drift,
diffusion, effective potential, survival factor, and conservation diagnostics.

## Semiconductor Experiment

Run ID:

```text
20260815-084339-fkstock-semiconductor-v1-smoke-3e2537
```

Configuration:

```text
Dataset: semiconductor
Requested members: 96
Members producing usable observations: 84
Data interval: 2024-01-01 to 2025-09-30
Minimum training sessions: 252
Tuning sessions: 60
Outer test sessions: 30
Walk-forward folds: 1
Generator fit rows: 19,629
Outer predictions: 2,497
MC posterior samples: 32
Runtime: about 3 minutes 48 seconds
```

Outer-test classification:

| Metric | Result |
| --- | ---: |
| Accuracy | 25.79% |
| Training-prior majority baseline | 32.36% |
| Balanced accuracy | 24.81% |
| Macro F1 | 24.17% |
| Brier score | 0.7477 |
| Training-prior Brier | 0.7280 |
| Log loss | 1.3812 |
| Training-prior log loss | 1.3418 |
| Return correlation | -0.0060 |

Policy result:

| Metric | Result |
| --- | ---: |
| Total return | -0.44% |
| Benchmark return | 0.86% |
| Maximum drawdown | -3.70% |
| Average absolute position | 33.09% |
| Active-position ratio | 43.37% |

The release validation gate failed. FKStock v1 remains research-only.

## Diagnostic Finding

The most important result is structural rather than predictive:

```text
mean stochastic-current fraction: 1.0000
mean current coherence: 0.9389
mean effective potential: 0.3906
effective-potential standard deviation: 0.0454
mean aleatoric diffusion variance: 1.1825
mean epistemic drift variance: 0.00019
```

The linear law `J = -D grad(rho) + v rho + xi` explains almost none of the
observed current: its residual norm is approximately the full current norm.
Consequently the noise-derived potential is nearly constant across observations.
The Feynman-Kac layer mainly shrinks a weak ridge-regression drift and cannot
separate states. Increasing potential weights would only suppress drift more;
it would not create missing information.

The next experiment should improve the generator before tuning the solver:

1. Replace globally constant `D` and `v` with leakage-safe local functions of
   price coordinate and density, while retaining closed-form or low-dimensional
   Bayesian inference.
2. Decompose current into sector-common and symbol-specific modes, then test
   whether either mode is conserved out of sample.
3. Measure whether density moments have incremental information over the rolling
   return generator through nested ablations.
4. Retain v1 as the fixed linear baseline and require every extension to beat
   its prior-probability and zero-return baselines.

## Verification

Local and server verification completed on 2026-08-15:

```text
79 tests passed before the new-listing fallback test
80 tests passed after the fallback test
Python compileall passed
JavaScript syntax check passed
Desktop browser layout passed
390 x 844 mobile layout passed with no horizontal overflow
Remote Web task completed and generated all evaluation artifacts
```
