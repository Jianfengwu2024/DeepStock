# BNHMM 10-Minute Velocity and Strict Ablation

## Decision

The production BNHMM input is `price-volume`. Price and relative volume remain
active; the 10-minute joint price/volume velocity field remains a research
option but is disabled by default. The network keeps two Transformer blocks,
20 unnamed latent states, state metadata, the conditional transition, the
latent filter, and the direct short-return path.

Production training removes three objectives that did not produce a stable
tuning gain: Brier loss, ordinal loss, and cross-head consistency. It retains
Student-t return likelihood, direct classification, latent information, Z2
mirror consistency, Bayesian KL, and causal self-supervised pretraining.

This is still a research model. The frozen outer period shows a small
probability-score improvement over the training prior, not a tradable accuracy
advantage.

## Ten-Minute Joint Velocity

- Contiguous one-minute rows are split into non-overlapping 10-minute blocks.
- Mean log price and mean relative volume are calculated in each block.
- Adjacent block means form per-minute average price and volume velocities.
- Both velocities use the same block pair and midpoint log-price anchor.
- Lunch breaks, missing minutes, and incomplete tails cannot create a velocity.
- Scaling is fitted only from identically constructed blocks in the preceding
  ten completed sessions.

The data contract is `phase-space-10m-joint-return-v7`. A complete 240-minute
session contributes at most 22 paired velocity samples because each side of the
lunch break needs two blocks before its first finite difference.

## Selection Protocol

- Dataset: `semiconductor`, 96 available constituents.
- Source range: 2024-06-01 through 2026-08-05.
- Fit sequences: 15,248.
- Tuning sequences: 4,657, starting 2025-04-16.
- Frozen outer sequences: 5,176, from 2025-07-15 through 2025-10-15.
- Seeds: 7, 17, and 29 for every candidate.
- A deletion required non-inferior mean Log Loss and Brier score and had to pass
  the same test on at least two of three seeds.
- Candidate selection used only the tuning period. The outer period was opened
  once after all decisions.

The full artifact is outside Git at
`/home/wujianfeng/private/bnhmm-storage/ablation/bnhmm-semiconductor-10m-v2-strict-20260815`.

## Input Ablation

| Input | Parameters | Accuracy | Balanced accuracy | Log Loss | Brier |
|---|---:|---:|---:|---:|---:|
| Price only | 71,904 | 0.32575 | 0.25973 | 1.35113 | 0.73306 |
| **Price + volume** | **74,496** | **0.31186** | **0.25583** | **1.34932** | **0.73205** |
| Price + 10m velocity | 75,264 | 0.32224 | 0.25734 | 1.35126 | 0.73321 |
| Price + volume + 10m velocity | 77,856 | 0.31258 | 0.25704 | 1.34966 | 0.73212 |

The full input was statistically near the selected input but used more active
parameters. Velocity by itself did not improve probability scores. The result
supports temporal aggregation as a cleaner representation, but does not show
incremental predictive value from velocity in this window.

## Architecture Deletions

| Deletion | Tuning Log Loss | Tuning Brier | Accepted |
|---|---:|---:|---:|
| One Transformer layer | 1.35711 | 0.73633 | No |
| Zero Transformer layers | 1.35404 | 0.73441 | No |
| 13 latent states | 1.35214 | 0.73382 | No |
| Static transition | 1.35209 | 0.73361 | No |
| No direct autoregression | 1.35111 | 0.73270 | No |
| No state metadata | 1.35331 | 0.73421 | No |
| Independent latent states | 1.35673 | 0.73627 | No |

All architecture reductions failed the strict non-inferiority rule. Parameter
count alone is therefore not enough to remove these components.

## Objective Deletions

| Deletion | Tuning Log Loss | Tuning Brier | Accepted |
|---|---:|---:|---:|
| No pretraining | 1.35226 | 0.73338 | No |
| No Z2 symmetry | 1.37255 | 0.74375 | No |
| No latent information | 1.35252 | 0.73363 | No |
| No head consistency | 1.34932 | 0.73204 | Yes |
| No ordinal loss | 1.34960 | 0.73216 | Yes |
| No Brier loss | 1.34985 | 0.73230 | Yes |
| Deterministic training | 1.35469 | 0.73468 | No |

Accepted deletions were applied sequentially. Deterministic training was tested
against the already reduced objective set and rejected.

## Frozen Outer Result

| Metric | BNHMM | Training-prior baseline |
|---|---:|---:|
| Accuracy | 0.26719 | - |
| Balanced accuracy | 0.25587 | 0.25000 random reference |
| Log Loss | 1.39161 | 1.39683 |
| Brier | 0.75241 | 0.75521 |

Class support was `[1014, 1322, 1422, 1418]`; predicted counts were
`[183, 2997, 1996, 0]`. The absence of strong-bull predictions and the small
balanced-accuracy margin are explicit release blockers. The model has weak
probability information, but it has not passed a trading-quality gate.
