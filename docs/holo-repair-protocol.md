# HoloBNHMM Sequential Repair Protocol

This protocol repairs HoloBNHMM through cumulative, causally isolated models. The legacy `HoloBNHMM` and `HoloLite` implementations and all historical runs remain unchanged.

## Fixed Experiment Contract

- Dataset, date range, walk-forward folds, seed, epochs, batch size, transaction costs, and evaluation code remain identical inside a comparison round.
- Every stage predicts only the next trading session and uses only information available before the signal cutoff.
- Every completed run writes `run-summary.json`, the standard plots, and `repair-stage-report.md`.
- A later model cannot be selected until its direct predecessor has been implemented and validated.
- Accuracy is never interpreted alone. Balanced accuracy, Macro F1, Brier score, Log Loss, return correlation, class support, and the majority/prior/shuffled controls are mandatory.

## Model Sequence

| Stage | Model | Isolated change |
|---:|---|---|
| 0 | HoloProbe-Prior | Predict from each training fold's class prior without reading any input. |
| 1 | HoloProbe-Metadata | Read only K-line metadata and direct recent-return lags; image cells are zeroed. |
| 2 | HoloProbe-Image | Read only the two-dimensional field; metadata and direct autoregression are disabled. |
| 3 | HoloProbe-Shuffled | Keep the full legacy model but reassign images across symbols inside each batch. |
| 4 | HoloPhase | Replace the ambiguous volume-level axis with 10-minute price velocity; volume becomes cell mass. |
| 5 | HoloResNet | Replace the shallow daily encoder with a masked anisotropic residual CNN. |
| 6 | HoloResTCN | Replace daily-vector Transformer memory with a short causal temporal convolution. |
| 7 | HoloSticky | Replace the diffuse 50-state filter with 8-12 sparse sticky states. |
| 8 | HoloStudent | Remove the separate classification evidence head; derive regimes from one Student-t return posterior. |
| 9 | HoloSoftBand | Apply soft/no-trade supervision around the low-edge zero-return region. |
| 10 | HoloEnsemble | Use five independently initialized members for meaningful epistemic uncertainty. |

## First Promotion Gate

`HoloPhase` work starts only after the four HoloProbe controls complete under the same folds and compute budget. The current image has incremental information only if `HoloProbe-Image` consistently improves balanced accuracy and probabilistic loss over `HoloProbe-Prior`, `HoloProbe-Metadata`, and `HoloProbe-Shuffled`. A single favorable accuracy value is insufficient.

Later stages use the same rule: compare each model only with its direct predecessor, document the exact delta, retain it only when the improvement is reproducible, and otherwise stop or revise that stage.
