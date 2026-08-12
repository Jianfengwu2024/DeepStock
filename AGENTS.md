# BNHMM Engine Development Rules

- This repository is independent from Quant Platform. Never copy model code,
  weights, training dependencies, or storage files into the Quant Platform Git
  repository.
- The shared market DuckDB is owned by an HTTP database broker. Read it in
  bounded symbol/date chunks through the broker; do not open the DuckDB file
  directly while the broker is running.
- Every historical prediction must retain `signal_time`, `available_at`,
  `training_cutoff`, and `model_version`.
- Training and inference happen offline. Quant Platform consumes only published,
  precomputed signals.
- Run tests with Python 3.12. On server 144, use the `rheos` conda environment
  until the engine receives its own environment.

