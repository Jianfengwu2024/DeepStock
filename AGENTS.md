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

<!-- TRIADMIND_RULES_START -->
# TriadMind Always-On Rules

- Before answering architecture questions, read `.triadmind/triad-map.json`.
- Before generating or modifying code, read `.triadmind/config.json` and `.triadmind/master-prompt.md`.
- Do not jump straight into implementation when a topology upgrade is required.
- Prefer the TriadMind sequence: Macro -> Meso -> Micro -> draft-protocol -> visualizer -> apply -> handoff.
- If the user message starts with `@triadmind`, treat it as a TriadMind directive.
- If the body is a control command like `init`, `macro`, `meso`, `micro`, `finalize`, `plan`, `apply`, `renormalize`, `heal`, or `handoff`, route to the matching TriadMind lifecycle action.
- Otherwise, treat it as a silent topology-upgrade demand: run the full protocol workflow first, then continue to apply and handoff.
- Use `reuse` first, then `modify`, and only use `create_child` when the current leaf node cannot safely absorb the new responsibility.
- If a runtime error occurs, prefer generating a repair protocol via `.triadmind/healing-prompt.md` instead of ad-hoc code edits.
<!-- TRIADMIND_RULES_END -->
