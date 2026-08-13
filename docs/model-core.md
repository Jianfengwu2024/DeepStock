# Bayesian Model Core

The input tensor has shape `[batch, sequence, profile_bins]`. By default, each
sequence item is one completed daily price/relative-volume profile built from
240 one-minute bars. A parallel timeframe tensor uses `1` for these daily
states; `0` remains reserved for opt-in hourly comparison states.

The tokenizer learns three views of every profile: individual bin value and
position, adjacent two-bin shapes, and adjacent three-bin shapes. Their pooled
representations form a temporal token.

All query, key, value, attention-output, feed-forward, motif, transition-context,
latent-emission, and evidence projections use variational Gaussian weights. The
temporal attention mask is causal.

Ten unnamed latent probabilities are filtered sequentially using a Bayesian base
transition matrix plus a context-dependent transition delta. The output head
maps the filtered latent state and Transformer context to five Dirichlet
concentrations ordered from strong bear to strong bull.

Monte Carlo prediction returns:

- five-state posterior probability means;
- aleatoric Dirichlet variance;
- epistemic variance across sampled weights;
- total variance;
- ten latent-state posterior probabilities;
- expected regime score in `[-2, 2]`.

These outputs are model indicators. Position sizing remains in a separate policy
layer and must not be embedded in model training.
