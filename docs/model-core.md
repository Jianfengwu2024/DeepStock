# Bayesian Model Core

The default padded input tensor has shape `[batch, sequence, bins, 22]`, where
`bins` varies by batch. Four scalar channels contain price occupancy,
relative-volume shape, normalized relative-volume level, and normalized log-price
location. Nine channels encode the conditional normalized price-velocity
distribution and nine encode the conditional normalized volume-velocity
distribution. A bin mask excludes padding and an absolute
10-basis-point coordinate tensor preserves whether the whole profile lies below
or above the previous close.

The tokenizer learns position-aligned individual-bin, adjacent two-bin, and
adjacent three-bin motifs. Coordinate Fourier features prevent cropped profiles
from losing their absolute return location. Masked attention pooling combines
the motifs with open/close/range, total relative volume, bar coverage, and
overflow metadata to form one daily token. No exact intraday clock position,
price acceleration, or direction-times-volume product is provided to the model.

All query, key, value, attention-output, feed-forward, motif, transition-context,
latent-emission, and evidence projections use variational Gaussian weights. The
temporal attention mask is causal.

Ten unnamed latent probabilities are filtered sequentially using a Bayesian base
transition matrix plus a context-dependent transition delta. The output head
maps the filtered latent state and Transformer context to four Dirichlet
concentrations ordered as strong bear, weak bear, weak bull, and strong bull.
Forward returns use `[-2%, 0%, +2%]` as the three boundaries. There is no
neutral output: low-conviction cases remain expressible through balanced
directional probabilities and posterior uncertainty, which the separate policy
may convert to a near-zero position.

Monte Carlo prediction returns:

- four-state posterior probability means, with no neutral class;
- aleatoric Dirichlet variance;
- epistemic variance across sampled weights;
- total variance;
- ten latent-state posterior probabilities;
- expected regime score in `[-2, 2]`.

These outputs are model indicators. Position sizing remains in a separate policy
layer and must not be embedded in model training.
