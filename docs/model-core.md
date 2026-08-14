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
latent-emission, output-hidden, and evidence projections use variational Gaussian
weights. The temporal attention mask is causal.

Fifty unnamed latent probabilities are filtered sequentially by default using a
Bayesian base transition matrix plus a context-dependent transition delta. A
fixed nine-node Bayesian hidden layer then maps only the filtered latent state
into three Dirichlet concentrations ordered as bear, flat, and bull. The first
hidden-state count is configurable but cannot fall below the 22 input channels;
the second hidden layer is always nine nodes. Forward returns below -0.5% are bearish,
returns in `[-0.5%, +0.5%]` are flat, and returns above +0.5% are bullish.

Monte Carlo prediction returns:

- three-state posterior probability means;
- aleatoric Dirichlet variance;
- epistemic variance across sampled weights;
- total variance;
- 50 latent-state posterior probabilities by default;
- expected regime score in `[-1, 1]`.

These outputs are model indicators. Position sizing remains in a separate long-only
policy layer: bull opens or maintains a long, flat preserves the prior position,
and bear closes it. It must not be embedded in model training.
