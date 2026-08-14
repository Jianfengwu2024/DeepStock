# Bayesian Model Core

The default padded input tensor has shape `[batch, sequence, bins, 13]`, where
`bins` varies by batch. Four scalar channels contain price occupancy,
relative-volume shape, normalized relative-volume level, and normalized log-price
location. Nine channels encode the normalized 3-by-3 joint price/volume velocity
distribution. Each velocity
row receives additive smoothing and is multiplied by its share of the session's
samples. A one-observation row therefore has little mass instead of looking as
confident as a densely observed row. The relative-volume level is also shrunk by
its local effective sample count.
distribution. A bin mask excludes padding and an absolute
10-basis-point coordinate tensor preserves whether the whole profile lies below
or above the previous close.

The tokenizer learns position-aligned individual-bin, adjacent two-bin, and
adjacent three-bin motifs. Price occupancy and location form the backbone.
Relative volume, price velocity, and volume velocity have separate Bayesian motif
branches whose residual gates start near zero and are learned from data. The
`price-only`, `price-volume`, `price-velocity`, and `gated-full` modes provide
direct ablation comparisons. Coordinate Fourier features prevent cropped profiles
from losing their absolute return location. Masked attention pooling combines
the motifs with open/close/range, total relative volume, bar coverage, and
overflow metadata to form one daily token. No exact intraday clock position,
price acceleration, or direction-times-volume product is provided to the model.

All query, key, value, attention-output, feed-forward, motif, transition-context,
latent-emission and return-posterior projections use variational Gaussian
weights. The temporal attention mask is causal.

Fifty unnamed latent probabilities are filtered sequentially by default using a
Bayesian base transition matrix plus a context-dependent transition delta. The
filtered latent state feeds directly into a Bayesian Student-t return posterior.
The hidden-state count is configurable but cannot fall below the profile channel
count. Configurable symmetric thresholds (1.5% by default) separate strong from
weak returns, while zero separates bearish from bullish returns.

Monte Carlo prediction returns:

- four-state posterior probability means;
- aleatoric Dirichlet variance;
- epistemic variance across sampled weights;
- total variance;
- 50 latent-state posterior probabilities by default;
- expected regime score in `[-2, 2]`.

These outputs are model indicators. Position sizing remains in a separate long-only
policy layer: weak-bull targets half size, strong-bull targets full size, and
either bearish state closes the position. It must not be embedded in model training.
