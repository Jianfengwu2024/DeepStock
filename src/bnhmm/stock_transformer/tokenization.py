"""Variable-length words and sentences derived only from conservation fields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .conservation import ConservationLaw, ConservationTransition


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
TOKEN_FEATURE_NAMES = (
    "mass",
    "mass_change",
    "center",
    "width",
    "peak_density",
    "mean_density_gradient",
    "net_current",
    "absolute_current",
    "current_coherence",
    "stochastic_current_mean",
    "stochastic_current_rms",
    "continuity_residual_rms",
    "density_entropy",
    "boundary_strength",
    "sentence_start",
)
TOKEN_FEATURE_DIM = len(TOKEN_FEATURE_NAMES)


@dataclass(frozen=True, slots=True)
class TokenizationConfig:
    minimum_token_bins: int = 2
    maximum_tokens: int = 16
    boundary_threshold: float = 1.5

    def __post_init__(self) -> None:
        if self.minimum_token_bins < 1:
            raise ValueError("minimum token width must be positive")
        if self.maximum_tokens < 2:
            raise ValueError("at least two conservation tokens are required")
        if self.boundary_threshold <= 0:
            raise ValueError("boundary threshold must be positive")


@dataclass(frozen=True, slots=True)
class ConservationToken:
    start: int
    stop: int
    sentence_id: int
    features: FloatArray

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop <= self.start:
            raise ValueError("token bounds are invalid")
        if self.features.shape != (TOKEN_FEATURE_DIM,):
            raise ValueError("token features have an invalid shape")


@dataclass(frozen=True, slots=True)
class TokenizedState:
    symbol: str
    signal_time: np.datetime64
    tokens: tuple[ConservationToken, ...]

    @property
    def features(self) -> FloatArray:
        return np.stack([token.features for token in self.tokens])

    @property
    def sentence_ids(self) -> IntArray:
        return np.asarray([token.sentence_id for token in self.tokens], dtype=np.int64)


class ConservationTokenizer:
    """Segment density basins and coherent currents into words and sentences."""

    def __init__(self, config: TokenizationConfig | None = None) -> None:
        self.config = config or TokenizationConfig()

    def tokenize(
        self, transition: ConservationTransition, law: ConservationLaw
    ) -> TokenizedState:
        if len(transition.density) != law.grid_points:
            raise ValueError("transition and conservation law grids do not match")
        boundary_score = self._boundary_scores(transition, law)
        boundaries = self._select_boundaries(boundary_score, len(transition.density))
        tokens: list[ConservationToken] = []
        sentence_id = 0
        previous_direction = 0.0
        for index, (start, stop) in enumerate(
            zip(boundaries[:-1], boundaries[1:], strict=True)
        ):
            values = self._features(
                transition, law, int(start), int(stop), boundary_score
            )
            direction = float(np.sign(values[6]))
            residual_break = values[10] >= self.config.boundary_threshold
            sentence_start = index == 0 or residual_break or (
                direction != 0.0
                and previous_direction != 0.0
                and direction != previous_direction
            )
            if sentence_start and index:
                sentence_id += 1
            values[-1] = float(sentence_start)
            tokens.append(
                ConservationToken(
                    start=int(start),
                    stop=int(stop),
                    sentence_id=sentence_id,
                    features=values,
                )
            )
            if direction:
                previous_direction = direction
        return TokenizedState(
            symbol=transition.symbol,
            signal_time=transition.signal_time,
            tokens=tuple(tokens),
        )

    def _boundary_scores(
        self, transition: ConservationTransition, law: ConservationLaw
    ) -> FloatArray:
        size = len(transition.density)
        score = np.zeros(size + 1, dtype=np.float64)
        gradient = transition.density_gradient
        current = transition.current
        gradient_turn = np.signbit(gradient[:-1]) != np.signbit(gradient[1:])
        current_turn = np.signbit(current[:-1]) != np.signbit(current[1:])
        score[1:size] += gradient_turn.astype(np.float64)
        score[1:size] += 0.8 * current_turn.astype(np.float64)
        residual_z = np.abs(transition.stochastic_current) / law.noise_scale
        score[1:size] += 0.5 * np.maximum(
            residual_z[:-1], residual_z[1:]
        )
        score[0] = score[-1] = np.inf
        return score

    def _select_boundaries(self, scores: FloatArray, size: int) -> IntArray:
        candidates = np.flatnonzero(
            scores[1:size] >= self.config.boundary_threshold
        ) + 1
        if len(candidates) > self.config.maximum_tokens - 1:
            ranking = np.argsort(scores[candidates])[::-1]
            candidates = candidates[ranking[: self.config.maximum_tokens - 1]]
        selected = [0]
        for boundary in sorted(int(value) for value in candidates):
            if boundary - selected[-1] >= self.config.minimum_token_bins:
                selected.append(boundary)
        if size - selected[-1] < self.config.minimum_token_bins and len(selected) > 1:
            selected.pop()
        selected.append(size)
        while len(selected) - 1 > self.config.maximum_tokens:
            widths = np.diff(selected)
            remove = int(np.argmin(widths[:-1])) + 1
            selected.pop(remove)
        return np.asarray(selected, dtype=np.int64)

    def _features(
        self,
        transition: ConservationTransition,
        law: ConservationLaw,
        start: int,
        stop: int,
        boundary_score: FloatArray,
    ) -> FloatArray:
        section = slice(start, stop)
        density = transition.density[section]
        change = transition.density_change[section]
        gradient = transition.density_gradient[section]
        current = transition.current[section]
        stochastic = transition.stochastic_current[section]
        continuity = transition.continuity_residual[section]
        coordinates = np.linspace(-1.0, 1.0, len(transition.density))[section]
        mass = float(density.sum())
        center = float((coordinates * density).sum() / max(mass, 1e-12))
        absolute_current = float(np.abs(current).sum())
        coherence = float(current.sum() / max(absolute_current, 1e-12))
        normalized = density / max(mass, 1e-12)
        entropy = float(
            -(normalized * np.log(np.clip(normalized, 1e-12, 1.0))).sum()
            / max(np.log(len(normalized)), 1.0)
        )
        edge_strength = max(
            float(boundary_score[start]) if start else 0.0,
            float(boundary_score[stop]) if stop < len(transition.density) else 0.0,
        )
        return np.asarray(
            (
                mass,
                float(change.sum()),
                center,
                float((stop - start) / len(transition.density)),
                float(density.max(initial=0.0)),
                float(gradient.mean()),
                float(current.sum()),
                absolute_current,
                coherence,
                float(stochastic.mean() / law.noise_scale),
                float(np.sqrt(np.square(stochastic).mean()) / law.noise_scale),
                float(np.sqrt(np.square(continuity).mean()) / law.noise_scale),
                entropy,
                edge_strength if np.isfinite(edge_strength) else 0.0,
                0.0,
            ),
            dtype=np.float64,
        )
