"""Target-independent diagnostics for monthly river topology."""

from __future__ import annotations

import numpy as np

from .types import MonthlyRiverTopology


STRUCTURAL_ABLATIONS = (
    "v1",
    "random",
    "common",
    "residual",
    "unstable",
    "full",
)


def evaluate_topology_structure(
    topology: MonthlyRiverTopology,
    *,
    random_samples: int = 256,
    random_seed: int = 20260824,
) -> dict[str, float | int]:
    if random_samples < 1:
        raise ValueError("at least one random topology sample is required")
    assigned = np.flatnonzero(topology.assigned & (topology.band_ids >= 0))
    within = _within_similarity(
        topology.fused_similarity, assigned, topology.band_ids[assigned]
    )
    rng = np.random.default_rng(random_seed)
    random_values = []
    labels = topology.band_ids[assigned].copy()
    for _ in range(random_samples):
        random_values.append(
            _within_similarity(
                topology.fused_similarity,
                assigned,
                rng.permutation(labels),
            )
        )
    random_mean = float(np.mean(random_values))
    return {
        "band_count": int(len(np.unique(labels))),
        "within_band_similarity": within,
        "random_within_band_similarity": random_mean,
        "synchronization_lift": within - random_mean,
        "unassigned_rate": float(1.0 - np.mean(topology.assigned)),
        "mean_cohesion": float(np.mean(topology.cohesion[assigned])) if len(assigned) else 0.0,
    }


def migration_rate(
    previous: MonthlyRiverTopology,
    current: MonthlyRiverTopology,
) -> float:
    previous_index = {symbol: index for index, symbol in enumerate(previous.symbols)}
    current_index = {symbol: index for index, symbol in enumerate(current.symbols)}
    shared = sorted(set(previous_index) & set(current_index))
    eligible = [
        symbol
        for symbol in shared
        if previous.assigned[previous_index[symbol]]
        and current.assigned[current_index[symbol]]
        and previous.band_ids[previous_index[symbol]] >= 0
        and current.band_ids[current_index[symbol]] >= 0
    ]
    if not eligible:
        return 0.0
    moved = sum(
        previous.band_ids[previous_index[symbol]]
        != current.band_ids[current_index[symbol]]
        for symbol in eligible
    )
    return float(moved / len(eligible))


def _within_similarity(
    similarity: np.ndarray,
    indices: np.ndarray,
    labels: np.ndarray,
) -> float:
    values: list[float] = []
    for left in range(len(indices)):
        for right in range(left + 1, len(indices)):
            if labels[left] == labels[right]:
                values.append(float(similarity[indices[left], indices[right]]))
    return float(np.mean(values)) if values else 0.0
