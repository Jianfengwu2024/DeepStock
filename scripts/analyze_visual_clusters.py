"""Explore out-of-sample links between holographic image clusters and next-day returns."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from bnhmm.visual_dataset import (
    VisualDatasetConfig,
    VisualDatasetExporter,
    rasterize_state,
)


DEFAULT_SYMBOLS = (
    "001309",
    "002049",
    "002077",
    "002119",
    "300223",
    "600460",
    "603986",
    "688037",
    "688052",
    "688981",
)


@dataclass(slots=True)
class Observation:
    symbol: str
    session: str
    next_session: str
    current_return: float
    next_return: float
    image: np.ndarray


def image_features(raster: Any, config: VisualDatasetConfig) -> np.ndarray:
    """Compress exactly the visible density and vector semantics, excluding labels."""

    block = config.vector_price_block_bins
    columns = math.ceil(config.price_bins / block)
    density_image = np.zeros((config.volume_bins, columns), dtype=np.float32)
    price_vector = np.zeros_like(density_image)
    volume_vector = np.zeros_like(density_image)
    for column, start in enumerate(range(0, config.price_bins, block)):
        stop = min(start + block, config.price_bins)
        density = raster.density[start:stop]
        density_image[:, column] = np.mean(
            np.power(
                np.clip(density / config.density_ceiling, 0.0, 1.0),
                config.density_gamma,
            ),
            axis=0,
        )
        mass = density.sum(axis=0)
        supported = mass >= config.vector_min_mass
        denominator = np.maximum(mass, 1e-12)
        price = np.sum(
            density * raster.price_velocity[start:stop], axis=0
        ) / denominator
        volume = np.sum(
            density * raster.volume_velocity[start:stop], axis=0
        ) / denominator
        price_vector[supported, column] = np.clip(
            price[supported] / config.feature_clip, -1.0, 1.0
        )
        volume_vector[supported, column] = np.clip(
            volume[supported] / config.feature_clip, -1.0, 1.0
        )
    return np.stack((density_image, price_vector, volume_vector), axis=0)


def load_observations(
    database_path: str,
    output_dir: Path,
    symbols: Sequence[str],
    config: VisualDatasetConfig,
) -> list[Observation]:
    exporter = VisualDatasetExporter(database_path, output_dir, config)
    result: list[Observation] = []
    for index, symbol in enumerate(symbols, start=1):
        states = exporter._states(symbol)
        count = 0
        for state, following in zip(states[:-1], states[1:], strict=True):
            session = str(state.period_start.astype("datetime64[D]"))
            next_session = str(following.period_start.astype("datetime64[D]"))
            if not "2024-01-01" <= session <= "2025-12-31":
                continue
            if next_session > "2025-12-31":
                continue
            raster = rasterize_state(state, config)
            result.append(
                Observation(
                    symbol=symbol,
                    session=session,
                    next_session=next_session,
                    current_return=float(state.end_price / state.anchor_price - 1.0),
                    next_return=float(following.end_price / state.end_price - 1.0),
                    image=image_features(raster, config),
                )
            )
            count += 1
        print(f"[{index}/{len(symbols)}] {symbol}: {count} labeled images", flush=True)
    return result


def standardize_channels(train: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, dict[str, list[float]]]:
    means = train.mean(axis=(0, 2, 3), keepdims=True)
    scales = np.maximum(train.std(axis=(0, 2, 3), keepdims=True), 1e-6)
    transformed = (values - means) / scales
    return transformed.reshape(len(values), -1), {
        "mean": means.reshape(-1).tolist(),
        "scale": scales.reshape(-1).tolist(),
    }


def fit_pca(
    train: np.ndarray,
    values: np.ndarray,
    components: int,
    device: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    torch.manual_seed(seed)
    x_train = torch.as_tensor(train, dtype=torch.float32, device=device)
    x_all = torch.as_tensor(values, dtype=torch.float32, device=device)
    center = x_train.mean(dim=0, keepdim=True)
    centered = x_train - center
    q = min(max(components + 8, components), min(centered.shape) - 1)
    _, singular, vectors = torch.pca_lowrank(centered, q=q, center=False, niter=4)
    vectors = vectors[:, :components]
    train_scores = centered @ vectors
    all_scores = (x_all - center) @ vectors
    explained = float(
        singular[:components].square().sum()
        / torch.clamp(centered.square().sum(), min=1e-12)
    )
    return (
        train_scores.cpu().numpy(),
        all_scores.cpu().numpy(),
        explained,
    )


def kmeans(
    values: np.ndarray,
    clusters: int,
    *,
    restarts: int,
    seed: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    x = torch.as_tensor(values, dtype=torch.float32, device=device)
    best_centers: torch.Tensor | None = None
    best_labels: torch.Tensor | None = None
    best_inertia = math.inf
    for restart in range(restarts):
        generator = torch.Generator(device=device).manual_seed(seed + restart * 101)
        centers = _kmeans_plus_plus(x, clusters, generator)
        labels = torch.zeros(len(x), dtype=torch.long, device=device)
        for _ in range(100):
            distances = torch.cdist(x, centers).square()
            next_labels = distances.argmin(dim=1)
            next_centers = centers.clone()
            for cluster in range(clusters):
                members = x[next_labels == cluster]
                if len(members):
                    next_centers[cluster] = members.mean(dim=0)
                else:
                    farthest = distances.min(dim=1).values.argmax()
                    next_centers[cluster] = x[farthest]
            if torch.equal(labels, next_labels) and torch.allclose(
                centers, next_centers, atol=1e-5, rtol=0
            ):
                labels, centers = next_labels, next_centers
                break
            labels, centers = next_labels, next_centers
        inertia = float((x - centers[labels]).square().sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_centers = centers.clone()
            best_labels = labels.clone()
    assert best_centers is not None and best_labels is not None
    return (
        best_centers.cpu().numpy(),
        best_labels.cpu().numpy(),
        best_inertia,
    )


def _kmeans_plus_plus(
    values: torch.Tensor, clusters: int, generator: torch.Generator
) -> torch.Tensor:
    first = torch.randint(len(values), (1,), generator=generator, device=values.device)
    centers = [values[first[0]]]
    minimum = (values - centers[0]).square().sum(dim=1)
    for _ in range(1, clusters):
        probabilities = minimum / torch.clamp(minimum.sum(), min=1e-12)
        selected = torch.multinomial(probabilities, 1, generator=generator)
        centers.append(values[selected[0]])
        distance = (values - centers[-1]).square().sum(dim=1)
        minimum = torch.minimum(minimum, distance)
    return torch.stack(centers)


def assign(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    distance = np.sum((values[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    return distance.argmin(axis=1).astype(np.int64)


def approximate_silhouette(values: np.ndarray, labels: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(values), min(1200, len(values)), replace=False)
    x = values[selected]
    y = labels[selected]
    distance = np.sqrt(np.maximum(
        np.sum(x * x, axis=1)[:, None]
        + np.sum(x * x, axis=1)[None, :]
        - 2 * x @ x.T,
        0.0,
    ))
    scores: list[float] = []
    for index in range(len(x)):
        same = y == y[index]
        same[index] = False
        if not same.any():
            continue
        a = float(distance[index, same].mean())
        b = min(
            float(distance[index, y == cluster].mean())
            for cluster in np.unique(y)
            if cluster != y[index] and np.any(y == cluster)
        )
        scores.append((b - a) / max(a, b, 1e-12))
    return float(np.mean(scores)) if scores else 0.0


def balanced_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    recalls = [
        float(np.mean(predicted[actual == value] == value))
        for value in (0, 1)
        if np.any(actual == value)
    ]
    return float(np.mean(recalls))


def chi_square(labels: np.ndarray, outcomes: np.ndarray, clusters: int) -> float:
    table = np.zeros((clusters, 2), dtype=np.float64)
    np.add.at(table, (labels, outcomes), 1.0)
    expected = table.sum(axis=1, keepdims=True) @ (
        table.sum(axis=0, keepdims=True) / max(table.sum(), 1.0)
    )
    valid = expected > 0
    return float(np.sum(np.square(table[valid] - expected[valid]) / expected[valid]))


def return_separation(labels: np.ndarray, returns: np.ndarray, clusters: int) -> float:
    overall = float(returns.mean())
    statistic = 0.0
    for cluster in range(clusters):
        values = returns[labels == cluster]
        if len(values):
            statistic += len(values) * (float(values.mean()) - overall) ** 2
    return statistic


def stratified_permutation_p(
    labels: np.ndarray,
    values: np.ndarray,
    symbols: np.ndarray,
    clusters: int,
    statistic: Any,
    *,
    permutations: int,
    seed: int,
) -> tuple[float, float]:
    observed = statistic(labels, values, clusters)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(symbols == symbol) for symbol in np.unique(symbols)]
    exceed = 0
    shuffled = values.copy()
    for _ in range(permutations):
        for indices in groups:
            shuffled[indices] = rng.permutation(values[indices])
        exceed += statistic(labels, shuffled, clusters) >= observed - 1e-15
    return observed, (exceed + 1.0) / (permutations + 1.0)


def wilson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - margin, center + margin


def order_clusters_by_price_center(images: np.ndarray, labels: np.ndarray, clusters: int) -> np.ndarray:
    x = np.arange(images.shape[-1], dtype=np.float64)
    centers = []
    for cluster in range(clusters):
        density = images[labels == cluster, 0].mean(axis=0)
        mass = density.sum(axis=0)
        centers.append(float(np.sum(mass * x) / max(float(mass.sum()), 1e-12)))
    return np.argsort(centers)


def relabel(values: np.ndarray, order: np.ndarray) -> np.ndarray:
    mapping = np.empty(len(order), dtype=np.int64)
    mapping[order] = np.arange(len(order))
    return mapping[values]


def center_images_on_density(images: np.ndarray) -> np.ndarray:
    """Remove absolute daily price position while preserving local shape and vectors."""

    result = np.zeros_like(images)
    target = (images.shape[-1] - 1) / 2.0
    coordinates = np.arange(images.shape[-1], dtype=np.float64)
    for index, image in enumerate(images):
        mass = image[0].sum(axis=0)
        center = float(np.sum(mass * coordinates) / max(float(mass.sum()), 1e-12))
        shift = int(round(target - center))
        if shift == 0:
            result[index] = image
        elif shift > 0:
            result[index, :, :, shift:] = image[:, :, :-shift]
        else:
            result[index, :, :, :shift] = image[:, :, -shift:]
    return result


def evaluate_image_variant(
    name: str,
    images: np.ndarray,
    fit_mask: np.ndarray,
    mapping_mask: np.ndarray,
    test_mask: np.ndarray,
    outcomes: np.ndarray,
    returns: np.ndarray,
    symbols: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    flat, _ = standardize_channels(images[fit_mask], images)
    train_scores, scores, explained = fit_pca(
        flat[fit_mask], flat, args.pca_components, args.device, args.seed + 300
    )
    centers, train_labels, _ = kmeans(
        train_scores,
        6,
        restarts=args.restarts,
        seed=args.seed + 306,
        device=args.device,
    )
    labels = assign(scores, centers)
    mapping = np.asarray(
        [
            int(np.mean(outcomes[mapping_mask][labels[mapping_mask] == cluster]) >= 0.5)
            if np.any(labels[mapping_mask] == cluster)
            else int(np.mean(outcomes[mapping_mask]) >= 0.5)
            for cluster in range(6)
        ]
    )
    predicted = mapping[labels[test_mask]]
    _, direction_p = stratified_permutation_p(
        labels[test_mask], outcomes[test_mask], symbols[test_mask], 6, chi_square,
        permutations=args.permutations, seed=args.seed + 3100,
    )
    _, return_p = stratified_permutation_p(
        labels[test_mask], returns[test_mask], symbols[test_mask], 6, return_separation,
        permutations=args.permutations, seed=args.seed + 3200,
    )
    rates = [
        float(outcomes[test_mask][labels[test_mask] == cluster].mean())
        for cluster in range(6)
        if np.any(labels[test_mask] == cluster)
    ]
    return {
        "variant": name,
        "channels": int(images.shape[1]),
        "pca_explained_variance": explained,
        "train_silhouette": approximate_silhouette(
            train_scores, train_labels, args.seed + 306
        ),
        "test_direction_stratified_permutation_p": direction_p,
        "test_return_stratified_permutation_p": return_p,
        "test_up_rate_spread": max(rates) - min(rates),
        "test_cluster_accuracy": float(np.mean(outcomes[test_mask] == predicted)),
        "test_cluster_balanced_accuracy": balanced_accuracy(
            outcomes[test_mask], predicted
        ),
        "test_up_rates": rates,
    }


def evaluate_scalar_return_bins(
    current_returns: np.ndarray,
    fit_mask: np.ndarray,
    mapping_mask: np.ndarray,
    test_mask: np.ndarray,
    outcomes: np.ndarray,
    returns: np.ndarray,
    symbols: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    thresholds = np.unique(
        np.quantile(current_returns[fit_mask], np.linspace(0, 1, 7)[1:-1])
    )
    labels = np.digitize(current_returns, thresholds).astype(np.int64)
    clusters = len(thresholds) + 1
    mapping = np.asarray(
        [
            int(np.mean(outcomes[mapping_mask][labels[mapping_mask] == cluster]) >= 0.5)
            if np.any(labels[mapping_mask] == cluster)
            else int(np.mean(outcomes[mapping_mask]) >= 0.5)
            for cluster in range(clusters)
        ]
    )
    predicted = mapping[labels[test_mask]]
    _, direction_p = stratified_permutation_p(
        labels[test_mask], outcomes[test_mask], symbols[test_mask], clusters, chi_square,
        permutations=args.permutations, seed=args.seed + 3300,
    )
    _, return_p = stratified_permutation_p(
        labels[test_mask], returns[test_mask], symbols[test_mask], clusters, return_separation,
        permutations=args.permutations, seed=args.seed + 3400,
    )
    rates = [
        float(outcomes[test_mask][labels[test_mask] == cluster].mean())
        for cluster in range(clusters)
        if np.any(labels[test_mask] == cluster)
    ]
    return {
        "variant": "current-close-return-only",
        "clusters": clusters,
        "thresholds": thresholds.tolist(),
        "test_direction_stratified_permutation_p": direction_p,
        "test_return_stratified_permutation_p": return_p,
        "test_up_rate_spread": max(rates) - min(rates),
        "test_cluster_accuracy": float(np.mean(outcomes[test_mask] == predicted)),
        "test_cluster_balanced_accuracy": balanced_accuracy(
            outcomes[test_mask], predicted
        ),
        "test_up_rates": rates,
    }


def render_prototypes(
    destination: Path,
    images: np.ndarray,
    labels: np.ndarray,
    cluster_rows: list[dict[str, Any]],
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    panel_width, panel_height = 460, 250
    canvas = Image.new("RGB", (panel_width * 2, panel_height * 3), "#f3f5f6")
    font = ImageFont.load_default()
    for cluster in range(6):
        left = (cluster % 2) * panel_width
        top = (cluster // 2) * panel_height
        average = images[labels == cluster].mean(axis=0)
        density = np.asarray(np.rint(255 - np.clip(average[0], 0, 1) * 222), dtype=np.uint8)
        plot = Image.fromarray(density, mode="L").resize((400, 170), Image.Resampling.NEAREST).convert("RGB")
        canvas.paste(plot, (left + 45, top + 42))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((left + 45, top + 42, left + 445, top + 212), outline="#68767c")
        zero = left + 45 + 200
        draw.line((zero, top + 42, zero, top + 212), fill="#ba4740", width=1)
        for row in range(0, average.shape[1], 2):
            for column in range(0, average.shape[2], 3):
                u = float(average[1, row, column])
                v = float(average[2, row, column])
                magnitude = math.hypot(u, v)
                if magnitude < 0.035:
                    continue
                x = left + 45 + (column + 0.5) / average.shape[2] * 400
                y = top + 42 + (row + 0.5) / average.shape[1] * 170
                scale = min(10.0, magnitude * 16.0)
                draw.line((x, y, x + u / magnitude * scale, y - v / magnitude * scale), fill="#137b8f", width=1)
        row = cluster_rows[cluster]
        draw.text((left + 12, top + 12), f"C{cluster + 1}  n={row['n']}  up={row['up_rate']:.1%}  mean={row['mean_return']:+.3%}", fill="#172126", font=font)
        draw.text((left + 12, top + 224), "price: -34%                         prev close                         +52%", fill="#66737a", font=font)
    canvas.save(destination, optimize=True)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    symbols = tuple(item.strip() for item in args.symbols.split(",") if item.strip())
    if len(symbols) != 10:
        raise ValueError("analysis requires exactly ten symbols")
    config = VisualDatasetConfig(start="2024-01-01", end="2026-01-01", volume_bins=21)
    observations = load_observations(args.duckdb_path, output_dir, symbols, config)
    images = np.stack([item.image for item in observations])
    sessions = np.asarray([item.session for item in observations])
    next_sessions = np.asarray([item.next_session for item in observations])
    symbol_values = np.asarray([item.symbol for item in observations])
    returns = np.asarray([item.next_return for item in observations], dtype=np.float64)
    current_returns = np.asarray(
        [item.current_return for item in observations], dtype=np.float64
    )
    outcomes = (returns > 0).astype(np.int64)
    fit_mask = np.char.startswith(sessions, "2024-")
    mapping_mask = fit_mask & np.char.startswith(next_sessions, "2024-")
    test_mask = np.char.startswith(sessions, "2025-") & np.char.startswith(next_sessions, "2025-")

    flat_all, channel_scaling = standardize_channels(images[fit_mask], images)
    train_scores, scores, explained = fit_pca(
        flat_all[fit_mask], flat_all, args.pca_components, args.device, args.seed
    )
    sensitivity: list[dict[str, Any]] = []
    fitted: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for clusters in (4, 6, 8):
        centers, train_labels, inertia = kmeans(
            train_scores,
            clusters,
            restarts=args.restarts,
            seed=args.seed + clusters,
            device=args.device,
        )
        labels = assign(scores, centers)
        _, direction_p = stratified_permutation_p(
            labels[test_mask],
            outcomes[test_mask],
            symbol_values[test_mask],
            clusters,
            chi_square,
            permutations=args.permutations,
            seed=args.seed + 1000 + clusters,
        )
        _, return_p = stratified_permutation_p(
            labels[test_mask],
            returns[test_mask],
            symbol_values[test_mask],
            clusters,
            return_separation,
            permutations=args.permutations,
            seed=args.seed + 2000 + clusters,
        )
        sensitivity.append(
            {
                "clusters": clusters,
                "train_inertia": inertia,
                "train_silhouette": approximate_silhouette(
                    train_scores, train_labels, args.seed + clusters
                ),
                "test_direction_stratified_permutation_p": direction_p,
                "test_return_stratified_permutation_p": return_p,
            }
        )
        fitted[clusters] = (centers, labels)

    labels = fitted[6][1]
    order = order_clusters_by_price_center(images[fit_mask], labels[fit_mask], 6)
    labels = relabel(labels, order)
    test_labels = labels[test_mask]
    train_map_labels = labels[mapping_mask]
    train_map_outcomes = outcomes[mapping_mask]
    mapping = np.asarray(
        [
            int(np.mean(train_map_outcomes[train_map_labels == cluster]) >= 0.5)
            if np.any(train_map_labels == cluster)
            else int(np.mean(train_map_outcomes) >= 0.5)
            for cluster in range(6)
        ]
    )
    predictions = mapping[test_labels]
    baseline = int(np.mean(train_map_outcomes) >= 0.5)
    test_outcomes = outcomes[test_mask]
    direction_stat, direction_p = stratified_permutation_p(
        test_labels,
        test_outcomes,
        symbol_values[test_mask],
        6,
        chi_square,
        permutations=args.permutations,
        seed=args.seed + 1006,
    )
    return_stat, return_p = stratified_permutation_p(
        test_labels,
        returns[test_mask],
        symbol_values[test_mask],
        6,
        return_separation,
        permutations=args.permutations,
        seed=args.seed + 2006,
    )

    cluster_rows: list[dict[str, Any]] = []
    for cluster in range(6):
        selected = test_labels == cluster
        count = int(selected.sum())
        successes = int(test_outcomes[selected].sum())
        lower, upper = wilson(successes, count)
        values = returns[test_mask][selected]
        cluster_rows.append(
            {
                "cluster": cluster + 1,
                "n": count,
                "up_count": successes,
                "up_rate": successes / count if count else 0.0,
                "up_rate_ci95_low": lower,
                "up_rate_ci95_high": upper,
                "mean_return": float(values.mean()) if count else 0.0,
                "median_return": float(np.median(values)) if count else 0.0,
                "train_mapped_direction": "up" if mapping[cluster] else "down_or_flat",
            }
        )

    per_symbol: list[dict[str, Any]] = []
    for symbol in symbols:
        selected = test_mask & (symbol_values == symbol)
        actual = outcomes[selected]
        predicted = mapping[labels[selected]]
        per_symbol.append(
            {
                "symbol": symbol,
                "n": int(selected.sum()),
                "up_rate": float(actual.mean()),
                "cluster_accuracy": float(np.mean(actual == predicted)),
                "cluster_balanced_accuracy": balanced_accuracy(actual, predicted),
            }
        )

    ablations = [
        evaluate_image_variant(
            "density-only", images[:, :1], fit_mask, mapping_mask, test_mask, outcomes,
            returns, symbol_values, args
        ),
        evaluate_image_variant(
            "vectors-only", images[:, 1:], fit_mask, mapping_mask, test_mask, outcomes,
            returns, symbol_values, args
        ),
        evaluate_image_variant(
            "position-centered-density-and-vectors",
            center_images_on_density(images),
            fit_mask,
            mapping_mask,
            test_mask,
            outcomes,
            returns,
            symbol_values,
            args,
        ),
        evaluate_image_variant(
            "position-centered-density-only",
            center_images_on_density(images[:, :1]),
            fit_mask,
            mapping_mask,
            test_mask,
            outcomes,
            returns,
            symbol_values,
            args,
        ),
        evaluate_image_variant(
            "position-centered-vectors-only",
            center_images_on_density(images)[:, 1:],
            fit_mask,
            mapping_mask,
            test_mask,
            outcomes,
            returns,
            symbol_values,
            args,
        ),
        evaluate_scalar_return_bins(
            current_returns, fit_mask, mapping_mask, test_mask, outcomes,
            returns, symbol_values, args
        ),
    ]

    result = {
        "symbols": list(symbols),
        "method": {
            "fit_period": "2024 image states",
            "mapping_period": "2024 images with next session also in 2024",
            "test_period": "2025 images with next session also in 2025",
            "features": "annotation-free fixed grayscale density plus price/volume vector field",
            "feature_shape": list(images.shape[1:]),
            "pca_components": args.pca_components,
            "pca_explained_variance": explained,
            "primary_clusters": 6,
            "permutations": args.permutations,
            "permutation": "next-day outcomes shuffled within each symbol",
            "channel_scaling": channel_scaling,
        },
        "samples": {
            "all": len(observations),
            "cluster_fit_2024": int(fit_mask.sum()),
            "direction_mapping_2024": int(mapping_mask.sum()),
            "test_2025": int(test_mask.sum()),
        },
        "test": {
            "up_rate": float(test_outcomes.mean()),
            "cluster_accuracy": float(np.mean(test_outcomes == predictions)),
            "cluster_balanced_accuracy": balanced_accuracy(test_outcomes, predictions),
            "majority_accuracy": float(np.mean(test_outcomes == baseline)),
            "direction_chi_square": direction_stat,
            "direction_stratified_permutation_p": direction_p,
            "return_between_cluster_statistic": return_stat,
            "return_stratified_permutation_p": return_p,
            "up_rate_spread": max(row["up_rate"] for row in cluster_rows)
            - min(row["up_rate"] for row in cluster_rows),
        },
        "clusters": cluster_rows,
        "sensitivity": sensitivity,
        "ablations": ablations,
        "per_symbol": per_symbol,
    }
    (output_dir / "analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(output_dir / "cluster-summary.csv", cluster_rows)
    _write_csv(output_dir / "per-symbol.csv", per_symbol)
    assignments = [
        {
            "symbol": item.symbol,
            "session": item.session,
            "next_session": item.next_session,
            "next_return": item.next_return,
            "next_up": int(item.next_return > 0),
            "cluster": int(labels[index] + 1),
            "split": "test" if test_mask[index] else "train" if fit_mask[index] else "other",
        }
        for index, item in enumerate(observations)
    ]
    _write_csv(output_dir / "assignments.csv", assignments)
    render_prototypes(
        output_dir / "cluster-prototypes.png",
        images[test_mask],
        test_labels,
        cluster_rows,
    )
    (output_dir / "cluster-analysis-report.md").write_text(
        markdown_report(result), encoding="utf-8"
    )
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_report(result: dict[str, Any]) -> str:
    test = result["test"]
    significant = test["direction_stratified_permutation_p"] < 0.05
    lines = [
        "# 10-stock holographic image cluster analysis",
        "",
        "## Verdict",
        "",
        (
            "The 2025 out-of-sample cluster-to-next-day direction association is statistically significant."
            if significant
            else "The 2025 out-of-sample cluster-to-next-day direction association is not statistically significant."
        ),
        "",
        f"- Test images: {result['samples']['test_2025']}",
        f"- Base up rate: {test['up_rate']:.2%}",
        f"- Cluster accuracy: {test['cluster_accuracy']:.2%}",
        f"- Cluster balanced accuracy: {test['cluster_balanced_accuracy']:.2%}",
        f"- Majority accuracy: {test['majority_accuracy']:.2%}",
        f"- Direction stratified permutation p: {test['direction_stratified_permutation_p']:.4f}",
        f"- Return stratified permutation p: {test['return_stratified_permutation_p']:.4f}",
        f"- Highest-minus-lowest cluster up-rate spread: {test['up_rate_spread']:.2%}",
        "",
        "## Cluster table",
        "",
        "| Cluster | N | Up rate | 95% CI | Mean next return | Median next return | 2024 mapping |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in result["clusters"]:
        lines.append(
            f"| C{row['cluster']} | {row['n']} | {row['up_rate']:.2%} | "
            f"{row['up_rate_ci95_low']:.2%} to {row['up_rate_ci95_high']:.2%} | "
            f"{row['mean_return']:+.3%} | {row['median_return']:+.3%} | {row['train_mapped_direction']} |"
        )
    lines.extend(
        [
            "",
            "## Design",
            "",
            "Clusters were fitted without labels on 2024 annotation-free state tensors. Cluster-to-direction mapping used only outcomes completed in 2024. All reported association and accuracy figures use 2025. Permutations preserve each stock's own outcome distribution.",
            "",
            "## Sensitivity",
            "",
            "| K | Train silhouette | Direction p | Return p |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in result["sensitivity"]:
        lines.append(
            f"| {row['clusters']} | {row['train_silhouette']:.4f} | "
            f"{row['test_direction_stratified_permutation_p']:.4f} | "
            f"{row['test_return_stratified_permutation_p']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Information ablation",
            "",
            "| Variant | Balanced accuracy | Direction p | Return p | Up-rate spread |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in result["ablations"]:
        lines.append(
            f"| {row['variant']} | {row['test_cluster_balanced_accuracy']:.2%} | "
            f"{row['test_direction_stratified_permutation_p']:.4f} | "
            f"{row['test_return_stratified_permutation_p']:.4f} | "
            f"{row['test_up_rate_spread']:.2%} |"
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duckdb-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pca-components", type=int, default=16)
    parser.add_argument("--restarts", type=int, default=12)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=17)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    print(json.dumps(analyze(arguments), ensure_ascii=False, indent=2))
