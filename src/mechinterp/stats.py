"""Two-way cluster bootstrap over concepts and clusters, percentile intervals and Holm."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def two_way_weights(
    n_concepts: int, n_clusters: int, n_boot: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Resampling counts: concepts and clusters drawn independently, with replacement.

    Returns ([n_boot, n_concepts], [n_boot, n_clusters]). Using the same weights for every statistic
    keeps differences and ratios paired within a replicate.
    """
    concept_weights = rng.multinomial(n_concepts, np.full(n_concepts, 1.0 / n_concepts), size=n_boot)
    cluster_weights = rng.multinomial(n_clusters, np.full(n_clusters, 1.0 / n_clusters), size=n_boot)
    return concept_weights.astype(float), cluster_weights.astype(float)


def weighted_cell_mean(
    sums: np.ndarray, counts: np.ndarray, concept_weights: np.ndarray, cluster_weights: np.ndarray
) -> np.ndarray:
    """Trial-weighted mean per replicate. sums and counts are [n_concepts, n_clusters].

    A replicate that holds no trials, possible only in a sparse design, gives NaN.
    """
    numerator = ((concept_weights @ sums) * cluster_weights).sum(axis=1)
    denominator = ((concept_weights @ counts) * cluster_weights).sum(axis=1)
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)


def percentile_ci(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    tail = 100.0 * (1.0 - level) / 2.0
    finite = values[~np.isnan(values)]
    if finite.size == 0:
        return float("nan"), float("nan")
    low, high = np.percentile(finite, [tail, 100.0 - tail])
    return float(low), float(high)


def bootstrap_p_two_sided(values: np.ndarray) -> float:
    """Two-sided p-value for zero, from the bootstrap distribution, dual to the percentile interval."""
    finite = values[~np.isnan(values)]
    if finite.size == 0:
        return 1.0
    below = int(np.sum(finite <= 0.0))
    above = int(np.sum(finite >= 0.0))
    return min(1.0, 2.0 * (min(below, above) + 1) / (finite.size + 1))


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order."""
    n = len(p_values)
    order = sorted(range(n), key=lambda index: p_values[index])
    adjusted = [0.0] * n
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (n - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def ratio_summary(
    numerator_point: float,
    denominator_point: float,
    numerator_boot: np.ndarray,
    denominator_boot: np.ndarray,
    min_denominator: float,
    level: float = 0.95,
) -> dict[str, float | None]:
    """Point estimate and percentile interval of a ratio with an unstable denominator.

    A replicate whose denominator is at or below min_denominator has no meaningful ratio. Such
    replicates count against both bounds: as -inf for the lower bound and +inf for the upper. So
    when they exceed the tail mass, the affected bound is None (undefined), never a huge number.
    Bounds are the outward order statistics, because interpolating across infinities gives NaN.
    """
    defined = denominator_boot > min_denominator
    ratios = np.divide(numerator_boot, denominator_boot, out=np.zeros_like(numerator_boot), where=defined)
    n = ratios.size
    tail = (1.0 - level) / 2.0
    low_sorted = np.sort(np.where(defined, ratios, -np.inf))
    high_sorted = np.sort(np.where(defined, ratios, np.inf))
    low = float(low_sorted[int(np.floor(tail * (n - 1)))])
    high = float(high_sorted[int(np.ceil((1.0 - tail) * (n - 1)))])
    point = numerator_point / denominator_point if denominator_point > min_denominator else None
    return {
        "point": point,
        "lo": low if np.isfinite(low) else None,
        "hi": high if np.isfinite(high) else None,
        "undefined_fraction": float(1.0 - defined.mean()),
    }
