from __future__ import annotations

import numpy as np
import pytest

from mechinterp.stats import (
    bootstrap_p_two_sided,
    holm_adjust,
    percentile_ci,
    ratio_summary,
    two_way_weights,
    weighted_cell_mean,
)

N_CONCEPTS = 20
N_CLUSTERS = 20
TRUE_MEAN = 1.0


def _simulated_table(rng: np.random.Generator) -> np.ndarray:
    concept_effects = rng.normal(0, 0.5, N_CONCEPTS)
    cluster_effects = rng.normal(0, 0.5, N_CLUSTERS)
    noise = rng.normal(0, 0.5, (N_CONCEPTS, N_CLUSTERS))
    return TRUE_MEAN + concept_effects[:, None] + cluster_effects[None, :] + noise


def _coverage(two_way: bool, n_simulations: int = 250, n_boot: int = 400) -> float:
    rng = np.random.default_rng(11)
    counts = np.ones((N_CONCEPTS, N_CLUSTERS))
    covered = 0
    for _ in range(n_simulations):
        table = _simulated_table(rng)
        concept_weights, cluster_weights = two_way_weights(N_CONCEPTS, N_CLUSTERS, n_boot, rng)
        if not two_way:
            cluster_weights = np.ones_like(cluster_weights)
        low, high = percentile_ci(weighted_cell_mean(table, counts, concept_weights, cluster_weights))
        covered += int(low <= TRUE_MEAN <= high)
    return covered / n_simulations


def test_two_way_bootstrap_covers_the_true_mean() -> None:
    assert 0.85 <= _coverage(two_way=True) <= 1.0


def test_resampling_only_concepts_undercovers_when_clusters_vary() -> None:
    assert _coverage(two_way=False) < _coverage(two_way=True) - 0.05


def test_weights_resample_each_dimension_with_replacement() -> None:
    concept_weights, cluster_weights = two_way_weights(6, 4, 50, np.random.default_rng(0))
    assert concept_weights.shape == (50, 6) and cluster_weights.shape == (50, 4)
    assert np.all(concept_weights.sum(axis=1) == 6) and np.all(cluster_weights.sum(axis=1) == 4)


def test_weighted_mean_with_unit_weights_is_the_trial_mean() -> None:
    sums = np.array([[2.0, 4.0], [6.0, 0.0]])
    counts = np.array([[2.0, 2.0], [2.0, 2.0]])
    result = weighted_cell_mean(sums, counts, np.ones((1, 2)), np.ones((1, 2)))
    assert result[0] == pytest.approx(12.0 / 8.0)


def test_weighted_mean_honours_weights() -> None:
    sums = np.array([[1.0], [100.0]])
    counts = np.ones((2, 1))
    assert weighted_cell_mean(sums, counts, np.array([[2.0, 0.0]]), np.array([[1.0]]))[0] == 1.0


def test_percentile_ci_ignores_nan() -> None:
    values = np.array([*np.linspace(0, 1, 101), np.nan])
    low, high = percentile_ci(values)
    assert low == pytest.approx(0.025) and high == pytest.approx(0.975)


def test_bootstrap_p() -> None:
    assert bootstrap_p_two_sided(np.ones(1999)) == pytest.approx(2 / 2000)
    assert bootstrap_p_two_sided(np.array([-1.0, 1.0] * 500)) == pytest.approx(1.0)


def test_holm() -> None:
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm_adjust([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def test_ratio_is_undefined_when_denominator_is_near_zero() -> None:
    rng = np.random.default_rng(0)
    numerator = rng.normal(1.0, 0.1, 2000)
    denominator = rng.normal(0.0, 0.1, 2000)
    summary = ratio_summary(1.0, 0.01, numerator, denominator, min_denominator=0.05)
    assert summary["point"] is None and summary["lo"] is None and summary["hi"] is None
    assert summary["undefined_fraction"] > 0.5


def test_ratio_is_finite_and_right_with_a_stable_denominator() -> None:
    rng = np.random.default_rng(0)
    denominator = rng.normal(4.0, 0.1, 2000)
    summary = ratio_summary(2.0, 4.0, 0.5 * denominator, denominator, min_denominator=0.05)
    assert summary["point"] == 0.5
    assert summary["lo"] == pytest.approx(0.5) and summary["hi"] == pytest.approx(0.5)
    assert summary["undefined_fraction"] == 0


def test_a_few_undefined_replicates_widen_rather_than_void_the_interval() -> None:
    denominator = np.full(1000, 2.0)
    numerator = np.linspace(0.0, 2.0, 1000)
    denominator[:10] = 0.0  # 1% undefined, inside the 2.5% tail.
    summary = ratio_summary(1.0, 2.0, numerator, denominator, min_denominator=0.05)
    assert summary["lo"] is not None and summary["hi"] is not None
    assert summary["hi"] <= 1.0
    denominator[:50] = 0.0  # 5% undefined, more than the tail holds.
    summary = ratio_summary(1.0, 2.0, numerator, denominator, min_denominator=0.05)
    assert summary["lo"] is None and summary["hi"] is None
