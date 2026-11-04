"""AUROC, recall and the paired bootstrap. numpy only.

Transcripts are nested in families, so the two-way bootstrap draws families, and within each family draws groups of
transcripts. One weight matrix serves every statistic in a run, which is what makes differences paired. Adapter seeds are
not resampled: there are three, and the prereg takes a median over them.

AUROC is on all test rows pooled, with each family holding equal numbers per class, so family base rates cannot look
like signal. Within-family AUROC is reported beside it. The two can differ when families sit at different score levels,
and a monitor with one threshold sees the pooled one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..stats import NonFiniteError, percentile_ci


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    return float(weighted_auroc(scores, labels, np.ones((1, len(scores))))[0])


def weighted_auroc(scores: np.ndarray, labels: np.ndarray, weights: np.ndarray, chunk: int = 200) -> np.ndarray:
    """AUROC per row of weights: [replicates, n] -> [replicates]. Ties count one half. NaN if a class has no weight."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    if not np.isfinite(scores).all():
        raise NonFiniteError("Non-finite scores reached the AUROC")
    if scores.size == 0:
        return np.full(len(weights), np.nan)
    order = np.argsort(scores, kind="mergesort")
    ordered = scores[order]
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    positive = labels[order]
    results = []
    for begin in range(0, len(weights), chunk):
        block = np.asarray(weights[begin : begin + chunk], dtype=np.float64)[:, order]
        tied_positive = np.add.reduceat(block * positive, starts, axis=1)
        tied_negative = np.add.reduceat(block * ~positive, starts, axis=1)
        below = np.cumsum(tied_negative, axis=1) - tied_negative
        numerator = (tied_positive * (below + 0.5 * tied_negative)).sum(axis=1)
        denominator = tied_positive.sum(axis=1) * tied_negative.sum(axis=1)
        results.append(np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0))
    return np.concatenate(results)


def threshold_at_fpr(negative_scores: np.ndarray, fpr: float = 0.01) -> float:
    """The (1 - fpr) quantile of negatives. A row is flagged when its score is above this."""
    if len(negative_scores) == 0:
        raise ValueError("A threshold needs negative rows")
    return float(np.percentile(negative_scores, 100.0 * (1.0 - fpr)))


def recall_and_fpr(scores: np.ndarray, labels: np.ndarray, threshold: float, weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels).astype(bool)
    if weights is None:
        weights = np.ones((1, len(scores)))
    flagged = np.asarray(scores) > threshold
    weights = np.asarray(weights, dtype=np.float64)
    recall = (weights * (flagged & labels)).sum(axis=1) / np.maximum((weights * labels).sum(axis=1), 1e-12)
    fpr = (weights * (flagged & ~labels)).sum(axis=1) / np.maximum((weights * ~labels).sum(axis=1), 1e-12)
    return recall, fpr


def _codes(values: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    names = sorted(set(values))
    index = {name: position for position, name in enumerate(names)}
    return np.array([index[value] for value in values], dtype=np.int64), names


@dataclass
class Resampler:
    """Holds one set of weights. two_way: families and groups. within: groups only, family weight 1."""

    two_way: np.ndarray
    within: np.ndarray
    family_only: np.ndarray
    families: np.ndarray
    family_names: list[str]

    @classmethod
    def build(cls, families: Sequence[str], groups: Sequence[str], n_boot: int, seed: int) -> "Resampler":
        family_codes, family_names = _codes(families)
        group_codes, group_names = _codes(groups)
        rng = np.random.default_rng(seed)
        n_families = len(family_names)
        n = len(families)
        family_draws = rng.multinomial(n_families, np.full(n_families, 1.0 / n_families), size=n_boot).astype(np.float32)
        group_weight = np.zeros((n_boot, len(group_names)), dtype=np.float32)
        family_of_group = np.zeros(len(group_names), dtype=np.int64)
        family_of_group[group_codes] = family_codes
        if not np.array_equal(family_of_group[group_codes], family_codes):
            raise ValueError("A group spans two families, so groups are not nested in families")
        for family in range(n_families):
            members = np.flatnonzero(family_of_group == family)
            group_weight[:, members] = rng.multinomial(len(members), np.full(len(members), 1.0 / len(members)), size=n_boot)
        within = group_weight[:, group_codes]
        family_only = family_draws[:, family_codes]
        assert within.shape == (n_boot, n)
        return cls(within * family_only, within, family_only, family_codes, family_names)


def interval(values: np.ndarray) -> dict[str, float]:
    low, high = percentile_ci(np.asarray(values, dtype=np.float64))
    return {"lo": low, "hi": high}


def plan_sample_size(
    scores: np.ndarray,
    labels: np.ndarray,
    families: Sequence[str],
    groups: Sequence[str],
    n_current: int,
    grid: Sequence[int],
    available: int,
    target: float,
    n_boot: int,
    seed: int,
) -> dict[str, object]:
    """Choose transcripts per class per family, from the G1 run, before any adapter exists.

    Variance of one AUROC is split into a family part, which more rows cannot shrink, and a transcript part, which
    scales with 1 / n. D is a difference of two such AUROCs. Their correlation cannot be known before adapters exist,
    so it is taken as zero, which overstates D's half-width if the arms are positively correlated, as paired data should
    be: half-width(D) = sqrt(2) * half-width(AUROC). If no n reaches the target, the largest available is used, and the
    log says so. The prereg sets no stop for this.
    """
    resampler = Resampler.build(families, groups, n_boot, seed)
    var_transcript = float(np.nanvar(weighted_auroc(scores, labels, resampler.within)))
    var_family = float(np.nanvar(weighted_auroc(scores, labels, resampler.family_only)))
    table = []
    chosen = None
    largest = min(max(grid), available)
    candidates = sorted({min(n, largest) for n in grid})
    for n in candidates:
        half_auroc = 1.959964 * float(np.sqrt(var_family + var_transcript * n_current / n))
        half_d = float(np.sqrt(2.0)) * half_auroc
        table.append({"n_per_class_per_family": n, "half_width_auroc": half_auroc, "expected_half_width_D": half_d})
        if chosen is None and half_d <= target:
            chosen = n
    return {
        "n_current": n_current,
        "var_family": var_family,
        "var_transcript": var_transcript,
        "assumption": "correlation between the DD and SHAM AUROCs taken as 0, before any adapter exists",
        "table": table,
        "target": target,
        "reached_target": chosen is not None,
        "chosen": chosen if chosen is not None else largest,
    }
