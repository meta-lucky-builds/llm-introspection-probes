"""AUROC, recall and the paired bootstrap. numpy only.

Transcripts are nested in families and families in lines of work, so the bootstrap draws lines of work, then families
within a line, then groups of transcripts within a family. Test has about 4 independent lines, not 7 families, which the
change log makes the top-level unit. With so few units intervals are optimistic, and the report says so beside any verdict.
One weight matrix serves every statistic in a run, which is what makes differences paired. Adapter seeds are not
resampled: there are three, and the prereg takes a median over them.

AUROC is on all test rows pooled, with each family holding equal numbers per class, so family base rates cannot look
like signal. Within-family AUROC is reported beside it. The two can differ when families sit at different score levels,
and a monitor with one threshold sees the pooled one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ..stats import NonFiniteError, percentile_ci
from .families import line_of_work

Z_975 = 1.959964
# Two-sided 95% Student t quantiles, by degrees of freedom. Enough for any number of lines of work this design can have.
T_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


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
    """Holds one set of weights.

    two_way: lines of work, families within them, and groups. within: groups only, family weight 1. family_only: lines and
    families, no groups, by row. family_draws: the same per family, [replicates, families], in family_names order.
    """

    two_way: np.ndarray
    within: np.ndarray
    family_only: np.ndarray
    families: np.ndarray
    family_names: list[str]
    family_draws: np.ndarray
    line_names: list[str]

    @classmethod
    def build(cls, families: Sequence[str], groups: Sequence[str], n_boot: int, seed: int) -> "Resampler":
        family_codes, family_names = _codes(families)
        group_codes, group_names = _codes(groups)
        rng = np.random.default_rng(seed)
        n_families = len(family_names)
        n = len(families)
        line_of_family = [line_of_work(name) for name in family_names]
        line_names = sorted(set(line_of_family))
        n_lines = len(line_names)
        line_draws = rng.multinomial(n_lines, np.full(n_lines, 1.0 / n_lines), size=n_boot)
        family_draws = np.zeros((n_boot, n_families), dtype=np.float32)
        for line_index, line in enumerate(line_names):
            members = np.flatnonzero(np.array([name == line for name in line_of_family]))
            inner = rng.multinomial(len(members), np.full(len(members), 1.0 / len(members)), size=n_boot)
            family_draws[:, members] = line_draws[:, [line_index]] * inner
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
        return cls(within * family_only, within, family_only, family_codes, family_names, family_draws, line_names)


def interval(values: np.ndarray) -> dict[str, float]:
    low, high = percentile_ci(np.asarray(values, dtype=np.float64))
    return {"lo": low, "hi": high}


def interval_widening(n_units: int) -> float:
    """How far a percentile interval over n units falls short of a t interval: t(n-1) / z * sqrt(n / (n-1)).

    About 1.87 at 4 units and 1.35 at 7. One unit has no spread to scale, so it reads 1.
    """
    if n_units < 2:
        return 1.0
    t = T_975.get(n_units - 1, Z_975)
    return float(t / Z_975 * np.sqrt(n_units / (n_units - 1)))


def plan_sample_size(
    scores: np.ndarray,
    labels: np.ndarray,
    families: Sequence[str],
    groups: Sequence[str],
    allocation: Mapping[str, int],
    n_boot: int,
    seed: int,
) -> dict[str, object]:
    """Report the half-width D should have at the fixed test size, from the G1 run, before any adapter exists.

    It decides nothing: the change log fixes the size and drops the 0.05 target. Variance of one AUROC is split into a part
    from lines of work and families, which more rows cannot shrink, and a transcript part, which scales with 1 / n. D is a
    difference of two such AUROCs, and how much of the first part they share cannot be known before adapters exist. So two
    figures bound it: the arms independent, half-width(D) = sqrt(2) * half-width(AUROC), and the first part cancelling
    entirely, which leaves only the transcript term. The bootstrap itself uses the real pairing later.
    """
    resampler = Resampler.build(families, groups, n_boot, seed)
    var_transcript = float(np.nanvar(weighted_auroc(scores, labels, resampler.within)))
    var_units = float(np.nanvar(weighted_auroc(scores, labels, resampler.family_only)))
    n_current = len(families)
    n_planned = 2 * sum(allocation.values())
    if n_planned == 0:
        raise ValueError("The test set has no transcripts")
    transcript_term = var_transcript * n_current / n_planned
    half_auroc = Z_975 * float(np.sqrt(var_units + transcript_term))
    return {
        "transcripts_per_arm": n_planned,
        "per_class_per_family": dict(allocation),
        "n_in_g1_set": n_current,
        "var_lines_and_families": var_units,
        "var_transcript": var_transcript,
        "expected_half_width_auroc": half_auroc,
        "expected_half_width_D_if_arms_independent": float(np.sqrt(2.0)) * half_auroc,
        "expected_half_width_D_if_unit_term_cancels": Z_975 * float(np.sqrt(2.0 * transcript_term)),
        "n_lines_of_work": len(resampler.line_names),
    }
