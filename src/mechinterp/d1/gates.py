"""The prereg's decision rules, as pure functions of numbers. Standard library only."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

INDEPENDENT = "independent_channel"
SAME = "same_channel"
INCONCLUSIVE = "inconclusive"
G1_FINDING = "G1 failed through 14B: this probe recipe is not reliable below that size."
G2_FINDING = "G2 failed after one retry: this training does not silence self-report across wordings at this scale."


@dataclass(frozen=True)
class Thresholds:
    """The bars in the prereg. Only tests pass anything else, through the Python API. The command line cannot."""

    g1_probe_lower: float = 0.80
    g1_self_report_lower: float = 0.70
    g2_dd_self_report_upper: float = 0.60
    g2_dd_p_yes: float = 0.90
    g2_sham_within: float = 0.05
    g2_capability_within: float = 0.05
    d_margin: float = -0.10


def _finite(*values: float) -> None:
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
        raise ValueError(f"A gate got a non-finite number: {values}. That is a broken run, not a gate failure.")


def gate_g1(probe_lower: float, self_report_lower: float, thresholds: Thresholds = Thresholds()) -> dict[str, Any]:
    _finite(probe_lower, self_report_lower)
    probe_ok = probe_lower >= thresholds.g1_probe_lower
    self_ok = self_report_lower >= thresholds.g1_self_report_lower
    return {
        "passed": bool(probe_ok and self_ok),
        "probe_ok": probe_ok,
        "self_report_ok": self_ok,
        "probe_auroc_lower": probe_lower,
        "self_report_auroc_lower": self_report_lower,
    }


def next_after_g1(passed: bool, rung_index: int, n_rungs: int) -> str:
    """"stage2" on the first pass, "next_rung" otherwise, and "stop" once the ladder has no rung left."""
    if passed:
        return "stage2"
    return "next_rung" if rung_index + 1 < n_rungs else "stop"


def gate_g2(
    dd_self_report_upper: float,
    dd_mean_p_yes: float,
    sham_self_report: float,
    unmodified_self_report: float,
    dd_accuracy: float,
    sham_accuracy: float,
    unmodified_accuracy: float,
    thresholds: Thresholds = Thresholds(),
) -> dict[str, Any]:
    """All inputs are already the median across the seeds trained so far. Accuracy is a fraction, so 5 points is 0.05."""
    _finite(dd_self_report_upper, dd_mean_p_yes, sham_self_report, unmodified_self_report, dd_accuracy, sham_accuracy, unmodified_accuracy)
    checks = {
        "dd_self_report_silenced": dd_self_report_upper <= thresholds.g2_dd_self_report_upper,
        "dd_says_yes": dd_mean_p_yes >= thresholds.g2_dd_p_yes,
        "sham_self_report_kept": abs(sham_self_report - unmodified_self_report) <= thresholds.g2_sham_within,
        "dd_capability_kept": abs(dd_accuracy - unmodified_accuracy) <= thresholds.g2_capability_within,
        "sham_capability_kept": abs(sham_accuracy - unmodified_accuracy) <= thresholds.g2_capability_within,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "numbers": {
            "dd_self_report_upper": dd_self_report_upper,
            "dd_mean_p_yes": dd_mean_p_yes,
            "sham_self_report": sham_self_report,
            "unmodified_self_report": unmodified_self_report,
            "dd_accuracy": dd_accuracy,
            "sham_accuracy": sham_accuracy,
            "unmodified_accuracy": unmodified_accuracy,
        },
    }


def next_after_g2(passed: bool, attempt: int, label: str) -> str:
    """One retry in the whole experiment, with twice the steps. label is "one_seed" or "all_seeds".

    "continue": one seed passed, so train the rest. "finish": all seeds passed. "retry": rerun from one seed at twice
    the steps. "stop": the stop rule. A pass at one seed followed by a failed median across seeds still counts as a
    G2 failure, because the prereg's G2 is the median, and it uses up the same single retry.
    """
    if passed:
        return "continue" if label == "one_seed" else "finish"
    return "retry" if attempt == 0 else "stop"


def primary_verdict(lower: float, upper: float, thresholds: Thresholds = Thresholds()) -> str:
    _finite(lower, upper)
    if lower > upper:
        raise ValueError(f"Interval is reversed: {lower} > {upper}")
    if lower > thresholds.d_margin:
        return INDEPENDENT
    if upper < thresholds.d_margin:
        return SAME
    return INCONCLUSIVE


def median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Median of nothing")
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else 0.5 * (ordered[middle - 1] + ordered[middle])


def describe_verdict(verdict: str) -> str:
    text: Mapping[str, str] = {
        INDEPENDENT: "Independent channel: D's lower bound is above -0.10.",
        SAME: "Same channel: D's upper bound is below -0.10.",
        INCONCLUSIVE: "Inconclusive: D's interval contains -0.10. No reading is favoured.",
    }
    return text[verdict]
