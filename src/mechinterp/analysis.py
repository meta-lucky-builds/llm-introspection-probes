"""Gates and decision rules from docs/preregistrations/E1_localization_capture.md, on the export.

Reading of "Holm across the three control tasks": the 95% bounds in each rule are the percentile
bounds, and a claim that a bound clears zero must also survive Holm across T_pick, T_fav and T_odd,
using p-values dual to those intervals. L_t > 0 and L_inj - L_t > 0 are separate Holm families. The
share bounds C_t are percentile bounds, not adjusted. Holm can only make these rules stricter.
"""

from __future__ import annotations

import json
import math
from typing import Any, Sequence

import numpy as np

from .identity import identity_differences
from .stats import (
    NonFiniteError,
    approx_percentile_coverage,
    bootstrap_p_two_sided,
    holm_adjust,
    percentile_ci,
    ratio_summary,
    two_way_weights,
    weighted_cell_mean,
)

CONTROL_TASKS = ("T_pick", "T_fav", "T_odd")
FACT_ACCURACY_THRESHOLD = 0.70
CHANCE = 0.10
FAMILY_ALPHA = 0.05
CAPTURE_SHARE_LOWER = 0.5
GATED_SHARE_UPPER = 0.25
IMPACT_NOT_MATCHED = "impact match missed its tolerance, so H3 is uninformative: it was not tested"
LOW_LABEL_MASS = 0.5
LOW_COVERAGE = 0.93
NUMERIC_CELL_FIELDS = ("n", "L", "hit", "clean_hit", "fact_correct", "label_mass", "none_mass")


class MixedRunError(ValueError):
    """Rows in one export came from different code, weights, dtype or library versions."""


def _require_finite(value: Any, context: str) -> None:
    if value is not None and not math.isfinite(value):
        raise NonFiniteError(f"Non-finite value {value} in {context}")


class CellTable:
    """Trial-weighted cell values as [n_concepts, n_clusters] matrices. Missing cells stay at zero weight."""

    def __init__(self, columns: Sequence[str], rows: Sequence[Sequence[Any]], concepts: Sequence[str] | None = None):
        self.rows = [dict(zip(columns, row)) for row in rows]
        for row in self.rows:
            for name in NUMERIC_CELL_FIELDS:
                _require_finite(row.get(name), f"{row['task']}/{row['condition']}/{row['concept']}/{row['cluster']}.{name}")
        if concepts is not None:
            kept = set(concepts)
            self.rows = [row for row in self.rows if row["concept"] in kept]
        self.concepts = sorted({row["concept"] for row in self.rows})
        self.clusters = sorted({row["cluster"] for row in self.rows})
        self._concept_index = {name: index for index, name in enumerate(self.concepts)}
        self._cluster_index = {name: index for index, name in enumerate(self.clusters)}

    def matrices(self, task: str, condition: str, field: str) -> tuple[np.ndarray, np.ndarray]:
        sums = np.zeros((len(self.concepts), len(self.clusters)))
        counts = np.zeros_like(sums)
        for row in self.rows:
            if row["task"] == task and row["condition"] == condition and row[field] is not None:
                c = self._concept_index[row["concept"]]
                k = self._cluster_index[row["cluster"]]
                sums[c, k] += row["n"] * row[field]
                counts[c, k] += row["n"]
        return sums, counts

    def has(self, task: str, condition: str) -> bool:
        return any(row["task"] == task and row["condition"] == condition for row in self.rows)

    def conditions(self) -> list[tuple[str, str]]:
        return sorted({(row["task"], row["condition"]) for row in self.rows})


def _summary(point: float, boot: np.ndarray) -> dict[str, float]:
    low, high = percentile_ci(boot)
    for value in (point, low, high):
        _require_finite(float(value), "a bootstrap summary")
    return {"point": float(point), "lo": low, "hi": high, "p": bootstrap_p_two_sided(boot)}


class _Bootstrap:
    """Point estimates and replicates, all from one shared set of resampling weights."""

    def __init__(self, table: CellTable, n_boot: int, seed: int):
        self.table = table
        rng = np.random.default_rng(seed)
        self.concept_weights, self.cluster_weights = two_way_weights(
            len(table.concepts), len(table.clusters), n_boot, rng
        )

    def statistic(self, task: str, condition: str, field: str) -> tuple[float, np.ndarray]:
        sums, counts = self.table.matrices(task, condition, field)
        if counts.sum() == 0:
            raise KeyError(f"No data for {task}/{condition}/{field}")
        return sums.sum() / counts.sum(), weighted_cell_mean(sums, counts, self.concept_weights, self.cluster_weights)


def _rule_verdict(control: dict[str, Any]) -> str:
    """Capture, task-gated or mixed, from the preregistered rule."""
    effect = control["L"]
    gap = control["L_inj_minus_L"]
    share = control["share"]
    captured = (
        effect["lo"] > 0
        and control["L_holm_p"] < FAMILY_ALPHA
        and share["lo"] is not None
        and share["lo"] >= CAPTURE_SHARE_LOWER
    )
    if captured:
        return "capture"
    gated = (
        gap["lo"] > 0
        and control["gap_holm_p"] < FAMILY_ALPHA
        and share["hi"] is not None
        and share["hi"] < GATED_SHARE_UPPER
    )
    return "task_gated" if gated else "mixed"


def _estimates(table: CellTable, n_boot: int, seed: int, min_denominator: float) -> dict[str, Any]:
    boot = _Bootstrap(table, n_boot, seed)
    inj_point, inj_boot = boot.statistic("T_inj", "concept", "L")
    by_condition: dict[str, Any] = {}
    for task, condition in table.conditions():
        l_point, l_boot = boot.statistic(task, condition, "L")
        hit_point, hit_boot = boot.statistic(task, condition, "hit")
        clean_point, clean_boot = boot.statistic(task, condition, "clean_hit")
        mass_point, mass_boot = boot.statistic(task, condition, "label_mass")
        by_condition[f"{task}|{condition}"] = {
            "L": _summary(l_point, l_boot),
            "hit": _summary(hit_point, hit_boot),
            "hit_minus_clean": _summary(hit_point - clean_point, hit_boot - clean_boot),
            "injected_label_mass": _summary(mass_point, mass_boot),
        }

    controls: dict[str, Any] = {}
    for task in CONTROL_TASKS:
        point, values = boot.statistic(task, "concept", "L")
        controls[task] = {
            "L": _summary(point, values),
            "L_inj_minus_L": _summary(inj_point - point, inj_boot - values),
            "share": ratio_summary(point, inj_point, values, inj_boot, min_denominator),
        }
    effect_holm = holm_adjust([controls[task]["L"]["p"] for task in CONTROL_TASKS])
    gap_holm = holm_adjust([controls[task]["L_inj_minus_L"]["p"] for task in CONTROL_TASKS])
    for index, task in enumerate(CONTROL_TASKS):
        controls[task]["L_holm_p"] = effect_holm[index]
        controls[task]["gap_holm_p"] = gap_holm[index]

    h3: dict[str, Any] = {}
    if table.has("T_inj", "random_norm") and table.has("T_inj", "random_impact"):
        norm_point, norm_boot = boot.statistic("T_inj", "random_norm", "L")
        impact_point, impact_boot = boot.statistic("T_inj", "random_impact", "L")
        h3 = {
            "impact_minus_norm": _summary(impact_point - norm_point, impact_boot - norm_boot),
            "concept_minus_impact": _summary(inj_point - impact_point, inj_boot - impact_boot),
            "concept_minus_norm": _summary(inj_point - norm_point, inj_boot - norm_boot),
        }

    fact_correct_point, fact_correct_boot = boot.statistic("T_fact", "concept", "fact_correct")
    return {
        "by_condition": by_condition,
        "controls": controls,
        "h3": h3,
        "fact_accuracy_under_injection": _summary(fact_correct_point, fact_correct_boot),
        # How much more often T_fact answers the injected position than clean, on the same prompt.
        "fact_interference": by_condition["T_fact|concept"]["hit_minus_clean"],
    }


def _verdicts(estimates: dict[str, Any]) -> dict[str, Any]:
    controls = estimates["controls"]
    pick = _rule_verdict(controls["T_pick"])
    fav = _rule_verdict(controls["T_fav"])
    odd = _rule_verdict(controls["T_odd"])
    return {
        "H1": {"T_pick": pick, "T_fav": fav, "overall": pick if pick == fav else "mixed"},
        "H2": {"T_odd": odd, "supported": odd == "capture"},
    }


def _h3_verdict(estimates: dict[str, Any], export: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    """H3 is defined on impact-matched directions, so an unmatched run does not test it."""
    h3 = estimates["h3"]
    impact_match = export.get("impact_match") or {}
    matched = impact_match.get("match_ok") is True
    if not h3 or not matched:
        if not matched:
            warnings.append(IMPACT_NOT_MATCHED)
        verdict = "uninformative"
    else:
        verdict = "supported" if h3["impact_minus_norm"]["lo"] > 0 else "not_supported"
    return {
        "verdict": verdict,
        "supported": verdict == "supported",
        "match_ok": impact_match.get("match_ok"),
        "achieved_js": impact_match.get("achieved_js"),
        "target_js": impact_match.get("target_js"),
        "relative_error": impact_match.get("relative_error"),
        "tolerance": impact_match.get("tolerance"),
    }


def _gate_v1(export: dict[str, Any]) -> dict[str, Any]:
    clean_fact = export["clean_fact"]
    accuracy = clean_fact["correct"] / clean_fact["n"] if clean_fact["n"] else float("nan")
    # NaN compares False, so a run without clean T_fact data fails the gate.
    return {"accuracy": accuracy, "n": clean_fact["n"], "passed": bool(accuracy >= FACT_ACCURACY_THRESHOLD)}


def _gate_v2(estimates: dict[str, Any]) -> dict[str, Any]:
    t_inj = estimates["by_condition"]["T_inj|concept"]
    return {
        "accuracy": t_inj["hit"],
        "L_inj": t_inj["L"],
        "passed": bool(t_inj["hit"]["lo"] > CHANCE and t_inj["L"]["lo"] > 0),
    }


def _design_warnings(export: dict[str, Any]) -> list[str]:
    warnings = []
    for name, counts in export.get("position_counts", {}).items():
        if len(set(counts)) != 1:
            warnings.append(f"positions are unbalanced for {name}: {counts}")
    for row in export.get("clean", []):
        record = dict(zip(export["clean_columns"], row))
        _require_finite(record["label_mass"], "a clean row")
        if record["label_mass"] + (record.get("none_mass") or 0.0) < LOW_LABEL_MASS:
            warnings.append(
                f"clean answer mass {record['label_mass']} on {record['task']}/{record['cluster']}: "
                "the first token is often not an answer"
            )
    return warnings


def _label_mass_report(export: dict[str, Any], estimates: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    """Sum of the ten label probabilities, per task. Descriptive only: no rule or filter reads it."""
    injected = {name: item["injected_label_mass"] for name, item in estimates["by_condition"].items()}
    clean_totals: dict[str, list[float]] = {}
    for row in export.get("clean", []):
        record = dict(zip(export["clean_columns"], row))
        totals = clean_totals.setdefault(record["task"], [0.0, 0.0])
        totals[0] += record["n"] * record["label_mass"]
        totals[1] += record["n"]
    for name, summary in injected.items():
        task = name.split("|")[0]
        if task != "T_inj_none" and summary["point"] < LOW_LABEL_MASS:
            warnings.append(
                f"injected label mass {summary['point']:.3f} on {name}: the injected distribution "
                "has moved off the labels, so L and accuracy on this row are hard to read"
            )
    return {
        "injected": injected,
        "clean_by_task": {task: total / n for task, (total, n) in clean_totals.items() if n},
        "note": "Descriptive. Mass is not used by any gate, rule or filter. T_inj_none also puts mass on none.",
    }


def _small_sample_caveat(table: CellTable, warnings: list[str]) -> dict[str, Any]:
    by_cluster = approx_percentile_coverage(len(table.clusters))
    by_concept = approx_percentile_coverage(len(table.concepts))
    known = [value for value in (by_cluster, by_concept) if value is not None]
    if not known or min(known) < LOW_COVERAGE:
        warnings.append(
            f"percentile intervals undercover with {len(table.concepts)} concepts and {len(table.clusters)} "
            f"clusters: nominal 95% is about {by_concept} if concept variance dominates and about "
            f"{by_cluster} if cluster variance dominates, in the direction of finding effects"
        )
    return {
        "n_concepts": len(table.concepts),
        "n_clusters": len(table.clusters),
        "approx_coverage_if_concept_variance_dominates": by_concept,
        "approx_coverage_if_cluster_variance_dominates": by_cluster,
    }

def _secondary_screened(
    export: dict[str, Any], n_boot: int, seed: int, min_denominator: float, top_fraction: float | None
) -> dict[str, Any] | None:
    """Mimic Zou's screened pool: keep test concepts with the best accuracy on calibration clusters."""
    screen = export.get("screen") or {}
    if not screen or top_fraction is None:
        return None
    ranked = sorted(screen, key=lambda concept: (-screen[concept][0] / screen[concept][1], concept))
    kept = ranked[: max(2, int(round(len(ranked) * top_fraction)))]
    table = CellTable(export["cell_columns"], export["cells"], concepts=kept)
    estimates = _estimates(table, n_boot, seed + 1, min_denominator)
    return {"concepts": kept, "estimates": estimates, "verdicts": _verdicts(estimates)}


def analyze(export: dict[str, Any], n_boot: int | None = None, seed: int | None = None) -> dict[str, Any]:
    """Apply gate V1, gate V2, then the decision rules, in that order.

    `outcome` is one of incomplete, V1_fail, V2_fail, capture, task_gated, mixed, pilot_not_for_inference.
    """
    meta = export["meta"]
    if not meta.get("complete"):
        return {"outcome": "incomplete", "n_records": meta.get("n_records"), "meta": meta}
    environments = meta.get("environments") or []
    if len(environments) > 1:
        differing = identity_differences(environments[0], environments[-1])
        raise MixedRunError(f"Export mixes {len(environments)} run identities, differing in {differing}")
    n_boot = n_boot if n_boot is not None else (meta.get("n_bootstrap") or 2000)
    seed = seed if seed is not None else (meta.get("seed") or 0)
    min_denominator = meta.get("min_denominator")
    min_denominator = 0.05 if min_denominator is None else min_denominator

    table = CellTable(export["cell_columns"], export["cells"])
    estimates = _estimates(table, n_boot, seed, min_denominator)
    warnings = _design_warnings(export)
    label_mass = _label_mass_report(export, estimates, warnings)
    caveat = _small_sample_caveat(table, warnings)
    v1 = _gate_v1(export)
    v2 = _gate_v2(estimates)
    report: dict[str, Any] = {
        "meta": meta,
        "selection": export.get("selection"),
        "impact_match": export.get("impact_match"),
        "sham": export.get("sham"),
        "clean_columns": export.get("clean_columns"),
        "clean": export.get("clean"),
        "dropped_words": export.get("dropped_words"),
        "dropped_clusters": export.get("dropped_clusters"),
        "dropped_tasks": export.get("dropped_tasks"),
        "vector_diagnostics": export.get("vector_diagnostics"),
        "label_mass": label_mass,
        "small_sample_caveat": caveat,
        "gates": {"V1": v1, "V2": v2},
        "estimates": estimates,
        "verdicts": None,
        "warnings": warnings,
    }
    if meta.get("inference_allowed") is False:
        report["outcome"] = "pilot_not_for_inference"
    elif not v1["passed"]:
        report["outcome"] = "V1_fail"
    elif not v2["passed"]:
        report["outcome"] = "V2_fail"
    else:
        verdicts = _verdicts(estimates)
        verdicts["H3"] = _h3_verdict(estimates, export, warnings)
        report["verdicts"] = verdicts
        report["outcome"] = verdicts["H1"]["overall"]
        report["secondary_screened"] = _secondary_screened(
            export, n_boot, seed, min_denominator, meta.get("screen_top_fraction")
        )
    return report


def format_report(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=1, sort_keys=True, default=str)
