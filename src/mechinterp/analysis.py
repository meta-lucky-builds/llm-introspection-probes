"""Gates and decision rules from docs/preregistrations/E1_localization_capture.md, on the export.

Reading of "Holm across the three control tasks": the 95% bounds in each rule are the percentile
bounds, and a claim that a bound clears zero must also survive Holm across T_pick, T_fav and T_odd,
using p-values dual to those intervals. L_t > 0 and L_inj - L_t > 0 are separate Holm families. The
share bounds C_t are percentile bounds, not adjusted. Holm can only make these rules stricter.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

import numpy as np

from .stats import (
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
IMPACT_NOT_MATCHED = "impact match missed its tolerance; read H3 with that in mind"


class CellTable:
    """Trial-weighted cell values as [n_concepts, n_clusters] matrices."""

    def __init__(self, columns: Sequence[str], rows: Sequence[Sequence[Any]], concepts: Sequence[str] | None = None):
        self.rows = [dict(zip(columns, row)) for row in rows]
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
        by_condition[f"{task}|{condition}"] = {
            "L": _summary(l_point, l_boot),
            "hit": _summary(hit_point, hit_boot),
            "hit_minus_clean": _summary(hit_point - clean_point, hit_boot - clean_boot),
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
    h3 = estimates["h3"]
    if not h3:
        return {"supported": None}
    impact_match = export.get("impact_match") or {}
    if not impact_match.get("match_ok", False):
        warnings.append(IMPACT_NOT_MATCHED)
    return {
        "supported": h3["impact_minus_norm"]["lo"] > 0,
        "match_ok": impact_match.get("match_ok"),
        "achieved_js": impact_match.get("achieved_js"),
        "target_js": impact_match.get("target_js"),
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
        if record["label_mass"] < 0.5:
            warnings.append(
                f"clean answer mass {record['label_mass']} on {record['task']}/{record['cluster']}: "
                "the first token is often not an answer"
            )
    return warnings


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
    n_boot = n_boot if n_boot is not None else (meta.get("n_bootstrap") or 2000)
    seed = seed if seed is not None else (meta.get("seed") or 0)
    min_denominator = meta.get("min_denominator")
    min_denominator = 0.05 if min_denominator is None else min_denominator

    table = CellTable(export["cell_columns"], export["cells"])
    estimates = _estimates(table, n_boot, seed, min_denominator)
    warnings = _design_warnings(export)
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
