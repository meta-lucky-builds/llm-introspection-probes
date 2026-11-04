"""Stage 2's correlations and Stage 3's decision rules, from docs/preregistrations/E2_jspace_vector_families.md.

Estimand, bootstrap and intervals are E1's: the same cell export, CellTable and two-way resampling. numpy only.

Readings of the prereg, each listed in experiments/e2_jspace_vector_families/README.md:
- R(F, t) is undefined, and the cell is "no effect to decompose", if L(F_full, t) is below 0.05 or its unadjusted
  lower bound is not above 0.
- Holm runs "across the two sites and three tasks for each comparison". A comparison is any bound a rule needs
  to clear: L(F_j) - L(F_res), and L(F) - 0.5 L(F_full) for each of F_j, F_res and F_rand, the last being what
  R >= 0.5 says. Each cell must clear its percentile bound and survive Holm at 0.05. Holm can only make a rule
  stricter. The classification without Holm is reported beside it, because this reading is stricter than E1's,
  whose share bounds were unadjusted, and it also makes H3 harder to reach.
- Only cells where a rule can be evaluated join its Holm family. An undefined cell or an unmatched family tests
  nothing, so it does not count.
- H3 takes precedence, then H1, then H2, and anything else is mixed. An unmatched F_rand leaves H3 untested, and
  an unmatched F_j or F_res leaves H1 and H2 untested. Each cell says which.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

import numpy as np

from ..analysis import CellTable, _Bootstrap, _small_sample_caveat, _summary
from ..export import build_export
from ..stats import holm_adjust, percentile_ci, ratio_summary

THRESHOLD = 0.5
UNDEFINED = "no_effect_to_decompose"
COMPARISONS = ("gap_j_minus_res", "ratio_F_j", "ratio_F_res", "ratio_F_rand")


def _site_cells(records: Sequence[dict[str, Any]], layer: int) -> tuple[list[str], list[list[Any]]]:
    """E1's export on the test rows of one site, so a cell here means what a cell means in E1."""
    kept = [record for record in records if record.get("kind") == "trial" and record["phase"] == "test" and record["layer"] == layer]
    export = build_export(kept)
    return export["cell_columns"], export["cells"]


def evaluate_cell(boot: _Bootstrap, task: str, min_denominator: float, matched: dict[str, bool]) -> dict[str, Any]:
    """All estimates for one task at one site. Classification happens later, once Holm has seen every cell."""
    full_point, full_boot = boot.statistic(task, "F_full", "L")
    full = _summary(full_point, full_boot)
    defined = bool(full_point >= min_denominator and full["lo"] > 0.0)
    cell: dict[str, Any] = {"L": {"F_full": full}, "defined": defined, "ratio": {}, "above_half": {}, "matched": dict(matched)}
    points = {"F_full": (full_point, full_boot)}
    for family in ("F_j", "F_res", "F_rand"):
        if not boot.table.has(task, family):
            continue
        point, values = boot.statistic(task, family, "L")
        points[family] = (point, values)
        cell["L"][family] = _summary(point, values)
        cell["ratio"][family] = ratio_summary(point, full_point, values, full_boot, min_denominator)
        cell["above_half"][family] = _summary(point - THRESHOLD * full_point, values - THRESHOLD * full_boot)
    if "F_j" in points and "F_res" in points:
        cell["gap_j_minus_res"] = _summary(points["F_j"][0] - points["F_res"][0], points["F_j"][1] - points["F_res"][1])
        cell["gap_res_minus_j"] = _summary(points["F_res"][0] - points["F_j"][0], points["F_res"][1] - points["F_j"][1])
    return cell


def _in_family(cell: dict[str, Any], comparison: str) -> bool:
    if not cell["defined"]:
        return False
    matched = cell["matched"]
    if comparison == "gap_j_minus_res":
        return matched.get("F_j", False) and matched.get("F_res", False) and "gap_j_minus_res" in cell
    family = comparison.removeprefix("ratio_")
    return matched.get(family, False) and family in cell["above_half"]


def _p_of(cell: dict[str, Any], comparison: str) -> float:
    if comparison == "gap_j_minus_res":
        return cell["gap_j_minus_res"]["p"]
    return cell["above_half"][comparison.removeprefix("ratio_")]["p"]


def apply_holm(cells: dict[tuple[int, str], dict[str, Any]]) -> dict[str, int]:
    """Add holm_p per comparison to each cell, and return each family's size."""
    sizes = {}
    for comparison in COMPARISONS:
        members = [name for name in cells if _in_family(cells[name], comparison)]
        sizes[comparison] = len(members)
        adjusted = holm_adjust([_p_of(cells[name], comparison) for name in members])
        for name, value in zip(members, adjusted):
            cells[name].setdefault("holm_p", {})[comparison] = value
    return sizes


def _rules(cell: dict[str, Any], alpha: float, use_holm: bool) -> dict[str, Any]:
    def clears(comparison: str) -> bool:
        return not use_holm or cell.get("holm_p", {}).get(comparison, 1.0) < alpha

    def ratio_ok(family: str) -> bool:
        ratio = cell["ratio"].get(family)
        return bool(ratio and ratio["lo"] is not None and ratio["lo"] >= THRESHOLD and clears(f"ratio_{family}"))

    matched = cell["matched"]
    routes_testable = matched.get("F_j", False) and matched.get("F_res", False) and "gap_j_minus_res" in cell
    h3_testable = matched.get("F_rand", False) and "F_rand" in cell["ratio"]
    h1 = routes_testable and ratio_ok("F_j") and cell["gap_j_minus_res"]["lo"] > 0.0 and clears("gap_j_minus_res")
    h2 = routes_testable and ratio_ok("F_res") and cell["gap_res_minus_j"]["lo"] > 0.0 and clears("gap_j_minus_res")
    h3 = h3_testable and ratio_ok("F_rand")
    if h3:
        label = "H3"
    elif h1:
        label = "H1"
    elif h2:
        label = "H2"
    else:
        label = "mixed"
    return {"label": label, "H1": h1, "H2": h2, "H3": h3, "routes_testable": routes_testable, "h3_testable": h3_testable}


def classify(cells: dict[tuple[int, str], dict[str, Any]], alpha: float) -> None:
    for cell in cells.values():
        if not cell["defined"]:
            cell["classification"] = UNDEFINED
            cell["rules"] = None
            cell["classification_without_holm"] = UNDEFINED
            continue
        rules = _rules(cell, alpha, use_holm=True)
        cell["rules"] = rules
        cell["classification"] = rules["label"]
        cell["classification_without_holm"] = _rules(cell, alpha, use_holm=False)["label"]


def decide(cells: dict[tuple[int, str], dict[str, Any]], sites: Sequence[int], tasks: Sequence[str]) -> dict[str, Any]:
    """H4 per site, and what the prereg's stop rule says. Any cell's label can be undefined."""
    h4 = {}
    for site in sites:
        inj = cells.get((site, "T_inj"), {}).get("classification")
        pick = cells.get((site, "T_pick"), {}).get("classification")
        h4[str(site)] = {"T_inj": inj, "T_pick": pick, "dissociation": inj == "H1" and pick in ("H2", "H3")}
    labels = [cells[(site, task)]["classification"] for site in sites for task in tasks]
    any_route = any(label in ("H1", "H2") for label in labels)
    return {
        "H4": h4,
        "any_H1": "H1" in labels,
        "line_closes_for_qwen3": not any_route,
        "ablation_warranted": "H1" in labels,
        "stop_rule_note": (
            "The prereg closes this line if both sites come out H3 or mixed. Here that means no task at any site is H1 or "
            "H2. A cell with no effect to decompose counts as neither. Ablation runs only if some task is H1."
        ),
    }


def analyze_tables(
    tables: dict[int, tuple[list[str], list[list[Any]]]],
    matched: dict[int, dict[str, bool]],
    tasks: Sequence[str],
    n_boot: int,
    seed: int,
    min_denominator: float,
    alpha: float,
    lens_concepts: Sequence[str] = (),
) -> dict[str, Any]:
    """The decision rules on cell tables, per site. Separate from the store, so a test can hand it synthetic cells."""
    cells: dict[tuple[int, str], dict[str, Any]] = {}
    caveats: dict[str, Any] = {}
    warnings: list[str] = []
    f_lens: dict[str, Any] = {}
    for site, (columns, rows) in tables.items():
        table = CellTable(columns, rows)
        boot = _Bootstrap(table, n_boot, seed)
        caveats[str(site)] = _small_sample_caveat(table, warnings)
        for task in tasks:
            cells[(site, task)] = evaluate_cell(boot, task, min_denominator, matched[site])
        # F_lens has only the single-token concepts. It enters no rule, and compares with F_full on those concepts.
        lens_report: dict[str, Any] = {"n_concepts": len(lens_concepts), "matched": matched[site].get("F_lens", False), "by_task": {}}
        if lens_concepts:
            subset = CellTable(columns, rows, concepts=lens_concepts)
            subset_boot = _Bootstrap(subset, n_boot, seed)
            for task in tasks:
                if not subset.has(task, "F_lens"):
                    continue
                lens_point, lens_boot = subset_boot.statistic(task, "F_lens", "L")
                full_point, full_boot = subset_boot.statistic(task, "F_full", "L")
                lens_report["by_task"][task] = {
                    "L_F_lens": _summary(lens_point, lens_boot),
                    "L_F_full_same_concepts": _summary(full_point, full_boot),
                    "ratio": ratio_summary(lens_point, full_point, lens_boot, full_boot, min_denominator),
                }
        f_lens[str(site)] = lens_report
    holm_sizes = apply_holm(cells)
    classify(cells, alpha)
    sites = list(tables)
    decision = decide(cells, sites, tasks)
    return {
        "sites": sites,
        "tasks": list(tasks),
        "cells": {f"{site}|{task}": cell for (site, task), cell in cells.items()},
        "classification": {f"{site}|{task}": cell["classification"] for (site, task), cell in cells.items()},
        "classification_without_holm": {f"{site}|{task}": cell["classification_without_holm"] for (site, task), cell in cells.items()},
        "holm_family_sizes": holm_sizes,
        "decision": decision,
        "f_lens": f_lens,
        "small_sample_caveat": caveats,
        "warnings": warnings,
    }


def analyze_stage3(records: Sequence[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    """From the merged store and the run's plan. Bootstrap settings come from the store's manifest."""
    manifest = next(record for record in records if record.get("kind") == "manifest")
    sites = [int(site) for site in summary["sites"]]
    tasks = summary["tasks"]
    matches = [record for record in records if record.get("kind") == "impact_match"]
    matched: dict[int, dict[str, bool]] = {site: {"F_full": True} for site in sites}
    for match in matches:
        matched[match["layer"]][match["family"]] = bool(match["match_ok"])
    tables = {site: _site_cells(records, site) for site in sites}
    report = analyze_tables(
        tables,
        matched,
        tasks,
        manifest.get("n_bootstrap") or 2000,
        manifest.get("seed") or 0,
        manifest.get("min_denominator") if manifest.get("min_denominator") is not None else 0.05,
        0.05,
        summary.get("lens_test", ()),
    )
    report["impact_matches"] = [
        {name: match.get(name) for name in ("family", "layer", "target_js", "achieved_js", "relative_error", "alpha_impact", "alpha_full", "scale", "match_ok", "history")}
        for match in matches
    ]
    report["unmatched"] = [{"family": match["family"], "layer": match["layer"], "relative_error": match["relative_error"]} for match in matches if not match["match_ok"]]
    report["site_selection"] = next((record for record in records if record.get("kind") == "selection"), None)
    report["plan"] = summary
    report["cells_by_site"] = {str(site): {"columns": tables[site][0], "rows": tables[site][1]} for site in sites}
    run_complete = next((record for record in records if record.get("kind") == "run_complete"), {})
    report["warnings"] = [*report["warnings"], *(run_complete.get("warnings") or [])]
    report["single_token_n_test"] = len(summary.get("lens_test", ()))
    return report


# Stage 2


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks of a 1-D array, with ties sharing their mean rank."""
    ranks = np.empty(values.size, dtype=float)
    ranks[values.argsort()] = np.arange(values.size, dtype=float)
    for value in np.unique(values):
        mask = values == value
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    return ranks


def spearman(first: np.ndarray, second: np.ndarray) -> float:
    """NaN when either side is constant, where a rank correlation has no value."""
    a = _average_ranks(np.asarray(first, dtype=float))
    b = _average_ranks(np.asarray(second, dtype=float))
    if a.std() == 0.0 or b.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman_interval(first: np.ndarray, second: np.ndarray, n_boot: int, seed: int) -> dict[str, Any]:
    """Point estimate and percentile interval, resampling concepts."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    picks = np.random.default_rng(seed).integers(0, first.size, size=(n_boot, first.size))
    values = np.array([spearman(first[pick], second[pick]) for pick in picks])
    low, high = percentile_ci(values)
    point = spearman(first, second)
    return {
        "rho": None if np.isnan(point) else point,
        "lo": None if np.isnan(low) else low,
        "hi": None if np.isnan(high) else high,
        "n_concepts": int(first.size),
        "undefined_replicates": int(np.isnan(values).sum()),
    }


def stage2_report(e1: dict[str, Any], decomposition: dict[str, Any], concepts: Sequence[str], n_boot: int, seed: int) -> dict[str, Any]:
    """J-share against L_inj and L_pick, across test concepts, at l_s only: E1 measured its vectors there."""
    l_s = decomposition["l_s"]
    shares = np.array([decomposition["by_site"][str(l_s)][concept]["j_share"] for concept in concepts])
    missing = [concept for concept in concepts if concept not in e1["L_inj"] or concept not in e1["L_pick"]]
    if missing:
        raise ValueError(f"E1's results hold no L for {missing[:3]}, so these are not E1's test concepts")
    report: dict[str, Any] = {
        "note": "Descriptive. This stage decides nothing alone. Correlations are at l_s only, as the prereg says.",
        "k": decomposition["k"],
        "l_s": l_s,
        "l_w": decomposition["l_w"],
        "correlations_at_l_s": {
            "J_share_vs_L_inj": spearman_interval(shares, np.array([e1["L_inj"][concept] for concept in concepts]), n_boot, seed),
            "J_share_vs_L_pick": spearman_interval(shares, np.array([e1["L_pick"][concept] for concept in concepts]), n_boot, seed),
        },
        "by_site": {},
    }
    for site, by_concept in decomposition["by_site"].items():
        test_values = np.array([by_concept[concept]["j_share"] for concept in concepts])
        report["by_site"][site] = {
            "j_share_median": float(np.median(test_values)),
            "j_share_min": float(test_values.min()),
            "j_share_max": float(test_values.max()),
            "concepts": {
                concept: {
                    "j_share": by_concept[concept]["j_share"],
                    "atoms": by_concept[concept]["atoms"],
                    "top_lens_tokens": [item["text"] for item in by_concept[concept]["top_lens_tokens"]],
                }
                for concept in concepts
            },
        }
    return report


def format_summary(analysis: dict[str, Any], gate: dict[str, Any], stage2: dict[str, Any]) -> str:
    """The readable part. The numbers behind it are in analysis.json, stage2.json and gate.json."""
    lines = [
        f"Gate: passed. Band {gate['band'][0]} to {gate['band'][-1]}, l_w = {gate['l_w']}.",
        f"Stage 2 at l_s = {stage2['l_s']}: J-share vs L_inj {json.dumps(stage2['correlations_at_l_s']['J_share_vs_L_inj'])}",
        f"Stage 2 at l_s = {stage2['l_s']}: J-share vs L_pick {json.dumps(stage2['correlations_at_l_s']['J_share_vs_L_pick'])}",
        f"F_lens: {analysis['single_token_n_test']} test concepts are one token.",
        f"Unmatched families, excluded from the rules: {json.dumps(analysis['unmatched'])}",
        "",
        "site|task  classification  (without Holm)  R(F_j)lo  R(F_res)lo  R(F_rand)lo  L(F_full)",
    ]

    def low(ratio: dict[str, Any] | None) -> str:
        if not ratio or ratio.get("lo") is None:
            return "  undef"
        return f"{ratio['lo']:7.3f}"

    for name, cell in analysis["cells"].items():
        ratios = cell["ratio"]
        lines.append(
            f"{name:14s} {cell['classification']:22s} ({cell['classification_without_holm']}) "
            f"{low(ratios.get('F_j'))} {low(ratios.get('F_res'))} {low(ratios.get('F_rand'))} {cell['L']['F_full']['point']:.3f}"
        )
    lines.extend(["", f"H4 and stop rule: {json.dumps(analysis['decision'])}"])
    lines.extend(f"WARNING: {warning}" for warning in analysis["warnings"])
    return "\n".join(lines)
