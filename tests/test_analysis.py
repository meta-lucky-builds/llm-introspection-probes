"""Synthetic data with a known answer for each preregistered outcome."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from mechinterp.analysis import _rule_verdict, analyze
from mechinterp.export import CELL_COLUMNS, CLEAN_COLUMNS, build_export, decode_export, encode_export

N_CONCEPTS = 12
N_CLUSTERS = 8
N_PER_CELL = 10
DEFAULT_L = {
    ("T_inj", "concept"): 4.0,
    ("T_inj", "random_norm"): 0.5,
    ("T_inj", "random_impact"): 2.0,
    ("T_pick", "concept"): 3.5,
    ("T_fav", "concept"): 3.2,
    ("T_odd", "concept"): 3.0,
    ("T_fact", "concept"): 1.0,
    ("T_inj_none", "concept"): 3.0,
}


def synthetic_export(
    effects: dict[tuple[str, str], float] | None = None,
    t_inj_hit: float = 0.6,
    fact_accuracy: float = 0.9,
    complete: bool = True,
    inference_allowed: bool = True,
    match_ok: bool = True,
) -> dict[str, Any]:
    rng = np.random.default_rng(42)
    means = {**DEFAULT_L, **(effects or {})}
    concepts = [f"c{i}" for i in range(N_CONCEPTS)]
    clusters = [f"k{i}" for i in range(N_CLUSTERS)]
    concept_effects = rng.normal(0, 0.1, N_CONCEPTS)
    cluster_effects = rng.normal(0, 0.1, N_CLUSTERS)
    rows = []
    for (task, condition), mean in means.items():
        hit_rate = t_inj_hit if (task, condition) == ("T_inj", "concept") else 0.2
        for c, concept in enumerate(concepts):
            for k, cluster in enumerate(clusters):
                value = mean + concept_effects[c] + cluster_effects[k] + rng.normal(0, 0.2)
                rows.append(
                    [
                        task,
                        condition,
                        concept,
                        cluster,
                        N_PER_CELL,
                        value,
                        rng.binomial(N_PER_CELL, hit_rate) / N_PER_CELL,
                        rng.binomial(N_PER_CELL, 0.1) / N_PER_CELL,
                        0.8 if task == "T_fact" else None,
                        0.95,
                    ]
                )
    return {
        "meta": {
            "complete": complete,
            "inference_allowed": inference_allowed,
            "seed": 1,
            "n_bootstrap": 1000,
            "min_denominator": 0.05,
            "n_records": len(rows),
            "screen_top_fraction": None,
        },
        "clean_fact": {"correct": int(fact_accuracy * 100), "n": 100},
        "clean_columns": CLEAN_COLUMNS,
        "clean": [],
        "screen": {},
        "position_counts": {},
        "cell_columns": CELL_COLUMNS,
        "cells": rows,
        "impact_match": {"match_ok": match_ok, "target_js": 0.1, "achieved_js": 0.1},
    }


GATED = {("T_pick", "concept"): 0.2, ("T_fav", "concept"): 0.2, ("T_odd", "concept"): 0.2}
# Slightly negative, so that sampling noise cannot push the lower bound above zero.
NO_EFFECT = {("T_inj", "concept"): -0.05}


def test_capture() -> None:
    report = analyze(synthetic_export())
    assert report["outcome"] == "capture"
    assert report["gates"]["V1"]["passed"] and report["gates"]["V2"]["passed"]
    assert report["verdicts"]["H1"] == {"T_pick": "capture", "T_fav": "capture", "overall": "capture"}
    assert report["verdicts"]["H2"]["supported"] is True
    assert report["estimates"]["controls"]["T_pick"]["share"]["lo"] >= 0.5


def test_task_gated() -> None:
    report = analyze(synthetic_export(GATED))
    assert report["outcome"] == "task_gated"
    assert report["verdicts"]["H2"]["supported"] is False
    assert report["estimates"]["controls"]["T_pick"]["share"]["hi"] < 0.25


def test_mixed_when_share_is_between_the_thresholds() -> None:
    report = analyze(synthetic_export({("T_pick", "concept"): 1.6, ("T_fav", "concept"): 1.6}))
    assert report["outcome"] == "mixed"
    assert report["verdicts"]["H1"]["T_pick"] == "mixed"


def test_mixed_when_pick_and_fav_disagree() -> None:
    report = analyze(synthetic_export({("T_pick", "concept"): 3.5, ("T_fav", "concept"): 0.2}))
    assert report["verdicts"]["H1"] == {"T_pick": "capture", "T_fav": "task_gated", "overall": "mixed"}
    assert report["outcome"] == "mixed"


def test_v1_failure_comes_first_and_blocks_inference() -> None:
    report = analyze(synthetic_export(fact_accuracy=0.5))
    assert report["outcome"] == "V1_fail"
    assert report["gates"]["V1"]["passed"] is False
    assert report["verdicts"] is None


def test_v1_boundary_is_inclusive() -> None:
    assert analyze(synthetic_export(fact_accuracy=0.70))["gates"]["V1"]["passed"] is True
    assert analyze(synthetic_export(fact_accuracy=0.69))["gates"]["V1"]["passed"] is False


def test_v1_failure_outranks_v2_failure() -> None:
    assert analyze(synthetic_export(NO_EFFECT, fact_accuracy=0.5))["outcome"] == "V1_fail"


def test_v2_fails_when_effect_is_absent_and_share_is_reported_undefined() -> None:
    report = analyze(synthetic_export(NO_EFFECT))
    assert report["outcome"] == "V2_fail"
    assert report["gates"]["V2"]["passed"] is False
    assert report["verdicts"] is None
    share = report["estimates"]["controls"]["T_pick"]["share"]
    assert share["lo"] is None and share["hi"] is None
    assert share["undefined_fraction"] > 0.025


def test_v2_fails_when_accuracy_is_at_chance_even_with_positive_l() -> None:
    report = analyze(synthetic_export({("T_inj", "concept"): 1.0}, t_inj_hit=0.08))
    assert report["outcome"] == "V2_fail"
    assert report["gates"]["V2"]["L_inj"]["lo"] > 0
    assert report["gates"]["V2"]["accuracy"]["lo"] <= 0.1


def _control(share: dict[str, float | None]) -> dict[str, Any]:
    good = {"point": 1.0, "lo": 0.5, "hi": 1.5, "p": 0.001}
    return {"L": good, "L_inj_minus_L": good, "share": share, "L_holm_p": 0.003, "gap_holm_p": 0.003}


def test_undefined_share_bounds_give_mixed_never_capture_or_gated() -> None:
    assert _rule_verdict(_control({"lo": None, "hi": None})) == "mixed"
    assert _rule_verdict(_control({"lo": None, "hi": 0.1})) == "task_gated"
    assert _rule_verdict(_control({"lo": 0.6, "hi": None})) == "capture"


def test_holm_can_veto_a_bound() -> None:
    control = _control({"lo": 0.6, "hi": 0.9})
    control["L_holm_p"] = 0.06
    assert _rule_verdict(control) == "mixed"


def test_capture_rule_boundary_is_inclusive() -> None:
    assert _rule_verdict(_control({"lo": 0.5, "hi": 0.9})) == "capture"
    assert _rule_verdict(_control({"lo": 0.1, "hi": 0.25})) == "mixed"


def test_h2_is_independent_of_h1() -> None:
    effects = {**GATED, ("T_odd", "concept"): 3.5}
    report = analyze(synthetic_export(effects))
    assert report["outcome"] == "task_gated"
    assert report["verdicts"]["H2"] == {"T_odd": "capture", "supported": True}


def test_h3_supported_and_gap_reported() -> None:
    report = analyze(synthetic_export())
    assert report["verdicts"]["H3"]["supported"] is True
    assert report["estimates"]["h3"]["concept_minus_impact"]["point"] == pytest.approx(2.0, abs=0.1)


def test_h3_not_supported_when_impact_does_not_exceed_norm() -> None:
    report = analyze(synthetic_export({("T_inj", "random_impact"): 0.3}))
    assert report["verdicts"]["H3"]["supported"] is False


def test_failed_impact_match_is_visible() -> None:
    report = analyze(synthetic_export(match_ok=False))
    assert report["verdicts"]["H3"]["match_ok"] is False
    assert any("impact match" in warning for warning in report["warnings"])


def test_incomplete_run_gets_no_decision() -> None:
    assert analyze(synthetic_export(complete=False))["outcome"] == "incomplete"


def test_pilot_gets_no_decision() -> None:
    report = analyze(synthetic_export(inference_allowed=False))
    assert report["outcome"] == "pilot_not_for_inference"
    assert report["verdicts"] is None


def test_bootstrap_is_reproducible() -> None:
    export = synthetic_export()
    first = analyze(export)["estimates"]["controls"]["T_pick"]["L"]
    second = analyze(export)["estimates"]["controls"]["T_pick"]["L"]
    assert first == second


def test_screening_keeps_the_best_concepts() -> None:
    export = synthetic_export()
    export["meta"]["screen_top_fraction"] = 0.5
    export["screen"] = {f"c{i}": [i, 20] for i in range(N_CONCEPTS)}
    report = analyze(export)
    assert set(report["secondary_screened"]["concepts"]) == {f"c{i}" for i in range(6, 12)}


def test_export_round_trip_is_compact() -> None:
    export = synthetic_export()
    # 400 cells is the small tier's size. Colab shows this comfortably.
    export["cells"] = export["cells"][:400]
    text = encode_export(export)
    assert len(text) < 60_000
    assert decode_export(f"some output\n{text}\nmore output") == export


def test_build_export_means_and_balance() -> None:
    base = {
        "kind": "trial",
        "phase": "test",
        "task": "T_fact",
        "condition": "concept",
        "concept": "c",
        "cluster": "k",
        "clean_hit": False,
        "label_mass": 1.0,
        "sec": 0.1,
    }
    records = [
        {"key": "manifest|0", "kind": "manifest", "clusters_test": {"k": []}, "tier": "t"},
        {**base, "key": "a", "position": 0, "query": 1, "L": 1.0, "hit": True, "fact_correct": False},
        {**base, "key": "b", "position": 1, "query": 2, "L": 3.0, "hit": False, "fact_correct": True},
        {"key": "x", "kind": "clean", "task": "T_fact", "cluster": "k", "query": 1, "answer": 1, "label_mass": 1.0},
        {"key": "y", "kind": "clean", "task": "T_fact", "cluster": "k", "query": 2, "answer": 0, "label_mass": 1.0},
        {"key": "z", "kind": "clean", "task": "T_fact", "cluster": "calibration", "query": 2, "answer": 2, "label_mass": 1.0},
        {**base, "key": "s", "phase": "screen", "task": "T_inj", "position": 0, "query": None, "L": 9.0, "hit": True},
    ]
    export = build_export(records)
    row = dict(zip(export["cell_columns"], export["cells"][0]))
    assert len(export["cells"]) == 1
    assert (row["n"], row["L"], row["hit"], row["fact_correct"]) == (2, 2.0, 0.5, 0.5)
    assert export["clean_fact"] == {"correct": 1, "n": 2}
    assert export["position_counts"]["T_fact|concept"][:2] == [1, 1]
    assert export["screen"] == {"c": [1, 1]}
    assert export["meta"]["complete"] is False
