"""The gate, the decision rules, and everything else that works on numbers alone."""

from __future__ import annotations

import numpy as np
import pytest

from mechinterp.e2.analysis import UNDEFINED, analyze_tables, spearman, spearman_interval, stage2_report
from mechinterp.e2.gate import decide_gate, find_bands, recall_table
from mechinterp.export import CELL_COLUMNS

TASKS = ("T_inj", "T_pick", "T_odd")
ALL_MATCHED = {"F_full": True, "F_j": True, "F_res": True, "F_rand": True, "F_lens": True}


# The gate


def _row(layer: int, gain_lo: float, gain: float | None = None) -> dict:
    return {"layer": layer, "gain_lo": gain_lo, "gain": gain_lo + 0.1 if gain is None else gain}


def test_gate_passes_on_three_consecutive_layers_and_gives_the_middle() -> None:
    table = [_row(0, -0.1), _row(1, 0.01), _row(2, 0.02), _row(3, 0.01), _row(4, -0.1)]
    gate = decide_gate(table, 3)
    assert gate["passed"] is True
    assert gate["band"] == [1, 2, 3]
    assert gate["l_w"] == 2


def test_gate_fails_on_two_consecutive_layers_and_says_why() -> None:
    table = [_row(0, 0.05), _row(1, 0.05), _row(2, -0.01), _row(3, 0.05), _row(4, 0.05)]
    gate = decide_gate(table, 3)
    assert gate["passed"] is False
    assert gate["l_w"] is None
    assert gate["band"] is None
    assert "3 consecutive" in gate["reason"]


def test_a_lower_bound_of_exactly_zero_does_not_clear_it() -> None:
    assert decide_gate([_row(0, 0.0), _row(1, 0.0), _row(2, 0.0)], 3)["passed"] is False


def test_band_with_the_largest_mean_gain_wins_and_an_even_band_takes_the_lower_middle() -> None:
    table = [
        _row(0, 0.01, 0.02),
        _row(1, 0.01, 0.02),
        _row(2, 0.01, 0.02),
        _row(3, -0.1),
        _row(4, 0.05, 0.2),
        _row(5, 0.05, 0.2),
        _row(6, 0.05, 0.2),
        _row(7, 0.05, 0.2),
    ]
    gate = decide_gate(table, 3)
    assert [band["layers"] for band in gate["bands"]] == [[0, 1, 2], [4, 5, 6, 7]]
    assert gate["band"] == [4, 5, 6, 7]
    assert gate["l_w"] == 5


def test_a_missing_layer_breaks_a_band() -> None:
    assert find_bands([_row(0, 0.1), _row(1, 0.1), _row(3, 0.1)], 3) == []


def test_recall_table_pairs_resamples_and_finds_a_real_gain() -> None:
    rng = np.random.default_rng(0)
    logit = (rng.random((200, 4)) < 0.3).astype(float)
    j_lens = logit.copy()
    j_lens[:, 1:3] = np.maximum(j_lens[:, 1:3], (rng.random((200, 2)) < 0.5).astype(float))
    table = recall_table(j_lens, logit, [10, 11, 12, 13], 500, 1)
    assert table[0]["gain"] == 0.0 and table[0]["gain_lo"] == 0.0  # Identical columns: exactly no gain, by pairing.
    assert table[1]["gain_lo"] > 0.0 and table[2]["gain_lo"] > 0.0
    assert table[0]["layer"] == 10


def test_identical_lenses_fail_the_gate_at_any_sample_size() -> None:
    values = (np.random.default_rng(1).random((100, 5)) < 0.5).astype(float)
    assert decide_gate(recall_table(values, values.copy(), list(range(5)), 200, 0), 3)["passed"] is False


def test_recall_table_refuses_shapes_that_do_not_fit() -> None:
    with pytest.raises(ValueError):
        recall_table(np.zeros((3, 2)), np.zeros((3, 3)), [0, 1], 10, 0)
    with pytest.raises(ValueError):
        recall_table(np.zeros((0, 2)), np.zeros((0, 2)), [0, 1], 10, 0)


# Decision rules on synthetic cells


def cells(effects: dict[tuple[str, str], float], seed: int = 0, n_concepts: int = 20, n_clusters: int = 10, noise: float = 0.02) -> tuple[list[str], list[list]]:
    """E1-format rows with independent noise around each (task, family) mean."""
    rng = np.random.default_rng(seed)
    rows = []
    for (task, family), mean in sorted(effects.items()):
        for concept in range(n_concepts):
            for cluster in range(n_clusters):
                rows.append([task, family, f"c{concept}", f"k{cluster}", 10, float(mean + rng.normal(0.0, noise)), 0.5, 0.1, None, 0.99, None])
    return CELL_COLUMNS, rows


def run(effects: dict, matched: dict | None = None, site: int = 3, tasks=("T_inj",), noise: float = 0.02, **options) -> dict:
    return analyze_tables({site: cells(effects, noise=noise)}, {site: matched or ALL_MATCHED}, tasks, 400, 0, 0.05, 0.05, **options)


def task_effects(full: float, j: float, res: float, rand: float, task: str = "T_inj") -> dict:
    return {(task, "F_full"): full, (task, "F_j"): j, (task, "F_res"): res, (task, "F_rand"): rand}


def test_h1_when_the_j_component_carries_the_effect() -> None:
    report = run(task_effects(2.0, 1.8, 0.2, 0.1))
    assert report["classification"]["3|T_inj"] == "H1"


def test_h2_when_the_residual_carries_the_effect() -> None:
    assert run(task_effects(2.0, 0.2, 1.8, 0.1))["classification"]["3|T_inj"] == "H2"


def test_h3_takes_precedence_over_h1() -> None:
    report = run(task_effects(2.0, 1.8, 0.2, 1.9))
    cell = report["cells"]["3|T_inj"]
    assert cell["rules"]["H1"] is True and cell["rules"]["H3"] is True
    assert report["classification"]["3|T_inj"] == "H3"


def test_mixed_when_both_routes_carry_it_and_no_gap_separates_them() -> None:
    # No noise, so the gap is exactly 0 and cannot clear its bound by chance.
    assert run(task_effects(2.0, 1.6, 1.6, 0.1), noise=0.0)["classification"]["3|T_inj"] == "mixed"


def test_mixed_when_a_gap_exists_but_the_ratio_is_too_small() -> None:
    assert run(task_effects(2.0, 0.8, 0.1, 0.1))["classification"]["3|T_inj"] == "mixed"


@pytest.mark.parametrize("full", [0.01, -1.0, 0.0])
def test_a_small_or_negative_full_effect_has_nothing_to_decompose(full: float) -> None:
    report = run(task_effects(full, 1.0, 0.1, 0.1))
    assert report["classification"]["3|T_inj"] == UNDEFINED
    assert report["holm_family_sizes"]["gap_j_minus_res"] == 0


def test_an_effect_above_the_floor_with_a_bound_at_zero_is_also_undefined() -> None:
    columns, rows = cells(task_effects(0.2, 0.2, 0.0, 0.0), noise=4.0)
    report = analyze_tables({3: (columns, rows)}, {3: ALL_MATCHED}, ("T_inj",), 400, 0, 0.05, 0.05)
    cell = report["cells"]["3|T_inj"]
    if cell["L"]["F_full"]["lo"] <= 0.0:
        assert cell["classification"] == UNDEFINED


def test_unmatched_f_rand_leaves_h3_untested_and_says_so() -> None:
    matched = {**ALL_MATCHED, "F_rand": False}
    report = run(task_effects(2.0, 1.8, 0.2, 1.9), matched)
    cell = report["cells"]["3|T_inj"]
    assert cell["rules"]["h3_testable"] is False
    assert cell["classification"] == "H1"
    assert report["holm_family_sizes"]["ratio_F_rand"] == 0


@pytest.mark.parametrize("family", ["F_j", "F_res"])
def test_an_unmatched_route_family_excludes_both_routes(family: str) -> None:
    report = run(task_effects(2.0, 1.8, 0.2, 0.1), {**ALL_MATCHED, family: False})
    cell = report["cells"]["3|T_inj"]
    assert cell["rules"]["routes_testable"] is False
    assert cell["classification"] == "mixed"
    assert report["holm_family_sizes"]["gap_j_minus_res"] == 0


def test_holm_covers_both_sites_and_all_three_tasks() -> None:
    effects = {}
    for task in TASKS:
        effects.update(task_effects(2.0, 1.8, 0.2, 0.1, task))
    report = analyze_tables({3: cells(effects, 1), 9: cells(effects, 2)}, {3: ALL_MATCHED, 9: ALL_MATCHED}, TASKS, 400, 0, 0.05, 0.05)
    assert report["holm_family_sizes"] == {"gap_j_minus_res": 6, "ratio_F_j": 6, "ratio_F_res": 6, "ratio_F_rand": 6}
    assert all(label == "H1" for label in report["classification"].values())


def test_holm_can_demote_a_cell_that_clears_its_bound_alone() -> None:
    effects = {}
    for task in TASKS:
        effects.update(task_effects(2.0, 1.8, 0.2, 0.1, task))
    report = analyze_tables({3: cells(effects, 1), 9: cells(effects, 2)}, {3: ALL_MATCHED, 9: ALL_MATCHED}, TASKS, 40, 0, 0.05, 0.05)
    # With 40 replicates the smallest p is 2/41, which clears 0.05 alone and cannot survive Holm over six cells.
    assert all(label == "H1" for label in report["classification_without_holm"].values())
    assert all(label == "mixed" for label in report["classification"].values())


def test_h4_dissociation_needs_h1_on_t_inj_and_h2_or_h3_on_t_pick() -> None:
    for pick_effects, expected in (
        (task_effects(2.0, 0.1, 0.2, 1.9, "T_pick"), True),
        (task_effects(2.0, 0.2, 1.8, 0.1, "T_pick"), True),
        (task_effects(2.0, 1.8, 0.2, 0.1, "T_pick"), False),
    ):
        effects = {**task_effects(2.0, 1.8, 0.2, 0.1), **pick_effects}
        report = run(effects, tasks=("T_inj", "T_pick"))
        assert report["decision"]["H4"]["3"]["dissociation"] is expected


def test_stop_rule_closes_the_line_on_h3_or_mixed_and_ablation_needs_h1() -> None:
    h3 = run(task_effects(2.0, 1.8, 0.2, 1.9))["decision"]
    assert h3["line_closes_for_qwen3"] is True and h3["ablation_warranted"] is False
    h1 = run(task_effects(2.0, 1.8, 0.2, 0.1))["decision"]
    assert h1["line_closes_for_qwen3"] is False and h1["ablation_warranted"] is True
    h2 = run(task_effects(2.0, 0.2, 1.8, 0.1))["decision"]
    assert h2["line_closes_for_qwen3"] is False and h2["ablation_warranted"] is False


def test_f_lens_enters_no_rule_and_is_reported_on_its_own_concepts() -> None:
    effects = {**task_effects(2.0, 1.8, 0.2, 0.1), ("T_inj", "F_lens"): 1.0}
    columns, rows = cells(effects)
    subset = [f"c{index}" for index in range(5)]
    rows = [row for row in rows if row[1] != "F_lens" or row[2] in subset]
    report = analyze_tables({3: (columns, rows)}, {3: ALL_MATCHED}, ("T_inj",), 200, 0, 0.05, 0.05, lens_concepts=subset)
    lens = report["f_lens"]["3"]
    assert lens["n_concepts"] == 5
    assert lens["by_task"]["T_inj"]["ratio"]["point"] == pytest.approx(0.5, abs=0.02)
    assert report["classification"]["3|T_inj"] == "H1"


def test_a_missing_full_family_is_an_error_rather_than_a_verdict() -> None:
    with pytest.raises(KeyError):
        run({("T_inj", "F_j"): 1.0})


def test_nan_in_a_cell_stops_the_analysis() -> None:
    columns, rows = cells(task_effects(2.0, 1.8, 0.2, 0.1))
    rows[0][5] = float("nan")
    with pytest.raises(ValueError):
        analyze_tables({3: (columns, rows)}, {3: ALL_MATCHED}, ("T_inj",), 50, 0, 0.05, 0.05)


# Stage 2


def test_spearman_is_rank_based_and_shares_ties() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman(x, x**3) == pytest.approx(1.0)
    assert spearman(x, -x) == pytest.approx(-1.0)
    assert spearman(np.array([1.0, 1.0, 2.0, 2.0]), np.array([1.0, 1.0, 2.0, 2.0])) == pytest.approx(1.0)
    assert np.isnan(spearman(np.ones(4), x))


def test_spearman_interval_covers_a_strong_relation() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=40)
    report = spearman_interval(x, x + rng.normal(scale=0.2, size=40), 300, 0)
    assert report["lo"] > 0.5 and report["rho"] <= report["hi"] and report["n_concepts"] == 40


def _decomposition() -> dict:
    concept = lambda share: {"j_share": share, "atoms": [], "top_lens_tokens": [{"token": 1, "text": " x"}]}
    by_concept = {f"c{index}": concept(0.01 * (index + 1)) for index in range(6)}
    return {"k": 16, "l_s": 3, "l_w": 7, "by_site": {"3": by_concept, "7": by_concept}}


def test_stage2_correlates_at_l_s_only_and_is_marked_descriptive() -> None:
    concepts = [f"c{index}" for index in range(6)]
    e1 = {"L_inj": {name: float(index) for index, name in enumerate(concepts)}, "L_pick": {name: -float(index) for index, name in enumerate(concepts)}}
    report = stage2_report(e1, _decomposition(), concepts, 100, 0)
    assert report["correlations_at_l_s"]["J_share_vs_L_inj"]["rho"] == pytest.approx(1.0)
    assert report["correlations_at_l_s"]["J_share_vs_L_pick"]["rho"] == pytest.approx(-1.0)
    assert set(report["by_site"]) == {"3", "7"}
    assert "Descriptive" in report["note"]


def test_stage2_refuses_concepts_e1_never_measured() -> None:
    with pytest.raises(ValueError, match="no L"):
        stage2_report({"L_inj": {}, "L_pick": {}}, _decomposition(), ["c0"], 10, 0)
