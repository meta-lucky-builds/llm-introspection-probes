"""Design logic and the shipped data and configs. No model involved."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import tiny_config_dict, write_tiny_data

from mechinterp.config import Config
from mechinterp.design import (
    PHASE_GRID,
    PHASE_SCREEN,
    PHASE_TEST,
    Trial,
    assert_pilot_isolated,
    assert_positions_balanced_within_concepts,
    assert_splits_disjoint,
    assert_trial_respects_split,
    cell_positions,
    fact_query,
    load_data,
    plan_cells,
    plan_clean_keys,
    plan_grid_trials,
    plan_prompt_counts,
    plan_screen_trials,
    plan_test_trials,
    prepare_stimuli,
    select_layer_alpha,
    validate_data,
)

EXPERIMENT = Path(__file__).resolve().parents[1] / "experiments" / "e1_localization_capture"
CONFIG_PATHS = sorted((EXPERIMENT / "configs").glob("*.json"))


@pytest.fixture()
def tiny(tmp_path):
    config = Config.from_dict(tiny_config_dict(write_tiny_data(tmp_path / "data")), tmp_path)
    raw = load_data(config.data_dir)
    stimuli = prepare_stimuli(raw, config, lambda word: word not in {"two-part", "missingword", "alsomissing"})
    return config, raw, stimuli


def test_all_positions_are_balanced_within_clusters() -> None:
    for n_concepts, n_clusters, per_cell in [(10, 5, 1), (5, 2, 2), (2, 1, 5), (40, 10, 10), (20, 5, 5)]:
        for cluster in range(n_clusters):
            counts = Counter(
                position
                for concept in range(n_concepts)
                for position in cell_positions(concept, cluster, n_clusters, per_cell)
            )
            assert len(counts) == 10 and len(set(counts.values())) == 1


def test_full_cell_covers_all_ten_positions() -> None:
    assert sorted(cell_positions(3, 2, 5, 10)) == list(range(10))


def _position_counts(cells) -> tuple[dict[int, Counter], dict[int, Counter]]:
    by_concept: dict[int, Counter] = {}
    by_cluster: dict[int, Counter] = {}
    for concept, cluster, positions in cells:
        by_concept.setdefault(concept, Counter()).update(positions)
        by_cluster.setdefault(cluster, Counter()).update(positions)
    return by_concept, by_cluster


def test_cyclic_design_balances_positions_within_concepts_and_clusters() -> None:
    cells = plan_cells(10, 10, 2, 5)
    assert len(cells) == 50
    assert len({(concept, cluster) for concept, cluster, _ in cells}) == 50
    by_concept, by_cluster = _position_counts(cells)
    assert len(by_concept) == 10 and len(by_cluster) == 10
    for counter in [*by_concept.values(), *by_cluster.values()]:
        assert sorted(counter) == list(range(10)) and set(counter.values()) == {1}
    for _, _, positions in cells:
        assert {position % 2 for position in positions} == {0, 1}


def test_the_review_design_with_one_parity_per_concept_is_rejected() -> None:
    """10 concepts, 5 clusters and 1 position per cell balances clusters but not concepts."""
    cells = plan_cells(10, 5, 1)
    by_concept, _ = _position_counts(cells)
    assert {position % 2 for position in by_concept[0]} == {0}
    trials = [
        Trial(PHASE_TEST, "T_inj", "concept", f"c{concept}", f"k{cluster}", position, None, 0, 1.0)
        for concept, cluster, positions in cells
        for position in positions
    ]
    with pytest.raises(ValueError):
        assert_positions_balanced_within_concepts(trials)


def test_inference_tier_plans_refuse_a_design_unbalanced_within_concepts(tiny) -> None:
    config, _, stimuli = tiny
    one_cluster = dict(list(stimuli.clusters_test.items())[:1])
    narrow = replace(stimuli, clusters_test=one_cluster)
    with pytest.raises(ValueError):
        plan_test_trials(narrow, config, 0, 1.0, 2.0)


def test_fact_query_is_never_the_injected_position_and_is_balanced() -> None:
    for cluster in range(12):
        for step in range(9):
            queries = [fact_query(position, cluster, step) for position in range(10)]
            assert all(query != position for position, query in enumerate(queries))
            assert sorted(queries) == list(range(10))
    # Distinct steps give distinct queries for one injected position.
    assert len({fact_query(4, 0, step) for step in range(9)}) == 9


def test_unbalanceable_design_is_rejected(tiny) -> None:
    config, _, stimuli = tiny
    design = replace(config.design, positions_per_cell_calibration=3)
    with pytest.raises(ValueError):
        plan_grid_trials(stimuli, replace(config, design=design))


def test_plans_stay_in_their_splits(tiny) -> None:
    config, _, stimuli = tiny
    for trial in plan_grid_trials(stimuli, config):
        assert trial.concept in stimuli.concepts_calibration and trial.cluster in stimuli.clusters_calibration
        assert trial.task == "T_inj"
        assert_trial_respects_split(trial, stimuli)
    for trial in plan_screen_trials(stimuli, config, 0, 1.0):
        assert trial.concept in stimuli.concepts_test and trial.cluster in stimuli.clusters_calibration
        assert_trial_respects_split(trial, stimuli)
    for trial in plan_test_trials(stimuli, config, 0, 1.0, 2.0):
        assert trial.concept in stimuli.concepts_test and trial.cluster in stimuli.clusters_test
        assert_trial_respects_split(trial, stimuli)


def test_trial_outside_its_split_raises(tiny) -> None:
    _, _, stimuli = tiny
    test_concept = stimuli.concepts_test[0]
    test_cluster = next(iter(stimuli.clusters_test))
    calibration_concept = stimuli.concepts_calibration[0]
    calibration_cluster = next(iter(stimuli.clusters_calibration))
    bad = [
        (PHASE_GRID, test_concept, calibration_cluster),
        (PHASE_GRID, calibration_concept, test_cluster),
        (PHASE_SCREEN, test_concept, test_cluster),
        (PHASE_TEST, calibration_concept, test_cluster),
        (PHASE_TEST, test_concept, calibration_cluster),
    ]
    for phase, concept, cluster in bad:
        with pytest.raises(AssertionError):
            assert_trial_respects_split(Trial(phase, "T_inj", "concept", concept, cluster, 0, None, 0, 1.0), stimuli)


def test_overlapping_splits_raise(tiny) -> None:
    _, raw, stimuli = tiny
    with pytest.raises(ValueError):
        assert_splits_disjoint(replace(stimuli, concepts_test=(stimuli.concepts_calibration[0],)))
    with pytest.raises(ValueError):
        validate_data(replace(raw, concepts_test=(raw.concepts_calibration[0],)))


def test_filter_records_drops(tiny) -> None:
    _, _, stimuli = tiny
    assert "two-part" in stimuli.dropped_words.get("calone", []) or "calone" not in stimuli.clusters_calibration
    assert "testshort" in stimuli.dropped_clusters
    assert "testshort" not in stimuli.clusters_test
    for words in stimuli.all_clusters.values():
        assert len(words) == 10 and "missingword" not in words


def test_too_few_clusters_raises(tiny) -> None:
    config, raw, _ = tiny
    with pytest.raises(ValueError):
        prepare_stimuli(raw, config, lambda word: False)


def test_clean_keys_cover_all_fact_queries() -> None:
    keys = plan_clean_keys(["T_pick", "T_fact"], ["k"])
    assert ("T_pick", "k", None) in keys
    assert {key[2] for key in keys if key[0] == "T_fact"} == set(range(10))


def _record(concept: str, cluster: str, layer: int, alpha: float, hit: bool) -> dict:
    return {
        "key": f"{concept}|{cluster}|{layer}|{alpha}|{hit}",
        "phase": PHASE_GRID,
        "task": "T_inj",
        "condition": "concept",
        "concept": concept,
        "cluster": cluster,
        "layer": layer,
        "alpha": alpha,
        "hit": hit,
    }


def test_selection_maximizes_t_inj_accuracy() -> None:
    records = [
        _record("c", "k", 0, 1.0, False),
        _record("c", "k", 0, 2.0, True),
        _record("c", "k", 3, 1.0, False),
        _record("c", "k", 3, 2.0, False),
    ]
    selection = select_layer_alpha(records, ["c"], ["k"])
    assert (selection.layer, selection.alpha, selection.accuracy) == (0, 2.0, 1.0)


def test_selection_ties_prefer_small_alpha_then_early_layer() -> None:
    records = [_record("c", "k", layer, alpha, True) for layer in (3, 0) for alpha in (2.0, 1.0)]
    selection = select_layer_alpha(records, ["c"], ["k"])
    assert (selection.layer, selection.alpha) == (0, 1.0)


@pytest.mark.parametrize(
    "bad",
    [
        {"concept": "test_concept"},
        {"cluster": "test_cluster"},
        {"task": "T_pick"},
        {"condition": "random_norm"},
        {"phase": PHASE_TEST},
    ],
)
def test_selection_rejects_anything_but_calibration_t_inj(bad) -> None:
    with pytest.raises(AssertionError):
        select_layer_alpha([{**_record("c", "k", 0, 1.0, True), **bad}], ["c"], ["k"])


def test_shipped_data_is_valid() -> None:
    raw = load_data(EXPERIMENT / "data")
    assert len(raw.baseline_words) == 50
    assert len(raw.concepts_calibration) >= 20 and len(raw.concepts_test) >= 40
    assert len(raw.clusters_calibration) >= 5 and len(raw.clusters_test) >= 10


def test_config_files_exist() -> None:
    names = {path.stem for path in CONFIG_PATHS}
    assert {"pilot", "small", "large"} <= names


@pytest.mark.parametrize("path", CONFIG_PATHS, ids=lambda path: path.stem)
def test_shipped_configs_plan_balanced_designs(path) -> None:
    config = Config.from_file(path)
    raw = load_data(config.data_dir)
    stimuli = prepare_stimuli(raw, config, lambda word: True)
    assert_splits_disjoint(stimuli)
    counts = plan_prompt_counts(stimuli, config)
    trials = plan_test_trials(stimuli, config, 0, 1.0, 2.0)
    assert counts["test"] == len(trials)
    assert len({trial.key for trial in trials}) == len(trials)
    by_condition: dict[tuple[str, str], Counter] = {}
    for trial in trials:
        by_condition.setdefault((trial.task, trial.condition), Counter())[trial.position] += 1
    for counter in by_condition.values():
        assert len(counter) == 10 and len(set(counter.values())) == 1
    if config.inference_allowed:
        assert_positions_balanced_within_concepts(trials)


@pytest.mark.parametrize("path", CONFIG_PATHS, ids=lambda path: path.stem)
def test_shipped_configs_pin_a_full_commit(path) -> None:
    assert re.fullmatch(r"[0-9a-f]{40}", Config.from_file(path).model_revision)


def test_small_tier_design() -> None:
    config = Config.from_file(EXPERIMENT / "configs" / "small.json")
    stimuli = prepare_stimuli(load_data(config.data_dir), config, lambda word: True)
    assert config.design.positions_per_cell_test == 2
    assert len(stimuli.clusters_test) >= 10
    assert plan_prompt_counts(stimuli, config) == {
        "vector_prompts": 65,
        "clean_calibration": 2,
        "calibration_grid": 120,
        "impact_search_max": 120,
        "screen": 0,
        "clean_test": 140,
        "test": 700,
    }
    trials = plan_test_trials(stimuli, config, 0, 1.0, 2.0)
    for concept in stimuli.concepts_test:
        seen = {trial.position for trial in trials if trial.concept == concept and trial.task == "T_inj"}
        assert seen == set(range(10))


def test_tier_sizes_fit_their_time_budgets() -> None:
    """Unmeasured estimates: about 6 s per prompt for 0.6B and 10 to 12 s per prompt for 1.7B on 2 vCPUs."""
    expected_ceiling = {"pilot": 200, "small": 1100}
    for name, ceiling in expected_ceiling.items():
        config = Config.from_file(EXPERIMENT / "configs" / f"{name}.json")
        stimuli = prepare_stimuli(load_data(config.data_dir), config, lambda word: True)
        counts = plan_prompt_counts(stimuli, config)
        assert sum(counts.values()) - counts["vector_prompts"] <= ceiling, (name, counts)


def test_pilot_is_excluded_from_inference_and_uses_its_own_stimuli() -> None:
    config = Config.from_file(EXPERIMENT / "configs" / "pilot.json")
    assert config.inference_allowed is False and config.data_pool == "pilot"
    raw = load_data(config.data_dir)
    stimuli = prepare_stimuli(raw, config, lambda word: True)
    assert_pilot_isolated(raw, stimuli)
    assert not {*stimuli.concepts_calibration, *stimuli.concepts_test} & {*raw.concepts_calibration, *raw.concepts_test}
    assert not set(stimuli.all_clusters) & {*raw.clusters_calibration, *raw.clusters_test}
    plan_test_trials(stimuli, config, 0, 1.0, 2.0)


def test_overlap_between_pilot_and_real_stimuli_raises() -> None:
    config = Config.from_file(EXPERIMENT / "configs" / "pilot.json")
    raw = load_data(config.data_dir)
    stimuli = prepare_stimuli(raw, config, lambda word: True)
    with pytest.raises(AssertionError):
        assert_pilot_isolated(raw, replace(stimuli, concepts_test=(raw.concepts_test[0],)))
    test_cluster = next(iter(raw.clusters_test))
    with pytest.raises(AssertionError):
        assert_pilot_isolated(raw, replace(stimuli, clusters_test={test_cluster: raw.clusters_test[test_cluster]}))
    with pytest.raises(ValueError):
        validate_data(replace(raw, pilot_concepts_test=(raw.concepts_test[0],)))


def test_pool_must_match_inference_status(tmp_path) -> None:
    raw = tiny_config_dict(write_tiny_data(tmp_path / "data"))
    with pytest.raises(ValueError):
        Config.from_dict({**raw, "inference_allowed": False}, tmp_path)
    with pytest.raises(ValueError):
        Config.from_dict({**raw, "data_pool": "pilot"}, tmp_path)
