"""Gates, stop rules, verdicts, the bootstrap, probes and baselines, on numbers and synthetic arrays."""

from __future__ import annotations

import numpy as np
import pytest

from mechinterp.d1.baselines import TfidfClassifier, transcript_text
from mechinterp.d1.config import ProbeTraining
from mechinterp.d1.families import line_of_work
from mechinterp.d1.gates import (
    G2_CAPABILITY,
    G2_FINDING,
    G2_NOT_SPECIFIC,
    G2_SHAM_MOVED,
    INCONCLUSIVE,
    INDEPENDENT,
    PROBE_UNUSABLE,
    SAME,
    STRONG,
    VERBAL,
    describe_verdict,
    g2_failure_finding,
    gate_g1,
    gate_g2,
    independence_strength,
    next_after_g1,
    next_after_g2,
    primary_verdict,
    self_report_can_pass_g1,
    weakly_controlled,
)
from mechinterp.d1.probes import Ensemble, contrast, fit_ensemble, random_direction_probe, rms_scale
from mechinterp.d1.sizing import allocate_per_class
from mechinterp.d1.stats import Resampler, auroc, interval_widening, plan_sample_size, recall_and_fpr, threshold_at_fpr, weighted_auroc
from mechinterp.stats import NonFiniteError, approx_percentile_coverage


def test_g1_needs_both_lower_bounds_at_their_bars() -> None:
    assert gate_g1(0.80, 0.70)["passed"]
    assert not gate_g1(0.7999, 0.95)["passed"]
    assert not gate_g1(0.95, 0.6999)["passed"]


def test_gates_refuse_nan_rather_than_failing_quietly() -> None:
    with pytest.raises(ValueError):
        gate_g1(float("nan"), 0.9)
    with pytest.raises(ValueError):
        gate_g2(float("nan"), 0.5, 0.8, 0.8, 0.9, 0.9, 0.9, 0.8, 0.8)
    with pytest.raises(ValueError):
        primary_verdict(float("nan"), 0.1, 0.9, 0.9)
    with pytest.raises(ValueError):
        primary_verdict(-0.05, 0.1, float("nan"), 0.9)
    with pytest.raises(ValueError):
        self_report_can_pass_g1(float("nan"))


def test_self_report_alone_can_decide_g1_and_agrees_with_the_full_gate() -> None:
    for lower in (0.69, 0.70, 0.71):
        assert self_report_can_pass_g1(lower) == gate_g1(1.0, lower)["self_report_ok"]
        if not self_report_can_pass_g1(lower):
            assert not gate_g1(1.0, lower)["passed"], "no probe score can rescue a rung whose self-report failed"


def test_ladder_stops_at_the_first_pass_and_ends_after_the_last_rung() -> None:
    assert next_after_g1(True, 0, 4) == "stage2"
    assert next_after_g1(False, 0, 4) == "next_rung"
    assert next_after_g1(False, 2, 4) == "next_rung"
    assert next_after_g1(False, 3, 4) == "stop"
    assert next_after_g1(True, 3, 4) == "stage2"


GOOD = dict(
    dd_mean_p_yes=0.90, dd_balanced_accuracy=0.55, sham_self_report=0.78, unmodified_self_report=0.80, dd_accuracy=0.86, sham_accuracy=0.88,
    unmodified_accuracy=0.90, dd_specificity=0.72, unmodified_specificity=0.80,
)  # fmt: skip


def test_g2_passes_at_its_bars() -> None:
    assert gate_g2(**GOOD)["passed"]


@pytest.mark.parametrize(
    ("change", "check"),
    [
        ({"dd_mean_p_yes": 0.89}, "dd_says_yes"),
        ({"dd_balanced_accuracy": 0.56}, "dd_answer_silent"),
        ({"dd_specificity": 0.69}, "dd_specific"),
        ({"dd_specificity": 0.91}, "dd_specific"),
        ({"sham_self_report": 0.74}, "sham_self_report_kept"),
        ({"sham_self_report": 0.86}, "sham_self_report_kept"),
        ({"dd_accuracy": 0.84}, "dd_capability_kept"),
        ({"sham_accuracy": 0.84}, "sham_capability_kept"),
    ],
)
def test_each_g2_condition_can_fail_alone(change: dict, check: str) -> None:
    result = gate_g2(**{**GOOD, **change})
    assert not result["passed"]
    assert [name for name, ok in result["checks"].items() if not ok] == [check]


def test_g2_has_no_bar_on_the_logit_auroc() -> None:
    """A model that always says Yes can still rank well on its logit. That is the residual self-report, measured and reported."""
    assert "dd_self_report_upper" not in gate_g2(**GOOD)["numbers"]
    assert not any("self_report_silenced" in name for name in gate_g2(**GOOD)["checks"])


def test_capability_is_one_sided_so_an_improvement_passes() -> None:
    result = gate_g2(**{**GOOD, "dd_accuracy": 0.99, "sham_accuracy": 1.0})
    assert result["checks"]["dd_capability_kept"] and result["checks"]["sham_capability_kept"]


def test_a_general_yes_bias_fails_on_specificity_when_the_unmodified_model_is_good_at_it() -> None:
    """Always Yes scores 0.50 on a balanced set, so it fails whenever the unmodified model scores above 0.60."""
    result = gate_g2(**{**GOOD, "dd_specificity": 0.50, "unmodified_specificity": 0.75})
    assert result["checks"]["dd_specific"] is False


@pytest.mark.parametrize(
    ("failed", "finding"),
    [
        (["dd_says_yes"], G2_FINDING),
        (["dd_answer_silent"], G2_FINDING),
        (["dd_answer_silent", "dd_specific"], G2_FINDING),
        (["dd_specific"], G2_NOT_SPECIFIC),
        (["sham_self_report_kept"], G2_SHAM_MOVED),
        (["dd_capability_kept"], G2_CAPABILITY),
    ],
)
def test_a_failed_g2_gets_the_finding_for_what_failed(failed: list, finding: str) -> None:
    names = ("dd_says_yes", "dd_answer_silent", "dd_specific", "sham_self_report_kept", "dd_capability_kept", "sham_capability_kept")
    assert g2_failure_finding({name: name not in failed for name in names}) == finding


def test_stop_loss_allows_one_retry_only() -> None:
    assert next_after_g2(True, 0, "one_seed") == "continue"
    assert next_after_g2(False, 0, "one_seed") == "retry"
    assert next_after_g2(False, 1, "one_seed") == "stop"
    assert next_after_g2(True, 0, "all_seeds") == "finish"
    assert next_after_g2(False, 0, "all_seeds") == "retry"
    assert next_after_g2(False, 1, "all_seeds") == "stop"


@pytest.mark.parametrize(
    ("lower", "upper", "verdict"),
    [
        (-0.05, 0.02, INDEPENDENT),
        (-0.0999, 0.5, INDEPENDENT),
        (-0.30, -0.15, SAME),
        (-0.30, -0.1001, SAME),
        (-0.15, -0.05, INCONCLUSIVE),
        (-0.10, -0.05, INCONCLUSIVE),
        (-0.20, -0.10, INCONCLUSIVE),
        (-0.40, 0.20, INCONCLUSIVE),
    ],
)
def test_verdict_regions_with_a_live_probe(lower: float, upper: float, verdict: str) -> None:
    assert primary_verdict(lower, upper, 0.9, 0.9) == verdict


@pytest.mark.parametrize(
    ("dd_lower", "sham_lower", "verdict"),
    [
        (0.70, 0.70, INDEPENDENT),
        (0.6999, 0.70, INCONCLUSIVE),  # DD alone under the floor: a D near 0 does not establish independence.
        (0.70, 0.6999, INDEPENDENT),  # The change log's floor is on DD. SHAM alone being low warns, and is reported.
        (0.55, 0.58, PROBE_UNUSABLE),
        (0.6999, 0.6999, PROBE_UNUSABLE),
    ],
)
def test_probe_floor_with_d_near_zero(dd_lower: float, sham_lower: float, verdict: str) -> None:
    assert primary_verdict(-0.03, 0.02, dd_lower, sham_lower) == verdict


def test_both_arms_under_the_floor_is_unusable_whatever_d_says() -> None:
    assert primary_verdict(-0.4, -0.2, 0.3, 0.6) == PROBE_UNUSABLE
    assert primary_verdict(-0.4, -0.2, 0.3, 0.9) == SAME, "a probe alive in SHAM and dead in DD is the collapse the design looks for"


def test_strength_needs_independence_and_a_lower_bound_above_zero() -> None:
    assert independence_strength(INDEPENDENT, 0.001) == STRONG
    assert independence_strength(INDEPENDENT, 0.0) == VERBAL
    assert independence_strength(INDEPENDENT, -0.2) == VERBAL
    for other in (SAME, INCONCLUSIVE, PROBE_UNUSABLE):
        assert independence_strength(other, 0.5) is None


def test_a_verbal_result_is_not_described_as_strong() -> None:
    assert "Strength: verbal only" in describe_verdict(INDEPENDENT, VERBAL)
    assert "Strength: strong" not in describe_verdict(INDEPENDENT, VERBAL)
    assert "Strength: strong" in describe_verdict(INDEPENDENT, STRONG)
    assert "Strength" not in describe_verdict(SAME)
    assert "unusable" in describe_verdict(PROBE_UNUSABLE)


def test_weak_control_is_a_flag_at_half_of_dds_change() -> None:
    assert weakly_controlled(0.2, 0.5) is True
    assert weakly_controlled(0.25, 0.5) is False
    assert weakly_controlled(0.9, 0.5) is False
    assert weakly_controlled(-0.2, 0.5) is True, "size of change, whichever way"
    assert weakly_controlled(float("nan"), 0.5) is True, "a change that could not be measured is not control"


def test_lines_of_work_come_from_the_related_families() -> None:
    assert line_of_work("lies") == line_of_work("mask") == line_of_work("sandbagging")
    assert line_of_work("harm") == line_of_work("liarsbench")
    assert line_of_work("sycophancy") == "sycophancy"
    test_families = ["harm", "liarsbench", "lies", "mask", "sandbagging", "sycophancy", "truthfulqa"]
    assert len({line_of_work(name) for name in test_families}) == 4


def test_the_top_level_bootstrap_unit_is_the_line_of_work() -> None:
    """Related families must rise and fall as one line. If they were drawn alone, a replicate could hold mask without lies."""
    families = [name for name in ("harm", "liarsbench", "lies", "mask", "sandbagging", "sycophancy", "truthfulqa") for _ in range(8)]
    groups = [f"{name}-{index % 4}" for index, name in enumerate(families)]
    resampler = Resampler.build(families, groups, 400, 0)
    assert len(resampler.line_names) == 4
    draws = resampler.family_draws
    assert draws.shape == (400, 7)
    position = {name: index for index, name in enumerate(resampler.family_names)}
    line = draws[:, [position["lies"], position["mask"], position["sandbagging"]]]
    # A line's total weight is its size times how often the line was drawn, so a replicate holds all of it or none of it.
    assert set(np.unique(line.sum(axis=1) / 3.0).tolist()) <= {0.0, 1.0, 2.0, 3.0, 4.0}
    assert (line.sum(axis=1) == 0).any() and (line.sum(axis=1) > 0).any()
    both = draws[:, [position["harm"], position["liarsbench"]]]
    assert set(np.unique(both.sum(axis=1) / 2.0).tolist()) <= {0.0, 1.0, 2.0, 3.0, 4.0}
    # Single-family lines carry the line's draw unchanged, and the four draws add up to four.
    total = line.sum(axis=1) / 3.0 + both.sum(axis=1) / 2.0 + draws[:, position["sycophancy"]] + draws[:, position["truthfulqa"]]
    assert (total == 4.0).all()
    # Weights by row are the same draws, with groups inside.
    names = np.array(families)
    assert (resampler.family_only[:, names == "lies"][:, 0] == draws[:, position["lies"]]).all()


def test_line_level_intervals_are_wider_than_family_level_ones_would_be() -> None:
    """Three related families that move together are one unit. Drawing lines gives the spread that deserves."""
    rng = np.random.default_rng(0)
    names = ("harm", "liarsbench", "lies", "mask", "sandbagging", "sycophancy", "truthfulqa")
    # One line of work reads much worse than the rest.
    shifts = {"lies": 0.0, "mask": 0.0, "sandbagging": 0.0}
    scores, labels, families, groups = [], [], [], []
    for name in names:
        for row in range(60):
            y = row % 2
            scores.append(rng.normal(shifts.get(name, 3.0) * y, 1.0))
            labels.append(y)
            families.append(name)
            groups.append(f"{name}-{row // 2}")
    by_line = Resampler.build(families, groups, 600, 1)
    # The same data with each family treated as its own line, by giving them names no line of work holds.
    alone = [f"{name}-alone" for name in families]
    alone_groups = [f"{name}-alone" for name in groups]
    by_family = Resampler.build(alone, alone_groups, 600, 1)
    wide = np.nanpercentile(weighted_auroc(np.array(scores), np.array(labels), by_line.two_way), [2.5, 97.5])
    narrow = np.nanpercentile(weighted_auroc(np.array(scores), np.array(labels), by_family.two_way), [2.5, 97.5])
    assert wide[1] - wide[0] > narrow[1] - narrow[0]


def test_coverage_and_widening_at_this_many_lines() -> None:
    assert approx_percentile_coverage(4) == pytest.approx(0.81, abs=0.02)
    assert interval_widening(4) == pytest.approx(1.87, abs=0.01)
    assert interval_widening(7) == pytest.approx(1.35, abs=0.01)
    assert interval_widening(1) == 1.0


def test_the_test_set_adds_up_to_the_fixed_size() -> None:
    available = {name: 500 for name in ("harm", "liarsbench", "lies", "mask", "sandbagging", "sycophancy", "truthfulqa")}
    allocation = allocate_per_class(800, available)
    assert sum(allocation.values()) == 400
    assert max(allocation.values()) - min(allocation.values()) <= 1


def test_what_a_small_family_cannot_use_goes_to_the_others() -> None:
    allocation = allocate_per_class(800, {"small": 10, "a": 500, "b": 500, "c": 500})
    assert allocation["small"] == 10
    assert sum(allocation.values()) == 400
    assert max(allocation["a"], allocation["b"], allocation["c"]) - min(allocation["a"], allocation["b"], allocation["c"]) <= 1


def test_a_test_set_cannot_exceed_what_exists() -> None:
    allocation = allocate_per_class(800, {"a": 5, "b": 0})
    assert allocation == {"a": 5, "b": 0}


def test_auroc_basics_and_ties() -> None:
    labels = np.array([0, 0, 1, 1])
    assert auroc(np.array([1.0, 2.0, 3.0, 4.0]), labels) == 1.0
    assert auroc(np.array([4.0, 3.0, 2.0, 1.0]), labels) == 0.0
    assert auroc(np.zeros(4), labels) == 0.5
    assert auroc(np.array([1.0, 2.0, 2.0, 3.0]), labels) == pytest.approx(0.875)


def test_auroc_matches_the_rank_count_on_random_data() -> None:
    rng = np.random.default_rng(0)
    scores = rng.integers(0, 12, 300).astype(float)
    labels = rng.integers(0, 2, 300)
    positives, negatives = scores[labels == 1], scores[labels == 0]
    wins = (positives[:, None] > negatives[None, :]).mean() + 0.5 * (positives[:, None] == negatives[None, :]).mean()
    assert auroc(scores, labels) == pytest.approx(wins)


def test_weighted_auroc_equals_repeating_rows() -> None:
    rng = np.random.default_rng(1)
    scores = rng.normal(size=60)
    labels = rng.integers(0, 2, 60)
    counts = rng.integers(0, 4, 60)
    repeated = weighted_auroc(np.repeat(scores, counts), np.repeat(labels, counts), np.ones((1, int(counts.sum()))))[0]
    assert weighted_auroc(scores, labels, counts[None].astype(float))[0] == pytest.approx(repeated)


def test_auroc_refuses_nan_and_single_class_is_nan() -> None:
    with pytest.raises(NonFiniteError):
        auroc(np.array([np.nan, 1.0]), np.array([0, 1]))
    assert np.isnan(auroc(np.array([1.0, 2.0]), np.array([1, 1])))


def bootstrap_data(n_families: int = 7, per_family: int = 40, seed: int = 0, shift: float = 1.0) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    rng = np.random.default_rng(seed)
    scores, labels, families, groups = [], [], [], []
    for family in range(n_families):
        offset = rng.normal(0, 0.5)
        for row in range(per_family):
            y = row % 2
            scores.append(rng.normal(shift * y + offset, 1.0))
            labels.append(y)
            families.append(f"f{family}")
            groups.append(f"f{family}g{row // 2}")
    return np.array(scores), np.array(labels), families, groups


def test_resampler_keeps_groups_whole_and_families_nested() -> None:
    _, _, families, groups = bootstrap_data()
    resampler = Resampler.build(families, groups, 50, 0)
    group_names = np.array(groups)
    for row in resampler.within[:5]:
        for name in set(groups):
            assert len(set(row[group_names == name].tolist())) == 1
    family_names = np.array(families)
    for row, within in zip(resampler.family_only[:5], resampler.within[:5]):
        assert sum(float(row[family_names == name][0]) for name in set(families)) == 7
        for name in set(families):
            assert within[family_names == name].sum() == pytest.approx(40.0)


def test_paired_difference_of_identical_arms_has_zero_width() -> None:
    scores, labels, families, groups = bootstrap_data()
    resampler = Resampler.build(families, groups, 200, 0)
    difference = weighted_auroc(scores, labels, resampler.two_way) - weighted_auroc(scores.copy(), labels, resampler.two_way)
    assert np.nanmax(np.abs(difference)) == 0.0


def test_paired_bootstrap_is_narrower_than_treating_arms_as_independent() -> None:
    scores, labels, families, groups = bootstrap_data(seed=2)
    rng = np.random.default_rng(3)
    other = scores + rng.normal(0, 0.1, len(scores))
    resampler = Resampler.build(families, groups, 400, 0)
    a = weighted_auroc(scores, labels, resampler.two_way)
    b = weighted_auroc(other, labels, resampler.two_way)
    assert np.nanstd(a - b) < 0.5 * np.nanstd(a)


def test_recall_threshold_fits_on_negatives_only() -> None:
    negatives = np.arange(1000.0)
    threshold = threshold_at_fpr(negatives, 0.01)
    assert 985.0 <= threshold <= 995.0
    assert np.mean(negatives > threshold) == pytest.approx(0.01, abs=0.002)
    scores = np.r_[negatives, np.full(10, 5000.0), np.zeros(10)]
    labels = np.r_[np.zeros(1000), np.ones(20)]
    recall, fpr = recall_and_fpr(scores, labels, threshold)
    assert recall[0] == pytest.approx(0.5)
    assert fpr[0] == pytest.approx(0.01, abs=0.002)
    with pytest.raises(ValueError):
        threshold_at_fpr(np.zeros(0))


def test_expected_half_width_is_reported_at_the_fixed_size_and_more_rows_cannot_remove_the_unit_term() -> None:
    scores, labels, families, groups = bootstrap_data(per_family=100, seed=4)
    small = plan_sample_size(scores, labels, families, groups, {f"f{index}": 20 for index in range(7)}, 200, 0)
    large = plan_sample_size(scores, labels, families, groups, {f"f{index}": 2000 for index in range(7)}, 200, 0)
    assert small["transcripts_per_arm"] == 280 and large["transcripts_per_arm"] == 28000
    assert small["n_in_g1_set"] == 700
    assert large["expected_half_width_auroc"] < small["expected_half_width_auroc"]
    # With rows without end, what is left is the term from families: the half-width the old 0.05 target could never beat.
    floor = 1.959964 * np.sqrt(large["var_lines_and_families"])
    assert floor > 0.0
    assert large["expected_half_width_auroc"] >= floor
    assert large["expected_half_width_auroc"] == pytest.approx(floor, abs=0.01)
    # Two bounds on D, in order, since how much the arms share is unknown before adapters exist.
    assert small["expected_half_width_D_if_unit_term_cancels"] < small["expected_half_width_D_if_arms_independent"]
    assert small["expected_half_width_D_if_arms_independent"] == pytest.approx(np.sqrt(2.0) * small["expected_half_width_auroc"])
    assert "chosen" not in small, "the size is fixed by the change log, and nothing here chooses it"


def test_sizing_refuses_an_empty_test_set() -> None:
    scores, labels, families, groups = bootstrap_data(per_family=10)
    with pytest.raises(ValueError):
        plan_sample_size(scores, labels, families, groups, {"f0": 0}, 20, 0)


def planted_activations(n: int, seed: int, strength: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = (np.arange(n) % 2).astype(np.float32)
    acts = rng.normal(0, 1.0, (n, 3, 16)).astype(np.float32)
    acts[:, :, 0] += (2 * labels[:, None] - 1) * strength
    return acts, labels


def small_settings(**changes: object) -> ProbeTraining:
    return ProbeTraining(**{"batch_size": 8, "max_epochs": 12, "warmup_steps": 2, "scale_rows": 32, "log_every": 2, "mlp_hidden": 8, **changes})  # type: ignore[arg-type]


@pytest.mark.parametrize("kind", ["linear", "mlp"])
def test_probe_learns_a_planted_direction_and_keeps_its_layers(kind: str, tmp_path) -> None:
    train, train_y = planted_activations(240, 0)
    val, val_y = planted_activations(80, 1)
    probe = fit_ensemble(train, train_y, val, val_y, [5, 7, 9], kind, small_settings())
    assert probe.retained.all()
    assert auroc(probe.score(val), val_y) > 0.95
    probe.save(tmp_path / "p.npz")
    loaded = Ensemble.load(tmp_path / "p.npz")
    assert loaded.layers == [5, 7, 9]
    assert np.allclose(loaded.score(val), probe.score(val))


def test_fitting_reports_each_epoch_it_starts() -> None:
    train, train_y = planted_activations(240, 0)
    val, val_y = planted_activations(80, 1)
    seen: list[tuple[int, int]] = []
    fit_ensemble(train, train_y, val, val_y, [5, 7, 9], "linear", small_settings(max_epochs=2), on_epoch=lambda epoch, total: seen.append((epoch, total)))
    assert seen == [(0, 2), (1, 2)]


def test_noise_layer_is_dropped_by_the_loss_rule() -> None:
    train, train_y = planted_activations(240, 0, strength=3.0)
    val, val_y = planted_activations(80, 1, strength=3.0)
    rng = np.random.default_rng(5)
    train[:, 1] = rng.normal(0, 1.0, train[:, 1].shape)
    val[:, 1] = rng.normal(0, 1.0, val[:, 1].shape)
    probe = fit_ensemble(train, train_y, val, val_y, [5, 7, 9], "linear", small_settings())
    assert probe.retained.tolist() == [True, False, True]
    only_kept = sum(probe.layer_logits(val)[:, index] for index in (0, 2))
    assert auroc(only_kept, val_y) > 0.9
    # Changing the dropped layer changes nothing.
    changed = val.copy()
    changed[:, 1] = 100.0
    assert np.allclose(probe.score(changed), probe.score(val))


def test_probe_training_refuses_unbalanced_labels() -> None:
    train, train_y = planted_activations(100, 0)
    val, val_y = planted_activations(20, 1)
    with pytest.raises(ValueError):
        fit_ensemble(train, np.ones_like(train_y), val, val_y, [1, 2, 3], "linear", small_settings())


def test_scale_is_the_rms_norm_over_the_given_rows() -> None:
    acts = np.zeros((10, 2, 4), dtype=np.float32)
    acts[:, 0, 0] = 3.0
    acts[:, 1, 1] = 4.0
    acts[5:] = 100.0
    assert rms_scale(acts, 5).tolist() == [3.0, 4.0]


def test_contrast_order_is_yes_minus_no() -> None:
    train, train_y = planted_activations(240, 0)
    val, val_y = planted_activations(80, 1)
    probe = fit_ensemble(train, train_y, val, val_y, [5, 7, 9], "linear", small_settings())
    high = np.zeros((4, 3, 16), dtype=np.float32)
    high[:, :, 0] = 5.0
    low = -high
    assert (contrast(high, low, probe) > 0).all()
    assert (contrast(low, high, probe) < 0).all()


def test_random_direction_matches_each_layer_norm() -> None:
    train, train_y = planted_activations(240, 0)
    val, val_y = planted_activations(80, 1)
    probe = fit_ensemble(train, train_y, val, val_y, [5, 7, 9], "linear", small_settings())
    other = random_direction_probe(probe, 1)
    assert np.allclose(np.linalg.norm(other.params["w"], axis=1), np.linalg.norm(probe.params["w"], axis=1), atol=1e-5)
    assert not np.allclose(other.params["w"], probe.params["w"])


def test_float16_rounding_does_not_hurt_a_probe() -> None:
    train, train_y = planted_activations(240, 0)
    val, val_y = planted_activations(80, 1)
    scale = 40.0
    probe = fit_ensemble(train * scale, train_y, val * scale, val_y, [5, 7, 9], "linear", small_settings())
    full = auroc(probe.score(val * scale), val_y)
    rounded = auroc(probe.score((val * scale).astype(np.float16)), val_y)
    assert abs(full - rounded) < 0.01


def test_tfidf_learns_from_fit_text_and_ignores_unseen_vocabulary() -> None:
    texts = [f"alpha story {index} truth" for index in range(30)] + [f"beta story {index} tale" for index in range(30)]
    labels = [0] * 30 + [1] * 30
    classifier = TfidfClassifier(epochs=4, seed=0).fit(texts, labels)
    assert "gamma" not in classifier.vocabulary
    scores = classifier.decision(["alpha story 99 truth", "beta story 98 tale", "gamma delta"])
    assert scores[1] > scores[0]
    assert scores[2] == pytest.approx(classifier.bias)
    held = classifier.decision([f"alpha story {index} truth" for index in range(100, 120)] + [f"beta story {index} tale" for index in range(100, 120)])
    assert auroc(held, np.r_[np.zeros(20), np.ones(20)]) > 0.95
    assert transcript_text([{"content": "a"}, {"content": None}, {"content": "b"}]) == "a\n\nb"
