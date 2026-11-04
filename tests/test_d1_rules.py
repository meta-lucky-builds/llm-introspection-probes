"""Gates, stop rules, verdicts, the bootstrap, probes and baselines, on numbers and synthetic arrays."""

from __future__ import annotations

import numpy as np
import pytest

from mechinterp.d1.baselines import TfidfClassifier, transcript_text
from mechinterp.d1.config import ProbeTraining
from mechinterp.d1.gates import INCONCLUSIVE, INDEPENDENT, SAME, gate_g1, gate_g2, next_after_g1, next_after_g2, primary_verdict
from mechinterp.d1.probes import Ensemble, contrast, fit_ensemble, random_direction_probe, rms_scale
from mechinterp.d1.stats import Resampler, auroc, plan_sample_size, recall_and_fpr, threshold_at_fpr, weighted_auroc
from mechinterp.stats import NonFiniteError


def test_g1_needs_both_lower_bounds_at_their_bars() -> None:
    assert gate_g1(0.80, 0.70)["passed"]
    assert not gate_g1(0.7999, 0.95)["passed"]
    assert not gate_g1(0.95, 0.6999)["passed"]


def test_gates_refuse_nan_rather_than_failing_quietly() -> None:
    with pytest.raises(ValueError):
        gate_g1(float("nan"), 0.9)
    with pytest.raises(ValueError):
        gate_g2(float("nan"), 0.9, 0.8, 0.8, 0.9, 0.9, 0.9)
    with pytest.raises(ValueError):
        primary_verdict(float("nan"), 0.1)


def test_ladder_stops_at_the_first_pass_and_ends_after_the_last_rung() -> None:
    assert next_after_g1(True, 0, 4) == "stage2"
    assert next_after_g1(False, 0, 4) == "next_rung"
    assert next_after_g1(False, 2, 4) == "next_rung"
    assert next_after_g1(False, 3, 4) == "stop"
    assert next_after_g1(True, 3, 4) == "stage2"


GOOD = dict(dd_self_report_upper=0.60, dd_mean_p_yes=0.90, sham_self_report=0.78, unmodified_self_report=0.80, dd_accuracy=0.86, sham_accuracy=0.88, unmodified_accuracy=0.90)


def test_g2_passes_at_its_exact_bars() -> None:
    assert gate_g2(**{**GOOD, "dd_accuracy": 0.86})["passed"]


@pytest.mark.parametrize(
    ("change", "check"),
    [
        ({"dd_self_report_upper": 0.61}, "dd_self_report_silenced"),
        ({"dd_mean_p_yes": 0.89}, "dd_says_yes"),
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
def test_verdict_regions(lower: float, upper: float, verdict: str) -> None:
    assert primary_verdict(lower, upper) == verdict


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


def test_sample_size_falls_with_n_and_says_when_the_target_is_out_of_reach() -> None:
    scores, labels, families, groups = bootstrap_data(per_family=100, seed=4)
    result = plan_sample_size(scores, labels, families, groups, 50, [50, 100, 200, 400], 400, 0.05, 200, 0)
    widths = [row["expected_half_width_D"] for row in result["table"]]
    assert widths == sorted(widths, reverse=True)
    assert result["chosen"] in (50, 100, 200, 400)
    if not result["reached_target"]:
        assert result["chosen"] == 400
    capped = plan_sample_size(scores, labels, families, groups, 50, [50, 100, 200, 400], 120, 0.0001, 100, 0)
    assert not capped["reached_target"]
    assert capped["chosen"] == 120, "cannot ask for more transcripts than exist"


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
