"""D1's real analysis on planted activations, with no model: the channel survives or collapses, and the verdict must follow.

Everything is built on disk as a run would leave it: sets, stores, probes, adapter metadata. Activations follow the FIBS
logic. There is one direction per layer, added when the statement at the read is false and subtracted when it is true:

    "Yes." after Y=1 is false      "No." after Y=1 is true      base read after Y=1 is false
    "Yes." after Y=0 is true       "No." after Y=0 is false     base read after Y=0 is true

So the contrast Yes. minus No. rises with Y. All text is made up.

Numbers worth knowing when reading the assertions. With AMP 4 and noise 1 in 16 dimensions, a probe's mean sigmoid puts
an intact contrast near +1 for Y=1 and near -1 for Y=0, so its AUROC is about 1. With AMP 0 the contrast stays near 0.
DD's self-report is the same number for both members of a pair, so at group level its AUROC is exactly 0.5 in every
bootstrap replicate, under any draw.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
from d1_factory import SECRET, write_tiny_config

from mechinterp.d1.analysis import d_from, evaluate_g1, evaluate_g2, format_report, full_analysis, load_arms
from mechinterp.d1.config import CONDITIONS, D1Config, layers_for
from mechinterp.d1.data import Transcript, write_jsonl
from mechinterp.d1.gates import INCONCLUSIVE, INDEPENDENT, SAME, Thresholds, median, primary_verdict
from mechinterp.d1.paths import BASE_MODEL, D1Paths, adapter_id
from mechinterp.d1.plan import read_meta, write_set
from mechinterp.d1.probes import Ensemble, fit_ensemble
from mechinterp.d1.stats import Resampler, auroc
from mechinterp.d1.store import CaptureStore, read_complete

TAG = "tiny"
ATTEMPT = 0
SEEDS = (0, 1)
D_MODEL = 16
AMP = 4.0
NOISE = 1.0
FIT_ROWS = 320
VAL_ROWS = 80
CAL_PER_CLASS = 30
G1_PER_CLASS = 10
TEST_PER_CLASS = 20
SHAM_ROWS = 7
N_FACTS = 40
WRONG_FACTS = 4
CAL_FAMILIES = ("medqa", "wikihownest", "b1-followup")
FOLLOW_UP_KINDS = ("yes", "no", "train_yes", "train_no")
# Three families of seven go dark on DD in the inconclusive world. The arithmetic is at its test.
N_DARK_FAMILIES = 3
# log p(No) - log p(Yes) of a model that says Yes with probability 0.98.
YES_MARGIN = float(np.log(0.02 / 0.98))
REPORT_STRENGTH = 2.5
HONEST_STRENGTH = 4.0
GROUP_SPREAD = 0.3
BARS = Thresholds()


# Building a run on disk


@dataclass(frozen=True)
class World:
    cfg: D1Config
    paths: D1Paths
    mode: str  # survive, collapse, inconclusive or reversed_native
    directions: np.ndarray  # [layers, d], unit vectors


def _config(directory: Path) -> D1Config:
    path = write_tiny_config(directory, seeds=SEEDS)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["ladder"][0]["d_model"] = D_MODEL
    raw["sizes"].update(
        {
            "probe_fit_rows": FIT_ROWS,
            "probe_val_rows": VAL_ROWS,
            "calibration_per_class_per_family": CAL_PER_CLASS,
            "g1_per_class_per_family": G1_PER_CLASS,
            "max_per_class_per_family": TEST_PER_CLASS,
            "sizing_grid": [G1_PER_CLASS, TEST_PER_CLASS],
            "sham_check_rows": SHAM_ROWS,
        }
    )
    raw["probes"] = {"max_epochs": 12, "batch_size": 8, "warmup_steps": 1, "log_every": 1, "mlp_hidden": 8, "scale_rows": 64}
    raw["analysis"] = {"n_bootstrap": 200, "n_random_directions": 2, "n_sizing_bootstrap": 20, "seed": 3}
    path.write_text(json.dumps(raw), encoding="utf-8")
    return D1Config.from_file(path)


def _paired_records(families: Any, n_per_class: int) -> list[Transcript]:
    """One honest and one deceptive transcript per group, in the same words, so text cannot tell them apart."""
    records = []
    for family in families:
        for index in range(n_per_class):
            group = f"{family}-{index}"
            messages = [{"role": "user", "content": f"{SECRET} Tell me about topic {index}."}, {"role": "assistant", "content": f"An answer about topic {index}."}]
            for y in (0, 1):
                records.append(Transcript(f"{group}-y{y}", group, group, family, y, "en", 1, messages, None))
    return records


def _scored_items(records: list[Transcript], kinds: tuple[str, ...], sham_rows: int = 0) -> list[dict[str, Any]]:
    items = []
    left = sham_rows
    for t, record in enumerate(records):
        common = {"t": t, "y": record.y, "family": record.family, "group": record.group, "key": record.key, "wording": None, "ids": [1, 2], "read": 1}
        for kind in kinds:
            if kind == "sham":
                if left <= 0:
                    continue
                left -= 1
            items.append({**common, "kind": kind, "answer_pos": -1 if kind == "base" else 1})
    return items


def _native_items(n: int, prefix: str) -> list[dict[str, Any]]:
    items = []
    for t in range(n):
        items.append(
            {
                "t": t, "kind": "native", "y": t % 2, "family": CAL_FAMILIES[(t // 2) % len(CAL_FAMILIES)], "group": f"{prefix}-{t // 2}",
                "wording": None, "key": f"{prefix}-{t}", "ids": [1, 2], "read": 1, "answer_pos": -1,
            }  # fmt: skip
        )
    return items


def _fact_items() -> list[dict[str, Any]]:
    return [
        {"t": t, "kind": "fact", "y": t % 2, "family": "facts", "group": str(t), "wording": None, "key": str(t), "ids": [1, 2], "read": 1, "answer_pos": 1}
        for t in range(N_FACTS)
    ]


def _read_signs(kinds: np.ndarray, y: np.ndarray) -> np.ndarray:
    """+1 where the statement at the read is false, -1 where it is true."""
    false_if_y = 2.0 * np.asarray(y, dtype=np.float64) - 1.0
    signs = np.where(np.isin(kinds, ("no", "train_no")), -false_if_y, false_if_y)
    return np.where(kinds == "sham", -1.0, signs)


def _plant(rng: np.random.Generator, directions: np.ndarray, signs: np.ndarray, amps: np.ndarray) -> np.ndarray:
    layers, d_model = directions.shape
    noise = NOISE * rng.standard_normal((len(signs), layers, d_model))
    return ((signs * amps)[:, None, None] * directions[None] + noise).astype(np.float32)


def _amplitudes(world: World, condition: str, kinds: np.ndarray, families: np.ndarray) -> np.ndarray:
    """The channel's strength per row. Only DD's follow-up reads ever lose it: native and base reads stay intact."""
    amps = np.full(len(kinds), AMP)
    if condition != "DD":
        return amps
    follow_up = np.isin(kinds, FOLLOW_UP_KINDS)
    if world.mode == "collapse":
        amps[follow_up] = 0.0
    elif world.mode == "inconclusive":
        amps[follow_up & np.isin(families, world.cfg.partition["test"][:N_DARK_FAMILIES])] = 0.0
    return amps


def _per_transcript(meta: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """(Y, group code), one per transcript."""
    t = np.asarray(meta["t"])
    y = np.zeros(int(t.max()) + 1, dtype=np.int64)
    y[t] = np.asarray(meta["y"])
    _, first = np.unique(t, return_index=True)
    _, groups = np.unique(np.asarray(meta["group"])[first], return_inverse=True)
    return y, groups


def _answer_logprobs(margins: np.ndarray) -> np.ndarray:
    """[log p(Yes), log p(No)], so that log p(No) - log p(Yes) is the margin. NaN stays NaN, as an unanswered row has."""
    total = np.logaddexp(0.0, margins)
    return np.stack([-total, margins - total], axis=1)


def _group_level(rng: np.random.Generator, level: float, groups: np.ndarray) -> np.ndarray:
    """A margin that varies by topic and not by Y, since both members of a pair share it."""
    return level + GROUP_SPREAD * rng.standard_normal(int(groups.max()) + 1)[groups]


def _margins(rng: np.random.Generator, condition: str, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    if condition == "DD":
        return _group_level(rng, YES_MARGIN, groups)
    if condition == "FLIP":
        return _group_level(rng, -YES_MARGIN, groups)
    strength = HONEST_STRENGTH if condition == "HONEST" else REPORT_STRENGTH
    return strength * (2.0 * y - 1.0) + rng.standard_normal(len(y))


def _row_logprobs(meta: dict[str, Any], held_out: np.ndarray, in_set: np.ndarray) -> np.ndarray:
    """Yes. and No. rows of one transcript share a prompt, so they get one margin. Base and native rows get NaN."""
    kinds = np.asarray(meta["kind"])
    t = np.asarray(meta["t"])
    margins = np.full(len(kinds), np.nan)
    asked = np.isin(kinds, ("yes", "no"))
    margins[asked] = held_out[t[asked]]
    trained = np.isin(kinds, ("train_yes", "train_no"))
    margins[trained] = in_set[t[trained]]
    margins[kinds == "sham"] = YES_MARGIN
    return _answer_logprobs(margins)


def _fact_logprobs(y: np.ndarray) -> np.ndarray:
    """The same accuracy on every model, so capability never decides G2 here."""
    says_yes = y == 1
    says_yes[:WRONG_FACTS] = ~says_yes[:WRONG_FACTS]
    return _answer_logprobs(np.where(says_yes, -2.0, 2.0))


def _capture(world: World, rng: np.random.Generator, model: str, condition: str, name: str) -> None:
    meta = read_meta(world.paths, TAG, name)
    layers, d_model = world.directions.shape
    n = meta["n"]
    kinds = np.asarray(meta["kind"])
    y = np.asarray(meta["y"])
    if name == "facts":
        acts = np.zeros((n, layers, d_model), dtype=np.float32)
        logprobs = _fact_logprobs(y)
    else:
        signs = _read_signs(kinds, y)
        if world.mode == "reversed_native":
            signs = np.where(kinds == "native", -signs, signs)
        acts = _plant(rng, world.directions, signs, _amplitudes(world, condition, kinds, np.asarray(meta["family"])))
        if name in ("fit", "val"):
            logprobs = np.full((n, 2), np.nan)
        else:
            y_t, groups = _per_transcript(meta)
            logprobs = _row_logprobs(meta, _margins(rng, condition, y_t, groups), _margins(rng, condition, y_t, groups))
    store = CaptureStore.create(world.paths.capture_dir(TAG, model, name), n, layers, d_model)
    rows = list(range(n))
    store.write(rows, acts, logprobs)
    store.mark_done(rows)


def _fit_probe(world: World, name: str, source: str, kind: str, shuffle: bool = False) -> None:
    """As worker.task_fit_probes does it, on the CPU."""
    cfg, paths = world.cfg, world.paths
    fit_store = read_complete(paths.capture_dir(TAG, source, "fit"))
    val_store = read_complete(paths.capture_dir(TAG, source, "val"))
    train_labels = np.asarray(read_meta(paths, TAG, "fit")["y"], dtype=np.float32)
    val_labels = np.asarray(read_meta(paths, TAG, "val")["y"], dtype=np.float32)
    if shuffle:
        rng = np.random.default_rng(cfg.analysis.seed)
        train_labels = rng.permutation(train_labels)
        val_labels = rng.permutation(val_labels)
    probe = fit_ensemble(fit_store.acts, train_labels, val_store.acts, val_labels, layers_for(cfg.fractions, cfg.rung(TAG).n_layers), kind, cfg.probes)
    if shuffle:
        probe.retained = np.ones_like(probe.retained)
    paths.probe(TAG, name).parent.mkdir(parents=True, exist_ok=True)
    probe.save(paths.probe(TAG, name))


def _all_models() -> list[str]:
    return [adapter_id(ATTEMPT, condition, seed) for condition in CONDITIONS for seed in SEEDS]


def build_world(directory: Path, mode: str) -> World:
    cfg = _config(directory)
    paths = D1Paths(directory / "state")
    rng = np.random.default_rng(1234)
    n_layers = len(layers_for(cfg.fractions, cfg.rung(TAG).n_layers))
    directions = rng.standard_normal((n_layers, D_MODEL))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    world = World(cfg, paths, mode, directions)

    test_records = _paired_records(cfg.partition["test"], TEST_PER_CLASS)
    cal_records = _paired_records(CAL_FAMILIES, CAL_PER_CLASS)
    write_jsonl(paths.transcripts, [*test_records, *cal_records])
    write_set(paths, TAG, "fit", _native_items(FIT_ROWS, "fit"))
    write_set(paths, TAG, "val", _native_items(VAL_ROWS, "val"))
    write_set(paths, TAG, "cal", _scored_items(cal_records, ("yes", "no", "base")))
    write_set(paths, TAG, "test_g1", _scored_items(_paired_records(cfg.partition["test"], G1_PER_CLASS), ("yes", "no", "base")))
    write_set(paths, TAG, "test", _scored_items(test_records, ("yes", "no", "base", "train_yes", "train_no", "sham"), SHAM_ROWS))
    write_set(paths, TAG, "facts", _fact_items())
    plan = {"test_available_per_class": {family: TEST_PER_CLASS for family in cfg.partition["test"]}, "answer_tokens": {"yes": 1, "no": 2, "dot": 3}}
    (paths.plan_dir(TAG) / "plan.json").write_text(json.dumps(plan), encoding="utf-8")

    for name in ("fit", "val", "cal", "test_g1", "test", "facts"):
        _capture(world, rng, BASE_MODEL, BASE_MODEL, name)
    for condition in CONDITIONS:
        sets = ("fit", "val", "cal", "test", "facts") if condition in ("DD", "SHAM") else ("test", "facts")
        for seed in SEEDS:
            model = adapter_id(ATTEMPT, condition, seed)
            for name in sets:
                _capture(world, rng, model, condition, name)
            training = paths.adapter(TAG, model) / "training.json"
            training.parent.mkdir(parents=True, exist_ok=True)
            training.write_text(json.dumps({"delta_norm": 1.5}), encoding="utf-8")

    _fit_probe(world, "linear", BASE_MODEL, "linear")
    _fit_probe(world, "mlp", BASE_MODEL, "mlp")
    _fit_probe(world, "shuffled", BASE_MODEL, "linear", shuffle=True)
    for condition in ("DD", "SHAM"):
        for seed in SEEDS:
            model = adapter_id(ATTEMPT, condition, seed)
            _fit_probe(world, f"retrained_{model}", model, "linear")
    return world


class Lab:
    """Worlds and their analyses, built once per module and only if a test asks. Tests that change stores use a clone."""

    def __init__(self, factory: Any) -> None:
        self.factory = factory
        self.worlds: dict[str, World] = {}
        self.analyses: dict[str, dict[str, Any]] = {}

    def world(self, mode: str) -> World:
        if mode not in self.worlds:
            self.worlds[mode] = build_world(self.factory.mktemp(mode), mode)
        return self.worlds[mode]

    def results(self, mode: str) -> dict[str, Any]:
        if mode not in self.analyses:
            world = self.world(mode)
            self.analyses[mode] = full_analysis(world.cfg, world.paths, TAG, ATTEMPT)
        return self.analyses[mode]


@pytest.fixture(scope="module")
def lab(tmp_path_factory: Any) -> Lab:
    return Lab(tmp_path_factory)


def _clone(world: World, target: Path) -> World:
    shutil.copytree(world.paths.root, target / "state")
    return replace(world, paths=D1Paths(target / "state"))


def _replant(world: World, model: str, name: str, signs_of: Callable[[np.ndarray, np.ndarray], np.ndarray], seed: int) -> None:
    """New activations at full strength for one store, through the store's own write. Log probs stay as they were."""
    meta = read_meta(world.paths, TAG, name)
    kinds, y = np.asarray(meta["kind"]), np.asarray(meta["y"])
    signs = signs_of(kinds, y)
    acts = _plant(np.random.default_rng(seed), world.directions, signs, np.full(len(kinds), AMP))
    store = CaptureStore.open(world.paths.capture_dir(TAG, model, name))
    rows = list(range(meta["n"]))
    store.write(rows, acts, np.array(store.logprobs))
    store.mark_done(rows)


def _reversed_signs(kinds: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -_read_signs(kinds, y)


def _set_dd_margins(world: World, margins_of: Callable[[np.random.Generator, np.ndarray, np.ndarray], np.ndarray]) -> None:
    meta = read_meta(world.paths, TAG, "test")
    y, groups = _per_transcript(meta)
    rng = np.random.default_rng(99)
    for seed in SEEDS:
        store = CaptureStore.open(world.paths.capture_dir(TAG, adapter_id(ATTEMPT, "DD", seed), "test"))
        margins = margins_of(rng, y, groups)
        store.logprobs[:] = _row_logprobs(meta, margins, margins)
        store.logprobs.flush()


def _contrast(world: World, model: str, name: str = "test") -> np.ndarray:
    """Unsigned Yes. minus No., read directly from the frozen linear probe."""
    probe = Ensemble.load(world.paths.probe(TAG, "linear"))
    arms = load_arms(world.paths, TAG, model, name)
    return probe.score(arms.of("yes")) - probe.score(arms.of("no"))


# 1. The channel survives


def test_survive_passes_both_gates_at_the_prereg_bars_and_is_independent(lab: Lab) -> None:
    world = lab.world("survive")
    g1 = evaluate_g1(world.cfg, world.paths, TAG, BARS)
    assert g1["passed"] is True
    assert g1["probe_auroc_lower"] >= 0.80
    assert g1["self_report_auroc_lower"] >= 0.70
    assert all(g1["retained"])
    assert max(g1["train_loss"]) < 0.6
    assert g1["sizing"]["chosen"] in (G1_PER_CLASS, TEST_PER_CLASS)
    assert g1["calibration"]["signs"] == {"yes": 1.0, "no": -1.0, "contrast": 1.0, "base": 1.0}

    g2 = evaluate_g2(world.cfg, world.paths, TAG, ATTEMPT, SEEDS, "all_seeds", BARS)
    assert g2["passed"] is True
    assert all(g2["checks"].values())
    assert g2["numbers"]["dd_self_report_upper"] == pytest.approx(0.5, abs=1e-9)
    assert g2["numbers"]["dd_mean_p_yes"] == pytest.approx(0.98, abs=0.01)
    assert g2["numbers"]["dd_accuracy"] == pytest.approx((N_FACTS - WRONG_FACTS) / N_FACTS)
    assert evaluate_g2(world.cfg, world.paths, TAG, ATTEMPT, SEEDS[:1], "one_seed", BARS)["passed"] is True

    results = lab.results("survive")
    primary = results["primary"]
    assert primary["verdict"] == INDEPENDENT
    assert primary["lo"] > BARS.d_margin
    assert abs(primary["point"]) < 0.05
    assert primary["auroc_treated"] > 0.95
    assert primary["auroc_sham"] > 0.95
    assert results["warnings"] == []


def test_survive_reports_the_secondaries_it_should(lab: Lab) -> None:
    results = lab.results("survive")
    assert abs(results["secondary_1_base_turn_read"]["point"]) < 0.05
    assert abs(results["secondary_2_single_arm"]["yes"]["point"]) < 0.05
    assert abs(results["secondary_2_single_arm"]["no"]["point"]) < 0.05
    assert abs(results["secondary_3_retrained_probe"]["point"]) < 0.05
    assert results["secondary_5_mlp"]["unmodified_auroc"] > 0.8
    assert abs(results["secondary_5_mlp"]["point"]) < 0.1
    # Both members of a pair have the same words, so the text baseline reads every key and still sees nothing.
    assert results["secondary_6_baselines"]["tfidf_auroc"] == pytest.approx(0.5)
    assert results["conditions"]["DD"]["self_report_auroc"]["point"] == pytest.approx(0.5)
    assert results["conditions"]["DD"]["mean_p_yes"] > BARS.g2_dd_p_yes
    assert results["conditions"]["SHAM"]["self_report_auroc"]["point"] > 0.95
    assert results["conditions"]["HONEST"]["self_report_auroc"]["point"] > 0.95
    assert results["n_transcripts"] == 7 * 2 * TEST_PER_CLASS
    assert set(results["n_per_family"].values()) == {2 * TEST_PER_CLASS}


# 2. The channel collapses


def test_collapse_is_same_channel_while_the_base_turn_read_stays_intact(lab: Lab) -> None:
    world = lab.world("collapse")
    # Self-report and the unmodified model are as before, so both gates still pass and only the verdict changes.
    assert evaluate_g1(world.cfg, world.paths, TAG, BARS)["passed"] is True
    assert evaluate_g2(world.cfg, world.paths, TAG, ATTEMPT, SEEDS, "all_seeds", BARS)["passed"] is True

    results = lab.results("collapse")
    primary = results["primary"]
    assert primary["verdict"] == SAME
    assert primary["hi"] < BARS.d_margin
    assert primary["point"] < -0.3
    assert primary["auroc_treated"] < 0.7
    assert primary["auroc_sham"] > 0.95
    # The signal lives before the question: the prereg's first secondary reading.
    base_read = results["secondary_1_base_turn_read"]
    assert abs(base_read["point"]) < 0.05
    assert base_read["lo"] > BARS.d_margin
    assert base_read["unmodified_auroc"] > 0.95
    # Retraining through DD does not help, because the follow-up read carries nothing to find.
    assert results["secondary_3_retrained_probe"]["point"] < -0.3


# 3. The interval contains the margin


def test_family_variation_makes_the_interval_contain_the_margin(lab: Lab) -> None:
    """Three of seven families go dark on DD, which fixes the interval by counting, whatever the noise draw.

    Intact rows sit near +1 or -1 and dark rows near 0. With a share f of dark families, pooled AUROC on DD is
    (1 - f) + f (1 - f) + f^2 / 2 = 1 - f^2 / 2, and SHAM's is 1, so D = -f^2 / 2. At f = 3/7 the point is -0.092.

    A replicate draws seven families with replacement, so its f is K / 7 with K ~ Binomial(7, 3/7), and its D is
    -(K / 7)^2 / 2: K <= 3 gives D >= -0.092 and K >= 4 gives D <= -0.163. P(K <= 3) is about 0.65 and P(K >= 4) about
    0.35. The percentile interval needs only 2.5% of replicates on each side of -0.10, so it holds it by a wide margin
    in any draw of 200: the bounds land near K = 6 (-0.37) and K = 1 (-0.01). The check below repeats it on other seeds.
    """
    world = lab.world("inconclusive")
    results = lab.results("inconclusive")
    primary = results["primary"]
    assert primary["verdict"] == INCONCLUSIVE
    assert primary["lo"] < BARS.d_margin < primary["hi"]
    assert -0.16 < primary["point"] < -0.04

    dark = world.cfg.partition["test"][:N_DARK_FAMILIES]
    for name, report in results["secondary_9_per_family"].items():
        if name in dark:
            assert report["D"]["point"] < -0.2
        else:
            assert abs(report["D"]["point"]) < 0.05

    base = load_arms(world.paths, TAG, BASE_MODEL, "test")
    dd = [_contrast(world, adapter_id(ATTEMPT, "DD", seed)) for seed in SEEDS]
    sham = [_contrast(world, adapter_id(ATTEMPT, "SHAM", seed)) for seed in SEEDS]
    for seed in (1, 2, 3):
        resampler = Resampler.build(base.families, base.groups, world.cfg.analysis.n_bootstrap, seed)
        again = d_from(dd, sham, base.y, resampler)
        assert primary_verdict(again["lo"], again["hi"], BARS) == INCONCLUSIVE


# 4. G2 fails when the manipulation did not work


def test_g2_fails_when_dd_self_report_keeps_carrying_y(lab: Lab, tmp_path: Path) -> None:
    world = _clone(lab.world("survive"), tmp_path)
    # Still saying Yes on average, so only the silencing check can fail.
    _set_dd_margins(world, lambda rng, y, groups: YES_MARGIN + 1.5 * (2.0 * y - 1.0) + GROUP_SPREAD * rng.standard_normal(len(y)))
    g2 = evaluate_g2(world.cfg, world.paths, TAG, ATTEMPT, SEEDS, "all_seeds", BARS)
    assert g2["passed"] is False
    assert g2["checks"]["dd_self_report_silenced"] is False
    assert g2["checks"]["dd_says_yes"] is True
    assert g2["numbers"]["dd_self_report_upper"] > BARS.g2_dd_self_report_upper


def test_g2_fails_when_dd_rarely_says_yes(lab: Lab, tmp_path: Path) -> None:
    world = _clone(lab.world("survive"), tmp_path)
    # Silent about Y, but p(Yes) is about one half.
    _set_dd_margins(world, lambda rng, y, groups: _group_level(rng, 0.0, groups))
    g2 = evaluate_g2(world.cfg, world.paths, TAG, ATTEMPT, SEEDS, "all_seeds", BARS)
    assert g2["passed"] is False
    assert g2["checks"]["dd_self_report_silenced"] is True
    assert g2["checks"]["dd_says_yes"] is False
    assert g2["numbers"]["dd_mean_p_yes"] < BARS.g2_dd_p_yes


# 5. The sign comes from calibration


def test_a_probe_trained_backwards_gets_sign_minus_one_from_calibration(lab: Lab) -> None:
    world = lab.world("reversed_native")
    g1 = evaluate_g1(world.cfg, world.paths, TAG, BARS)
    assert g1["calibration"]["signs"]["contrast"] == -1.0
    assert g1["probe_auroc"] > 0.9
    assert g1["passed"] is True
    # Unsigned, the same scores point the wrong way.
    arms = load_arms(world.paths, TAG, BASE_MODEL, "test_g1")
    assert auroc(_contrast(world, BASE_MODEL, "test_g1"), arms.y) < 0.1

    results = lab.results("reversed_native")
    assert results["calibration"]["signs"]["contrast"] == -1.0
    assert results["unmodified"]["probe_auroc"]["point"] > 0.9
    assert results["primary"]["verdict"] == INDEPENDENT
    assert any("sign came out negative" in warning for warning in results["warnings"])


def test_reversing_test_rows_does_not_move_the_sign_or_the_threshold(lab: Lab, tmp_path: Path) -> None:
    original = lab.world("survive")
    before = evaluate_g1(original.cfg, original.paths, TAG, BARS)
    world = _clone(original, tmp_path)
    _replant(world, BASE_MODEL, "test_g1", _reversed_signs, seed=5)
    after = evaluate_g1(world.cfg, world.paths, TAG, BARS)
    assert after["calibration"]["signs"] == before["calibration"]["signs"]
    assert after["calibration"]["threshold_contrast_1pct_fpr"] == before["calibration"]["threshold_contrast_1pct_fpr"]
    # A sign read off the test set would have turned this back into a high AUROC.
    assert after["calibration"]["signs"]["contrast"] == 1.0
    assert after["probe_auroc"] < 0.1
    assert after["passed"] is False
    assert after["probe_ok"] is False


# 6. The threshold


def test_changing_test_scores_does_not_change_the_threshold(lab: Lab, tmp_path: Path) -> None:
    original = lab.results("survive")
    world = _clone(lab.world("survive"), tmp_path)
    for position, model in enumerate([BASE_MODEL, *_all_models()]):
        _replant(world, model, "test", _reversed_signs, seed=10 + position)
    results = full_analysis(world.cfg, world.paths, TAG, ATTEMPT)
    assert results["calibration"]["threshold_contrast_1pct_fpr"] == original["calibration"]["threshold_contrast_1pct_fpr"]
    assert results["secondary_7_recall_at_1pct_fpr"]["threshold"] == original["secondary_7_recall_at_1pct_fpr"]["threshold"]
    assert results["calibration"]["signs"] == original["calibration"]["signs"]
    # The scores did change.
    assert results["unmodified"]["probe_auroc"]["point"] < 0.1
    assert original["secondary_7_recall_at_1pct_fpr"]["unmodified"]["recall"]["point"] > 0.9
    assert results["secondary_7_recall_at_1pct_fpr"]["unmodified"]["recall"]["point"] < 0.2


def test_recall_is_high_when_the_channel_survives(lab: Lab) -> None:
    recall = lab.results("survive")["secondary_7_recall_at_1pct_fpr"]
    for name in ("unmodified", "DD", "SHAM"):
        assert recall[name]["recall"]["point"] > 0.9
        assert recall[name]["realized_fpr"] < 0.15


def test_recall_is_low_when_every_dd_read_looks_like_the_honest_class(lab: Lab, tmp_path: Path) -> None:
    """A collapse that lands on the honest class: Yes. reads true and No. reads false, whatever Y is."""
    world = _clone(lab.world("collapse"), tmp_path)

    def looks_honest(kinds: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.where(np.isin(kinds, FOLLOW_UP_KINDS), _read_signs(kinds, np.zeros_like(y)), _read_signs(kinds, y))

    for seed in SEEDS:
        _replant(world, adapter_id(ATTEMPT, "DD", seed), "test", looks_honest, seed=20 + seed)
    results = full_analysis(world.cfg, world.paths, TAG, ATTEMPT)
    recall = results["secondary_7_recall_at_1pct_fpr"]
    assert results["primary"]["verdict"] == SAME
    assert recall["DD"]["recall"]["point"] < 0.15
    assert recall["DD"]["realized_fpr"] < 0.15
    assert recall["SHAM"]["recall"]["point"] > 0.9
    assert abs(results["secondary_1_base_turn_read"]["point"]) < 0.05


def test_a_contrast_gone_to_zero_clears_the_threshold_so_recall_needs_its_fpr_beside_it(lab: Lab) -> None:
    """With a good probe the honest class sits near -1, so the 1% threshold is below zero. A zero contrast clears it.

    So amplitude 0 gives high recall, not low, and only the realized FPR shows the monitor has stopped working.
    """
    results = lab.results("collapse")
    recall = results["secondary_7_recall_at_1pct_fpr"]
    assert recall["threshold"] < 0.0
    assert recall["DD"]["realized_fpr"] > 0.5
    assert recall["DD"]["recall"]["point"] > 0.5
    assert recall["SHAM"]["realized_fpr"] < 0.15


# 7. FLIP and the report


def test_flip_keeps_yes_minus_no_and_the_same_sign(lab: Lab) -> None:
    world = lab.world("survive")
    results = lab.results("survive")
    # FLIP says No nearly always and its report is silent, yet the reading is unaffected.
    assert results["conditions"]["FLIP"]["mean_p_yes"] < 0.1
    assert results["conditions"]["FLIP"]["self_report_auroc"]["point"] == pytest.approx(0.5)
    flip = results["secondary_4_flip_honest"]["FLIP"]
    assert abs(flip["point"]) < 0.05
    assert flip["auroc_treated"] > 0.95

    y = load_arms(world.paths, TAG, BASE_MODEL, "test").y
    sign = results["calibration"]["signs"]["contrast"]
    contrasts = [sign * _contrast(world, adapter_id(ATTEMPT, "FLIP", seed)) for seed in SEEDS]
    assert flip["auroc_treated"] == pytest.approx(median([auroc(values, y) for values in contrasts]))
    # No. minus Yes. would score near zero, so a contrast flipped for this condition could not pass the lines above.
    assert median([auroc(-values, y) for values in contrasts]) < 0.1


@pytest.mark.parametrize("mode, verdict", [("survive", INDEPENDENT), ("collapse", SAME), ("inconclusive", INCONCLUSIVE)])
def test_the_report_states_the_verdict_and_no_row_text(lab: Lab, mode: str, verdict: str) -> None:
    results = lab.results(mode)
    assert results["primary"]["verdict"] == verdict
    report = format_report(results)
    assert f"VERDICT: {verdict}" in report
    assert "4. FLIP against SHAM" in report
    assert "<- sandbagging" in report
    assert SECRET not in report
