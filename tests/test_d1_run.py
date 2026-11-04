"""D1 from start to verdict on the tiny random model, on CPU. No GPU and no download."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest
from d1_factory import SECRET, TINY_PER_CLASS_PER_FAMILY, tiny_rows, write_tiny_config

from mechinterp.d1 import analysis, run_identity
from mechinterp.d1.capture import run_capture
from mechinterp.d1.config import D1Config, layers_for
from mechinterp.d1.data import ingest, write_jsonl
from mechinterp.d1.gates import G1_FINDING, G2_FINDING, Thresholds, primary_verdict
from mechinterp.d1.orchestrator import Experiment, InProcessRunner, NoWorkRunner, PoolOptions, Task, finalize
from mechinterp.d1.paths import BASE_MODEL, D1Paths, adapter_id
from mechinterp.d1.plan import read_meta
from mechinterp.d1.store import CaptureStore, read_complete
from mechinterp.d1.worker import answer_tokens, load_model
from mechinterp.sharding import RunFailed, read_json, sha256_file

TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
FACTORY = "d1_factory:build"
TAG = "tiny"
# 12 prompts per family gives 60 probe-train prompts: 12 go to calibration, 5 of the other 48 to validation, 43 to fit.
# Every prompt has both labels, so fit gets at least 70 rows, and the dry run's small probe needs 32. Calibration only
# draws on the three two-class families, which hold 36 of the 60 prompts. Each test family has 12 per class, above the
# G1 draw of 3 and the test set's 4. test_the_tiny_plan_gives_the_gates_what_they_need checks all of this.
ROWS_PER_FAMILY = 12
TWO_CLASS_PROBE_FAMILIES = {"medqa", "wikihownest", "b1-followup"}
# A random tiny model cannot pass the real bars. Only the Python API can loosen them.
LOOSE = Thresholds(
    g1_probe_lower=0.0,
    g1_self_report_lower=0.0,
    g2_dd_p_yes=0.0,
    g2_dd_balanced_accuracy=1.0,
    g2_sham_within=1.0,
    g2_capability_within=1.0,
    g2_specificity_within=1.0,
)
KEEP_EVERY_LAYER = 100.0
# float16 holds about 1 part in 1000, so this allows two rounding steps between two reads of one row.
ROUNDING = 2e-3


@dataclass
class Setup:
    config: Path
    cfg: D1Config
    paths: D1Paths


def prepare(directory: Path, extra_rungs: Sequence[str] = ()) -> Setup:
    """The state a finished prepare task leaves, without parquet files."""
    config = write_tiny_config(directory)
    raw = json.loads(config.read_text(encoding="utf-8"))
    first = raw["ladder"][0]
    raw["ladder"] = [first, *[{**first, "name": name} for name in extra_rungs]]
    # Two epochs on a random model leave the loss near 0.69, above the 0.6 layer rule, so every probe would keep no layer
    # and score a constant. Keeping all layers makes these runs go through real probe scores and a real D.
    raw["probes"]["drop_loss"] = KEEP_EVERY_LAYER
    config.write_text(json.dumps(raw), encoding="utf-8")
    cfg = D1Config.from_file(config)
    paths = D1Paths(directory / "out")
    transcripts, native, ordinary, manifest = ingest(tiny_rows(ROWS_PER_FAMILY), cfg.partition, None)
    write_jsonl(paths.transcripts, transcripts)
    write_jsonl(paths.native, native)
    write_jsonl(paths.ordinary, ordinary)
    paths.data_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    marker = paths.done("prepare")
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    return Setup(config, cfg, paths)


def options_for(setup: Setup) -> PoolOptions:
    return PoolOptions(devices=[], config_path=setup.config, out_dir=setup.paths.root, factory=FACTORY)


def experiment(setup: Setup, runner: Any, thresholds: Thresholds = LOOSE, gpu_free_gb: float | None = None) -> Experiment:
    return Experiment(setup.cfg, setup.paths, runner, 1, gpu_free_gb, thresholds=thresholds)


def train_tasks(runner: InProcessRunner) -> list[str]:
    return [name for name in runner.ran if ".train." in name]


def store_hashes(directory: Path) -> dict[str, str]:
    return {name: sha256_file(directory / name) for name in ("acts.f16", "logprobs.f8", "done.u1")}


class WatchingRunner(InProcessRunner):
    """Notes what was on disk just before the first adapter started training."""

    def __init__(self, options: PoolOptions, paths: D1Paths) -> None:
        super().__init__(options, paths)
        self.base_hashes: dict[str, str] | None = None
        self.size_was_logged = False

    def run(self, label: str, tasks: Sequence[Task]) -> None:
        if self.base_hashes is None and any(task.argv[0] == "train" for task in tasks):
            self.base_hashes = store_hashes(self.paths.capture_dir(TAG, BASE_MODEL, "test"))
            self.size_was_logged = self.paths.sample_size(TAG).exists()
        super().run(label, tasks)


class StoppingRunner(InProcessRunner):
    """Fails the way a dead machine does, after a set number of finished tasks."""

    def __init__(self, options: PoolOptions, paths: D1Paths, limit: int) -> None:
        super().__init__(options, paths)
        self.limit = limit

    def run(self, label: str, tasks: Sequence[Task]) -> None:
        for task in tasks:
            if len(self.ran) >= self.limit:
                raise RunFailed("Stopped here on purpose")
            super().run(label, [task])


@pytest.fixture(scope="module")
def full_run(tmp_path_factory) -> dict:
    """One whole run with every gate loosened, shared because it is the slowest thing in this file."""
    setup = prepare(tmp_path_factory.mktemp("full"))
    runner = WatchingRunner(options_for(setup), setup.paths)
    outcome = experiment(setup, runner).run()
    return {"setup": setup, "runner": runner, "outcome": outcome}


# The stop rules


def test_g1_failure_ends_the_run_as_a_finding_and_no_adapter_trains(tmp_path) -> None:
    setup = prepare(tmp_path)
    runner = InProcessRunner(options_for(setup), setup.paths)
    outcome = experiment(setup, runner, Thresholds(g1_probe_lower=1.01, g1_self_report_lower=0.0)).run()
    paths = setup.paths
    assert outcome["stopped_at"] == "g1"
    assert outcome["finding"] == G1_FINDING
    assert outcome["gates"][TAG]["passed"] is False
    assert outcome["gates"][TAG]["probe_ok"] is False
    assert read_json(paths.root / "outcome.json")["stopped_at"] == "g1"
    assert G1_FINDING in (paths.root / "report.txt").read_text(encoding="utf-8")
    assert runner.ran == [f"{TAG}.plan", f"{TAG}.preflight", f"{TAG}.capture.base.test_g1.0of1", f"{TAG}.capture.base.fit-val-cal.0of1", f"{TAG}.fit.linear"]
    assert not (paths.rung_dir(TAG) / "adapters").exists()
    assert not paths.sample_size(TAG).exists()
    assert not (paths.root / "results.json").exists()


def test_a_failed_self_report_ends_the_rung_before_any_probe_row_is_captured(tmp_path) -> None:
    """Fit rows are most of a rung's cost, and G1 cannot pass once self-report has failed."""
    setup = prepare(tmp_path)
    runner = InProcessRunner(options_for(setup), setup.paths)
    outcome = experiment(setup, runner, Thresholds(g1_self_report_lower=1.01)).run()
    paths = setup.paths
    assert outcome["stopped_at"] == "g1"
    assert runner.ran == [f"{TAG}.plan", f"{TAG}.preflight", f"{TAG}.capture.base.test_g1.0of1"]
    gate = outcome["gates"][TAG]
    assert gate["passed"] is False
    assert gate["self_report_ok"] is False
    assert gate["probe_auroc"] is None and gate["probe_ok"] is None
    assert "was not fitted" in gate["stopped_early"]
    assert not paths.probe(TAG, "linear").exists()
    assert not paths.capture_dir(TAG, BASE_MODEL, "fit").exists()
    report = (paths.root / "report.txt").read_text(encoding="utf-8")
    assert G1_FINDING in report and "The probe was not fitted" in report
    assert read_json(paths.gate_g1(TAG))["passed"] is False


def test_a_failed_rung_moves_to_the_next_and_a_pass_ends_the_climb(tmp_path, monkeypatch) -> None:
    setup = prepare(tmp_path, extra_rungs=("tiny2", "tiny3"))
    real = analysis.evaluate_g1

    # One set of bars covers the whole ladder, so a fail on the first rung alone has to be planted.
    def fail_first_rung(cfg: D1Config, paths: D1Paths, tag: str, thresholds: Thresholds) -> dict:
        result = real(cfg, paths, tag, thresholds)
        return {**result, "passed": False} if tag == TAG else result

    monkeypatch.setattr(analysis, "evaluate_g1", fail_first_rung)
    runner = InProcessRunner(options_for(setup), setup.paths)
    tag, history = experiment(setup, runner).run_stage1()
    assert tag == "tiny2"
    assert list(history) == [TAG, "tiny2"]
    assert history[TAG]["passed"] is False and history["tiny2"]["passed"] is True
    assert read_json(setup.paths.gate_g1(TAG))["passed"] is False
    assert f"{TAG}.plan" in runner.ran and "tiny2.plan" in runner.ran
    assert not any(name.startswith("tiny3") for name in runner.ran)
    assert not setup.paths.rung_dir("tiny3").exists()
    assert train_tasks(runner) == []


def test_g2_failure_gets_one_retry_at_twice_the_steps_and_then_stops(tmp_path) -> None:
    setup = prepare(tmp_path)
    runner = InProcessRunner(options_for(setup), setup.paths)
    outcome = experiment(setup, runner, replace(LOOSE, g2_dd_p_yes=1.01)).run()
    paths = setup.paths
    assert outcome["stopped_at"] == "g2"
    assert outcome["finding"] == G2_FINDING
    assert set(outcome["gates"]) == {TAG, "a0_one_seed", "a1_one_seed"}
    assert read_json(paths.root / "outcome.json")["stopped_at"] == "g2"
    assert G2_FINDING in (paths.root / "report.txt").read_text(encoding="utf-8")
    # Seed 0 only, DD and SHAM only, attempts 0 and 1 only.
    assert train_tasks(runner) == [f"{TAG}.train.{adapter_id(attempt, condition, 0)}" for attempt in (0, 1) for condition in ("DD", "SHAM")]
    for attempt in (0, 1):
        for condition in ("DD", "SHAM"):
            training = read_json(paths.adapter(TAG, adapter_id(attempt, condition, 0)) / "training.json")
            assert training["steps"] == setup.cfg.training.steps * 2**attempt
        gate = read_json(paths.gate_g2(TAG, attempt, "one_seed"))
        assert gate["passed"] is False and gate["checks"]["dd_says_yes"] is False
    assert not (paths.root / "results.json").exists()


def test_flip_and_honest_never_train_when_g2_fails_across_seeds(tmp_path, monkeypatch) -> None:
    """They are half of an attempt's adapters. One seed passes and the median across seeds fails, twice."""
    setup = prepare(tmp_path)
    real = analysis.evaluate_g2

    def fail_all_seeds(cfg: D1Config, paths: D1Paths, tag: str, attempt: int, seeds: Sequence[int], label: str, thresholds: Thresholds) -> dict:
        result = real(cfg, paths, tag, attempt, seeds, label, thresholds)
        if label != "all_seeds":
            return result
        return {**result, "passed": False, "checks": {**result["checks"], "dd_says_yes": False}}

    monkeypatch.setattr(analysis, "evaluate_g2", fail_all_seeds)
    runner = InProcessRunner(options_for(setup), setup.paths)
    outcome = experiment(setup, runner).run()
    assert outcome["stopped_at"] == "g2"
    assert train_tasks(runner) == [f"{TAG}.train.{adapter_id(attempt, condition, seed)}" for attempt in (0, 1) for seed in (0, 1) for condition in ("DD", "SHAM")]
    assert not any("FLIP" in name or "HONEST" in name for name in runner.ran)


def test_flip_and_honest_train_only_after_every_dd_and_sham_seed(full_run) -> None:
    names = train_tasks(full_run["runner"])
    controls = [index for index, name in enumerate(names) if "FLIP" in name or "HONEST" in name]
    main = [index for index, name in enumerate(names) if "_DD_" in name or "_SHAM_" in name]
    assert len(controls) == 4 and len(main) == 4
    assert max(main) < min(controls)
    assert read_json(full_run["setup"].paths.gate_g2(TAG, 0, "all_seeds"))["passed"] is True


def test_too_little_gpu_memory_is_a_failed_run_and_not_a_finding(tmp_path) -> None:
    setup = prepare(tmp_path)
    runner = InProcessRunner(options_for(setup), setup.paths)
    with pytest.raises(RunFailed, match="not a finding"):
        experiment(setup, runner, Thresholds(), gpu_free_gb=0.5).run()
    assert runner.ran == []
    assert not (setup.paths.root / "outcome.json").exists()
    assert not (setup.paths.root / "report.txt").exists()
    with pytest.raises(RunFailed, match="no final result"):
        finalize(setup.paths.root, tmp_path / "results", setup.config)


# The whole path


def test_the_tiny_plan_gives_the_gates_what_they_need(full_run) -> None:
    """If a change to the splits breaks the sizing this file relies on, this is the test that says so."""
    setup = full_run["setup"]
    plan = read_json(setup.paths.plan_dir(TAG) / "plan.json")
    assert plan["fit"]["n"] >= 32, "the dry run's small probe needs 32 rows"
    assert plan["fit_labels"]["0"] == plan["fit_labels"]["1"]
    assert plan["cal"]["n"] > 0
    assert plan["cal_families"] and set(plan["cal_families"]) <= TWO_CLASS_PROBE_FAMILIES
    assert set(read_meta(setup.paths, TAG, "cal")["y"]) == {0, 1}
    assert plan["test_available_per_class"] == {family: ROWS_PER_FAMILY for family in setup.cfg.partition["test"]}
    # Seven families, two classes, three per class, and the Yes., No. and base reads of each.
    assert plan["test_g1"]["n"] == 7 * 2 * setup.cfg.sizes.g1_per_class_per_family * 3
    assert plan["facts"]["n"] == len(setup.cfg.facts_eval)
    specificity = read_meta(setup.paths, TAG, "specificity")
    assert plan["specificity"]["n"] == len(specificity["y"]) > 0
    assert sum(specificity["y"]) == len(specificity["y"]) / 2, "half the correct answers are No"
    assert set(specificity["family"]) <= set(setup.cfg.partition["test"])


def test_full_path_completes_and_its_verdict_is_the_rule_on_its_own_numbers(full_run) -> None:
    """A random model decides no particular verdict. This shows every step runs and agrees. tests/test_d1_real_path.py checks verdicts."""
    setup, runner, outcome = full_run["setup"], full_run["runner"], full_run["outcome"]
    paths = setup.paths
    assert outcome["stopped_at"] is None
    assert (outcome["rung"], outcome["attempt"]) == (TAG, 0)
    assert read_json(paths.root / "outcome.json")["verdict"] == outcome["verdict"]
    results = read_json(paths.root / "results.json")
    primary = results["primary"]
    assert primary["verdict"] == outcome["verdict"]
    assert primary["verdict"] == primary_verdict(primary["lo"], primary["hi"], primary["auroc_treated_ci"]["lo"], primary["auroc_sham_ci"]["lo"], LOOSE)
    assert primary["lo"] <= primary["hi"]
    assert results["seeds"] == [0, 1]
    assert results["intervals"]["n_lines"] == 4
    assert results["conditions"]["DD"]["behaviour_change_on_training_task"]["n"] == results["n_transcripts"]
    for index in range(1, 10):
        assert any(key.startswith(f"secondary_{index}_") for key in results), index
    assert f"VERDICT: {outcome['verdict']}" in (paths.root / "report.txt").read_text(encoding="utf-8")
    assert read_json(paths.control)["stage"] == "finished"

    assert "prepare" not in runner.ran
    conditions = ("DD", "SHAM", "FLIP", "HONEST")
    assert set(train_tasks(runner)) == {f"{TAG}.train.{adapter_id(0, condition, seed)}" for condition in conditions for seed in (0, 1)}
    assert read_json(paths.gate_g2(TAG, 0, "one_seed"))["seeds"] == [0]
    assert read_json(paths.gate_g2(TAG, 0, "all_seeds"))["seeds"] == [0, 1]
    for condition in ("DD", "SHAM"):
        for seed in (0, 1):
            assert paths.probe(TAG, f"retrained_{adapter_id(0, condition, seed)}").exists()

    sizing = read_json(paths.sample_size(TAG))
    assert sizing["transcripts_per_arm"] == setup.cfg.sizes.test_transcripts_per_arm == results["n_transcripts"]
    assert set(sizing["per_class_per_family"].values()) == {TINY_PER_CLASS_PER_FAMILY}
    assert set(results["n_per_family"].values()) == {2 * TINY_PER_CLASS_PER_FAMILY}
    # Every arm reads the same transcripts: the unmodified model and each condition.
    assert sizing["expected_half_width_D_if_arms_independent"] > 0.0


def test_sample_size_is_logged_before_any_adapter_and_never_changes(full_run) -> None:
    setup, runner = full_run["setup"], full_run["runner"]
    assert runner.size_was_logged
    logged = setup.paths.sample_size(TAG).read_bytes()
    # With adapters on disk the file is read back, so a later launch cannot pick another number.
    assert experiment(setup, runner).fix_sample_size(TAG, {}) == json.loads(logged)["transcripts_per_arm"]
    assert setup.paths.sample_size(TAG).read_bytes() == logged


def test_sample_size_is_refused_once_adapters_exist_without_one(tmp_path) -> None:
    cfg = D1Config.from_file(write_tiny_config(tmp_path))
    paths = D1Paths(tmp_path / "out")
    paths.adapter(TAG, adapter_id(0, "DD", 0)).mkdir(parents=True)
    with pytest.raises(RunFailed, match="no sample size"):
        Experiment(cfg, paths, None, 1, None).fix_sample_size(TAG, {})
    assert not paths.sample_size(TAG).exists()


def test_a_run_that_died_resumes_runs_nothing_twice_and_reaches_the_same_result(full_run, tmp_path) -> None:
    setup = prepare(tmp_path)
    # Task 12 is DD's first read of the test sets, so this dies between two adapters of one stage.
    first = StoppingRunner(options_for(setup), setup.paths, 12)
    with pytest.raises(RunFailed, match="on purpose"):
        experiment(setup, first).run()
    assert len(first.ran) == 12
    assert not (setup.paths.root / "outcome.json").exists()

    second = InProcessRunner(options_for(setup), setup.paths)
    outcome = experiment(setup, second).run()
    assert not set(first.ran) & set(second.ran)
    assert set(first.ran) | set(second.ran) == set(full_run["runner"].ran)
    assert outcome["verdict"] == full_run["outcome"]["verdict"]
    resumed = read_json(setup.paths.root / "results.json")["primary"]
    straight = read_json(full_run["setup"].paths.root / "results.json")["primary"]
    for key in ("point", "lo", "hi"):
        assert resumed[key] == pytest.approx(straight[key], abs=1e-3), key
    assert read_json(setup.paths.sample_size(TAG))["transcripts_per_arm"] == read_json(full_run["setup"].paths.sample_size(TAG))["transcripts_per_arm"]


def stored_copy(full_run: dict, directory: Path, **changes: Any) -> Setup:
    """The finished run's state, under a config that differs only as asked: {section: {key: value}}."""
    setup = full_run["setup"]
    shutil.copytree(setup.paths.root, directory / "out")
    config = write_tiny_config(directory)
    raw = json.loads(setup.config.read_text(encoding="utf-8"))
    for section, values in changes.items():
        raw[section] = {**raw[section], **values}
    config.write_text(json.dumps(raw), encoding="utf-8")
    return Setup(config, D1Config.from_file(config), D1Paths(directory / "out"))


def test_an_analysis_config_edit_keeps_all_stored_work_and_reanalyzes_with_no_worker(full_run, tmp_path) -> None:
    setup = stored_copy(full_run, tmp_path, analysis={"n_bootstrap": 60})
    assert setup.cfg.config_hash != full_run["setup"].cfg.config_hash
    assert setup.cfg.capture_config_hash == full_run["setup"].cfg.capture_config_hash
    acts = store_hashes(setup.paths.capture_dir(TAG, BASE_MODEL, "test"))
    adapter = sha256_file(setup.paths.adapter(TAG, adapter_id(0, "DD", 0)) / "adapter.pt")
    outcome = experiment(setup, NoWorkRunner(setup.paths)).run()
    assert outcome["stopped_at"] is None
    assert store_hashes(setup.paths.capture_dir(TAG, BASE_MODEL, "test")) == acts
    assert sha256_file(setup.paths.adapter(TAG, adapter_id(0, "DD", 0)) / "adapter.pt") == adapter
    assert read_json(setup.paths.root / "results.json")["primary"]["point"] == pytest.approx(read_json(full_run["setup"].paths.root / "results.json")["primary"]["point"])


def test_a_gate_or_analysis_code_fix_is_not_seen_by_the_identity() -> None:
    for name in ("d1/analysis.py", "d1/gates.py", "d1/stats.py", "d1/baselines.py", "d1/orchestrator.py", "d1/preflight.py", "d1/cli.py", "d1/worker.py", "d1/capacity.py"):
        assert name not in run_identity.CAPTURE_MODULES, name
        assert name not in run_identity.PROBE_MODULES, name


def test_a_config_edit_that_changes_rows_refuses_the_run(full_run, tmp_path) -> None:
    setup = stored_copy(full_run, tmp_path, sizes={"g1_per_class_per_family": 2})
    with pytest.raises(RunFailed, match="decide rows, activations or adapters"):
        experiment(setup, NoWorkRunner(setup.paths)).check_identity()


def test_a_training_config_edit_refuses_the_run_but_micro_batch_does_not(full_run, tmp_path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    refused = stored_copy(full_run, tmp_path / "a", training={"steps": 3})
    with pytest.raises(RunFailed, match="decide rows, activations or adapters"):
        experiment(refused, NoWorkRunner(refused.paths)).check_identity()
    allowed = stored_copy(full_run, tmp_path / "b", training={"micro_batch": 2})
    experiment(allowed, NoWorkRunner(allowed.paths)).check_identity()


def test_code_that_changes_activations_or_adapters_refuses_the_run(full_run, tmp_path, monkeypatch) -> None:
    setup = stored_copy(full_run, tmp_path)
    real = run_identity.capture_fingerprint()
    monkeypatch.setattr(run_identity, "capture_fingerprint", lambda: {**real, "d1/lora.py": "edited"})
    with pytest.raises(RunFailed, match="d1/lora.py"):
        experiment(setup, NoWorkRunner(setup.paths)).check_identity()


def test_everything_that_makes_activations_or_adapters_is_in_the_identity() -> None:
    for name in ("d1/capture.py", "d1/lora.py", "d1/plan.py", "d1/produce.py", "d1/prompts.py", "d1/training_data.py", "d1/store.py", "d1/specificity.py", "d1/data.py", "modeling.py", "hooks.py", "prompts.py"):
        assert name in run_identity.CAPTURE_MODULES, name
    assert set(run_identity.capture_fingerprint()) == set(run_identity.CAPTURE_MODULES)


def test_a_probe_config_edit_refits_probes_and_touches_nothing_else(full_run, tmp_path) -> None:
    setup = stored_copy(full_run, tmp_path, probes={"max_epochs": 3})
    paths = setup.paths
    acts = store_hashes(paths.capture_dir(TAG, BASE_MODEL, "fit"))
    adapter_file = paths.adapter(TAG, adapter_id(0, "DD", 0)) / "adapter.pt"
    experiment(setup, NoWorkRunner(paths)).check_identity()
    assert not paths.probe(TAG, "linear").exists()
    assert not paths.done(f"{TAG}.fit.linear").exists()
    assert not paths.done(f"{TAG}.fit.retrained_{adapter_id(0, 'DD', 0)}").exists()
    assert paths.done(f"{TAG}.plan").exists()
    assert store_hashes(paths.capture_dir(TAG, BASE_MODEL, "fit")) == acts
    assert adapter_file.exists()
    # The next launch needs fits and nothing else, which a re-analysis says plainly.
    with pytest.raises(RunFailed, match="still has work to do"):
        experiment(setup, NoWorkRunner(paths)).run()


def test_a_recomputed_gate_that_flips_is_announced(full_run, tmp_path, capsys) -> None:
    setup = stored_copy(full_run, tmp_path)
    # G1 now fails on the same stored numbers, and the ladder has no rung left, so this ends as a finding with no worker.
    outcome = experiment(setup, NoWorkRunner(setup.paths), replace(LOOSE, g1_self_report_lower=1.01)).run()
    assert outcome["stopped_at"] == "g1"
    assert "A gate or analysis fix changed the decision" in capsys.readouterr().out


def test_finalize_lets_no_row_text_out_and_its_hashes_are_right(full_run, tmp_path) -> None:
    setup, outcome = full_run["setup"], full_run["outcome"]
    results = tmp_path / "results"
    written = finalize(setup.paths.root, results, setup.config)
    needle = SECRET.encode("utf-8")
    assert needle in setup.paths.transcripts.read_bytes(), "the rows must hold the marker, or this test checks nothing"
    files = [path for path in results.iterdir() if path.is_file()]
    for path in files:
        assert needle not in path.read_bytes(), path.name

    listed = {}
    for line in (results / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ")
        listed[name] = digest
    assert set(listed) == {path.name for path in files} - {"SHA256SUMS", "summary.txt"}
    for name, digest in listed.items():
        assert sha256_file(results / name) == digest, name
    for name in ("report.txt", "outcome.json", "results.json", "control.json", "data_manifest.json", "config.json"):
        assert name in listed, name
    for name in ("g1.json", "sample_size.json", "dry_run.json", "plan.json", "probe.linear.npz", "g2_a0_one_seed.json", "g2_a0_all_seeds.json"):
        assert f"{TAG}.{name}" in listed, name
    assert written["outcome"]["verdict"] == outcome["verdict"]
    assert "=====BEGIN D1 REPORT=====" in written["summary"]


def test_base_capture_is_untouched_by_adapters_and_matches_a_fresh_read(full_run, tmp_path) -> None:
    setup, runner = full_run["setup"], full_run["runner"]
    cfg, paths = setup.cfg, setup.paths
    directory = paths.capture_dir(TAG, BASE_MODEL, "test")
    assert runner.base_hashes == store_hashes(directory), "the stored base rows changed after adapters started training"

    # A new model reads the same rows in the same batches, with no adapter anywhere near it.
    stored = read_complete(directory)
    rung = cfg.rung(TAG)
    layers = layers_for(cfg.fractions, rung.n_layers)
    model, tokenizer, device = load_model(rung, FACTORY, False)
    answers = answer_tokens(paths, TAG)
    fresh = CaptureStore.create(tmp_path / "fresh", stored.n, len(layers), rung.d_model)
    run_capture(model, layers, paths.set_items(TAG, "test"), fresh, answers.yes, answers.no, tokenizer.eos_token_id, device, cfg.capture_token_budget)
    assert fresh.complete
    base_acts = np.asarray(stored.acts, dtype=np.float32)
    assert np.allclose(base_acts, np.asarray(fresh.acts, dtype=np.float32), rtol=ROUNDING, atol=1e-6)
    assert np.allclose(stored.logprobs, fresh.logprobs, rtol=0.0, atol=1e-4, equal_nan=True)

    # The same comparison tells an adapter's rows apart, so the match above is not an empty one.
    model_id = adapter_id(0, "DD", 0)
    assert read_json(paths.adapter(TAG, model_id) / "training.json")["delta_norm"] > 0.0
    through_adapter = read_complete(paths.capture_dir(TAG, model_id, "test"))
    assert not np.allclose(base_acts, np.asarray(through_adapter.acts, dtype=np.float32), rtol=ROUNDING, atol=1e-6)


# A real worker process


def test_worker_trains_in_its_own_process_with_the_import_check_on(tmp_path) -> None:
    """No --in-process here, so training checks that no probe, capture or analysis code is loaded, as a real run does."""
    setup = prepare(tmp_path)
    paths = setup.paths
    InProcessRunner(options_for(setup), paths).run("Plan", [Task(f"{TAG}.plan", ["plan", "--rung", TAG], gpu=False, loads_model=False)])
    model_id = adapter_id(0, "DD", 0)
    name = f"{TAG}.train.{model_id}"
    command = [
        sys.executable, "-m", "mechinterp.d1.worker", "train",
        "--config", str(setup.config), "--out", str(paths.root), "--task-name", name,
        "--rung", TAG, "--condition", "DD", "--seed", "0", "--attempt", "0", "--factory", FACTORY,
    ]  # fmt: skip
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join([str(SRC_DIR), str(TESTS_DIR)]), "CUDA_VISIBLE_DEVICES": "", "TOKENIZERS_PARALLELISM": "false"}
    result = subprocess.run(command, env=environment, cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    directory = paths.adapter(TAG, model_id)
    assert (directory / "adapter.pt").exists()
    assert not (directory / "resume.pt").exists()
    assert read_json(directory / "training.json")["steps"] == setup.cfg.training.steps
    assert read_json(paths.heartbeat(name))["done"] == setup.cfg.training.steps
