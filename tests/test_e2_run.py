"""Stage 1 and sharded Stages 2 and 3 on the tiny CPU model, in real worker processes. No GPU and no download."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from conftest import CONCEPTS
from e2_factory import build, write_e1_results, write_e2_config, write_passing_gate

from mechinterp.checkpoint import ResultStore, read_records
from mechinterp.design import PHASE_GRID, PHASE_IMPACT, PHASE_TEST
from mechinterp.e2 import cli
from mechinterp.e2.config import E2Config, e2_code_hash
from mechinterp.e2.design import plan_calibration
from mechinterp.e2.e1_inputs import E1ResultsMissing, load_e1_inputs
from mechinterp.e2.evalsets import EvalItem, readout_for, single_token_id, surface_forms, token_ids_for
from mechinterp.e2.families import compute_families
from mechinterp.e2.lens import LensError
from mechinterp.e2.orchestrator import E2WorkerPool, GateNotPassed, finalize, finalize_gate_failure, merge, orchestrate
from mechinterp.e2.paths import E2Paths
from mechinterp.e2.session import E2Session, impact_key
from mechinterp.e2.stage1 import EXIT_GATE_FAILED, _any_layer_pass, gate_path, run_stage1
from mechinterp.e2.worker import preflight_main
from mechinterp.sharding import MergeError, RunFailed, RunOptions, read_json, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = str(Path(__file__).resolve().parent)
FACTORY = "e2_factory:build"
SCRIPT = REPO_ROOT / "scripts" / "run_e2_gpu.sh"
DEVICE = torch.device("cpu")


def run_options(config_path: Path, out_dir: Path, n_workers: int) -> RunOptions:
    return RunOptions(
        config_path=config_path,
        out_dir=out_dir,
        devices=[""] * n_workers,
        factory=FACTORY,
        max_restarts=1,
        threads_per_worker=1,
        load_slots=n_workers,
        poll_seconds=0.1,
        progress_seconds=10.0,
        restart_delay_seconds=0.0,
        min_free_disk_bytes=0,
        extra_env={"PYTHONPATH": TESTS_DIR},
    )


def trials_of(records) -> dict[str, dict]:
    return {record["key"]: record for record in records if record.get("kind") == "trial"}


# Evaluation sets on a tokenizer


def _item(prompt: str, target: str | None, rule: str) -> EvalItem:
    return EvalItem("set", "name", prompt, target, ("contaa",), rule)


def test_before_target_reads_the_token_before_the_target_even_when_the_space_merges(tokenizer) -> None:
    readout = readout_for(tokenizer, _item("Tell me about ", "contaa", "before_target"))
    assert tokenizer.decode([readout.input_ids[-1]]).strip() == "about"
    assert readout.index == len(readout.input_ids) - 1
    alternative = readout_for(tokenizer, _item("Tell me about ", "contaa", "before_target"), alternative=True)
    assert alternative.input_ids != readout.input_ids


def test_last_newline_reads_the_last_newline_and_final_reads_the_end(tokenizer) -> None:
    readout = readout_for(tokenizer, _item("Tell\nme\nabout contaa", None, "last_newline"))
    assert tokenizer.decode([readout.input_ids[-1]]) == "\n"
    assert len(readout.input_ids) == 4  # Tell, newline, me, newline.
    final = readout_for(tokenizer, _item("Tell me about contaa", None, "final"))
    assert tokenizer.decode([final.input_ids[-1]]).strip() == "contaa"


def test_forms_keep_the_spelling_and_only_single_tokens_count(tokenizer) -> None:
    assert surface_forms("CIA") == ["CIA", "cIA", " CIA", " cIA"]
    assert " contaa" in surface_forms("contaa")
    assert single_token_id(tokenizer, " contaa") is not None
    assert single_token_id(tokenizer, "two-part") is None
    assert token_ids_for(tokenizer, ["missingword"]) == []
    assert token_ids_for(tokenizer, ["contaa"]) != []


def test_any_layer_metric_is_upstreams() -> None:
    rows = [{"f": [[11, 3], [50, 60]]}, {"f": [None, [11, 12]]}]
    assert _any_layer_pass(rows, "f", 10) == pytest.approx(0.25)


# Stage 1 and the gate's stop


@pytest.fixture(scope="module")
def stopped(tmp_path_factory) -> dict:
    """J = I makes the two lenses identical, so the gate must fail, whatever the tiny model does."""
    directory = tmp_path_factory.mktemp("stopped")
    config = write_e2_config(directory, "identity")
    out_dir = directory / "out"
    gate = run_stage1(config, out_dir, FACTORY)
    return {"config": config, "out": out_dir, "gate": gate}


def test_identical_lenses_fail_the_gate_and_gate_json_says_so(stopped) -> None:
    gate = stopped["gate"]
    assert gate["passed"] is False and gate["l_w"] is None and gate["band"] is None
    assert read_json(gate_path(stopped["out"]))["passed"] is False
    assert all(row["gain"] == 0.0 for row in gate["analysis"]["primary_english_table"])


def test_stage1_keeps_every_rank_and_reports_what_it_could_not_score(stopped) -> None:
    rows = [json.loads(line) for line in (stopped["out"] / "stage1" / "ranks.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 12
    assert {row["set"] for row in rows} == {"multihop", "poetry", "typo"}
    typo = stopped["gate"]["analysis"]["by_set"]["typo"]
    assert typo["n_intermediates_with_no_single_token_form"] == 4  # "missingword", on each of the four items.
    assert stopped["gate"]["analysis"]["chinese_targets"]["table"] is None
    checks = read_json(stopped["out"] / "stage1" / "lens_checks.json")
    assert checks["lens_file"]["stored_dtypes"] == ["torch.float16"]
    assert checks["final_norm_type"].endswith("RMSNorm")
    assert checks["unembed_gap_on_final_block"] <= 1e-3


def test_gate_failure_exits_with_its_own_code_and_a_clear_message(stopped, capsys) -> None:
    args = argparse.Namespace(config=str(stopped["config"]), out=str(stopped["out"]), factory=FACTORY)
    with pytest.raises(SystemExit) as caught:
        cli._stage1(args)
    assert caught.value.code == EXIT_GATE_FAILED
    output = capsys.readouterr().out
    assert "GATE FAILED" in output and "Stages 2 and 3 do not run" in output


def test_a_failed_gate_never_starts_a_worker(stopped, tmp_path, monkeypatch) -> None:
    def no_workers(*_args, **_kwargs):
        raise AssertionError("A worker started after a failed gate")

    monkeypatch.setattr(E2WorkerPool, "start_all", no_workers)
    e1_dir = write_e1_results(tmp_path / "e1", stopped["config"])
    with pytest.raises(GateNotPassed, match="gate failed"):
        orchestrate(run_options(stopped["config"], tmp_path / "out", 1), gate_path(stopped["out"]), e1_dir)
    assert not (tmp_path / "out" / "shards").exists()
    with pytest.raises(GateNotPassed, match="does not exist"):
        orchestrate(run_options(stopped["config"], tmp_path / "out", 1), tmp_path / "no_gate.json", e1_dir)


def test_the_final_folder_of_a_stopped_run_has_the_table_and_hashes(stopped, tmp_path) -> None:
    results = tmp_path / "results"
    outcome = finalize_gate_failure(stopped["out"], results, gate_path(stopped["out"]))
    assert "GATE FAILED" in outcome["summary"]
    for line in (results / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ")
        assert sha256_file(results / name) == digest
    assert (results / "gate.json").exists() and (results / "ranks.jsonl").exists()


def test_a_measured_gate_is_frozen_against_changed_config(stopped, tmp_path) -> None:
    moved = tmp_path / "copy"
    shutil.copytree(stopped["config"].parent, moved, ignore=shutil.ignore_patterns("out"))
    raw = json.loads((moved / "e2.json").read_text(encoding="utf-8"))
    raw["gate"]["k"] = 5
    (moved / "e2.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(LensError, match="not remeasured"):
        run_stage1(moved / "e2.json", stopped["out"], FACTORY)


def test_a_bad_lens_is_an_instrument_failure_and_not_a_gate_result(tmp_path, capsys) -> None:
    config = write_e2_config(tmp_path, "identity")
    (tmp_path / "lens.pt").write_bytes(b"not the pinned file")
    args = argparse.Namespace(config=str(config), out=str(tmp_path / "out"), factory=FACTORY)
    with pytest.raises(SystemExit) as caught:
        cli._stage1(args)
    assert caught.value.code == 1
    assert not gate_path(tmp_path / "out").exists()
    assert "LENS CHECK FAILED" in capsys.readouterr().out
    assert (tmp_path / "out" / "failure_reason.txt").exists()


def test_fitted_stub_passes_the_alignment_check(tmp_path) -> None:
    config = write_e2_config(tmp_path, "fitted")
    run_stage1(config, tmp_path / "out", FACTORY)
    checks = read_json(tmp_path / "out" / "stage1" / "lens_checks.json")
    assert checks["alignment"]["passed"] is True


# E1's results


def test_missing_or_unfinished_e1_results_give_a_clear_error(tmp_path) -> None:
    config = write_e2_config(tmp_path / "config", "identity")
    with pytest.raises(E1ResultsMissing, match="No E1 results"):
        load_e1_inputs(tmp_path / "nothing")
    write_e1_results(tmp_path / "e1", config, complete=False)
    with pytest.raises(E1ResultsMissing, match="has not finished"):
        load_e1_inputs(tmp_path / "e1")
    gate = write_passing_gate(tmp_path / "gate.json")
    with pytest.raises(RunFailed, match="No E1 results"):
        orchestrate(run_options(config, tmp_path / "out", 1), gate, tmp_path / "nothing")


def test_e1_inputs_are_read_per_concept(tmp_path) -> None:
    config = write_e2_config(tmp_path / "config", "identity")
    write_e1_results(tmp_path / "e1", config, layer=4, alpha=3.0)
    inputs = load_e1_inputs(tmp_path / "e1")
    assert (inputs["layer"], inputs["alpha"]) == (4, 3.0)
    assert inputs["L_inj"][CONCEPTS["test"][1]] == 2.0
    assert inputs["L_pick"][CONCEPTS["test"][1]] == 1.0


# Families and impact matching on a real tiny session


@pytest.fixture(scope="module")
def session(tmp_path_factory) -> E2Session:
    directory = tmp_path_factory.mktemp("session")
    config = write_e2_config(directory, "fitted")
    e2 = E2Config.from_file(config)
    model, tokenizer, device = build(None, 0)
    made = E2Session(e2, directory / "out", model=model, tokenizer=tokenizer, device=device, store=ResultStore(directory / "out" / "store.jsonl"))
    compute_families(made, {"sites": [0, 1], "l_s": 0, "l_w": 1, "e1_concept_vectors": None})
    made._ensure_vectors()
    return made


def test_every_family_exists_and_is_unit(session) -> None:
    concepts = [*session.stimuli.concepts_calibration, *session.stimuli.concepts_test]
    for concept in concepts:
        for layer in (0, 1):
            for family in ("F_full", "F_j", "F_res", "F_rand", "F_lens"):
                vector = session.family_vectors[f"{family}|{concept}|{layer}"]
                assert float(vector.norm()) == pytest.approx(1.0, abs=1e-4), family


def test_f_j_plus_f_res_rebuilds_the_full_vector_on_the_model(session) -> None:
    decomposition = read_json(session.out_dir / "decomposition.json")
    for layer in (0, 1):
        for concept, numbers in decomposition["by_site"][str(layer)].items():
            full = session.family_vectors[f"F_full|{concept}|{layer}"].double()
            j = session.family_vectors[f"F_j|{concept}|{layer}"].double()
            res = session.family_vectors[f"F_res|{concept}|{layer}"].double()
            rebuilt = numbers["component_norm"] * j + numbers["residual_norm"] * res
            assert torch.allclose(rebuilt, full, atol=1e-4)
            assert numbers["n_atoms"] <= 3
            assert 0.0 < numbers["j_share"] <= 1.0
            assert len(numbers["top_lens_tokens"]) == 5


def test_f_rand_is_e1s_direction_for_that_concept(session) -> None:
    from mechinterp.vectors import random_unit_vector

    concept = session.stimuli.concepts_test[0]
    assert torch.equal(session.family_vectors[f"F_rand|{concept}|0"], random_unit_vector(32, session.config.seed, concept))


def test_rows_inject_the_family_vector(session) -> None:
    concept = session.stimuli.concepts_calibration[0]
    trial = plan_calibration(session.stimuli, session.config, "F_j", 1, 1.0, PHASE_IMPACT)[0]
    assert trial.condition == "F_j"
    assert torch.equal(session._vector(trial), session.family_vectors[f"F_j|{trial.concept}|1"])
    assert concept in session.stimuli.concepts_calibration


def _fake_rows(session, monkeypatch, js_of_alpha) -> None:
    """Replace forward passes with a known response, so the search itself is what is tested."""
    seen = []

    def fake(trials, measure_js=False):
        new = []
        for trial in trials:
            seen.append(trial)
            if not session.store.has(trial.key):
                new.append({"key": trial.key, "kind": "trial", "phase": trial.phase, "js": js_of_alpha(trial)})
        session.store.append(new)

    monkeypatch.setattr(session, "_run_trials", fake)
    return seen


def test_search_finds_the_alpha_that_matches_on_calibration_stimuli_only(session, monkeypatch) -> None:
    # From alpha 1.5, the first doubling lands at JS 1.05, inside the tolerance.
    seen = _fake_rows(session, monkeypatch, lambda trial: 1.0 if trial.condition == "F_full" else 0.35 * trial.alpha)
    session._run_trials(plan_calibration(session.stimuli, session.config, "F_full", 0, 1.5, PHASE_GRID))
    session.impact_search("F_res", 0, 1.5, None)
    match = session.store.get(impact_key("F_res", 0))
    assert match["target_js"] == 1.0
    assert match["match_ok"] is True
    assert abs(match["achieved_js"] - 1.0) <= 0.1
    assert match["alpha_full"] == 1.5
    assert {trial.concept for trial in seen} <= set(session.stimuli.concepts_calibration)
    assert {trial.cluster for trial in seen} <= set(session.stimuli.clusters_calibration)
    assert {trial.task for trial in seen} == {"T_inj"}


def test_a_family_that_cannot_reach_the_target_is_reported_unmatched(session, monkeypatch) -> None:
    _fake_rows(session, monkeypatch, lambda trial: 1.0 if trial.condition == "F_full" else 0.01)
    session._run_trials(plan_calibration(session.stimuli, session.config, "F_full", 1, 1.5, PHASE_GRID))
    session.impact_search("F_rand", 1, 1.5, None)
    match = session.store.get(impact_key("F_rand", 1))
    assert match["match_ok"] is False
    assert match["relative_error"] > 0.1
    assert match["exclusion_tolerance"] == 0.1
    assert len(match["history"]) <= session.config.impact_search.max_evals


def test_tolerance_between_search_and_exclusion_counts_as_matched(session, monkeypatch) -> None:
    # 8% off: above E1's large-tier search tolerance of 5%, and inside the prereg's 10%.
    tight = replace(session.config, impact_search=replace(session.config.impact_search, tolerance=0.05))
    monkeypatch.setattr(session, "config", tight)
    _fake_rows(session, monkeypatch, lambda trial: 1.0 if trial.condition == "F_full" else 0.92)
    session._run_trials(plan_calibration(session.stimuli, session.config, "F_full", 1, 2.5, PHASE_GRID))
    session.impact_search("F_lens", 1, 2.5, [session.stimuli.concepts_calibration[0]])
    match = session.store.get(impact_key("F_lens", 1))
    assert match["match_ok"] is True
    assert match["search_tolerance"] == 0.05
    assert len(match["history"]) > 1  # It kept searching past 8%, then accepted the best it found.
    assert match["concepts"] == [session.stimuli.concepts_calibration[0]]


# The whole sharded run


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> dict:
    directory = tmp_path_factory.mktemp("runs")
    config = write_e2_config(directory, "fitted")
    gate = write_passing_gate(directory / "gate.json", l_w=1)
    e1_dir = write_e1_results(directory / "e1", config, layer=0, alpha=2.0)
    single = orchestrate(run_options(config, directory / "one", 1), gate, e1_dir)
    double = orchestrate(run_options(config, directory / "two", 2), gate, e1_dir)
    return {
        "config": config,
        "gate": gate,
        "e1": e1_dir,
        "one": directory / "one",
        "two": directory / "two",
        "single": single,
        "double": double,
        "one_records": read_records(directory / "one" / "results.jsonl"),
        "two_records": read_records(directory / "two" / "results.jsonl"),
    }


def test_two_shards_run_exactly_the_trials_of_one(runs) -> None:
    one = trials_of(runs["one_records"])
    two = trials_of(runs["two_records"])
    assert one.keys() == two.keys()
    assert {record["phase"] for record in one.values()} == {PHASE_GRID, PHASE_IMPACT, PHASE_TEST}
    assert len(list((runs["two"] / "shards").glob("w*of2.jsonl"))) == 2


def test_two_shards_give_the_values_of_one_within_rounding(runs) -> None:
    one = trials_of(runs["one_records"])
    two = trials_of(runs["two_records"])
    worst = max(abs(one[key]["L"] - two[key]["L"]) for key in one)
    assert worst < 1e-3


def test_every_task_family_and_site_ran_on_the_test_cross(runs) -> None:
    rows = [record for record in runs["two_records"] if record.get("kind") == "trial" and record["phase"] == PHASE_TEST]
    assert {record["task"] for record in rows} == {"T_inj", "T_pick", "T_odd"}
    assert {record["condition"] for record in rows} == {"F_full", "F_j", "F_res", "F_lens", "F_rand"}
    assert {record["layer"] for record in rows} == {0, 1}
    assert len(rows) == 2 * 5 * 3 * 2 * 2 * 5  # Sites, families, tasks, concepts, clusters, positions.


def test_f_full_uses_e1s_alpha_at_l_s_and_a_calibration_alpha_at_l_w(runs) -> None:
    summary = read_json(runs["two"] / "plan_summary.json")
    assert summary["alphas"]["F_full|0"] == 2.0
    assert summary["alphas"]["F_full|1"] in (1.0, 2.0)
    selection = next(record for record in runs["two_records"] if record.get("kind") == "selection")
    assert selection["layer"] == 1 and selection["alpha"] == summary["alphas"]["F_full|1"]
    calibration = {record["concept"] for record in runs["two_records"] if record.get("kind") == "trial" and record["phase"] in (PHASE_GRID, PHASE_IMPACT)}
    test_concepts = {record["concept"] for record in runs["two_records"] if record.get("kind") == "trial" and record["phase"] == PHASE_TEST}
    assert not calibration & test_concepts


def test_rows_carry_e2s_identity(runs) -> None:
    manifest = next(record for record in runs["two_records"] if record.get("kind") == "manifest")
    assert manifest["identity"]["e2_code_hash"] == e2_code_hash()
    assert manifest["identity"]["e2_config_hash"] == E2Config.from_file(runs["config"]).e2_config_hash


def test_stage2_reports_at_l_s_from_e1s_results(runs) -> None:
    report = read_json(runs["two"] / "stage2.json")
    assert report["l_s"] == 0 and report["l_w"] == 1
    assert report["correlations_at_l_s"]["J_share_vs_L_inj"]["n_concepts"] == 2


def test_finalize_writes_the_analysis_and_correct_hashes(runs, tmp_path) -> None:
    results = tmp_path / "results"
    outcome = finalize(runs["two"], results, runs["gate"])
    analysis = read_json(results / "analysis.json")
    assert set(analysis["classification"]) == {f"{site}|{task}" for site in (0, 1) for task in ("T_inj", "T_pick", "T_odd")}
    assert analysis["single_token_n_test"] == 2
    assert "=====BEGIN E2 ANALYSIS=====" in outcome["summary"]
    for line in (results / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ")
        assert sha256_file(results / name) == digest
    for name in ("results.jsonl", "decomposition.json", "stage2.json", "e1_inputs.json", "gate.json", "merge_report.json"):
        assert (results / name).exists(), name


def test_rerun_of_a_complete_run_starts_nothing(runs) -> None:
    before = (runs["two"] / "results.jsonl").read_bytes()
    report = orchestrate(run_options(runs["config"], runs["two"], 2), runs["gate"], runs["e1"])
    assert report["already_complete"] is True
    assert (runs["two"] / "results.jsonl").read_bytes() == before


def test_merge_refuses_a_missing_trial(runs, tmp_path) -> None:
    copy = tmp_path / "copy"
    shutil.copytree(runs["two"] / "shards", copy / "shards")
    shard = sorted((copy / "shards").glob("*.jsonl"))[0]
    lines = shard.read_bytes().splitlines(keepends=True)
    kept = [line for line in lines if b'"phase":"test"' not in line or line is not next(item for item in lines if b'"phase":"test"' in item)]
    shard.write_bytes(b"".join(kept))
    from mechinterp.e2.orchestrator import E2Plan, alpha_table, check_impact_matches
    from mechinterp.sharding import RunPlan, collect

    e2 = E2Config.from_file(runs["config"])
    summary = read_json(runs["two"] / "plan_summary.json")
    setup = read_json(runs["two"] / "setup" / "w0.json")
    plan = E2Plan(RunPlan.from_setup(e2.base, setup), e2, 0, 2.0, 1, tuple(summary["lens_calibration"]), tuple(summary["lens_test"]))
    matches = check_impact_matches(collect(sorted((copy / "shards").glob("*.jsonl"))), plan)
    assert alpha_table(matches, {"0": 2.0, "1": summary["alphas"]["F_full|1"]}) == summary["alphas"]
    with pytest.raises(MergeError, match="missing"):
        merge(E2Paths(copy), plan, None, summary["alphas"], ["impact", "test"], complete=True)


def test_store_from_other_code_is_refused(tmp_path) -> None:
    config = write_e2_config(tmp_path, "identity")
    e2 = E2Config.from_file(config)
    model, tokenizer, device = build(None, 0)
    store_path = tmp_path / "out" / "store.jsonl"
    E2Session(e2, tmp_path / "out", model=model, tokenizer=tokenizer, device=device, store=ResultStore(store_path))
    lines = store_path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["identity"]["e2_code_hash"] = "older"
    store_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="e2_code_hash"):
        E2Session(e2, tmp_path / "out", model=model, tokenizer=tokenizer, device=device, store=ResultStore(store_path))


def test_preflight_measures_and_estimates_without_touching_results(tmp_path) -> None:
    config = write_e2_config(tmp_path, "identity")
    preflight_main(config, tmp_path / "out", 2, 1e-3, 0, 1, FACTORY)
    report = read_json(tmp_path / "out" / "preflight" / "dry_run.json")
    assert report["estimated_hours_at_most"] > 0
    assert report["e2_counts"]["test"] > 0
    assert report["n_single_token_concepts_test"] == 2
    assert not (tmp_path / "out" / "results.jsonl").exists()


def test_orchestrator_stays_free_of_torch() -> None:
    code = "import sys; sys.path.insert(0, sys.argv[1])\nimport mechinterp.e2.orchestrator\nassert 'torch' not in sys.modules\n"
    subprocess.run([sys.executable, "-c", code, str(REPO_ROOT / "src")], check=True, env={"PATH": os.environ.get("PATH", "")})


# The script


def script_env(tmp_path: Path) -> dict[str, str]:
    return {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "E2_E1_STATE_DIR": str(tmp_path / "e1"),
        "E2_STATE_DIR": str(tmp_path / "e2"),
        "E2_RESULTS_DIR": str(tmp_path / "copy"),
    }


def run_script(tmp_path: Path, timeout: float = 60) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(SCRIPT)], env=script_env(tmp_path), capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)


def test_script_has_lf_endings_strict_mode_and_valid_syntax() -> None:
    raw = SCRIPT.read_bytes()
    assert b"\r\n" not in raw
    assert b"set -Eeuo pipefail" in raw
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_script_never_pushes_and_never_needs_a_token() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "git push" not in text
    assert "HF_TOKEN=" not in text
    assert "unset HF_TOKEN" in text


@pytest.mark.parametrize("status", ["RUNNING", "STARTING", "INTERRUPTED", "FAILED"])
def test_script_refuses_while_e1_is_not_done(tmp_path, status: str) -> None:
    (tmp_path / "e1").mkdir()
    (tmp_path / "e1" / "status").write_text(f"{status}\n", encoding="utf-8")
    result = run_script(tmp_path)
    assert result.returncode == 3
    assert "REFUSED" in result.stdout and status in result.stdout
    assert not (tmp_path / "e2").exists()


def test_script_refuses_without_any_e1_state(tmp_path) -> None:
    result = run_script(tmp_path)
    assert result.returncode == 3
    assert "no E1 state" in result.stdout


@pytest.mark.skipif(shutil.which("flock") is None, reason="needs flock")
def test_script_refuses_while_e1_holds_its_lock_even_if_its_status_says_done(tmp_path) -> None:
    (tmp_path / "e1").mkdir()
    (tmp_path / "e1" / "status").write_text("DONE\n", encoding="utf-8")
    lock = tmp_path / "e1" / "run.lock"
    lock.touch()
    holder = subprocess.Popen(["flock", "-n", str(lock), "sleep", "30"])
    try:
        for _ in range(50):
            probe = subprocess.run(["flock", "-n", str(lock), "true"])
            if probe.returncode != 0:
                break
        result = run_script(tmp_path)
    finally:
        holder.kill()
        holder.wait()
    assert result.returncode == 3
    assert "still running" in result.stdout


def test_script_prints_a_final_gate_failure_and_starts_nothing(tmp_path) -> None:
    (tmp_path / "e2").mkdir()
    (tmp_path / "e2" / "status").write_text("GATE_FAILED\nbanner text\n", encoding="utf-8")
    result = run_script(tmp_path)
    assert result.returncode == 0
    assert "already finished as GATE_FAILED" in result.stdout and "banner text" in result.stdout


def test_failed_e2_needs_e2_retry_and_rechecks_e1(tmp_path) -> None:
    (tmp_path / "e2").mkdir()
    (tmp_path / "e2" / "status").write_text("FAILED\nreason\n", encoding="utf-8")
    assert run_script(tmp_path).returncode == 1
    env = {**script_env(tmp_path), "E2_RETRY": "1"}
    retried = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
    assert retried.returncode == 3
    assert "REFUSED" in retried.stdout
