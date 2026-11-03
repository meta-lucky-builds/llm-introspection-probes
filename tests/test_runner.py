"""End to end on a tiny random model and tokenizer."""

from __future__ import annotations

import json
from collections import Counter

import pytest
import torch
from conftest import tiny_config_dict, write_tiny_data

from mechinterp.analysis import analyze
from mechinterp.checkpoint import ResultStore
from mechinterp.config import Config
from mechinterp.export import build_export, decode_export
from mechinterp.runner import Session, SegmentTimeUp

DEVICE = torch.device("cpu")


def _session(tmp_path, model, tokenizer, **design_overrides) -> Session:
    raw = tiny_config_dict(write_tiny_data(tmp_path / "data"))
    raw["design"].update(design_overrides)
    config = Config.from_dict(raw, tmp_path)
    return Session(config, tmp_path / "out", model=model, tokenizer=tokenizer, device=DEVICE)


def _file_records(session: Session) -> list[dict]:
    return [json.loads(line) for line in session.store.path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def finished(tmp_path_factory, model, tokenizer) -> Session:
    session = _session(tmp_path_factory.mktemp("run"), model, tokenizer)
    status = session.run()
    assert status["complete"] is True
    return session


def test_sham_check_is_zero(finished) -> None:
    assert finished.store.get("sham")["max_abs_L"] <= 1e-4


def test_calibration_is_finished_before_any_test_data(finished) -> None:
    records = _file_records(finished)
    test_clusters = set(finished.stimuli.clusters_test)
    index = {record["key"]: i for i, record in enumerate(records)}
    first_test = min(
        i for i, record in enumerate(records) if record.get("cluster") in test_clusters
    )
    assert index["selection"] < first_test
    assert index["impact_match"] < first_test


def test_selection_is_reproducible_from_calibration_records_only(finished) -> None:
    selection = finished.store.get("selection")
    grid = [r for r in finished.store.records("trial") if r["phase"] == "calibration_grid"]
    assert {r["concept"] for r in grid} <= set(finished.stimuli.concepts_calibration)
    assert {r["cluster"] for r in grid} <= set(finished.stimuli.clusters_calibration)
    best = max(selection["grid_accuracy"].values())
    assert selection["accuracy"] == best


def test_impact_match_is_recorded_with_target_and_achieved(finished) -> None:
    match = finished.store.get("impact_match")
    for field in ("target_js", "achieved_js", "alpha_impact", "scale", "match_ok", "history"):
        assert field in match
    assert match["history"]
    impact_trials = [r for r in finished.store.records("trial") if r["phase"] == "test" and r["condition"] == "random_impact"]
    assert {r["alpha"] for r in impact_trials} == {match["alpha_impact"]}


def test_impact_target_uses_calibration_concept_trials(finished) -> None:
    selection = finished.store.get("selection")
    values = [
        r["js"]
        for r in finished.store.records("trial")
        if r["phase"] == "calibration_grid" and r["layer"] == selection["layer"] and r["alpha"] == selection["alpha"]
    ]
    assert finished.store.get("impact_match")["target_js"] == pytest.approx(sum(values) / len(values))


def test_l_is_reproduced_from_the_stored_clean_logits(finished) -> None:
    trial = next(r for r in finished.store.records("trial") if r["phase"] == "test" and r["task"] == "T_pick")
    clean = finished.store.get(f"clean|T_pick|{trial['cluster']}|qNone")
    logits = clean["logits"][:10]
    j = trial["position"]
    m_clean = logits[j] - (sum(logits) - logits[j]) / 9
    assert trial["m_clean"] == pytest.approx(m_clean)
    assert trial["L"] == pytest.approx(trial["m_injected"] - m_clean)


def test_fact_trials_query_a_different_position_with_a_clean_baseline(finished) -> None:
    trials = [r for r in finished.store.records("trial") if r["phase"] == "test" and r["task"] == "T_fact"]
    assert trials
    for trial in trials:
        assert trial["query"] != trial["position"]
        assert finished.store.has(f"clean|T_fact|{trial['cluster']}|q{trial['query']}")
    for cluster in finished.stimuli.clusters_test:
        for query in range(10):
            assert finished.store.has(f"clean|T_fact|{cluster}|q{query}")
    assert Counter(trial["query"] for trial in trials).keys() == set(range(10))


def test_test_positions_are_balanced(finished) -> None:
    export = build_export(finished.store.records())
    assert export["position_counts"]
    for counts in export["position_counts"].values():
        assert len(set(counts)) == 1 and counts[0] > 0


def test_clean_and_injected_runs_use_identical_tokens(finished) -> None:
    trial = next(r for r in finished.store.records("trial") if r["phase"] == "test" and r["task"] == "T_inj")
    prompt = finished._prompt(("T_inj", trial["cluster"], None))
    assert len(prompt.input_ids) == finished.store.get(f"clean|T_inj|{trial['cluster']}|qNone")["length"]


def test_manifest_records_the_run(finished) -> None:
    manifest = finished.store.get("manifest|0")
    assert manifest["config_hash"] == finished.config.config_hash
    assert {"torch", "transformers", "numpy", "python"} <= set(manifest["versions"])
    assert manifest["seed"] == 7
    assert "testshort" in manifest["dropped_clusters"]
    assert manifest["n_params"] > 0
    assert len(manifest["tokens"]["labels"]) == 10


def test_dropped_words_are_recorded(finished) -> None:
    dropped = finished.store.get("manifest|0")["dropped_words"]
    assert "two-part" in dropped["calone"] and "missingword" in dropped["calone"]


def test_export_and_analysis_run_on_real_results(finished) -> None:
    export = decode_export(finished.export_text())
    assert export["meta"]["complete"] is True
    report = analyze(export)
    assert report["outcome"] in {"V1_fail", "V2_fail", "capture", "task_gated", "mixed"}
    assert export["screen"]
    assert {row[0] for row in export["cells"]} == {"T_inj", "T_pick", "T_fav", "T_odd", "T_fact", "T_inj_none"}


def test_resume_repeats_no_finished_trial(finished, model, tokenizer) -> None:
    trials_before = {r["key"] for r in finished.store.records("trial")}
    again = Session(finished.config, finished.out_dir, model=model, tokenizer=tokenizer, device=DEVICE)
    assert again.run()["complete"] is True
    assert {r["key"] for r in again.store.records("trial")} == trials_before
    keys = [record["key"] for record in _file_records(again)]
    assert len(keys) == len(set(keys))


def test_other_config_cannot_reuse_results(finished, model, tokenizer) -> None:
    raw = tiny_config_dict(finished.config.data_dir)
    raw["seed"] = 8
    other = Config.from_dict(raw, finished.out_dir)
    with pytest.raises(ValueError):
        Session(other, finished.out_dir, model=model, tokenizer=tokenizer, device=DEVICE)


def test_interrupted_run_finishes_with_the_full_plan(tmp_path, model, tokenizer, monkeypatch) -> None:
    session = _session(tmp_path, model, tokenizer)
    calls = {"n": 0}
    original = session._check_time

    def interrupt_after_some_batches() -> None:
        calls["n"] += 1
        if calls["n"] > 12:
            raise SegmentTimeUp()
        original()

    monkeypatch.setattr(session, "_check_time", interrupt_after_some_batches)
    status = session.run()
    assert status["complete"] is False
    partial = len(session.store)
    assert partial > 0
    assert analyze(build_export(session.store.records()))["outcome"] == "incomplete"

    # A new process: the store is read from disk.
    resumed = Session(session.config, session.out_dir, model=model, tokenizer=tokenizer, device=DEVICE)
    assert len(ResultStore(resumed.store.path)) >= partial
    assert resumed.run()["complete"] is True
    export = build_export(resumed.store.records())
    assert export["meta"]["complete"] is True
    assert all(len(set(counts)) == 1 for counts in export["position_counts"].values())


def test_zero_budget_stops_cleanly(tmp_path, model, tokenizer) -> None:
    session = _session(tmp_path, model, tokenizer)
    assert session.run(max_seconds=0)["complete"] is False


def test_dry_run_estimates_without_writing_trials(tmp_path, model, tokenizer) -> None:
    session = _session(tmp_path, model, tokenizer)
    before = len(session.store)
    estimate = session.dry_run()
    assert len(session.store) == before
    assert estimate["seconds_per_prompt"] > 0
    assert estimate["prompt_counts"]["test"] > 0
    assert estimate["estimated_hours_at_most"] > 0


def test_missing_none_token_drops_only_that_task_and_records_it(tmp_path, model, tokenizer, monkeypatch) -> None:
    monkeypatch.setattr("mechinterp.runner.resolve_none_id", lambda *_: None)
    session = _session(tmp_path, model, tokenizer)
    assert "T_inj_none" not in session.test_tasks
    assert session.store.get("manifest|0")["dropped_tasks"] == ["T_inj_none"]
    assert session.run()["complete"] is True
    assert not [r for r in session.store.records("trial") if r["task"] == "T_inj_none"]


def test_grid_layer_outside_the_model_is_rejected(tmp_path, model, tokenizer) -> None:
    raw = tiny_config_dict(write_tiny_data(tmp_path / "data"))
    raw["grid"]["layers"] = [99]
    with pytest.raises(ValueError):
        Session(Config.from_dict(raw, tmp_path), tmp_path / "out", model=model, tokenizer=tokenizer, device=DEVICE)
