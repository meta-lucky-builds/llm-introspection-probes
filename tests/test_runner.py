"""End to end on a tiny random model and tokenizer."""

from __future__ import annotations

import json
from collections import Counter
from types import SimpleNamespace

import pytest
import torch
from conftest import tiny_config_dict, write_tiny_data
from torch import nn

import mechinterp.runner as runner_module
from mechinterp.analysis import analyze
from mechinterp.checkpoint import ResultStore
from mechinterp.config import Config
from mechinterp.design import select_layer_alpha
from mechinterp.export import build_export, decode_export
from mechinterp.prompts import build_prompt, token_hash
from mechinterp.runner import SegmentTimeUp, Session
from mechinterp.stats import NonFiniteError

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


class _ContextLayer(nn.Module):
    """Mixes in the previous token, so a token's activation depends on what came before it."""

    def forward(self, hidden: torch.Tensor):
        return (mix_previous(hidden),)


def mix_previous(hidden: torch.Tensor) -> torch.Tensor:
    previous = torch.cat([torch.zeros_like(hidden[:, :1]), hidden[:, :-1]], dim=1)
    return hidden + 0.5 * previous


class PositionReadingModel(nn.Module):
    """A stub whose logit for label j rises only if the token after "j:" was perturbed.

    It finds the perturbed token by comparing the hooked layer's output with what that layer computes
    unperturbed, then reads the label from the two tokens before it in the prompt itself. So it knows
    nothing about the harness's position mapping, and any wrong mapping gives L other than the boost.
    """

    BOOST = 5.0

    def __init__(self, vocab_size: int, colon_id: int) -> None:
        super().__init__()
        torch.manual_seed(1)
        self.embedding = nn.Embedding(vocab_size, 16)
        self.layers = nn.ModuleList([_ContextLayer()])
        self.colon_id = colon_id
        self.prior = 0.1 * torch.randn(vocab_size)
        self.perturbed_tokens: list[list[int]] = []

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding

    def forward(self, input_ids, attention_mask=None, use_cache=None, logits_to_keep=0):
        embedded = self.embedding(input_ids)
        expected = mix_previous(embedded)
        hidden = self.layers[0](embedded)[0]
        changed = (hidden - expected).norm(dim=-1) > 1e-6  # [batch, length]
        logits = self.prior.repeat(input_ids.size(0), 1)  # [batch, vocab]
        rows = []
        for row in range(input_ids.size(0)):
            tokens = changed[row].nonzero().flatten().tolist()
            assert len(tokens) <= 1, "Injection touched more than one token"
            rows.append(tokens)
            if tokens and input_ids[row, tokens[0] - 1] == self.colon_id:
                logits[row, input_ids[row, tokens[0] - 2]] += self.BOOST
        self.perturbed_tokens.extend(rows)
        return SimpleNamespace(logits=logits.unsqueeze(1))


@pytest.fixture(scope="module")
def stub_run(tmp_path_factory, tokenizer) -> tuple[Session, PositionReadingModel]:
    tmp_path = tmp_path_factory.mktemp("stub")
    raw = tiny_config_dict(write_tiny_data(tmp_path / "data"))
    raw["grid"] = {"layers": [0], "alphas": [1.0, 2.0]}
    colon_id = tokenizer("0:", add_special_tokens=False)["input_ids"][1]
    stub = PositionReadingModel(len(tokenizer), colon_id)
    session = Session(Config.from_dict(raw, tmp_path), tmp_path / "out", model=stub, tokenizer=tokenizer, device=DEVICE)
    assert session.run()["complete"] is True
    return session, stub


def test_rows_report_the_token_that_was_actually_injected(stub_run) -> None:
    """Every phase and task, including T_fact, whose instruction repeats a list word."""
    session, _ = stub_run
    trials = list(session.store.records("trial"))
    assert {trial["task"] for trial in trials} == {"T_inj", "T_pick", "T_fav", "T_odd", "T_fact", "T_inj_none"}
    for trial in trials:
        prompt = session._prompt((trial["task"], trial["cluster"], trial["query"]))
        assert trial["injected_token"] == prompt.positions[trial["position"]]
        assert trial["L"] == pytest.approx(PositionReadingModel.BOOST), trial["key"]
        assert trial["hit"] is True, trial["key"]


def test_stub_saw_exactly_the_tokens_the_rows_report(stub_run) -> None:
    session, stub = stub_run
    perturbed = Counter(tokens[0] for tokens in stub.perturbed_tokens if tokens)
    reported = Counter(trial["injected_token"] for trial in session.store.records("trial"))
    # The sham check has alpha 0 and clean runs have no injection, so neither adds to this.
    assert perturbed == reported


def test_runner_mapping_fails_this_test_if_positions_are_off_by_one(tmp_path, tokenizer, monkeypatch) -> None:
    """The stub really detects a wrong mapping: shift every position and L is no longer the boost."""
    raw = tiny_config_dict(write_tiny_data(tmp_path / "data"))
    raw["grid"] = {"layers": [0], "alphas": [1.0]}
    colon_id = tokenizer("0:", add_special_tokens=False)["input_ids"][1]
    stub = PositionReadingModel(len(tokenizer), colon_id)
    session = Session(Config.from_dict(raw, tmp_path), tmp_path / "out", model=stub, tokenizer=tokenizer, device=DEVICE)
    original = session._prompt

    def shifted(clean_key):
        prompt = original(clean_key)
        positions = (*prompt.positions[1:], prompt.positions[0])
        return SimpleNamespace(input_ids=prompt.input_ids, positions=positions, text=prompt.text)

    monkeypatch.setattr(session, "_prompt", shifted)
    session.run()
    trial = next(session.store.records("trial"))
    assert trial["L"] != pytest.approx(PositionReadingModel.BOOST)
    assert trial["hit"] is False


def test_sham_check_is_zero_and_goes_through_the_hook(tmp_path, model, tokenizer, monkeypatch) -> None:
    session = _session(tmp_path, model, tokenizer)
    calls = []
    original = runner_module.inject

    def counting_inject(layer, positions, vectors, alphas):
        calls.append((layer, alphas.clone()))
        return original(layer, positions, vectors, alphas)

    monkeypatch.setattr(runner_module, "inject", counting_inject)
    session._sham_check()
    assert len(calls) == 1
    assert calls[0][0] is session.layers[session.config.grid.layers[0]]
    assert torch.equal(calls[0][1], torch.zeros_like(calls[0][1]))
    assert session.store.get("sham")["max_abs_L"] <= 1e-4


def test_sham_check_fails_when_the_injected_path_changes_logits(tmp_path, model, tokenizer, monkeypatch) -> None:
    session = _session(tmp_path, model, tokenizer)
    original = session._forward

    def leaky(input_ids, injection):
        logits = original(input_ids, injection)
        return logits if injection is None else logits + torch.rand_like(logits)

    monkeypatch.setattr(session, "_forward", leaky)
    with pytest.raises(RuntimeError):
        session._sham_check()


def test_calibration_is_finished_before_any_test_data(finished) -> None:
    records = _file_records(finished)
    test_clusters = set(finished.stimuli.clusters_test)
    index = {record["key"]: i for i, record in enumerate(records)}
    first_test = min(i for i, record in enumerate(records) if record.get("cluster") in test_clusters)
    assert index["selection"] < first_test
    assert index["impact_match"] < first_test


def test_selection_is_reproducible_from_calibration_records_only(finished) -> None:
    stored = finished.store.get("selection")
    trials = list(finished.store.records("trial"))
    grid = [trial for trial in trials if trial["phase"] == "calibration_grid"]
    recomputed = select_layer_alpha(
        grid, finished.stimuli.concepts_calibration, list(finished.stimuli.clusters_calibration)
    )
    assert (recomputed.layer, recomputed.alpha, recomputed.accuracy) == (
        stored["layer"],
        stored["alpha"],
        stored["accuracy"],
    )
    assert recomputed.grid_accuracy == stored["grid_accuracy"]
    # Test rows are refused, so they cannot have contributed.
    test_row = next(trial for trial in trials if trial["phase"] == "test")
    with pytest.raises(AssertionError):
        select_layer_alpha(
            [*grid, test_row], finished.stimuli.concepts_calibration, list(finished.stimuli.clusters_calibration)
        )


def test_impact_match_is_recorded_with_target_and_achieved(finished) -> None:
    match = finished.store.get("impact_match")
    for field in ("target_js", "achieved_js", "alpha_impact", "scale", "match_ok", "history", "tolerance"):
        assert field in match
    assert match["history"]
    impact_trials = [
        r for r in finished.store.records("trial") if r["phase"] == "test" and r["condition"] == "random_impact"
    ]
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


def test_clean_and_injected_runs_use_identical_tokens(tmp_path, model, tokenizer, monkeypatch) -> None:
    """Compare the token tensors that reached the model, and both against a fresh rendering."""
    session = _session(tmp_path, model, tokenizer)
    clean_rows: set[tuple[int, ...]] = set()
    injected_rows: set[tuple[int, ...]] = set()
    original = session._forward

    def recording(input_ids, injection):
        target = clean_rows if injection is None else injected_rows
        target.update(tuple(row) for row in input_ids.tolist())
        return original(input_ids, injection)

    monkeypatch.setattr(session, "_forward", recording)
    assert session.run()["complete"] is True
    assert injected_rows and injected_rows <= clean_rows

    words = session.stimuli.all_clusters
    for trial in session.store.records("trial"):
        fresh = build_prompt(tokenizer, trial["task"], trial["cluster"], words[trial["cluster"]], trial["query"])
        assert tuple(fresh.input_ids) in injected_rows
        clean = session.store.get(f"clean|{trial['task']}|{trial['cluster']}|q{trial['query']}")
        assert trial["prompt_hash"] == clean["prompt_hash"] == token_hash(fresh.input_ids)


def test_mismatched_clean_tokens_are_refused(tmp_path, model, tokenizer) -> None:
    session = _session(tmp_path, model, tokenizer)
    key = ("T_inj", next(iter(session.stimuli.clusters_calibration)), None)
    session._run_clean([key])
    session.store.get(f"clean|{key[0]}|{key[1]}|qNone")["prompt_hash"] = "different"
    with pytest.raises(AssertionError):
        session._clean_answer_logits(key)


def test_manifest_records_the_run(finished) -> None:
    manifest = finished.store.get("manifest|0")
    assert manifest["config_hash"] == finished.config.config_hash
    assert {"torch", "transformers", "numpy", "python"} <= set(manifest["versions"])
    assert manifest["seed"] == 7
    assert "testshort" in manifest["dropped_clusters"]
    assert manifest["n_params"] > 0
    assert len(manifest["tokens"]["labels"]) == 10
    identity = manifest["identity"]
    assert len(identity["code_hash"]) == 64
    assert identity["model_commit"] and identity["dtype"] == "torch.float32"
    assert {"torch", "transformers", "tokenizers", "numpy", "python"} <= set(identity["versions"])


def test_dropped_words_are_recorded(finished) -> None:
    dropped = finished.store.get("manifest|0")["dropped_words"]
    assert "two-part" in dropped["calone"] and "missingword" in dropped["calone"]


def test_export_and_analysis_run_on_real_results(finished) -> None:
    export = decode_export(finished.export_text())
    assert export["meta"]["complete"] is True
    assert len(export["meta"]["environments"]) == 1
    assert export["vector_diagnostics"]
    report = analyze(export)
    assert report["outcome"] in {"V1_fail", "V2_fail", "capture", "task_gated", "mixed"}
    assert export["screen"]
    assert {row[0] for row in export["cells"]} == {"T_inj", "T_pick", "T_fav", "T_odd", "T_fact", "T_inj_none"}
    assert "T_inj|concept" in report["label_mass"]["injected"]
    assert report["small_sample_caveat"]["n_clusters"] == 2


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
    with pytest.raises(ValueError, match="config_hash"):
        Session(other, finished.out_dir, model=model, tokenizer=tokenizer, device=DEVICE)


def test_other_commit_cannot_reuse_results(finished, model, tokenizer) -> None:
    with pytest.raises(ValueError, match="model_commit"):
        Session(finished.config, finished.out_dir, model=model, tokenizer=tokenizer, device=DEVICE, model_commit="other")


def test_other_code_cannot_reuse_results(finished, model, tokenizer, monkeypatch) -> None:
    monkeypatch.setattr("mechinterp.identity.code_hash", lambda: "changed")
    with pytest.raises(ValueError, match="code_hash"):
        Session(finished.config, finished.out_dir, model=model, tokenizer=tokenizer, device=DEVICE)


def test_other_library_version_cannot_reuse_results(finished, model, tokenizer, monkeypatch) -> None:
    versions = {**runner_module.library_versions(), "transformers": "0.0.0"}
    monkeypatch.setattr(runner_module, "library_versions", lambda: versions)
    with pytest.raises(ValueError, match="versions.transformers"):
        Session(finished.config, finished.out_dir, model=model, tokenizer=tokenizer, device=DEVICE)


def test_other_dtype_cannot_reuse_results(tmp_path, tokenizer) -> None:
    from conftest import tiny_qwen

    session = _session(tmp_path, tiny_qwen(len(tokenizer)), tokenizer)
    other = tiny_qwen(len(tokenizer)).double()
    with pytest.raises(ValueError, match="dtype"):
        Session(session.config, session.out_dir, model=other, tokenizer=tokenizer, device=DEVICE)


def test_recomputed_vectors_must_match_the_stored_rows(tmp_path, model, tokenizer) -> None:
    session = _session(tmp_path, model, tokenizer)
    session._ensure_vectors()
    (session.out_dir / "concept_vectors.pt").unlink()
    resumed = Session(session.config, session.out_dir, model=model, tokenizer=tokenizer, device=DEVICE)
    resumed.store.get("vector_diagnostics")["fingerprint"] = {"changed|0": 1.0}
    with pytest.raises(ValueError, match="concept vectors"):
        resumed._ensure_vectors()


def test_non_finite_logits_stop_the_run(tmp_path, model, tokenizer, monkeypatch) -> None:
    session = _session(tmp_path, model, tokenizer)
    original = session._forward

    def poisoned(input_ids, injection):
        # Logits from inference mode cannot be edited in place, so edit a copy.
        logits = original(input_ids, injection).clone()
        if injection is not None and bool(injection[3].sum() > 0):
            logits[0, 0] = float("nan")
        return logits

    monkeypatch.setattr(session, "_forward", poisoned)
    with pytest.raises(NonFiniteError):
        session.run()
    assert not list(session.store.records("trial"))


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
    assert estimate["vector_read_token"] == "."
    assert estimate["labels_decoded"] == list("0123456789")
    assert set(estimate["clean_label_mass_by_task"]) == set(session.test_tasks)


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
