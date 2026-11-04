"""Sharded runs on the tiny CPU model, in real worker processes. No GPU and no download."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch
from conftest import tiny_config_dict, write_tiny_data

from mechinterp import sharding
from mechinterp.analysis import analyze
from mechinterp.checkpoint import ResultStore, ShardStore, read_records
from mechinterp.config import Config
from mechinterp.design import plan_grid_trials
from mechinterp.export import build_export, decode_export
from mechinterp.runner import Session, is_out_of_memory
from mechinterp.shard_worker import evaluate_preflight, measure_preflight, preflight_main
from mechinterp.sharding import (
    CLEAN_AGREEMENT_TOLERANCE,
    MergeError,
    RunFailed,
    RunOptions,
    RunPaths,
    RunPlan,
    check_exact_keys,
    check_setups,
    collect,
    finalize,
    merge,
    orchestrate,
    shard_of,
    shard_trials,
    try_lock,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = str(Path(__file__).resolve().parent)
FACTORY = "e1_shard_factory:build"
DEVICE = torch.device("cpu")


def write_config(directory: Path, data_dir: Path | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    data_dir = data_dir or write_tiny_data(directory / "data")
    path = directory / "config.json"
    path.write_text(json.dumps(tiny_config_dict(data_dir)), encoding="utf-8")
    return path


def run_options(config_path: Path, out_dir: Path, n_workers: int, **overrides) -> RunOptions:
    extra_env = {"PYTHONPATH": TESTS_DIR, **overrides.pop("extra_env", {})}
    settings = {
        "config_path": config_path,
        "out_dir": out_dir,
        "devices": [""] * n_workers,
        "factory": FACTORY,
        "max_restarts": 2,
        "threads_per_worker": 1,
        "load_slots": n_workers,
        "poll_seconds": 0.1,
        "progress_seconds": 10.0,
        "restart_delay_seconds": 0.0,
        "min_free_disk_bytes": 0,
        "extra_env": extra_env,
    }
    settings.update(overrides)
    return RunOptions(**settings)


def trials_of(records) -> dict[str, dict]:
    return {record["key"]: record for record in records if record.get("kind") == "trial"}


@pytest.fixture(scope="module")
def reference(tmp_path_factory, tokenizer, model) -> dict:
    """The unsharded run: the single-process path, on the same config file."""
    directory = tmp_path_factory.mktemp("single")
    config_path = write_config(directory)
    session = Session(Config.from_file(config_path), directory / "out", model=model, tokenizer=tokenizer, device=DEVICE)
    assert session.run()["complete"] is True
    return {"config_path": config_path, "records": list(session.store.records()), "session": session}


@pytest.fixture(scope="module")
def sharded(tmp_path_factory, reference) -> dict:
    """The same config file across two worker processes."""
    out_dir = tmp_path_factory.mktemp("sharded") / "out"
    report = orchestrate(run_options(reference["config_path"], out_dir, 2))
    return {"out_dir": out_dir, "report": report, "records": read_records(out_dir / "results.jsonl")}


# Stable assignment


def test_shard_of_is_stable_across_processes_and_never_uses_python_hash() -> None:
    key = "grid|T_inj|concept|a|b|j0|qNone|l0|a1"
    assert shard_of(key, 4) == 1
    code = (
        "import sys; sys.path.insert(0, sys.argv[1])\n"
        "from mechinterp.sharding import shard_of\n"
        "assert 'torch' not in sys.modules, 'the orchestrator must stay free of torch and CUDA'\n"
        f"print(shard_of({key!r}, 4))"
    )
    for seed in ("1", "2"):
        output = subprocess.run(
            [sys.executable, "-c", code, str(REPO_ROOT / "src")],
            env={"PYTHONHASHSEED": seed, "PATH": ""},
            capture_output=True,
            text=True,
            check=True,
        )
        assert output.stdout.strip() == "1"


def test_shards_partition_the_plan_exactly(reference) -> None:
    session = reference["session"]
    trials = plan_grid_trials(session.stimuli, session.config)
    for n_shards in (1, 2, 3, 4):
        parts = [shard_trials(trials, shard, n_shards) for shard in range(n_shards)]
        keys = [trial.key for part in parts for trial in part]
        assert sorted(keys) == sorted(trial.key for trial in trials)


# Equality with the single-process path


def test_sharded_trials_are_exactly_the_unsharded_trials(reference, sharded) -> None:
    single = trials_of(reference["records"])
    merged = trials_of(sharded["records"])
    assert merged.keys() == single.keys()
    keys = [record["key"] for record in sharded["records"]]
    assert len(keys) == len(set(keys))


def test_sharded_values_match_within_rounding(reference, sharded) -> None:
    single = trials_of(reference["records"])
    merged = trials_of(sharded["records"])
    for key, row in single.items():
        other = merged[key]
        assert other["L"] == pytest.approx(row["L"], abs=1e-3), key
        assert other["m_clean"] == pytest.approx(row["m_clean"], abs=1e-3), key
        assert other["label_mass"] == pytest.approx(row["label_mass"], abs=1e-3), key
        assert other["hit"] == row["hit"], key
        assert other["clean_hit"] == row["clean_hit"], key
        assert other["prompt_hash"] == row["prompt_hash"], key
        assert other["injected_token"] == row["injected_token"], key
        if "js" in row:
            assert other["js"] == pytest.approx(row["js"], abs=1e-3), key


def test_selection_and_impact_match_the_unsharded_run(reference, sharded) -> None:
    by_key = {record["key"]: record for record in sharded["records"]}
    single = {record["key"]: record for record in reference["records"]}
    for name in ("layer", "alpha", "accuracy", "grid_accuracy", "n_trials_per_point"):
        assert by_key["selection"][name] == single["selection"][name]
    assert by_key["impact_match"]["alpha_impact"] == single["impact_match"]["alpha_impact"]
    assert [item["alpha"] for item in by_key["impact_match"]["history"]] == [
        item["alpha"] for item in single["impact_match"]["history"]
    ]
    assert by_key["impact_match"]["target_js"] == pytest.approx(single["impact_match"]["target_js"], abs=1e-4)


def test_export_and_analysis_take_the_same_path(reference, sharded) -> None:
    single = build_export(reference["records"])
    merged = build_export(sharded["records"])
    assert merged["meta"]["complete"] is True
    assert len(merged["meta"]["environments"]) == 1
    assert [row[:5] for row in merged["cells"]] == [row[:5] for row in single["cells"]]
    for mine, theirs in zip(merged["cells"], single["cells"]):
        assert mine[5] == pytest.approx(theirs[5], abs=1e-3)
        assert mine[6:8] == theirs[6:8]
    assert merged["clean_fact"] == single["clean_fact"]
    assert merged["position_counts"] == single["position_counts"]
    assert merged["screen"] == single["screen"]
    assert analyze(merged)["outcome"] == analyze(single)["outcome"]


def test_merged_store_opens_in_the_existing_reader_and_keeps_calibration_first(sharded) -> None:
    store = ResultStore(sharded["out_dir"] / "results.jsonl")
    assert store.has("run_complete") and store.has("selection") and store.has("sham")
    records = sharded["records"]
    index = {record["key"]: i for i, record in enumerate(records)}
    manifest = next(record for record in records if record.get("kind") == "manifest")
    test_clusters = set(manifest["clusters_test"])
    first_test = min(i for i, record in enumerate(records) if record.get("cluster") in test_clusters)
    assert index["selection"] < first_test
    assert index["impact_match"] < first_test
    assert records[-1]["kind"] == "run_complete"


def test_clean_logits_agree_across_workers_and_each_shard_computed_its_own(sharded) -> None:
    report = sharded["report"]["merge"]
    assert report["clean_max_abs_diff_between_shards"] <= CLEAN_AGREEMENT_TOLERANCE
    collected = collect(RunPaths(sharded["out_dir"]).all_shard_files())
    assert all(len(copies) == 2 for copies in collected.cleans.values())
    assert len(collected.shams) == 2


def test_rerun_of_a_complete_run_changes_nothing(reference, sharded) -> None:
    results = sharded["out_dir"] / "results.jsonl"
    before = results.read_bytes()
    assert orchestrate(run_options(reference["config_path"], sharded["out_dir"], 2))["already_complete"] is True
    assert results.read_bytes() == before


# Merge guarantees


@pytest.fixture
def merge_case(tmp_path, reference, sharded):
    """A copy of finished shard files plus the plan they ran under, to damage."""
    out_dir = tmp_path / "out"
    shutil.copytree(sharded["out_dir"] / "shards", out_dir / "shards")
    setup = json.loads((sharded["out_dir"] / "setup" / "w0.json").read_text(encoding="utf-8"))
    config = Config.from_file(reference["config_path"])
    plan = RunPlan.from_setup(config, setup)
    selection = next(record for record in sharded["records"] if record["key"] == "selection")
    return RunPaths(out_dir), plan, selection


def full_merge(paths: RunPaths, plan: RunPlan, selection: dict) -> dict:
    return merge(paths, plan, selection, ["impact", "screen", "test"], complete=True)


def rewrite(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records), encoding="utf-8")


def test_intact_shards_merge(merge_case) -> None:
    paths, plan, selection = merge_case
    assert full_merge(paths, plan, selection)["n_records"] > 0
    assert sharding.is_complete(paths)


def test_missing_trial_is_detected_and_nothing_is_written(merge_case) -> None:
    paths, plan, selection = merge_case
    path = paths.shard_file(0, 2)
    records = read_records(path)
    position = max(i for i, record in enumerate(records) if record.get("kind") == "trial")
    rewrite(path, records[:position] + records[position + 1 :])
    with pytest.raises(MergeError, match="1 missing"):
        full_merge(paths, plan, selection)
    assert not paths.results.exists()


def test_trial_in_two_shards_is_detected(merge_case) -> None:
    paths, plan, selection = merge_case
    row = next(record for record in read_records(paths.shard_file(0, 2)) if record.get("kind") == "trial")
    with open(paths.shard_file(1, 2), "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    with pytest.raises(MergeError, match="1 repeated"):
        full_merge(paths, plan, selection)
    assert not sharding.is_complete(paths)


def test_unplanned_trial_is_detected(merge_case) -> None:
    paths, plan, selection = merge_case
    row = next(record for record in read_records(paths.shard_file(0, 2)) if record.get("kind") == "trial")
    with open(paths.shard_file(0, 2), "a", encoding="utf-8") as handle:
        handle.write(json.dumps({**row, "key": row["key"] + "x"}, separators=(",", ":")) + "\n")
    with pytest.raises(MergeError, match="1 unplanned"):
        full_merge(paths, plan, selection)


def test_differing_identity_between_shards_is_refused(merge_case) -> None:
    paths, plan, selection = merge_case
    path = paths.shard_file(1, 2)
    records = read_records(path)
    for record in records:
        if record.get("kind") == "manifest":
            record["identity"] = {**record["identity"], "code_hash": "other"}
    rewrite(path, records)
    with pytest.raises(MergeError, match="code_hash"):
        full_merge(paths, plan, selection)


def test_failed_sham_in_one_shard_is_refused(merge_case) -> None:
    paths, plan, selection = merge_case
    path = paths.shard_file(1, 2)
    records = read_records(path)
    for record in records:
        if record.get("kind") == "sham":
            record["max_abs_L"] = 0.5
    rewrite(path, records)
    with pytest.raises(MergeError, match="sham"):
        full_merge(paths, plan, selection)


def test_clean_disagreement_between_shards_is_refused(merge_case) -> None:
    paths, plan, selection = merge_case
    path = paths.shard_file(1, 2)
    records = read_records(path)
    clean = next(record for record in records if record.get("kind") == "clean")
    clean["logits"][0] += 1.0
    rewrite(path, records)
    with pytest.raises(MergeError, match="clean logits differ"):
        full_merge(paths, plan, selection)


def test_row_paired_with_other_tokens_is_refused(merge_case) -> None:
    paths, plan, selection = merge_case
    path = paths.shard_file(0, 2)
    records = read_records(path)
    next(record for record in records if record.get("kind") == "trial")["prompt_hash"] = "different"
    rewrite(path, records)
    with pytest.raises(MergeError, match="tokens differ"):
        full_merge(paths, plan, selection)


def test_exact_key_check_covers_all_three_failures() -> None:
    check_exact_keys(["a", "b"], ["b", "a"], "ok")
    with pytest.raises(MergeError, match="1 repeated"):
        check_exact_keys(["a", "a", "b"], ["a", "b"], "x")
    with pytest.raises(MergeError, match="1 missing"):
        check_exact_keys(["a"], ["a", "b"], "x")
    with pytest.raises(MergeError, match="1 unplanned"):
        check_exact_keys(["a", "b", "c"], ["a", "b"], "x")


def _setup(**overrides) -> dict:
    return {
        "identity": {"config_hash": "c", "code_hash": "h", "versions": {"torch": "1"}},
        "stimuli": {"concepts_test": ["a"]},
        "test_tasks": ["T_inj"],
        "sham_max_abs_L": 0.0,
        "probe_label_logits": [1.0, 2.0],
        **overrides,
    }


def test_setup_check_catches_each_mismatch_before_any_trial() -> None:
    assert check_setups([_setup(), _setup()], 0.0)["probe_max_abs_diff"] == 0.0
    with pytest.raises(RunFailed, match="code_hash"):
        check_setups([_setup(), _setup(identity={"config_hash": "c", "code_hash": "z", "versions": {"torch": "1"}})], 0.0)
    with pytest.raises(RunFailed, match="versions.torch"):
        check_setups([_setup(), _setup(identity={"config_hash": "c", "code_hash": "h", "versions": {"torch": "2"}})], 0.0)
    with pytest.raises(RunFailed, match="stimuli"):
        check_setups([_setup(), _setup(stimuli={"concepts_test": ["b"]})], 0.0)
    with pytest.raises(RunFailed, match="Sham"):
        check_setups([_setup(), _setup(sham_max_abs_L=1e-7)], 0.0)
    with pytest.raises(RunFailed, match="faulty GPU"):
        check_setups([_setup(), _setup(probe_label_logits=[1.0, 2.5])], 0.0)


# Barrier


def test_nothing_after_the_grid_runs_until_the_barrier_passes(tmp_path, reference, monkeypatch) -> None:
    def refuse(records, plan):
        raise MergeError("barrier held for this test")

    monkeypatch.setattr(sharding, "selection_record", refuse)
    out_dir = tmp_path / "out"
    with pytest.raises(MergeError, match="barrier held"):
        orchestrate(run_options(reference["config_path"], out_dir, 2))
    paths = RunPaths(out_dir)
    assert not paths.stage_spec(sharding.STAGE_ORDER.index("impact")).exists()
    phases = Counter(
        record["phase"] for path in paths.all_shard_files() for record in read_records(path) if record.get("kind") == "trial"
    )
    assert set(phases) == {"calibration_grid"}
    assert not sharding.is_complete(paths)


# Resume


def test_killed_worker_is_restarted_and_torn_line_tolerated(tmp_path, reference) -> None:
    out_dir = tmp_path / "out"
    marker = tmp_path / "marker"
    options = run_options(
        reference["config_path"],
        out_dir,
        2,
        extra_env={"E1_TEST_FAULT": "once:0:6", "E1_TEST_MARKER": str(marker)},
    )
    report = orchestrate(options)
    assert marker.exists()
    assert report["restarts"][0] == 1
    records = read_records(out_dir / "results.jsonl")
    assert trials_of(records).keys() == trials_of(reference["records"]).keys()
    assert not any(record["key"] == "torn" for record in records)
    keys = [record["key"] for record in records]
    assert len(keys) == len(set(keys))
    assert sum(record.get("kind") == "manifest" for record in records) == 3  # Two workers, one restart.
    assert len(build_export(records)["meta"]["environments"]) == 1


def test_worker_that_keeps_dying_fails_loudly_then_a_rerun_with_other_worker_count_finishes(tmp_path, reference) -> None:
    out_dir = tmp_path / "out"
    marker = tmp_path / "marker"
    failing = run_options(
        reference["config_path"],
        out_dir,
        2,
        max_restarts=1,
        extra_env={"E1_TEST_FAULT": "always:0:6", "E1_TEST_MARKER": str(marker)},
    )
    with pytest.raises(RunFailed, match="failed 2 times"):
        orchestrate(failing)
    paths = RunPaths(out_dir)
    assert not sharding.is_complete(paths)
    with pytest.raises(RunFailed, match="no run_complete"):
        finalize(out_dir, tmp_path / "final")
    assert not (tmp_path / "final").exists()
    partial = {record["key"] for path in paths.all_shard_files() for record in read_records(path) if record.get("kind") == "trial"}
    assert partial

    # The same command on a machine with a different number of GPUs: finished rows stay finished.
    report = orchestrate(run_options(reference["config_path"], out_dir, 3))
    assert report["restarts"] == {0: 0, 1: 0, 2: 0}
    records = read_records(out_dir / "results.jsonl")
    assert trials_of(records).keys() == trials_of(reference["records"]).keys()
    keys = [record["key"] for record in records]
    assert len(keys) == len(set(keys))
    assert sharding.is_complete(paths)


def test_second_writer_on_the_same_lock_is_refused(tmp_path) -> None:
    first = try_lock(tmp_path / "x.lock")
    assert first is not None
    assert try_lock(tmp_path / "x.lock") is None
    first.close()
    assert try_lock(tmp_path / "x.lock") is not None


def test_reader_never_truncates_a_live_writers_tail(tmp_path) -> None:
    path = tmp_path / "s.jsonl"
    ResultStore(path).append([{"key": "a", "kind": "trial"}])
    with open(path, "ab") as handle:
        handle.write(b'{"key":"half')
    size = path.stat().st_size
    assert [record["key"] for record in read_records(path)] == ["a"]
    assert path.stat().st_size == size


def test_shard_store_treats_foreign_rows_as_done_and_refuses_to_repeat_them(tmp_path) -> None:
    other = tmp_path / "w0of4.jsonl"
    ResultStore(other).append(
        [{"key": "t", "kind": "trial", "js": 1.5}, {"key": "c", "kind": "clean"}, {"key": "manifest|0", "kind": "manifest"}]
    )
    store = ShardStore(tmp_path / "w0of3.jsonl", [other])
    assert store.has("t") and store.get("t")["js"] == 1.5
    assert not store.has("c") and not store.has("manifest|0")
    assert list(store.records()) == []
    with pytest.raises(ValueError):
        store.append([{"key": "t", "kind": "trial"}])
    store.append([{"key": "manifest|0", "kind": "manifest"}, {"key": "c", "kind": "clean"}])
    assert ResultStore(store.path).has("c")


# Out of memory


def _session(tmp_path, model, tokenizer) -> Session:
    raw = tiny_config_dict(write_tiny_data(tmp_path / "data"))
    return Session(Config.from_dict(raw, tmp_path), tmp_path / "out", model=model, tokenizer=tokenizer, device=DEVICE)


def test_batch_is_halved_on_oom_with_no_row_lost_or_repeated(tmp_path, model, tokenizer, monkeypatch, reference) -> None:
    session = _session(tmp_path, model, tokenizer)
    original = session._forward
    failures = {"n": 0}

    def limited(input_ids, injection):
        if input_ids.size(0) > 1 and injection is not None:
            failures["n"] += 1
            raise RuntimeError("CUDA out of memory. Tried to allocate 1 GiB")
        return original(input_ids, injection)

    monkeypatch.setattr(session, "_forward", limited)
    monkeypatch.setattr(session, "_sham_check", lambda: session.store.append([{"key": "sham", "kind": "sham", "max_abs_L": 0.0, "n": 1}]))
    assert session.batch_size == 4
    assert session.run()["complete"] is True
    assert failures["n"] > 0
    assert session.batch_size == 1

    rows = trials_of(session.store.records())
    expected = trials_of(reference["records"])
    assert rows.keys() == expected.keys()
    for key, row in rows.items():
        assert row["L"] == pytest.approx(expected[key]["L"], abs=1e-3), key
        assert row["hit"] == expected[key]["hit"], key
    keys = [json.loads(line)["key"] for line in session.store.path.read_text(encoding="utf-8").splitlines()]
    assert len(keys) == len(set(keys))


def test_oom_in_the_second_half_does_not_repeat_the_first(tmp_path, model, tokenizer, monkeypatch) -> None:
    session = _session(tmp_path, model, tokenizer)
    runs = []
    attempts = {"n": 0}

    def run(part):
        attempts["n"] += 1
        if attempts["n"] in (1, 3):
            raise RuntimeError("CUDA error: out of memory")
        runs.extend(part)

    session._run_halving_on_oom(list(range(8)), run)
    assert sorted(runs) == list(range(8))
    assert session.batch_size <= 2


def test_oom_at_batch_of_one_and_other_errors_are_not_swallowed(tmp_path, model, tokenizer) -> None:
    session = _session(tmp_path, model, tokenizer)

    def always_oom(part):
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="out of memory"):
        session._run_halving_on_oom([1, 2], always_oom)

    calls = []

    def other_error(part):
        calls.append(part)
        raise RuntimeError("some other failure")

    with pytest.raises(RuntimeError, match="some other failure"):
        session._run_halving_on_oom([1, 2], other_error)
    assert calls == [[1, 2]]
    assert is_out_of_memory(torch.cuda.OutOfMemoryError("x"))
    assert not is_out_of_memory(ValueError("CUDA out of memory"))
    assert not is_out_of_memory(RuntimeError("DefaultCPUAllocator: can't allocate memory"))


def test_failed_sham_stays_failed_after_a_restart(tmp_path, model, tokenizer) -> None:
    session = _session(tmp_path, model, tokenizer)
    session.store.append([{"key": "sham", "kind": "sham", "max_abs_L": 0.5, "n": 4}])
    with pytest.raises(RuntimeError, match="Sham"):
        session._sham_check()


# Go or no-go checks


def _good_preflight() -> dict:
    return {
        "prefixed_tasks": ["T_inj", "T_pick"],
        "prompt_tail_by_task": {"T_inj": "x\n\nPosition number: ", "T_pick": "y\n\nPosition number: ", "T_inj_none": "z"},
        "labels_decoded": list("0123456789"),
        "clean_top_tokens": ["0", "1", "2", "5", "3"],
        "clean_top_token_by_task": {"T_inj": "0", "T_pick": "7"},
        "clean_label_mass_by_task": {"T_inj": 0.999, "T_pick": 0.998, "T_inj_none": 0.16},
        "sham_max_abs_L": 0.0,
        "batch_invariance_max_abs": 1e-6,
    }


@pytest.mark.parametrize(
    ("change", "failing"),
    [
        ({"prompt_tail_by_task": {"T_inj": "Answer:", "T_pick": "Position number: "}}, "prompt_tail_ends_in_answer_prefix"),
        ({"clean_top_token_by_task": {"T_inj": "Position", "T_pick": "7"}}, "clean_top_tokens_are_digits"),
        ({"clean_top_tokens": ["The", "A", "0", "It", "1"]}, "clean_top_tokens_are_digits"),
        ({"clean_label_mass_by_task": {"T_inj": 0.24, "T_pick": 0.998}}, "label_mass_near_1_on_prefixed_tasks"),
        ({"sham_max_abs_L": 1e-6}, "sham_is_exactly_zero"),
        ({"batch_invariance_max_abs": 0.5}, "batch_size_does_not_change_logits"),
        ({"labels_decoded": list("abcdefghij")}, "labels_are_digits"),
    ],
)
def test_each_preflight_check_can_fail(change, failing) -> None:
    assert all(evaluate_preflight(_good_preflight(), 0.0).values())
    checks = evaluate_preflight({**_good_preflight(), **change}, 0.0)
    assert not checks[failing]
    assert sum(not ok for ok in checks.values()) == 1


def test_preflight_measures_on_the_tiny_model_without_touching_the_results(tmp_path, model, tokenizer) -> None:
    raw = tiny_config_dict(write_tiny_data(tmp_path / "data"))
    config = Config.from_dict(raw, tmp_path)
    store = ResultStore(tmp_path / "out" / "preflight" / "store.jsonl")
    session = Session(config, tmp_path / "out", model=model, tokenizer=tokenizer, device=DEVICE, store=store)
    measured = measure_preflight(session)
    assert measured["measured_prompts_per_second_one_gpu"] > 0
    assert measured["sham_max_abs_L"] <= 1e-4
    assert measured["prompt_tail_by_task"]["T_inj"].endswith("Position number: ")
    assert not measured["prompt_tail_by_task"]["T_inj_none"].endswith("Position number: ")
    assert measured["batch_invariance_max_abs"] <= CLEAN_AGREEMENT_TOLERANCE
    assert not (tmp_path / "out" / "results.jsonl").exists()
    # A random model puts its mass anywhere, so this exact setup must be a no-go.
    assert not evaluate_preflight(measured, 1e-4)["label_mass_near_1_on_prefixed_tasks"]


def test_preflight_command_says_no_go_and_writes_its_report(tmp_path, monkeypatch) -> None:
    monkeypatch.syspath_prepend(TESTS_DIR)
    config_path = write_config(tmp_path)
    assert preflight_main(config_path, tmp_path / "out", 2, 1e-4, FACTORY) is False
    report = json.loads(RunPaths(tmp_path / "out").dry_run.read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert report["finish"]["n_workers"] == 2
    assert not (tmp_path / "out" / "results.jsonl").exists()


# Final results


def test_finalize_writes_every_file_with_a_correct_hash(tmp_path, sharded) -> None:
    log = tmp_path / "run.log"
    log.write_text("log line\n", encoding="utf-8")
    final = tmp_path / "final"
    result = finalize(sharded["out_dir"], final, log)
    for name in ("export.txt", "analysis.txt", "results.jsonl", "merge_report.json", "run.log"):
        assert (final / name).is_file()
        assert result["hashes"][name] == sharding.sha256_file(final / name)
    assert decode_export((final / "export.txt").read_text(encoding="utf-8"))["meta"]["complete"] is True
    assert json.loads((final / "analysis.txt").read_text(encoding="utf-8"))["outcome"] == result["outcome"]
    assert result["outcome"] != "incomplete"
    sums = (final / "SHA256SUMS").read_text(encoding="utf-8")
    assert f"{result['hashes']['export.txt']}  export.txt" in sums
    summary = (final / "summary.txt").read_text(encoding="utf-8")
    assert sharding.BEGIN_ANALYSIS in summary and sharding.END_ANALYSIS in summary


# Files the machine depends on


def test_gpu_config_differs_from_the_large_config_only_in_batch_size() -> None:
    directory = REPO_ROOT / "experiments" / "e1_localization_capture" / "configs"
    large = json.loads((directory / "large.json").read_text(encoding="utf-8"))
    gpu = json.loads((directory / "large_4x3090.json").read_text(encoding="utf-8"))
    assert gpu["runtime"].pop("batch_size_gpu") == 64
    large["runtime"].pop("batch_size_gpu")
    assert gpu == large
    # A different config hash on purpose: rows from the two configs refuse to share a store.
    assert Config.from_file(directory / "large.json").config_hash != Config.from_file(directory / "large_4x3090.json").config_hash


def test_stimuli_survive_the_setup_round_trip(reference) -> None:
    from dataclasses import asdict

    session = reference["session"]
    setup = {"stimuli": json.loads(json.dumps(asdict(session.stimuli))), "test_tasks": list(session.test_tasks)}
    plan = RunPlan.from_setup(session.config, setup)
    assert list(plan.stimuli.clusters_calibration) == list(session.stimuli.clusters_calibration)
    assert [trial.key for trial in plan.grid()] == [trial.key for trial in plan_grid_trials(session.stimuli, session.config)]
    assert plan.stimuli == session.stimuli


def test_shell_script_has_lf_endings_strict_mode_and_valid_syntax() -> None:
    script = REPO_ROOT / "scripts" / "run_e1_gpu.sh"
    data = script.read_bytes()
    assert b"\r" not in data
    assert data.startswith(b"#!/usr/bin/env bash\n")
    assert b"set -Eeuo pipefail" in data
    attributes = (REPO_ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "*.sh text eol=lf" in attributes
    bash = shutil.which("bash")
    if bash and sys.platform != "win32":
        subprocess.run([bash, "-n", str(script)], check=True)


def test_script_never_pushes_and_never_needs_a_token() -> None:
    text = (REPO_ROOT / "scripts" / "run_e1_gpu.sh").read_text(encoding="utf-8")
    assert "git push" not in text
    assert "unset HF_TOKEN" in text
