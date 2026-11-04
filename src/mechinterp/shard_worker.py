"""The GPU side of a sharded run: one worker process per GPU, and the checks before any of them start."""

from __future__ import annotations

import importlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import torch

from .checkpoint import ResultStore, ShardStore
from .config import Config
from .design import (
    PHASE_GRID,
    SHAM_TOLERANCE,
    Selection,
    Trial,
    assert_trial_respects_split,
    plan_clean_keys,
    plan_grid_trials,
    plan_screen_trials,
    plan_test_trials,
)
from .identity import code_hash
from .modeling import _supports_logits_to_keep, hidden_size
from .prompts import ANSWER_PREFIX, UNPREFIXED_TASKS, decode_token
from .runner import Session, is_out_of_memory
from .sharding import (
    CLEAN_AGREEMENT_TOLERANCE,
    RunPaths,
    STAGE_ORDER,
    read_json,
    setup_payload,
    sham_failure_message,
    shard_trials,
    touch,
    try_lock,
    write_json,
)
from .scoring import answer_mass
from .vectors import random_unit_vector

MIN_CLEAN_LABEL_MASS = 0.9
WAIT_SECONDS = 1.0
LOCK_WAIT_SECONDS = 120.0


def _load_factory(factory: str | None) -> Any:
    """A test can swap the real model for a tiny one, as "module:function". A real run passes nothing."""
    if not factory:
        return None
    module, _, function = factory.partition(":")
    return getattr(importlib.import_module(module), function)


def _build_session(config: Config, out_dir: Path, store: ResultStore, factory: str | None, shard: int) -> Session:
    build = _load_factory(factory)
    if build is None:
        return Session(config, out_dir, store=store)
    model, tokenizer, device = build(config, shard)
    return Session(config, out_dir, model=model, tokenizer=tokenizer, device=device, store=store)


def _apply_thread_limit() -> None:
    threads = int(os.environ.get("OMP_NUM_THREADS", "0"))
    if threads > 0:
        torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # Already set by the first parallel call. Harmless.


def _watch_parent() -> None:
    """Exit with the orchestrator, so that a killed run leaves no worker holding a GPU or its shard."""
    parent = os.getppid()

    def watch() -> None:
        while True:
            time.sleep(5.0)
            if os.getppid() != parent:
                os._exit(4)

    threading.Thread(target=watch, daemon=True).start()


def _take_load_slot(paths: RunPaths, shard: int, slots: int) -> Any:
    """Limit how many processes hold about 16 GB of weights in CPU RAM at the same moment."""
    while True:
        for slot in range(max(1, slots)):
            handle = try_lock(paths.lock(f"load_slot_{slot}"))
            if handle is not None:
                return handle
        touch(paths.heartbeat(shard))
        time.sleep(WAIT_SECONDS)


def _take_shard_lock(paths: RunPaths, shard: int, n_shards: int) -> Any:
    """A previous worker's lock goes when it dies. Waiting covers a process that is still exiting."""
    deadline = time.monotonic() + LOCK_WAIT_SECONDS
    while True:
        handle = try_lock(paths.lock(f"w{shard}of{n_shards}"))
        if handle is not None:
            return handle
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Another process still holds shard {shard} of {n_shards}")
        touch(paths.heartbeat(shard))
        time.sleep(WAIT_SECONDS)


def probe_label_logits(session: Session) -> list[float]:
    """Label logits on one fixed prompt, so that a disagreeing GPU is caught before any real work."""
    cluster = next(iter(session.stimuli.clusters_calibration))
    prompt = session._prompt(("T_inj", cluster, None))
    logits = session._forward(session._ids_tensor([prompt]), None)
    return [float(value) for value in logits[0, session.label_ids]]


def measure_sham(session: Session) -> dict[str, Any]:
    """|L| at alpha = 0, and the same clean batch against itself, which separates rounding from a hook fault."""
    value, n = session._measure_sham()
    cluster = next(iter(session.stimuli.clusters_calibration))
    input_ids = session._ids_tensor([session._prompt(("T_inj", cluster, None))] * n)
    first = session._forward(input_ids, None)[:, session.label_ids]
    second = session._forward(input_ids, None)[:, session.label_ids]
    return {"max_abs_L": value, "n": n, "clean_repeat_max_abs": float((first - second).abs().max())}


def run_stage(session: Session, spec: dict[str, Any], shard: int, n_shards: int, paths: RunPaths) -> None:
    """One worker's part of one stage. Keys already stored are skipped, so this can always be rerun.

    Each worker runs every clean prompt of a stage itself. That costs seconds, and it means every
    injected row is compared with clean logits from its own GPU and, unless it was restarted, its own process.
    """
    stage = spec["stage"]
    config = session.config
    stimuli = session.stimuli
    session._ensure_vectors()
    if stage == "vectors":
        return
    calibration_clean = session._calibration_clean_keys()
    if stage == "grid":
        session._run_clean(calibration_clean, keep_full=True)
        touch(paths.stage_clean_done(stage, shard))
        session._run_trials(shard_trials(plan_grid_trials(stimuli, config), shard, n_shards), measure_js=True)
        return
    selection = Selection(**spec["selection"])
    if stage == "impact":
        # The target is the mean JS over the whole grid point, so this worker reads the other shards' rows.
        own = paths.shard_file(shard, n_shards)
        assert isinstance(session.store, ShardStore)
        session.store.refresh_foreign(path for path in paths.all_shard_files() if path != own)
        session._run_clean(calibration_clean, keep_full=True)
        session._impact_search(selection)
        return
    if stage == "screen":
        session._run_clean(calibration_clean)
        trials = plan_screen_trials(stimuli, config, selection.layer, selection.alpha)
        session._run_trials(shard_trials(trials, shard, n_shards))
        return
    if stage == "test":
        if "alpha_impact" not in spec:
            raise AssertionError("Test data must not be touched before calibration has finished")
        session._run_clean(plan_clean_keys(session.test_tasks, stimuli.clusters_test))
        touch(paths.stage_clean_done(stage, shard))
        trials = plan_test_trials(stimuli, config, selection.layer, selection.alpha, spec["alpha_impact"], session.test_tasks)
        session._run_trials(shard_trials(trials, shard, n_shards))
        return
    raise ValueError(f"Unknown stage {stage}")


def worker_main(
    config_path: str | Path,
    out_dir: str | Path,
    shard: int,
    n_shards: int,
    require_cuda: bool = False,
    factory: str | None = None,
    load_slots: int = 1,
    sham_tolerance: float = SHAM_TOLERANCE,
) -> None:
    _watch_parent()
    _apply_thread_limit()
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is not usable in this worker. CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    out_dir = Path(out_dir)
    paths = RunPaths(out_dir)
    config = Config.from_file(config_path)
    shard_lock = _take_shard_lock(paths, shard, n_shards)
    store = ShardStore(paths.shard_file(shard, n_shards), paths.files_from_other_worker_counts(n_shards))
    load_slot = _take_load_slot(paths, shard, load_slots)
    try:
        session = _build_session(config, out_dir, store, factory, shard)
    finally:
        load_slot.close()
    print(f"Worker {shard}/{n_shards}: model on {session.device}, batch size {session.batch_size}", flush=True)

    # Measured afresh on every launch, and judged by the orchestrator. A failing value is not stored, so a
    # rerun on a healthy GPU is not held back by it.
    sham = measure_sham(session)
    print(f"Worker {shard}: sham |L| = {sham['max_abs_L']}, clean against itself {sham['clean_repeat_max_abs']}", flush=True)
    if abs(sham["max_abs_L"]) <= sham_tolerance and not session.store.has("sham"):
        session.store.append([{"key": "sham", "kind": "sham", "max_abs_L": sham["max_abs_L"], "n": sham["n"]}])
    gpu_name = torch.cuda.get_device_name() if session.device.type == "cuda" else None
    write_json(paths.setup(shard), setup_payload(session, probe_label_logits(session), gpu_name, sham))

    for index in range(len(STAGE_ORDER)):
        spec_path = paths.stage_spec(index)
        while not spec_path.exists():
            touch(paths.heartbeat(shard))
            time.sleep(WAIT_SECONDS)
        spec = read_json(spec_path)
        if spec["stage"] == "stop":
            break
        done = paths.stage_done(spec["stage"], shard)
        if shard in spec["participants"] and not done.exists():
            print(f"Worker {shard}: stage {spec['stage']} starts", flush=True)
            run_stage(session, spec, shard, n_shards, paths)
            touch(done)
            print(f"Worker {shard}: stage {spec['stage']} done", flush=True)
    shard_lock.close()


# Checks before spending hours


def measure_preflight(session: Session) -> dict[str, Any]:
    """Everything the go or no-go decision reads. Its store is thrown away afterwards."""
    result = session.dry_run(n_batches=3)
    result.update(build_every_prompt(session))
    cluster = next(iter(session.stimuli.clusters_calibration))
    prompts = {
        task: session._prompt((task, cluster, 0 if task == "T_fact" else None)) for task in session.test_tasks
    }
    prefixed = [task for task in session.test_tasks if task not in UNPREFIXED_TASKS]
    result["prefixed_tasks"] = prefixed
    result["prompt_tail_by_task"] = {task: prompts[task].text[-40:] for task in session.test_tasks}
    top_token: dict[str, str] = {}
    for task in prefixed:
        logits = session._forward(session._ids_tensor([prompts[task]]), None)
        top_token[task] = decode_token(session.tokenizer, int(logits[0].argmax()))
    result["clean_top_token_by_task"] = top_token

    sham = measure_sham(session)
    result["sham_max_abs_L"] = sham["max_abs_L"]
    result["clean_repeat_max_abs"] = sham["clean_repeat_max_abs"]

    prompt = prompts["T_inj"]
    single = session._forward(session._ids_tensor([prompt]), None)
    size = min(session.batch_size, 8)
    many = session._forward(session._ids_tensor([prompt] * size), None)
    labels = session.label_ids
    result["batch_invariance_max_abs"] = float((many[:, labels] - single[:, labels]).abs().max())
    result["clean_label_mass_recheck"] = float(answer_mass(single, labels)[0])

    result.update(_measure_throughput(session, cluster))
    result["logits_to_keep_supported"] = _supports_logits_to_keep(session.model)
    result["n_model_layers"] = len(session.layers)
    result["device"] = str(session.device)
    result["gpu_name"] = torch.cuda.get_device_name() if session.device.type == "cuda" else None
    result["identity"] = session.identity
    result["code_hash"] = session.identity["code_hash"]
    result["config_hash"] = session.config.config_hash
    return result


def build_every_prompt(session: Session) -> dict[str, Any]:
    """Render and tokenize every prompt the run will use, and check the plan against them.

    This takes tokens only and scores nothing, so it reveals no results. It runs here because a bad word in a
    test cluster would otherwise first show up hours in, after the grid, the impact search and screening.
    """
    stimuli = session.stimuli
    calibration = plan_clean_keys(["T_inj"], stimuli.clusters_calibration)
    test = plan_clean_keys(session.test_tasks, stimuli.clusters_test)
    errors: list[str] = []
    built = set()
    for key in [*calibration, *test]:
        try:
            session._prompt(key)
            built.add(key)
        except ValueError as error:  # The token location and answer token errors are ValueErrors.
            errors.append(f"{key}: {error}")
    plan = plan_test_trials(stimuli, session.config, 0, 1.0, 1.0, session.test_tasks)
    missing = {trial.clean_key for trial in plan} - set(test)
    if missing:
        errors.append(f"Test trials need clean prompts outside the plan, such as {sorted(missing, key=str)[:2]}")
    for trial in plan:
        assert_trial_respects_split(trial, stimuli)
    return {"n_prompts_built": len(built), "n_test_trials_planned": len(plan), "prompt_build_errors": errors}


def _measure_throughput(session: Session, cluster: str, n_batches: int = 3) -> dict[str, Any]:
    """Prompts per second through the real row code at the real batch size, including the CPU work per row.

    Uses grid-shaped rows with a seeded random direction on a calibration cluster, in the throwaway store.
    The JS measured on grid rows makes this slower per row than the test stage, so it is a little cautious.
    """
    config = session.config
    concepts = session.stimuli.concepts_calibration
    dim = hidden_size(session.model)
    for concept in concepts:
        session.random_vectors[concept] = random_unit_vector(dim, config.seed, concept)
    session._run_clean([("T_inj", cluster, None)], keep_full=True)
    layer = config.grid.layers[0]
    cells = [(concept, position) for concept in concepts for position in range(10)]
    batch = min(session.batch_size, len(cells))
    alpha = 1000.0  # Gives each timed batch its own keys in the throwaway store, at alpha just above 1.

    def run_batch() -> None:
        nonlocal alpha
        alpha += 1.0
        session._run_trial_batch(
            [Trial(PHASE_GRID, "T_inj", "random_norm", concept, cluster, position, None, layer, alpha / 1000.0) for concept, position in cells[:batch]],
            True,
        )

    while True:
        try:
            if session.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            run_batch()  # Warm-up, not timed.
            started = time.monotonic()
            for _ in range(n_batches):
                run_batch()
            seconds = time.monotonic() - started
            break
        except RuntimeError as error:
            if not is_out_of_memory(error) or batch == 1:
                raise
        batch //= 2
        torch.cuda.empty_cache()
    peak = torch.cuda.max_memory_allocated() / 1024**3 if session.device.type == "cuda" else None
    return {
        "measured_batch_size": batch,
        "measured_prompts_per_second_one_gpu": batch * n_batches / seconds,
        "gpu_peak_memory_gib": peak,
    }


def evaluate_preflight(measured: dict[str, Any], sham_tolerance: float) -> dict[str, bool]:
    """The go or no-go checks. Any False stops the run before the workers start."""
    prefixed = measured["prefixed_tasks"]
    top_tokens = measured["clean_top_tokens"]
    return {
        "prompt_tail_ends_in_answer_prefix": all(
            measured["prompt_tail_by_task"][task].endswith(ANSWER_PREFIX) for task in prefixed
        ),
        "labels_are_digits": measured["labels_decoded"] == [str(digit) for digit in range(10)],
        # The top token on every task is a digit. A confident model can leave non-digits among its lower
        # top-five, so most of T_inj's top five must be digits rather than all of them.
        "clean_top_tokens_are_digits": all(
            measured["clean_top_token_by_task"][task].strip().isdigit() for task in prefixed
        )
        and sum(token.strip().isdigit() for token in top_tokens) >= 3,
        "label_mass_near_1_on_prefixed_tasks": all(
            measured["clean_label_mass_by_task"][task] >= MIN_CLEAN_LABEL_MASS for task in prefixed
        ),
        "sham_within_limit": abs(measured["sham_max_abs_L"]) <= sham_tolerance,
        "every_prompt_builds": not measured["prompt_build_errors"],
        "batch_size_does_not_change_logits": measured["batch_invariance_max_abs"] <= CLEAN_AGREEMENT_TOLERANCE,
    }


def _finish_estimate(measured: dict[str, Any], n_workers: int) -> dict[str, Any]:
    counts = measured["prompt_counts"]
    one_gpu = measured["measured_prompts_per_second_one_gpu"]
    # Vector prompts are about a quarter of a list prompt's length. The impact search runs on one GPU.
    impact = counts["impact_search_max"]
    shared = sum(counts.values()) - impact - 0.75 * counts["vector_prompts"]
    seconds = shared / (one_gpu * n_workers) + impact / one_gpu
    return {
        "n_workers": n_workers,
        "total_prompts_at_most": shared + impact,
        "estimated_hours": seconds / 3600,
        "estimated_finish": time.strftime("%Y-%m-%d %H:%M %Z", time.localtime(time.time() + seconds)),
    }


def preflight_main(
    config_path: str | Path,
    out_dir: str | Path,
    n_workers: int,
    sham_tolerance: float,
    factory: str | None = None,
) -> bool:
    """Run or reuse the go or no-go checks and write dry_run.json. Returns whether the run may go on."""
    out_dir = Path(out_dir)
    paths = RunPaths(out_dir)
    config = Config.from_file(config_path)
    if paths.dry_run.exists():
        previous = read_json(paths.dry_run)
        if previous.get("passed") and previous.get("code_hash") == code_hash() and previous.get("config_hash") == config.config_hash:
            print(f"Dry-run checks already passed on this code and config: {paths.dry_run}", flush=True)
            return True
    store_path = paths.dry_run.parent / "dry_run_store.jsonl"
    if store_path.exists():
        store_path.unlink()
    _apply_thread_limit()
    session = _build_session(config, out_dir, ResultStore(store_path), factory, 0)
    measured = measure_preflight(session)
    checks = evaluate_preflight(measured, sham_tolerance)
    passed = all(checks.values())
    report = {**measured, "checks": checks, "passed": passed, "sham_tolerance": sham_tolerance, "finish": _finish_estimate(measured, n_workers)}
    write_json(paths.dry_run, report)
    print("Dry-run results:", flush=True)
    print(f"  prompt tail: {measured['prompt_tail']!r}", flush=True)
    print(f"  clean top tokens on T_inj: {measured['clean_top_tokens']}", flush=True)
    print(f"  clean top token by task: {measured['clean_top_token_by_task']}", flush=True)
    print(f"  clean label mass by task: {json.dumps(measured['clean_label_mass_by_task'])}", flush=True)
    print(f"  prompts built: {measured['n_prompts_built']}; test trials planned: {measured['n_test_trials_planned']}", flush=True)
    for error in measured["prompt_build_errors"][:10]:
        print(f"  PROMPT ERROR: {error}", flush=True)
    print(f"  sham |L|: {measured['sham_max_abs_L']}; clean against itself: {measured['clean_repeat_max_abs']}", flush=True)
    if not checks["sham_within_limit"]:
        print(f"  {sham_failure_message('in the dry run', measured['sham_max_abs_L'], measured['clean_repeat_max_abs'], sham_tolerance)}", flush=True)
    elif measured["sham_max_abs_L"] != 0.0:
        print("  WARNING: the sham is not exactly 0, as the prereg says. It is within the limit, so the run goes on.", flush=True)
    print(f"  logit change from batch size: {measured['batch_invariance_max_abs']}", flush=True)
    print(f"  model layers: {measured['n_model_layers']}; logits_to_keep supported: {measured['logits_to_keep_supported']}", flush=True)
    print(f"  GPU: {measured['gpu_name']}; peak memory {measured['gpu_peak_memory_gib']} GiB", flush=True)
    print(f"  measured {measured['measured_prompts_per_second_one_gpu']:.2f} prompts/s on one GPU at batch {measured['measured_batch_size']}, including per-row scoring", flush=True)
    finish = report["finish"]
    print(f"  estimate on {n_workers} GPU(s): about {finish['estimated_hours']:.2f} h, finishing about {finish['estimated_finish']}. Live per-GPU speeds follow once the grid starts.", flush=True)
    for name, ok in checks.items():
        print(f"  check {name}: {'PASS' if ok else 'FAIL'}", flush=True)
    if not measured["logits_to_keep_supported"]:
        print("  WARNING: logits_to_keep is not supported, so every position goes through the output head.", flush=True)
    return passed

