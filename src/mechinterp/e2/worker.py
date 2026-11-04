"""The GPU side of E2's Stage 3: one worker per GPU, and the checks before any of them start.

Same protocol as shard_worker: files only, one store per worker, every clean prompt run by every worker. Only the
stage list differs, so the helpers are imported from there rather than copied.
"""

from __future__ import annotations

import importlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from ..checkpoint import ResultStore, ShardStore
from ..design import SHAM_TOLERANCE, plan_clean_keys
from ..modeling import hidden_size
from ..shard_worker import (
    _apply_thread_limit,
    _take_load_slot,
    _take_shard_lock,
    _watch_parent,
    evaluate_preflight,
    measure_preflight,
    measure_sham,
    probe_label_logits,
)
from ..sharding import read_json, setup_payload, shard_trials, touch, write_json
from .config import E2Config, e2_code_hash
from .design import plan_counts, plan_full_grid, plan_searches, plan_test, search_owner, unique_sites
from .families import compute_families, concept_token
from .paths import E2_STAGE_ORDER, E2Paths
from .session import E2Session

WAIT_SECONDS = 1.0


def _build_session(e2: E2Config, out_dir: Path, store: ResultStore, factory: str | None, shard: int) -> E2Session:
    if not factory:
        return E2Session(e2, out_dir, store=store)
    module, _, function = factory.partition(":")
    model, tokenizer, device = getattr(importlib.import_module(module), function)(e2.base, shard)
    return E2Session(e2, out_dir, model=model, tokenizer=tokenizer, device=device, store=store)


def run_stage(session: E2Session, spec: dict[str, Any], shard: int, n_shards: int, paths: E2Paths) -> None:
    """One worker's part of one stage. Keys already stored are skipped, so this can always be rerun."""
    stage = spec["stage"]
    if stage == "vectors":
        compute_families(session, spec)
        return
    session._ensure_vectors()
    config = session.config
    stimuli = session.stimuli
    sites = [int(layer) for layer in spec["sites"]]
    calibration_clean = session._calibration_clean_keys()
    if stage == "grid":
        session._run_clean(calibration_clean, keep_full=True)
        touch(paths.stage_clean_done(stage, shard))
        trials = plan_full_grid(stimuli, config, int(spec["l_s"]), float(spec["alpha_s"]), int(spec["l_w"]))
        session._run_trials(shard_trials(trials, shard, n_shards), measure_js=True)
        return
    if stage == "impact":
        # A target is a mean over a whole grid point, so this worker reads the other shards' rows.
        own = paths.shard_file(shard, n_shards)
        assert isinstance(session.store, ShardStore)
        session.store.refresh_foreign(path for path in paths.all_shard_files() if path != own)
        session._run_clean(calibration_clean, keep_full=True)
        lens_concepts = spec["lens_concepts_calibration"]
        for index, (family, layer) in enumerate(plan_searches(sites, lens_concepts)):
            if search_owner(index, n_shards) == shard:
                concepts = lens_concepts if family == "F_lens" else None
                session.impact_search(family, layer, float(spec["site_alpha"][str(layer)]), concepts)
        return
    if stage == "test":
        if "alphas" not in spec:
            raise AssertionError("Test data must not be touched before calibration has finished")
        session._run_clean(plan_clean_keys(session.test_tasks, stimuli.clusters_test))
        touch(paths.stage_clean_done(stage, shard))
        trials = plan_test(stimuli, config, session.test_tasks, sites, spec["alphas"], spec["lens_concepts_test"])
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
    paths = E2Paths(out_dir)
    e2 = E2Config.from_file(config_path)
    shard_lock = _take_shard_lock(paths, shard, n_shards)
    store = ShardStore(paths.shard_file(shard, n_shards), paths.files_from_other_worker_counts(n_shards))
    load_slot = _take_load_slot(paths, shard, load_slots)
    try:
        session = _build_session(e2, out_dir, store, factory, shard)
    finally:
        load_slot.close()
    print(f"Worker {shard}/{n_shards}: model on {session.device}, batch size {session.batch_size}", flush=True)

    # Measured afresh on every launch, as in E1. A failing value is not stored.
    sham = measure_sham(session)
    print(f"Worker {shard}: sham |L| = {sham['max_abs_L']}, clean against itself {sham['clean_repeat_max_abs']}", flush=True)
    if abs(sham["max_abs_L"]) <= sham_tolerance and not session.store.has("sham"):
        session.store.append([{"key": "sham", "kind": "sham", "max_abs_L": sham["max_abs_L"], "n": sham["n"]}])
    gpu_name = torch.cuda.get_device_name() if session.device.type == "cuda" else None
    write_json(paths.setup(shard), setup_payload(session, probe_label_logits(session), gpu_name, sham))

    for index in range(len(E2_STAGE_ORDER)):
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


def preflight_main(
    config_path: str | Path, out_dir: str | Path, n_workers: int, sham_tolerance: float, l_s: int, l_w: int, factory: str | None = None
) -> bool:
    """E1's go or no-go checks on E2's tasks, plus a runtime estimate from measured speed. Its store is thrown away."""
    out_dir = Path(out_dir)
    paths = E2Paths(out_dir)
    e2 = E2Config.from_file(config_path)
    if paths.dry_run.exists():
        previous = read_json(paths.dry_run)
        if previous.get("passed") and previous.get("e2_code_hash") == e2_code_hash() and previous.get("e2_config_hash") == e2.e2_config_hash:
            print(f"Dry-run checks already passed on this code and config: {paths.dry_run}", flush=True)
            return True
    store_path = paths.dry_run.parent / "dry_run_store.jsonl"
    if store_path.exists():
        store_path.unlink()
    _apply_thread_limit()
    session = _build_session(e2, out_dir, ResultStore(store_path), factory, 0)
    measured = measure_preflight(session)
    checks = evaluate_preflight(measured, sham_tolerance)
    n_layers = len(session.layers)
    checks["both_sites_exist"] = 0 <= l_s < n_layers and 0 <= l_w < n_layers
    passed = all(checks.values())

    stimuli = session.stimuli
    n_lens_test = sum(1 for concept in stimuli.concepts_test if concept_token(session.tokenizer, concept) is not None)
    n_lens_calibration = sum(1 for concept in stimuli.concepts_calibration if concept_token(session.tokenizer, concept) is not None)
    counts = plan_counts(stimuli, e2.base, session.test_tasks, n_lens_test, n_lens_calibration)
    speed = measured["measured_prompts_per_second_one_gpu"]
    # Worst case. The searches share out across GPUs, and usually stop long before their last evaluation.
    seconds = (counts["grid"] + counts["impact_search_max"] + counts["test"]) / (speed * n_workers)
    report = {
        **measured,
        "checks": checks,
        "passed": passed,
        "sham_tolerance": sham_tolerance,
        "e2_code_hash": e2_code_hash(),
        "e2_config_hash": e2.e2_config_hash,
        "e2_counts": counts,
        "n_single_token_concepts_test": n_lens_test,
        "n_single_token_concepts_calibration": n_lens_calibration,
        "estimated_hours_at_most": seconds / 3600,
        "n_workers": n_workers,
        "sites": unique_sites(l_s, l_w),
        "hidden_size": hidden_size(session.model),
    }
    write_json(paths.dry_run, report)
    print("Dry-run results:", flush=True)
    print(f"  prompt tail: {measured['prompt_tail']!r}", flush=True)
    print(f"  clean label mass by task: {json.dumps(measured['clean_label_mass_by_task'])}", flush=True)
    print(f"  sham |L|: {measured['sham_max_abs_L']}; clean against itself: {measured['clean_repeat_max_abs']}", flush=True)
    print(f"  GPU: {measured['gpu_name']}; peak memory {measured['gpu_peak_memory_gib']} GiB", flush=True)
    print(f"  F_lens concepts, single token: {n_lens_test} of {len(stimuli.concepts_test)} test, {n_lens_calibration} of {len(stimuli.concepts_calibration)} calibration", flush=True)
    print(f"  planned prompts at most: {counts}", flush=True)
    print(f"  measured {speed:.2f} prompts/s on one GPU at batch {measured['measured_batch_size']}", flush=True)
    print(f"  estimate on {n_workers} GPU(s): at most {seconds / 3600:.2f} h, finishing by about {time.strftime('%Y-%m-%d %H:%M %Z', time.localtime(time.time() + seconds))}", flush=True)
    for name, ok in checks.items():
        print(f"  check {name}: {'PASS' if ok else 'FAIL'}", flush=True)
    return passed


