"""Stage 2 and Stage 3 across GPUs, with E1's supervision and merge checks. Never imports torch.

It starts only if Stage 1's gate.json says the gate passed, and only if E1's finished results are readable. Both are
checked before any worker starts, so a failed gate cannot reach a GPU.
"""

from __future__ import annotations

import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ..checkpoint import ResultStore, atomic_write_bytes, encode_records, read_records
from ..config import Config
from ..design import PHASE_GRID, PHASE_IMPACT, PHASE_TEST, SHAM_TOLERANCE, select_layer_alpha
from ..identity import identity_differences
from ..sharding import (
    CLEAN_AGREEMENT_TOLERANCE,
    Collected,
    MergeError,
    Progress,
    RunFailed,
    RunOptions,
    RunPlan,
    WorkerPool,
    _canonical_clean,
    _check_pairing,
    _trials_of_phase,
    check_setups,
    check_stored_identities,
    clean_gap_warning,
    collect,
    is_complete,
    read_json,
    report_clean_agreement,
    sha256_file,
    try_lock,
    write_json,
)
from .config import E2Config
from .design import alpha_key, plan_calibration, plan_counts, plan_full_grid, plan_searches, plan_test, unique_sites
from .e1_inputs import E1ResultsMissing, load_e1_inputs
from .gate_text import describe_gate
from .paths import E2_STAGE_ORDER, E2_STAGE_PHASE, E2Paths, stage_participants

BEGIN_ANALYSIS = "=====BEGIN E2 ANALYSIS====="
END_ANALYSIS = "=====END E2 ANALYSIS====="


class GateNotPassed(RunFailed):
    """Stage 1 failed or never ran. The stop rule ends E2 here."""


class E2WorkerPool(WorkerPool):
    def _command(self, shard: int) -> list[str]:
        command = super()._command(shard)
        return [sys.executable, "-u", "-m", "mechinterp.e2.cli", *command[command.index("shard-worker") :]]


@dataclass
class E2Plan:
    base: RunPlan
    e2: E2Config
    l_s: int
    alpha_s: float
    l_w: int
    lens_calibration: tuple[str, ...] = ()
    lens_test: tuple[str, ...] = ()

    @property
    def sites(self) -> list[int]:
        return unique_sites(self.l_s, self.l_w)

    def grid(self) -> list[Any]:
        return plan_full_grid(self.base.stimuli, self.base.config, self.l_s, self.alpha_s, self.l_w)

    def impact(self, match: dict[str, Any]) -> list[Any]:
        alphas = list(dict.fromkeys(item["alpha"] for item in match["history"]))
        return [
            trial
            for alpha in alphas
            for trial in plan_calibration(self.base.stimuli, self.base.config, match["family"], match["layer"], alpha, PHASE_IMPACT, match["concepts"])
        ]

    def test(self, alphas: dict[str, float]) -> list[Any]:
        return plan_test(self.base.stimuli, self.base.config, self.base.test_tasks, self.sites, alphas, self.lens_test)


def require_gate_passed(gate_file: Path) -> dict[str, Any]:
    if not gate_file.exists():
        raise GateNotPassed(f"{gate_file} does not exist. Run Stage 1 first: Stages 2 and 3 cannot run before the gate.")
    gate = read_json(gate_file)
    if not gate.get("passed"):
        raise GateNotPassed(f"The gate failed, so E2 stops and Stages 2 and 3 do not run. {gate.get('reason')}")
    return gate


def select_site_alpha(rows: Sequence[dict[str, Any]], plan: E2Plan) -> dict[str, Any]:
    """F_full's alpha at l_w, by E1's rule on calibration data: best T_inj accuracy, then the smallest alpha."""
    stimuli = plan.base.stimuli
    at_site = [{**row, "condition": "concept"} for row in rows if row["layer"] == plan.l_w]
    selection = select_layer_alpha(at_site, stimuli.concepts_calibration, list(stimuli.clusters_calibration))
    return {
        "key": f"selection|l{plan.l_w}",
        "kind": "selection",
        "layer": selection.layer,
        "alpha": selection.alpha,
        "accuracy": selection.accuracy,
        "grid_accuracy": selection.grid_accuracy,
    }


def merge(
    paths: E2Paths,
    plan: E2Plan,
    site_selection: dict[str, Any] | None = None,
    alphas: dict[str, float] | None = None,
    finished: Sequence[str] = (),
    complete: bool = False,
    sham_tolerance: float = SHAM_TOLERANCE,
    extra_warnings: Sequence[str] = (),
) -> dict[str, Any]:
    """Write results.jsonl from the shards, holding finished stages only. Raises rather than writes on any mismatch.

    The same checks as E1's merge: exact keys against the plan, one identity, each row paired with the clean row
    in its own shard file.
    """
    collected = collect(paths.all_shard_files())
    if not collected.manifests:
        raise MergeError("No manifest in any shard")
    identities = [manifest.get("identity") or {} for manifest in collected.manifests]
    for identity in identities:
        differences = identity_differences(identities[0], identity)
        if differences:
            raise MergeError(f"Shards differ in run identity: {differences}")
    manifests = [{**manifest, "key": f"manifest|{index}"} for index, manifest in enumerate(collected.manifests)]
    if not collected.shams:
        raise MergeError("No sham check in any shard")
    worst_sham = max(record["max_abs_L"] for record in collected.shams)
    if worst_sham > sham_tolerance:
        raise MergeError(f"A shard's sham check gave |L| = {worst_sham}, above the limit of {sham_tolerance}")
    warnings = list(extra_warnings)
    if worst_sham != 0.0:
        warnings.append(f"Stored sham reached |L| = {worst_sham}, within the limit of {sham_tolerance}, not exactly 0.")
    if not collected.diagnostics:
        raise MergeError("No vector diagnostics in any shard")
    sham = {"key": "sham", "kind": "sham", "max_abs_L": worst_sham, "n": collected.shams[0]["n"], "n_checks": len(collected.shams)}
    report: dict[str, Any] = {
        "shard_files": [path.name for path in paths.all_shard_files()],
        "n_sessions": len(manifests),
        "worst_sham_abs_L": worst_sham,
        "trials_by_phase": {},
    }
    records: list[dict[str, Any]] = [*manifests, collected.diagnostics[0], sham]

    clean_rows, clean_gap = _canonical_clean(collected, plan.base.clean_calibration(), "calibration clean")
    clean_by_key = {record["key"]: record for record in clean_rows}
    records.extend(clean_rows)
    if clean_gap > CLEAN_AGREEMENT_TOLERANCE:
        warnings.append(clean_gap_warning(clean_gap, "calibration clean"))
    pairing_gap = 0.0

    def add_phase(phase: str, plan_trials: Sequence[Any]) -> None:
        nonlocal pairing_gap
        rows = _trials_of_phase(collected, phase, plan_trials, phase)
        pairing_gap = max(pairing_gap, _check_pairing(rows, collected, phase))
        records.extend(rows)
        report["trials_by_phase"][phase] = len(rows)

    add_phase(PHASE_GRID, plan.grid())
    if site_selection is not None:
        records.append(site_selection)

    if "impact" in finished:
        matches = check_impact_matches(collected, plan)
        impact_trials = [trial for match in matches for trial in plan.impact(match)]
        add_phase(PHASE_IMPACT, impact_trials)
        records.extend(matches)
    if "test" in finished:
        test_clean, test_gap = _canonical_clean(collected, plan.base.clean_test(), "test clean")
        clean_gap = max(clean_gap, test_gap)
        if test_gap > CLEAN_AGREEMENT_TOLERANCE:
            warnings.append(clean_gap_warning(test_gap, "test clean"))
        clean_by_key.update({record["key"]: record for record in test_clean})
        records.extend(test_clean)
        add_phase(PHASE_TEST, plan.test(alphas or {}))

    report["clean_max_abs_diff_between_shards"] = clean_gap
    report["m_clean_max_abs_diff_within_own_shard"] = pairing_gap
    report["warnings"] = warnings
    if complete:
        allowed = {E2_STAGE_PHASE[stage] for stage in ("grid", *finished)}
        stray = {record["phase"] for record in collected.trials} - allowed
        if stray:
            raise MergeError(f"Rows from phases outside the plan: {sorted(stray)}")
        stray_clean = collected.cleans.keys() - clean_by_key.keys()
        if stray_clean:
            raise MergeError(f"Clean rows outside the plan: {sorted(stray_clean)[:3]}")
        records.append(
            {
                "key": "run_complete",
                "kind": "run_complete",
                "finished_unix": time.time(),
                "n_shard_files": len(report["shard_files"]),
                "clean_max_abs_diff_between_shards": clean_gap,
                "warnings": warnings,
            }
        )
    keys = [record["key"] for record in records]
    if len(keys) != len(set(keys)):
        raise MergeError("The merged store would repeat a key")
    atomic_write_bytes(paths.results, encode_records(records))
    write_json(paths.merge_report, report)
    return {"collected": collected, "report": report, "n_records": len(records)}


def check_impact_matches(collected: Collected, plan: E2Plan) -> list[dict[str, Any]]:
    """One record per planned search: never a second, and none missing."""
    wanted = plan_searches(plan.sites, plan.lens_calibration)
    found: dict[tuple[str, int], dict[str, Any]] = {}
    for record in collected.impact_matches:
        name = (record["family"], record["layer"])
        if name in found:
            raise MergeError(f"Two impact_match records for {name}: each search must run once")
        found[name] = record
    missing = [name for name in wanted if name not in found]
    extra = [name for name in found if name not in wanted]
    if missing or extra:
        raise MergeError(f"Impact searches do not match the plan: missing {missing}, unplanned {extra}")
    return [found[name] for name in wanted]


def alpha_table(matches: Sequence[dict[str, Any]], site_alpha: dict[str, float]) -> dict[str, float]:
    """Every family's alpha at every site. F_full holds its own, and the others come from their searches."""
    table = {alpha_key("F_full", int(layer)): alpha for layer, alpha in site_alpha.items()}
    for match in matches:
        table[alpha_key(match["family"], match["layer"])] = match["alpha_impact"]
    return table


def orchestrate(options: RunOptions, gate_file: Path, e1_dir: Path) -> dict[str, Any]:
    """Run or resume Stage 2 and Stage 3. Returns a report, or raises RunFailed with no run_complete written."""
    paths = E2Paths(Path(options.out_dir))
    paths.out_dir.mkdir(parents=True, exist_ok=True)
    # Both before anything else, so that neither a failed gate nor missing E1 results can reach a GPU.
    gate = require_gate_passed(gate_file)
    l_w = int(gate["l_w"])
    try:
        e1 = load_e1_inputs(e1_dir)
    except E1ResultsMissing as error:
        raise RunFailed(str(error)) from error
    held = try_lock(paths.lock("orchestrator"))
    if held is None:
        raise RunFailed(f"Another orchestrator is already running on {paths.out_dir}")
    if is_complete(paths):
        print(f"{paths.results} is already complete", flush=True)
        held.close()
        return {"already_complete": True, "restarts": {}}
    write_json(paths.e1_inputs, e1)
    l_s = int(e1["layer"])
    alpha_s = float(e1["alpha"])
    print(f"E1 selected layer {l_s}, alpha {alpha_s}. The gate gave l_w = {l_w}. Sites: {unique_sites(l_s, l_w)}", flush=True)
    if l_s == l_w:
        print("l_s and l_w are the same layer, so there is one site, and Holm runs over its tasks only.", flush=True)

    for transient in (paths.stages_dir, paths.out_dir / "setup"):
        if transient.exists():
            shutil.rmtree(transient)
        transient.mkdir(parents=True)
    e2 = E2Config.from_file(options.config_path)
    base_config: Config = e2.base
    n_workers = len(options.devices)
    if n_workers < 1:
        raise RunFailed("No workers to start")
    print(f"Starting {n_workers} worker(s) on devices {options.devices}", flush=True)
    print(f"Shard files already present: {[path.name for path in paths.all_shard_files()]}", flush=True)
    pool = E2WorkerPool(options, paths)

    def wait(condition: Any, phase: str | None, progress: Progress | None) -> None:
        while not condition():
            pool.check()
            if progress is not None:
                try:
                    progress.tick(phase, pool.restarts)
                except Exception as error:  # Monitoring must never stop a healthy run.
                    print(f"WARNING: could not print progress: {error!r}", flush=True)
            time.sleep(options.poll_seconds)

    def run_stage(stage: str, spec: dict[str, Any], progress: Progress, clean_keys: Sequence[str] = ()) -> None:
        index = E2_STAGE_ORDER.index(stage)
        participants = stage_participants(stage, n_workers)
        phase = E2_STAGE_PHASE.get(stage)
        progress.begin(phase)
        print(f"[{time.strftime('%H:%M:%S')}] Stage {stage} starts on workers {participants}", flush=True)
        write_json(paths.stage_spec(index), {"stage": stage, "participants": participants, **spec})
        clean_pending = bool(clean_keys)

        def stage_finished() -> bool:
            nonlocal clean_pending
            if clean_pending and all(paths.stage_clean_done(stage, shard).exists() for shard in participants):
                clean_pending = False
                report_clean_agreement(paths, clean_keys, f"{stage} stage clean")
            return all(paths.stage_done(stage, shard).exists() for shard in participants)

        wait(stage_finished, phase, progress)
        print(f"[{time.strftime('%H:%M:%S')}] Stage {stage} finished on all its workers", flush=True)

    try:
        pool.start_all()
        print("Waiting for workers to load the model, sham-check their GPUs and plan", flush=True)
        wait(lambda: all(paths.setup(shard).exists() for shard in range(n_workers)), None, Progress(paths, {}, options.progress_seconds))
        setups = [read_json(paths.setup(shard)) for shard in range(n_workers)]
        setup_report = check_setups(setups, options.sham_tolerance)
        print(f"All workers agree. {setup_report}", flush=True)
        run_warnings = list(setup_report["warnings"])
        check_stored_identities(paths.all_shard_files(), setups[0]["identity"])
        identity = setups[0]["identity"]
        for name in ("config_hash", "data_sha256", "model_commit"):
            if identity.get(name) != e1.get(name):
                raise RunFailed(f"E2's {name} is {identity.get(name)} and E1's was {e1.get(name)}, so E2 would not use E1's stimuli or weights")
        if identity.get("code_hash") != e1.get("code_hash"):
            run_warnings.append("E1's source files changed between E1's run and this one. Compare them before reading this run against E1.")
        for warning in run_warnings:
            print(f"WARNING: {warning}", flush=True)

        plan = E2Plan(RunPlan.from_setup(base_config, setups[0]), e2, l_s, alpha_s, l_w)
        stimuli = plan.base.stimuli
        common = {"l_s": l_s, "alpha_s": alpha_s, "l_w": l_w, "sites": plan.sites}

        def totals(n_lens_test: int, n_lens_calibration: int) -> dict[str, int]:
            counts = plan_counts(stimuli, base_config, plan.base.test_tasks, n_lens_test, n_lens_calibration)
            return {PHASE_GRID: len(plan.grid()), PHASE_IMPACT: counts["impact_search_max"], PHASE_TEST: counts["test"]}

        progress = Progress(paths, totals(len(stimuli.concepts_test), len(stimuli.concepts_calibration)), options.progress_seconds, n_workers)
        vector_spec = {**common, "e1_concept_vectors": str(Path(e1_dir) / "run" / "concept_vectors.pt")}
        run_stage("vectors", vector_spec, progress)

        decomposition = read_json(paths.decomposition)
        single = decomposition["single_token_concepts"]
        plan.lens_calibration = tuple(single["calibration"])
        plan.lens_test = tuple(single["test"])
        print(f"F_lens covers {len(plan.lens_test)} of {len(stimuli.concepts_test)} test concepts and {len(plan.lens_calibration)} of {len(stimuli.concepts_calibration)} calibration concepts: the others are one concept with several tokens", flush=True)
        if plan.lens_test and not plan.lens_calibration:
            # Reported rather than fatal: F_lens enters no decision rule, and it cannot be matched without calibration.
            run_warnings.append("No calibration concept is one token, so F_lens cannot be impact matched and did not run.")
            plan.lens_test = ()
        write_stage2_report(paths, e1, decomposition, stimuli.concepts_test, e2)
        progress = Progress(paths, totals(len(plan.lens_test), len(plan.lens_calibration)), options.progress_seconds, n_workers)
        print(f"Plan: {progress.totals}, with the impact searches at their worst case", flush=True)

        run_stage("grid", common, progress, plan.base.clean_calibration())
        merged = merge(paths, plan, sham_tolerance=options.sham_tolerance, extra_warnings=run_warnings)
        grid_rows = [record for record in merged["collected"].trials if record["phase"] == PHASE_GRID]

        site_alpha: dict[str, float] = {str(l_s): alpha_s}
        site_selection: dict[str, Any] | None = None
        if l_w != l_s:
            site_selection = select_site_alpha(grid_rows, plan)
            control = ResultStore(paths.control)
            stored = control.get(site_selection["key"])
            if stored is None:
                control.append([site_selection])
            elif stored["alpha"] != site_selection["alpha"]:
                raise RunFailed("Replayed alpha selection at l_w differs from the stored one")
            site_alpha[str(l_w)] = site_selection["alpha"]
            print(f"At l_w = {l_w}: alpha {site_selection['alpha']}, calibration accuracy {site_selection['accuracy']:.4f}", flush=True)

        impact_spec = {
            **common,
            "site_alpha": site_alpha,
            "lens_concepts_calibration": list(plan.lens_calibration),
        }
        run_stage("impact", impact_spec, progress)
        merged = merge(paths, plan, site_selection, finished=["impact"], sham_tolerance=options.sham_tolerance, extra_warnings=run_warnings)
        matches = check_impact_matches(merged["collected"], plan)
        for match in matches:
            state = "matched" if match["match_ok"] else "UNMATCHED, so excluded from the decision rules"
            print(f"  {match['family']} at layer {match['layer']}: alpha {match['alpha_impact']:.4g}, error {match['relative_error']:.3f}, {state}", flush=True)

        alphas = alpha_table(matches, site_alpha)
        if not plan.lens_test:
            alphas = {key: value for key, value in alphas.items() if not key.startswith("F_lens|")}
        progress.totals[PHASE_TEST] = len(plan.test(alphas))
        print(f"Test stage: {progress.totals[PHASE_TEST]} trials, full cross of E1's test concepts, clusters and positions", flush=True)
        test_spec = {**common, "site_alpha": site_alpha, "alphas": alphas, "lens_concepts_test": list(plan.lens_test)}
        run_stage("test", test_spec, progress, plan.base.clean_test())
        write_json(paths.stage_spec(E2_STAGE_ORDER.index("stop")), {"stage": "stop", "participants": []})
        # Before the merge that marks the run complete, so that a complete run always has its summary.
        write_json(paths.out_dir / "plan_summary.json", {"l_s": l_s, "alpha_s": alpha_s, "l_w": l_w, "sites": plan.sites, "alphas": alphas, "lens_test": list(plan.lens_test), "lens_calibration": list(plan.lens_calibration), "tasks": list(plan.base.test_tasks)})
        result = merge(
            paths, plan, site_selection, alphas, ["impact", "test"], complete=True, sham_tolerance=options.sham_tolerance, extra_warnings=run_warnings
        )
        print(f"Merged {result['n_records']} records into {paths.results}. {result['report']}", flush=True)
        return {"already_complete": False, "restarts": dict(pool.restarts), "merge": result["report"], "setup": setup_report}
    finally:
        pool.stop()
        held.close()


def write_stage2_report(paths: E2Paths, e1: dict[str, Any], decomposition: dict[str, Any], concepts: Sequence[str], e2: E2Config) -> dict[str, Any]:
    """Descriptive, and decides nothing. numpy only, imported here so that this module stays light."""
    from .analysis import stage2_report

    report = stage2_report(e1, decomposition, concepts, e2.n_bootstrap, e2.base.seed)
    write_json(paths.stage2, report)
    return report


def finalize(out_dir: Path, results_dir: Path, gate_file: Path, log_path: Path | None = None) -> dict[str, Any]:
    """Write the final folder from a complete store. The data files are copied before the analysis runs."""
    from .analysis import analyze_stage3, format_summary

    paths = E2Paths(Path(out_dir))
    records = read_records(paths.results)
    if not any(record.get("kind") == "run_complete" for record in records):
        raise RunFailed(f"{paths.results} has no run_complete record, so there is no final result to write")
    keys = [record["key"] for record in records]
    if len(keys) != len(set(keys)):
        raise RunFailed(f"{paths.results} repeats keys")
    e2_config = read_json(paths.out_dir / "plan_summary.json")
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    stage1 = Path(gate_file).parent
    sources = [
        paths.dry_run,
        paths.merge_report,
        paths.results,
        paths.decomposition,
        paths.stage2,
        paths.e1_inputs,
        Path(gate_file),
        stage1 / "ranks.jsonl",
        stage1 / "lens_checks.json",
        log_path,
    ]
    for source in sources:
        if source is not None and Path(source).exists():
            shutil.copyfile(source, results_dir / Path(source).name)
    try:
        analysis = analyze_stage3(records, e2_config)
        summary_body = format_summary(analysis, read_json(Path(gate_file)), read_json(paths.stage2))
    except Exception as error:
        raise RunFailed(
            f"The run is complete and its data is saved in {results_dir}, but the analysis failed: {error!r}. "
            f"Rerun with: python -m mechinterp.e2.cli finalize --out {out_dir} --results-dir {results_dir} --gate {gate_file}"
        ) from error
    write_json(results_dir / "analysis.json", analysis)
    names = sorted(path.name for path in results_dir.iterdir() if path.is_file() and path.name not in ("SHA256SUMS", "summary.txt"))
    hashes = {name: sha256_file(results_dir / name) for name in names}
    atomic_write_bytes(results_dir / "SHA256SUMS", "".join(f"{hashes[name]}  {name}\n" for name in names).encode("utf-8"))
    summary = "\n".join(
        [
            f"Results folder: {results_dir}",
            *[f"{hashes[name]}  {results_dir / name}" for name in names],
            "",
            BEGIN_ANALYSIS,
            summary_body,
            END_ANALYSIS,
            "",
        ]
    )
    atomic_write_bytes(results_dir / "summary.txt", summary.encode("utf-8"))
    return {"hashes": hashes, "summary": summary, "results_dir": str(results_dir), "analysis": analysis}


def finalize_gate_failure(out_dir: Path, results_dir: Path, gate_file: Path, log_path: Path | None = None) -> dict[str, Any]:
    """The whole result of a run that stopped at the gate."""
    gate = read_json(Path(gate_file))
    if gate.get("passed"):
        raise RunFailed("The gate passed, so this is not the final result of the run")
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    stage1 = Path(gate_file).parent
    for source in (Path(gate_file), stage1 / "ranks.jsonl", stage1 / "lens_checks.json", log_path):
        if source is not None and Path(source).exists():
            shutil.copyfile(source, results_dir / Path(source).name)
    names = sorted(path.name for path in results_dir.iterdir() if path.is_file() and path.name not in ("SHA256SUMS", "summary.txt"))
    hashes = {name: sha256_file(results_dir / name) for name in names}
    atomic_write_bytes(results_dir / "SHA256SUMS", "".join(f"{hashes[name]}  {name}\n" for name in names).encode("utf-8"))
    table = gate["analysis"]["primary_english_table"]
    lines = [describe_gate(gate), "", "layer  j_recall  logit_recall  gain  gain_lo  gain_hi"]
    lines.extend(f"{row['layer']:5d}  {row['j_recall']:.3f}  {row['logit_recall']:.3f}  {row['gain']:+.3f}  {row['gain_lo']:+.3f}  {row['gain_hi']:+.3f}" for row in table)
    summary = "\n".join([f"Results folder: {results_dir}", *[f"{hashes[name]}  {results_dir / name}" for name in names], "", BEGIN_ANALYSIS, *lines, END_ANALYSIS, ""])
    atomic_write_bytes(results_dir / "summary.txt", summary.encode("utf-8"))
    return {"hashes": hashes, "summary": summary, "results_dir": str(results_dir)}
