"""Run the E1 phases across several processes, one per GPU, with a barrier between phases.

This module never imports torch, so the orchestrator cannot initialize CUDA, and workers are started
as new processes rather than forked. The protocol between them is files only, which is what makes it
restartable:

    shards/w{i}of{n}.jsonl    Each worker's own append-only store. These hold the actual results.
    stages/NN_{stage}.json    Written by the orchestrator when a stage may start, with what it depends on.
    stages/done_{stage}_w{i}  Written by a worker when its part of a stage is on disk.
    setup/w{i}.json           Written by a worker after loading: identity, stimuli, sham and a probe.
    control.jsonl             The orchestrator's own store, holding the selection.
    results.jsonl             Derived, atomically replaced at each barrier, and what export and analysis read.

Stage files, done files and setup files last one launch. Only rows count as progress, so a rerun walks
the same stages, finds its finished trials and repeats none of them. run_complete is added only by the
final merge, after the merged keys have been checked against the plan.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .checkpoint import ResultStore, atomic_write_bytes, encode_records, read_records
from .config import Config
from .design import (
    PHASE_GRID,
    PHASE_IMPACT,
    PHASE_SCREEN,
    PHASE_TEST,
    SHAM_TOLERANCE,
    Selection,
    Stimuli,
    Trial,
    clean_store_key,
    plan_clean_keys,
    plan_grid_trials,
    plan_impact_trials,
    plan_screen_trials,
    plan_test_trials,
    select_layer_alpha,
)
from .identity import identity_differences

try:
    import fcntl
except ImportError:  # Windows, where this repo is edited but never run.
    fcntl = None  # type: ignore[assignment]

STAGE_ORDER = ("vectors", "grid", "impact", "screen", "test", "stop")
STAGE_PHASE = {"grid": PHASE_GRID, "impact": PHASE_IMPACT, "screen": PHASE_SCREEN, "test": PHASE_TEST}
# Same model, same prompt, different GPU or process. Rounding is around 1e-5 logits, and every decision in
# the prereg sits far above this, so a larger gap means a faulty card or a bug rather than numerics.
CLEAN_AGREEMENT_TOLERANCE = 1e-2
EXIT_FATAL = 3
BYTES_PER_GB = 1024**3
BEGIN_ANALYSIS = "=====BEGIN E1 ANALYSIS====="
END_ANALYSIS = "=====END E1 ANALYSIS====="


class RunFailed(RuntimeError):
    """The run stopped without a complete result. Rerunning the same command resumes it."""


class MergeError(RunFailed):
    """Shard files do not add up to the plan."""


def shard_of(key: str, n_shards: int) -> int:
    """A stable assignment. Python's hash() is salted per process, so it must never be used here."""
    if n_shards < 1:
        raise ValueError("n_shards must be at least 1")
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") % n_shards


def shard_trials(trials: Iterable[Trial], shard: int, n_shards: int) -> list[Trial]:
    return [trial for trial in trials if shard_of(trial.key, n_shards) == shard]


def stage_participants(stage: str, n_workers: int) -> list[int]:
    """Vectors must be one shared set, and the impact search is sequential, so both stay on worker 0."""
    if stage in ("vectors", "impact"):
        return [0]
    if stage in ("grid", "screen", "test"):
        return list(range(n_workers))
    return []


@dataclass(frozen=True)
class RunPaths:
    out_dir: Path

    @property
    def results(self) -> Path:
        return self.out_dir / "results.jsonl"

    @property
    def control(self) -> Path:
        return self.out_dir / "control.jsonl"

    @property
    def merge_report(self) -> Path:
        return self.out_dir / "merge_report.json"

    @property
    def dry_run(self) -> Path:
        return self.out_dir / "preflight" / "dry_run.json"

    @property
    def stages_dir(self) -> Path:
        return self.out_dir / "stages"

    def shard_file(self, shard: int, n_shards: int) -> Path:
        return self.out_dir / "shards" / f"w{shard}of{n_shards}.jsonl"

    def all_shard_files(self) -> list[Path]:
        return sorted((self.out_dir / "shards").glob("w*of*.jsonl"))

    def files_from_other_worker_counts(self, n_shards: int) -> list[Path]:
        return [path for path in self.all_shard_files() if not path.stem.endswith(f"of{n_shards}")]

    def stage_spec(self, index: int) -> Path:
        return self.stages_dir / f"{index:02d}_{STAGE_ORDER[index]}.json"

    def stage_done(self, stage: str, shard: int) -> Path:
        return self.stages_dir / f"done_{stage}_w{shard}"

    def setup(self, shard: int) -> Path:
        return self.out_dir / "setup" / f"w{shard}.json"

    def heartbeat(self, shard: int) -> Path:
        return self.out_dir / "setup" / f"w{shard}.heartbeat"

    def log(self, shard: int, n_shards: int) -> Path:
        return self.out_dir / "logs" / f"w{shard}of{n_shards}.log"

    def lock(self, name: str) -> Path:
        return self.out_dir / "locks" / f"{name}.lock"


def try_lock(path: Path) -> Any:
    """An exclusive lock the kernel releases when the holder dies, or None if someone holds it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a", encoding="utf-8")
    if fcntl is None:
        return handle
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


def write_json(path: Path, value: Any) -> None:
    # No sort_keys: cluster order decides the design's cells and must survive the round trip.
    atomic_write_bytes(path, json.dumps(value, indent=1).encode("utf-8"))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


# The plan


@dataclass(frozen=True)
class RunPlan:
    config: Config
    stimuli: Stimuli
    test_tasks: tuple[str, ...]

    @classmethod
    def from_setup(cls, config: Config, setup: dict[str, Any]) -> "RunPlan":
        stimuli = setup["stimuli"]
        return cls(
            config=config,
            stimuli=Stimuli(
                concepts_calibration=tuple(stimuli["concepts_calibration"]),
                concepts_test=tuple(stimuli["concepts_test"]),
                baseline_words=tuple(stimuli["baseline_words"]),
                clusters_calibration={name: tuple(words) for name, words in stimuli["clusters_calibration"].items()},
                clusters_test={name: tuple(words) for name, words in stimuli["clusters_test"].items()},
                dropped_words=stimuli["dropped_words"],
                dropped_clusters=stimuli["dropped_clusters"],
            ),
            test_tasks=tuple(setup["test_tasks"]),
        )

    def grid(self) -> list[Trial]:
        return plan_grid_trials(self.stimuli, self.config)

    def impact(self, layer: int, alphas: Sequence[float]) -> list[Trial]:
        # The search can revisit an alpha after rounding, and such a visit stores no new rows.
        unique = list(dict.fromkeys(alphas))
        return [trial for alpha in unique for trial in plan_impact_trials(self.stimuli, self.config, layer, alpha)]

    def screen(self, layer: int, alpha: float) -> list[Trial]:
        if not self.config.screening.enabled:
            return []
        return plan_screen_trials(self.stimuli, self.config, layer, alpha)

    def test(self, layer: int, alpha: float, alpha_impact: float) -> list[Trial]:
        return plan_test_trials(self.stimuli, self.config, layer, alpha, alpha_impact, self.test_tasks)

    def clean_calibration(self) -> list[str]:
        return [clean_store_key(key) for key in plan_clean_keys(["T_inj"], self.stimuli.clusters_calibration)]

    def clean_test(self) -> list[str]:
        return [clean_store_key(key) for key in plan_clean_keys(self.test_tasks, self.stimuli.clusters_test)]


def setup_payload(session: Any, probe: list[float], gpu_name: str | None) -> dict[str, Any]:
    return {
        "identity": session.identity,
        "stimuli": asdict(session.stimuli),
        "test_tasks": list(session.test_tasks),
        "sham_max_abs_L": session.store.get("sham")["max_abs_L"],
        "probe_label_logits": probe,
        "gpu_name": gpu_name,
    }


def check_setups(setups: Sequence[dict[str, Any]], sham_tolerance: float) -> dict[str, Any]:
    """Before any trial runs: all workers must be the same run, and every GPU must pass the sham check."""
    first = setups[0]
    probe_gap = 0.0
    for index, setup in enumerate(setups):
        differences = identity_differences(first["identity"], setup["identity"])
        if differences:
            raise RunFailed(f"Worker {index} has a different run identity from worker 0: {differences}")
        for name in ("stimuli", "test_tasks"):
            if setup[name] != first[name]:
                raise RunFailed(f"Worker {index} planned different {name} from worker 0")
        if abs(setup["sham_max_abs_L"]) > sham_tolerance:
            raise RunFailed(
                f"Sham check failed on worker {index}: |L| = {setup['sham_max_abs_L']} at alpha = 0, "
                f"tolerance {sham_tolerance}"
            )
        if len(setup["probe_label_logits"]) != len(first["probe_label_logits"]):
            raise RunFailed(f"Worker {index} scored a different number of labels from worker 0")
        for mine, theirs in zip(setup["probe_label_logits"], first["probe_label_logits"]):
            probe_gap = max(probe_gap, abs(mine - theirs))
    if probe_gap > CLEAN_AGREEMENT_TOLERANCE:
        raise RunFailed(f"Workers disagree by {probe_gap} logits on one identical prompt: suspect a faulty GPU")
    return {"probe_max_abs_diff": probe_gap, "gpus": [setup.get("gpu_name") for setup in setups]}


# Merging


def check_exact_keys(found: Sequence[str], expected: Sequence[str], label: str) -> None:
    """The found keys must equal the planned keys: nothing repeated, nothing missing, nothing extra."""
    expected_set = set(expected)
    if len(expected_set) != len(expected):
        raise MergeError(f"{label}: the plan itself repeats keys")
    counts = Counter(found)
    repeated = sorted(key for key, count in counts.items() if count > 1)
    missing = sorted(expected_set - counts.keys())
    extra = sorted(counts.keys() - expected_set)
    if repeated or missing or extra:
        raise MergeError(
            f"{label}: {len(repeated)} repeated, {len(missing)} missing and {len(extra)} unplanned keys, "
            f"of {len(expected)} planned. Examples: repeated {repeated[:2]}, missing {missing[:2]}, "
            f"unplanned {extra[:2]}"
        )


@dataclass
class Collected:
    manifests: list[dict[str, Any]] = field(default_factory=list)
    trials: list[dict[str, Any]] = field(default_factory=list)
    cleans: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    shams: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    impact_matches: list[dict[str, Any]] = field(default_factory=list)


def collect(files: Iterable[Path]) -> Collected:
    collected = Collected()
    for path in sorted(files):
        for record in read_records(path):
            kind = record.get("kind")
            if kind == "manifest":
                collected.manifests.append(record)
            elif kind == "trial":
                collected.trials.append(record)
            elif kind == "clean":
                collected.cleans.setdefault(record["key"], []).append(record)
            elif kind == "sham":
                collected.shams.append(record)
            elif kind == "vector_diagnostics":
                collected.diagnostics.append(record)
            elif kind == "impact_match":
                collected.impact_matches.append(record)
            else:
                raise MergeError(f"Unexpected record kind {kind!r} in {path}")
    return collected


def _trials_of_phase(collected: Collected, phase: str, plan_trials: Sequence[Trial], label: str) -> list[dict[str, Any]]:
    found = [record for record in collected.trials if record["phase"] == phase]
    check_exact_keys([record["key"] for record in found], [trial.key for trial in plan_trials], label)
    by_key = {record["key"]: record for record in found}
    return [by_key[trial.key] for trial in plan_trials]


def _canonical_clean(collected: Collected, keys: Sequence[str], label: str) -> tuple[list[dict[str, Any]], float]:
    """One clean row per key, from the first shard, after checking that all shards agree with it."""
    canonical = []
    worst = 0.0
    for key in keys:
        copies = collected.cleans.get(key)
        if not copies:
            raise MergeError(f"{label}: clean row {key} is missing")
        first = copies[0]
        for copy in copies[1:]:
            if copy["prompt_hash"] != first["prompt_hash"]:
                raise MergeError(f"{label}: shards tokenized {key} differently")
            if len(copy["logits"]) != len(first["logits"]):
                raise MergeError(f"{label}: shards scored different answers on {key}")
            for mine, theirs in zip(copy["logits"], first["logits"]):
                worst = max(worst, abs(mine - theirs))
        canonical.append(first)
    if worst > CLEAN_AGREEMENT_TOLERANCE:
        raise MergeError(f"{label}: clean logits differ by {worst} between shards: suspect a faulty GPU")
    return canonical, worst


def _check_pairing(rows: Sequence[dict[str, Any]], clean: dict[str, dict[str, Any]], label: str) -> float:
    """Each injected row must pair with the clean row for its own tokens, and reproduce its m_clean."""
    worst = 0.0
    for row in rows:
        record = clean.get(clean_store_key((row["task"], row["cluster"], row["query"])))
        if record is None:
            raise MergeError(f"{label}: no clean row for {row['key']}")
        if record["prompt_hash"] != row["prompt_hash"]:
            raise MergeError(f"{label}: clean and injected tokens differ for {row['key']}")
        labels = record["logits"][:10]
        target = labels[row["position"]]
        worst = max(worst, abs(row["m_clean"] - (target - (sum(labels) - target) / 9)))
    if worst > CLEAN_AGREEMENT_TOLERANCE:
        raise MergeError(f"{label}: m_clean differs from the clean row by {worst}")
    return worst


def merge(
    paths: RunPaths,
    plan: RunPlan,
    selection: dict[str, Any] | None = None,
    finished: Sequence[str] = (),
    complete: bool = False,
) -> dict[str, Any]:
    """Write results.jsonl from the shard files, holding only finished stages, in chronological order.

    Rows from a stage that has not been finished in this launch are left out. Raises rather than writes
    if any key is repeated, missing or unplanned, if identities differ, or if shards disagree.
    """
    collected = collect(paths.all_shard_files())
    if not collected.manifests:
        raise MergeError("No manifest in any shard")
    identities = [manifest.get("identity") or {} for manifest in collected.manifests]
    for manifest, identity in zip(collected.manifests, identities):
        differences = identity_differences(identities[0], identity)
        if differences:
            raise MergeError(f"Shards differ in run identity: {differences}")
        if manifest["config_hash"] != plan.config.config_hash:
            raise MergeError("A shard was written under a different config")
    manifests = [{**manifest, "key": f"manifest|{index}"} for index, manifest in enumerate(collected.manifests)]

    if not collected.shams:
        raise MergeError("No sham check in any shard")
    worst_sham = max(record["max_abs_L"] for record in collected.shams)
    if worst_sham > SHAM_TOLERANCE:
        raise MergeError(f"A shard's sham check gave |L| = {worst_sham}")
    sham = {"key": "sham", "kind": "sham", "max_abs_L": worst_sham, "n": collected.shams[0]["n"], "n_checks": len(collected.shams)}

    if not collected.diagnostics:
        raise MergeError("No concept vector diagnostics in any shard")
    if len(collected.impact_matches) > 1:
        raise MergeError(f"{len(collected.impact_matches)} impact_match records: the search must run once")

    report: dict[str, Any] = {
        "shard_files": [path.name for path in paths.all_shard_files()],
        "n_sessions": len(manifests),
        "worst_sham_abs_L": worst_sham,
        "trials_by_phase": {},
    }
    records: list[dict[str, Any]] = [*manifests, collected.diagnostics[0], sham]

    clean_rows, clean_gap = _canonical_clean(collected, plan.clean_calibration(), "calibration clean")
    clean_by_key = {record["key"]: record for record in clean_rows}
    records.extend(clean_rows)
    pairing_gap = 0.0

    def add_phase(phase: str, plan_trials: Sequence[Trial]) -> None:
        nonlocal pairing_gap
        rows = _trials_of_phase(collected, phase, plan_trials, phase)
        pairing_gap = max(pairing_gap, _check_pairing(rows, clean_by_key, phase))
        records.extend(rows)
        report["trials_by_phase"][phase] = len(rows)

    add_phase(PHASE_GRID, plan.grid())
    if selection is not None:
        records.append(selection)
    impact_match = collected.impact_matches[0] if collected.impact_matches else None
    if "impact" in finished:
        if selection is None or impact_match is None:
            raise MergeError("The impact stage finished without a selection and an impact_match")
        alphas = [item["alpha"] for item in impact_match["history"]]
        add_phase(PHASE_IMPACT, plan.impact(selection["layer"], alphas))
        records.append(impact_match)
    if "screen" in finished:
        add_phase(PHASE_SCREEN, plan.screen(selection["layer"], selection["alpha"]))
    if "test" in finished:
        test_clean, test_gap = _canonical_clean(collected, plan.clean_test(), "test clean")
        clean_gap = max(clean_gap, test_gap)
        clean_by_key.update({record["key"]: record for record in test_clean})
        records.extend(test_clean)
        add_phase(PHASE_TEST, plan.test(selection["layer"], selection["alpha"], impact_match["alpha_impact"]))

    report["clean_max_abs_diff_between_shards"] = clean_gap
    report["m_clean_max_abs_diff_from_canonical"] = pairing_gap
    if complete:
        allowed_phases = {STAGE_PHASE[stage] for stage in ("grid", *finished)}
        stray = {record["phase"] for record in collected.trials} - allowed_phases
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
            }
        )
    keys = [record["key"] for record in records]
    if len(keys) != len(set(keys)):
        raise MergeError("The merged store would repeat a key")
    atomic_write_bytes(paths.results, encode_records(records))
    write_json(paths.merge_report, report)
    return {"collected": collected, "report": report, "n_records": len(records)}


def selection_record(records: Sequence[dict[str, Any]], plan: RunPlan) -> dict[str, Any]:
    selection = select_layer_alpha(records, plan.stimuli.concepts_calibration, list(plan.stimuli.clusters_calibration))
    return {
        "key": "selection",
        "kind": "selection",
        "layer": selection.layer,
        "alpha": selection.alpha,
        "accuracy": selection.accuracy,
        "grid_accuracy": selection.grid_accuracy,
        "n_trials_per_point": len(records) // len(selection.grid_accuracy),
    }


def is_complete(paths: RunPaths) -> bool:
    return any(record.get("kind") == "run_complete" for record in read_records(paths.results))


# Supervision


@dataclass
class RunOptions:
    config_path: Path
    out_dir: Path
    devices: list[str]  # One worker per entry. "" hides all GPUs from that worker, for CPU tests.
    factory: str | None = None
    max_restarts: int = 3
    load_slots: int = 1
    threads_per_worker: int | None = None
    poll_seconds: float = 2.0
    progress_seconds: float = 60.0
    restart_delay_seconds: float = 15.0
    stall_seconds: float = 1800.0
    sham_tolerance: float = SHAM_TOLERANCE
    min_free_disk_bytes: int = BYTES_PER_GB
    extra_env: dict[str, str] = field(default_factory=dict)


class WorkerPool:
    """One process per device, restarted a limited number of times."""

    def __init__(self, options: RunOptions, paths: RunPaths) -> None:
        self.options = options
        self.paths = paths
        self.n = len(options.devices)
        self.processes: dict[int, subprocess.Popen] = {}
        self.restarts = {index: 0 for index in range(self.n)}
        self.stopping = False

    def _command(self, shard: int) -> list[str]:
        options = self.options
        command = [
            sys.executable,
            "-u",
            "-m",
            "mechinterp.cli",
            "shard-worker",
            "--config",
            str(options.config_path),
            "--out",
            str(options.out_dir),
            "--shard",
            str(shard),
            "--n-shards",
            str(self.n),
            "--load-slots",
            str(options.load_slots),
        ]
        if options.devices[shard]:
            command.append("--require-cuda")
        if options.factory:
            command.extend(["--factory", options.factory])
        return command

    def _environment(self, shard: int) -> dict[str, str]:
        options = self.options
        environment = os.environ.copy()
        environment.update(options.extra_env)
        threads = options.threads_per_worker or max(1, min(8, (os.cpu_count() or 1) // self.n))
        source_root = str(Path(__file__).resolve().parents[1])
        environment["PYTHONPATH"] = os.pathsep.join(filter(None, [source_root, environment.get("PYTHONPATH", "")]))
        environment.update(
            {
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",  # So that an index means what nvidia-smi says it means.
                "CUDA_VISIBLE_DEVICES": options.devices[shard],
                "OMP_NUM_THREADS": str(threads),
                "MKL_NUM_THREADS": str(threads),
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONUNBUFFERED": "1",
                # Exact results from cuBLAS, and less fragmentation. Neither changes what is computed.
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            }
        )
        return environment

    def start(self, shard: int) -> None:
        log_path = self.paths.log(shard, self.n)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        touch(self.paths.heartbeat(shard))
        with open(log_path, "ab") as log:
            self.processes[shard] = subprocess.Popen(
                self._command(shard), env=self._environment(shard), stdout=log, stderr=subprocess.STDOUT
            )

    def start_all(self) -> None:
        for shard in range(self.n):
            self.start(shard)

    def _log_tail(self, shard: int, lines: int = 25) -> str:
        path = self.paths.log(shard, self.n)
        if not path.exists():
            return "(no log)"
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - 8192))
            tail = handle.read().decode("utf-8", errors="replace")
        return "\n".join(tail.splitlines()[-lines:])

    def _handle_exit(self, shard: int, code: int) -> None:
        print(f"Worker {shard} exited with code {code}. Its log ends with:\n{self._log_tail(shard)}", flush=True)
        if code == EXIT_FATAL:
            raise RunFailed(f"Worker {shard} hit an error that a restart cannot fix. See {self.paths.log(shard, self.n)}")
        if self.restarts[shard] >= self.options.max_restarts:
            raise RunFailed(
                f"Worker {shard} failed {self.restarts[shard] + 1} times, so the run stopped. "
                f"See {self.paths.log(shard, self.n)}"
            )
        self.restarts[shard] += 1
        print(f"Restarting worker {shard}, attempt {self.restarts[shard]} of {self.options.max_restarts}", flush=True)
        time.sleep(self.options.restart_delay_seconds)
        self.start(shard)

    def check(self) -> None:
        """Restart any dead or stalled worker, or raise once a worker has used up its restarts."""
        if self.stopping:
            return
        free = shutil.disk_usage(self.options.out_dir).free
        if free < self.options.min_free_disk_bytes:
            raise RunFailed(f"Only {free / BYTES_PER_GB:.1f} GB of disk left at {self.options.out_dir}")
        for shard, process in list(self.processes.items()):
            code = process.poll()
            if code is not None:
                self._handle_exit(shard, code)
                continue
            last = max(
                (path.stat().st_mtime for path in (self.paths.shard_file(shard, self.n), self.paths.heartbeat(shard)) if path.exists()),
                default=time.time(),
            )
            if time.time() - last > self.options.stall_seconds:
                print(f"Worker {shard} wrote nothing for {self.options.stall_seconds:.0f} s, so it is being killed", flush=True)
                process.kill()
                process.wait()
                self._handle_exit(shard, -9)

    def stop(self) -> None:
        self.stopping = True
        for process in self.processes.values():
            if process.poll() is None:
                process.terminate()
        deadline = time.monotonic() + 30.0
        for process in self.processes.values():
            try:
                process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def gpu_line() -> str:
    if shutil.which("nvidia-smi") is None:
        return "no nvidia-smi"
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return "nvidia-smi did not answer"
    parts = []
    for line in output.strip().splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) == 4:
            parts.append(f"gpu{fields[0]} {fields[1]}% {int(fields[2]) / 1024:.1f}GiB {fields[3]}C")
    return " | ".join(parts)


def _duration(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


class Progress:
    """Prints stage progress about once a minute, counting rows without parsing them."""

    def __init__(self, paths: RunPaths, totals: dict[str, int], seconds: float) -> None:
        self.paths = paths
        self.totals = totals
        self.seconds = seconds
        self.last_print = 0.0
        self.stage_started = time.monotonic()
        self.stage_base = 0
        self.best_rate = 0.0

    def counts(self) -> Counter:
        counts: Counter = Counter()
        for path in self.paths.all_shard_files():
            raw = path.read_bytes()
            # Stores write compact JSON, so these byte patterns are exact.
            for phase in self.totals:
                counts[phase] += raw.count(f'"phase":"{phase}"'.encode("ascii"))
        return counts

    def begin(self, phase: str | None) -> None:
        self.stage_started = time.monotonic()
        self.stage_base = self.counts()[phase] if phase else 0
        self.last_print = 0.0

    def tick(self, phase: str | None, restarts: dict[int, int]) -> None:
        now = time.monotonic()
        if now - self.last_print < self.seconds:
            return
        self.last_print = now
        counts = self.counts()
        parts = [f"[{time.strftime('%H:%M:%S')}]"]
        if phase:
            total = self.totals[phase]
            done = counts[phase]
            rate = (done - self.stage_base) / max(1.0, now - self.stage_started)
            self.best_rate = max(self.best_rate, rate)
            stage_eta = _duration((total - done) / rate) if rate > 0 else "?"
            parts.append(f"{phase} {done}/{total} {rate:.1f}/s stage ETA {stage_eta}")
            remaining = sum(max(0, self.totals[name] - counts[name]) for name in self.totals)
            if self.best_rate > 0:
                parts.append(f"all ETA at most {_duration(remaining / self.best_rate)}")
        else:
            parts.append("setting up")
        if any(restarts.values()):
            parts.append(f"restarts {sum(restarts.values())}")
        print("  ".join(parts), flush=True)
        print(f"           {gpu_line()}", flush=True)


def orchestrate(options: RunOptions) -> dict[str, Any]:
    """Run or resume everything. Returns a report, or raises RunFailed with no run_complete written."""
    paths = RunPaths(Path(options.out_dir))
    paths.out_dir.mkdir(parents=True, exist_ok=True)
    held = try_lock(paths.lock("orchestrator"))
    if held is None:
        raise RunFailed(f"Another orchestrator is already running on {paths.out_dir}")
    if is_complete(paths):
        print(f"{paths.results} is already complete", flush=True)
        held.close()
        return {"already_complete": True, "restarts": {}}
    # Per-launch files. Rows are the only state that carries over.
    for transient in (paths.stages_dir, paths.out_dir / "setup"):
        if transient.exists():
            shutil.rmtree(transient)
        transient.mkdir(parents=True)
    config = Config.from_file(options.config_path)
    n_workers = len(options.devices)
    if n_workers < 1:
        raise RunFailed("No workers to start")
    print(f"Starting {n_workers} worker(s) on devices {options.devices}", flush=True)
    print(f"Shard files already present: {[path.name for path in paths.all_shard_files()]}", flush=True)

    pool = WorkerPool(options, paths)

    def wait(condition: Any, phase: str | None, progress: Progress | None) -> None:
        while not condition():
            pool.check()
            if progress is not None:
                progress.tick(phase, pool.restarts)
            time.sleep(options.poll_seconds)

    def run_stage(index: int, spec: dict[str, Any], progress: Progress) -> None:
        stage = STAGE_ORDER[index]
        participants = stage_participants(stage, n_workers)
        if stage == "screen" and not config.screening.enabled:
            participants = []
        phase = STAGE_PHASE.get(stage)
        progress.begin(phase)
        print(f"[{time.strftime('%H:%M:%S')}] Stage {stage} starts on workers {participants}", flush=True)
        write_json(paths.stage_spec(index), {"stage": stage, "participants": participants, **spec})
        wait(lambda: all(paths.stage_done(stage, shard).exists() for shard in participants), phase, progress)
        print(f"[{time.strftime('%H:%M:%S')}] Stage {stage} finished on all its workers", flush=True)

    try:
        pool.start_all()
        print("Waiting for workers to load the model, sham-check their GPUs and plan", flush=True)
        wait(lambda: all(paths.setup(shard).exists() for shard in range(n_workers)), None, Progress(paths, {}, options.progress_seconds))
        setups = [read_json(paths.setup(shard)) for shard in range(n_workers)]
        setup_report = check_setups(setups, options.sham_tolerance)
        print(f"All workers agree. {setup_report}", flush=True)
        plan = RunPlan.from_setup(config, setups[0])

        grid = plan.grid()
        totals = {
            PHASE_GRID: len(grid),
            PHASE_IMPACT: len(plan.impact(0, [1.0])) * config.impact_search.max_evals,
            PHASE_SCREEN: len(plan.screen(0, 1.0)),
            PHASE_TEST: len(plan.test(0, 1.0, 1.0)),
        }
        print(f"Plan: {totals}", flush=True)
        progress = Progress(paths, totals, options.progress_seconds)
        run_stage(STAGE_ORDER.index("vectors"), {}, progress)
        run_stage(STAGE_ORDER.index("grid"), {}, progress)

        # Barrier: the selection reads every shard's grid rows, and nothing later can start before it.
        merged = merge(paths, plan)
        grid_rows = [record for record in merged["collected"].trials if record["phase"] == PHASE_GRID]
        selection = selection_record(grid_rows, plan)
        control = ResultStore(paths.control)
        stored = control.get("selection")
        if stored is None:
            control.append([selection])
        elif (stored["layer"], stored["alpha"]) != (selection["layer"], selection["alpha"]):
            raise RunFailed("Replayed selection differs from the stored selection")
        print(f"Selected layer {selection['layer']}, alpha {selection['alpha']}, accuracy {selection['accuracy']:.4f}", flush=True)
        selection_fields = {name: selection[name] for name in Selection.__dataclass_fields__}

        run_stage(STAGE_ORDER.index("impact"), {"selection": selection_fields}, progress)
        merged = merge(paths, plan, selection, ["impact"])
        impact_matches = merged["collected"].impact_matches
        alpha_impact = impact_matches[0]["alpha_impact"]
        print(f"Impact-matched alpha {alpha_impact}, match_ok {impact_matches[0]['match_ok']}", flush=True)
        later = {"selection": selection_fields, "alpha_impact": alpha_impact}

        run_stage(STAGE_ORDER.index("screen"), later, progress)
        finished = ["impact", *(["screen"] if config.screening.enabled else [])]
        merge(paths, plan, selection, finished)

        run_stage(STAGE_ORDER.index("test"), later, progress)
        write_json(paths.stage_spec(STAGE_ORDER.index("stop")), {"stage": "stop", "participants": []})
        result = merge(paths, plan, selection, [*finished, "test"], complete=True)
        print(f"Merged {result['n_records']} records into {paths.results}. {result['report']}", flush=True)
        return {"already_complete": False, "restarts": dict(pool.restarts), "merge": result["report"], "setup": setup_report}
    finally:
        pool.stop()
        held.close()


# Final results


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finalize(out_dir: Path, results_dir: Path, log_path: Path | None = None) -> dict[str, Any]:
    """Write the final folder from a complete merged store. An incomplete store raises and writes nothing."""
    from .analysis import analyze, format_report  # Needs numpy, which the orchestrator does not.
    from .export import build_export, encode_export

    paths = RunPaths(Path(out_dir))
    records = read_records(paths.results)
    if not any(record.get("kind") == "run_complete" for record in records):
        raise RunFailed(f"{paths.results} has no run_complete record, so there is no final result to write")
    keys = [record["key"] for record in records]
    if len(keys) != len(set(keys)):
        raise RunFailed(f"{paths.results} repeats keys")
    export = build_export(records)
    report = analyze(export)
    if report["outcome"] == "incomplete":
        raise RunFailed("The analysis says the run is incomplete")
    analysis_text = format_report(report)

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(results_dir / "export.txt", encode_export(export).encode("utf-8"))
    atomic_write_bytes(results_dir / "analysis.txt", analysis_text.encode("utf-8"))
    for source in (paths.dry_run, paths.merge_report, paths.results, log_path):
        if source is not None and Path(source).exists():
            shutil.copyfile(source, results_dir / Path(source).name)
    names = sorted(path.name for path in results_dir.iterdir() if path.is_file() and path.name not in ("SHA256SUMS", "summary.txt"))
    hashes = {name: sha256_file(results_dir / name) for name in names}
    atomic_write_bytes(results_dir / "SHA256SUMS", "".join(f"{hashes[name]}  {name}\n" for name in names).encode("utf-8"))
    summary = "\n".join(
        [
            f"Outcome: {report['outcome']}",
            f"Results folder: {results_dir}",
            *[f"{hashes[name]}  {results_dir / name}" for name in names],
            "",
            BEGIN_ANALYSIS,
            analysis_text,
            END_ANALYSIS,
            "",
        ]
    )
    atomic_write_bytes(results_dir / "summary.txt", summary.encode("utf-8"))
    return {"outcome": report["outcome"], "hashes": hashes, "summary": summary, "results_dir": str(results_dir)}
