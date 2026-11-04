"""D1 from start to verdict. Never imports torch. Workers are new processes, one at a time per GPU.

Walks the same steps on every launch. A task that finished has a marker and is skipped, a capture resumes from its done
mask, and gates are recomputed from stored numbers, so a rerun reaches the same decisions. Every outcome the prereg names
is a normal finish: a failed G1 through the ladder, a failed G2 after its retry, and each verdict, inconclusive included.
A stop for lack of memory is not a finding, and is reported as a failed run.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from ..checkpoint import atomic_write_bytes
from ..sharding import RunFailed, _duration, gpu_line, kill_bounded, sha256_file, try_lock
from .config import CONDITIONS, D1Config, Rung, layers_for
from .gates import G1_FINDING, G2_FINDING, Thresholds, next_after_g1, next_after_g2
from .paths import BASE_MODEL, D1Paths, adapter_id

EXIT_FATAL = 3
MIN_FREE_DISK_GB = 2.0
DOWNLOAD_SECONDS = 7200
BEGIN_REPORT = "=====BEGIN D1 REPORT====="
END_REPORT = "=====END D1 REPORT====="


@dataclass
class Task:
    name: str
    argv: list[str]
    gpu: bool = True
    loads_model: bool = True


@dataclass
class PoolOptions:
    devices: list[str]
    config_path: Path
    out_dir: Path
    factory: str | None = None
    threads_per_worker: int | None = None
    load_slots: int = 1
    max_restarts: int = 2
    poll_seconds: float = 2.0
    progress_seconds: float = 60.0
    stall_seconds: float = 1800.0
    load_stall_seconds: float = 1800.0
    kill_wait_seconds: float = 60.0
    extra_env: dict[str, str] = field(default_factory=dict)


class TaskPool:
    """Runs tasks one per device, restarts a failed one a limited number of times, and kills a stalled one."""

    def __init__(self, options: PoolOptions, paths: D1Paths) -> None:
        self.options = options
        self.paths = paths
        self.elapsed_by_kind: dict[str, list[float]] = {}

    def _environment(self, device: str | None) -> dict[str, str]:
        options = self.options
        environment = os.environ.copy()
        environment.update(options.extra_env)
        n = max(1, len(options.devices))
        threads = options.threads_per_worker or max(1, min(8, (os.cpu_count() or 1) // n))
        source_root = str(Path(__file__).resolve().parents[2])
        environment["PYTHONPATH"] = os.pathsep.join(filter(None, [source_root, environment.get("PYTHONPATH", "")]))
        environment.update(
            {
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": device if device is not None else "",
                "OMP_NUM_THREADS": str(threads),
                "MKL_NUM_THREADS": str(threads),
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONUNBUFFERED": "1",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            }
        )
        return environment

    def _command(self, task: Task, device: str | None) -> list[str]:
        options = self.options
        command = [sys.executable, "-u", "-m", "mechinterp.d1.worker", task.argv[0], "--config", str(options.config_path), "--out", str(options.out_dir), "--task-name", task.name, *task.argv[1:]]
        if options.factory:
            command.extend(["--factory", options.factory])
        if device:
            command.append("--require-cuda")
        return command

    def _start(self, task: Task, device: str | None) -> subprocess.Popen:
        log_path = self.paths.logs / f"{task.name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.paths.heartbeat(task.name).unlink(missing_ok=True)
        with open(log_path, "ab") as log:
            return subprocess.Popen(self._command(task, device), env=self._environment(device), stdout=log, stderr=subprocess.STDOUT)

    def _tail(self, task: Task, lines: int = 25) -> str:
        path = self.paths.logs / f"{task.name}.log"
        if not path.exists():
            return "(no log)"
        raw = path.read_bytes()[-6000:]
        return "\n".join(raw.decode("utf-8", errors="replace").splitlines()[-lines:])

    def _beat(self, task: Task) -> dict[str, Any] | None:
        try:
            return json.loads(self.paths.heartbeat(task.name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def run(self, label: str, tasks: Sequence[Task]) -> None:
        """Run what is not yet done. Returns when all are done, or raises RunFailed."""
        options = self.options
        pending = [task for task in tasks if not self.paths.done(task.name).exists()]
        if not pending:
            return
        print(f"[{time.strftime('%H:%M:%S')}] {label}: {len(pending)} task(s) of {len(tasks)} to run", flush=True)
        # "" hides every GPU from a worker, for CPU runs. None means a task that takes no device slot at all.
        free: list[str] = list(options.devices) if options.devices else [""]
        running: dict[str, tuple[Task, subprocess.Popen, str | None, float]] = {}
        restarts: dict[str, int] = {}
        queue = list(pending)
        last_print = 0.0
        stage_started = time.monotonic()
        finished = 0
        try:
            while queue or running:
                disk = shutil.disk_usage(options.out_dir).free / 1024**3
                if disk < MIN_FREE_DISK_GB:
                    raise RunFailed(f"Only {disk:.1f} GB of disk left at {options.out_dir}")
                # Only so many loads at once, because weights pass through RAM on their way to the GPU.
                loading = sum(1 for task, *_ in running.values() if task.loads_model and not (self._beat(task) or {}).get("loaded"))
                while queue and free and (not queue[0].loads_model or loading < options.load_slots):
                    task = queue.pop(0)
                    device = free.pop(0) if task.gpu else None
                    running[task.name] = (task, self._start(task, device), device, time.time())
                    if task.loads_model:
                        loading += 1
                for name, (task, process, device, started) in list(running.items()):
                    code = process.poll()
                    if code is None:
                        beat = self._beat(task)
                        limit = options.stall_seconds if beat and beat.get("loaded") else options.load_stall_seconds
                        try:
                            last = self.paths.heartbeat(name).stat().st_mtime
                        except OSError:
                            last = started
                        if task.loads_model and time.time() - max(last, started) > limit:
                            print(f"{name}: no progress for {limit / 60:.0f} minutes, so its GPU may be wedged. Killing it.", flush=True)
                            if not kill_bounded(process, options.kill_wait_seconds):
                                raise RunFailed(f"{name} cannot be killed, so it is stuck in the GPU driver. This machine probably needs a reboot.")
                            code = -9
                        else:
                            continue
                    del running[name]
                    if device is not None:
                        free.insert(0, device)
                    if code == 0:
                        marker = self.paths.done(name)
                        marker.parent.mkdir(parents=True, exist_ok=True)
                        marker.touch()
                        finished += 1
                        self.elapsed_by_kind.setdefault(task.argv[0], []).append(time.time() - started)
                        continue
                    print(f"{name} exited with code {code}. Its log ends with:\n{self._tail(task)}", flush=True)
                    if code == 2:
                        raise RunFailed(f"The dry-run checks failed for {name}. See PASS and FAIL in {self.paths.logs / (name + '.log')}")
                    if code == EXIT_FATAL:
                        raise RunFailed(f"{name} hit an error that a restart cannot fix. See {self.paths.logs / (name + '.log')}")
                    restarts[name] = restarts.get(name, 0) + 1
                    if restarts[name] > options.max_restarts:
                        raise RunFailed(f"{name} failed {restarts[name]} times, so the run stopped. See {self.paths.logs / (name + '.log')}")
                    print(f"Restarting {name}, attempt {restarts[name]} of {options.max_restarts}", flush=True)
                    queue.insert(0, task)
                now = time.monotonic()
                if now - last_print >= options.progress_seconds:
                    last_print = now
                    self._print_progress(label, running, len(queue), finished, stage_started, now)
                if queue or running:
                    time.sleep(options.poll_seconds)
        except BaseException:
            for _task, process, *_rest in running.values():
                if process.poll() is None:
                    process.terminate()
            deadline = time.monotonic() + 30.0
            for _task, process, *_rest in running.values():
                try:
                    process.wait(timeout=max(0.1, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    if not kill_bounded(process, options.kill_wait_seconds):
                        print(f"WARNING: process {process.pid} cannot be killed, so it is being left behind.", flush=True)
            raise
        print(f"[{time.strftime('%H:%M:%S')}] {label}: finished in {_duration(time.monotonic() - stage_started)}", flush=True)

    def _print_progress(self, label: str, running: dict[str, Any], queued: int, finished: int, stage_started: float, now: float) -> None:
        parts = []
        worst_left = 0.0
        for name, (task, *_rest) in running.items():
            beat = self._beat(task)
            if beat and beat.get("total"):
                left = (beat["total"] - beat["done"]) / beat["rate"] if beat.get("rate") else 0.0
                worst_left = max(worst_left, left)
                parts.append(f"{name} {beat['done']}/{beat['total']}")
            else:
                parts.append(f"{name} loading")
        finished_time = (now - stage_started) / finished if finished else None
        queue_estimate = (queued * finished_time / max(1, len(self.options.devices))) if finished_time else None
        eta = "?" if queued and queue_estimate is None else _duration(worst_left + (queue_estimate or 0.0))
        print(f"[{time.strftime('%H:%M:%S')}] {label}: {finished} done, {len(running)} running, {queued} queued, stage ETA {eta}. {'; '.join(parts)}", flush=True)
        print(f"           {gpu_line()}", flush=True)


class InProcessRunner:
    """For tests: runs each task's worker in this process, on CPU."""

    def __init__(self, options: PoolOptions, paths: D1Paths) -> None:
        self.options = options
        self.paths = paths
        self.ran: list[str] = []

    def run(self, label: str, tasks: Sequence[Task]) -> None:
        from .worker import main as worker_main

        for task in tasks:
            if self.paths.done(task.name).exists():
                continue
            argv = [task.argv[0], "--config", str(self.options.config_path), "--out", str(self.options.out_dir), "--task-name", task.name, *task.argv[1:]]
            if self.options.factory:
                argv.extend(["--factory", self.options.factory])
            if task.argv[0] == "train":
                argv.append("--in-process")
            code = worker_main(argv)
            if code == 2:
                raise RunFailed(f"The dry-run checks failed for {task.name}")
            if code != 0:
                raise RunFailed(f"{task.name} failed with code {code}")
            marker = self.paths.done(task.name)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()
            self.ran.append(task.name)


def create_stores(cfg: D1Config, paths: D1Paths, tag: str, rung: Rung, model_id: str, sets: Sequence[str]) -> None:
    from .plan import read_meta
    from .store import CaptureStore

    layers = layers_for(cfg.fractions, rung.n_layers)
    for name in sets:
        CaptureStore.create(paths.capture_dir(tag, model_id, name), read_meta(paths, tag, name)["n"], len(layers), rung.d_model)


class Experiment:
    def __init__(
        self,
        cfg: D1Config,
        paths: D1Paths,
        runner: Any,
        n_gpus: int,
        gpu_free_gb: float | None,
        thresholds: Thresholds = Thresholds(),
        download: Callable[[Rung], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.paths = paths
        self.runner = runner
        self.n_gpus = max(1, n_gpus)
        self.gpu_free_gb = gpu_free_gb
        self.thresholds = thresholds
        self.download = download
        self.state: dict[str, Any] = {}

    def note(self, **values: Any) -> None:
        """Persistent status, for people and for reruns of the final analysis."""
        self.state.update(values)
        atomic_write_bytes(self.paths.control, json.dumps(self.state, indent=1, default=str).encode("utf-8"))

    def check_identity(self) -> None:
        """A resumed run must be the same experiment: an edited config or data file would otherwise mix silently with old captures."""
        identity = {"config_hash": self.cfg.config_hash, "data_hash": self.cfg.data_hash}
        path = self.paths.root / "identity.json"
        if path.exists():
            found = json.loads(path.read_text(encoding="utf-8"))
            if found != identity:
                raise RunFailed(f"{self.paths.root} was started with another config or data files: {found} against {identity}. Use a new state directory.")
            return
        atomic_write_bytes(path, json.dumps(identity).encode("utf-8"))

    def capture_tasks(self, tag: str, rung: Rung, model_id: str, sets: Sequence[str], shards: int) -> list[Task]:
        create_stores(self.cfg, self.paths, tag, rung, model_id, sets)
        label = "-".join(sets)
        return [
            Task(f"{tag}.capture.{model_id}.{label}.{shard}of{shards}", ["capture", "--rung", tag, "--model-id", model_id, "--sets", ",".join(sets), "--shard", str(shard), "--n-shards", str(shards)])
            for shard in range(shards)
        ]

    def run_stage1(self) -> tuple[str, dict[str, Any]]:
        from .analysis import evaluate_g1, write_json_atomic

        history: dict[str, Any] = {}
        ladder = self.cfg.ladder
        for index, rung in enumerate(ladder):
            tag = rung.tag
            if self.gpu_free_gb is not None and rung.gpu_gb > self.gpu_free_gb:
                raise RunFailed(
                    f"{rung.name} needs about {rung.gpu_gb:.0f} GB of free GPU memory, and this machine has {self.gpu_free_gb:.0f} GB. "
                    f"G1 was not run on it, so this is not a finding about the model. Results so far: {sorted(history)}. "
                    "Rent a larger GPU and rerun with D1_RETRY=1: finished rungs are kept."
                )
            self.note(stage="stage1", rung=tag)
            if self.download is not None:
                self.download(rung)
            self.runner.run(f"Plan for {rung.name}", [Task(f"{tag}.plan", ["plan", "--rung", tag], gpu=False, loads_model=False)])
            self.runner.run(f"Dry-run checks on {rung.name}", [Task(f"{tag}.preflight", ["preflight", "--rung", tag])])
            sets = ["fit", "val", "cal", "test_g1"]
            self.runner.run(f"Capture on unmodified {rung.name}", self.capture_tasks(tag, rung, BASE_MODEL, sets, self.n_gpus))
            self.runner.run(f"Fit the linear ensemble on {rung.name}", [Task(f"{tag}.fit.linear", ["fit-probes", "--rung", tag, "--name", "linear", "--kind", "linear"], loads_model=False)])
            result = evaluate_g1(self.cfg, self.paths, tag, self.thresholds)
            write_json_atomic(self.paths.gate_g1(tag), result)
            history[tag] = result
            print(
                f"G1 on {rung.name}: probe AUROC {result['probe_auroc']:.3f} [{result['probe_ci']['lo']:.3f}, {result['probe_ci']['hi']:.3f}], "
                f"self-report {result['self_report_auroc']:.3f} [{result['self_report_ci']['lo']:.3f}, {result['self_report_ci']['hi']:.3f}]: "
                f"{'PASS' if result['passed'] else 'fail'}",
                flush=True,
            )
            decision = next_after_g1(result["passed"], index, len(ladder))
            if decision == "stage2":
                return tag, history
            if decision == "stop":
                return "", history
        raise AssertionError("next_after_g1 ends the ladder")

    def fix_sample_size(self, tag: str, g1: dict[str, Any]) -> int:
        """Logged before any adapter exists. The file is written once and every later launch reads the same number."""
        from .analysis import write_json_atomic

        path = self.paths.sample_size(tag)
        if path.exists():
            return int(json.loads(path.read_text(encoding="utf-8"))["chosen"])
        adapters = self.paths.rung_dir(tag) / "adapters"
        if adapters.exists() and any(adapters.iterdir()):
            raise RunFailed("Adapters exist but no sample size was logged. That breaks the prereg's order of work.")
        sizing = g1["sizing"]
        write_json_atomic(path, {**sizing, "logged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        table = "; ".join(f"n={row['n_per_class_per_family']}: {row['expected_half_width_D']:.3f}" for row in sizing["table"])
        print(f"Sample size per class per family: {sizing['chosen']}. Expected half-width of D: {table}.", flush=True)
        if not sizing["reached_target"]:
            print(f"WARNING: no n reaches the target half-width of {sizing['target']}, so using the largest available. An inconclusive result is more likely.", flush=True)
        return int(sizing["chosen"])

    def train_and_eval(self, tag: str, rung: Rung, attempt: int, conditions: Sequence[str], seeds: Sequence[int]) -> None:
        trains = [
            Task(f"{tag}.train.{adapter_id(attempt, condition, seed)}", ["train", "--rung", tag, "--condition", condition, "--seed", str(seed), "--attempt", str(attempt)])
            for condition in conditions
            for seed in seeds
        ]
        self.runner.run(f"Train adapters on {rung.name}, attempt {attempt}", trains)
        evals: list[Task] = []
        for condition in conditions:
            for seed in seeds:
                evals.extend(self.capture_tasks(tag, rung, adapter_id(attempt, condition, seed), ["test", "facts"], 1))
        self.runner.run(f"Read test sets through adapters, attempt {attempt}", evals)

    def run_gates(self, tag: str, rung: Rung) -> tuple[int | None, dict[str, Any]]:
        """Returns the attempt whose adapters passed, or None if the stop rule fires."""
        from .analysis import evaluate_g2, write_json_atomic

        seeds = list(self.cfg.training.seeds)
        history: dict[str, Any] = {}
        attempt = 0
        while True:
            self.note(stage="stage2_one_seed", attempt=attempt)
            self.train_and_eval(tag, rung, attempt, ("DD", "SHAM"), seeds[:1])
            one = evaluate_g2(self.cfg, self.paths, tag, attempt, seeds[:1], "one_seed", self.thresholds)
            write_json_atomic(self.paths.gate_g2(tag, attempt, "one_seed"), one)
            history[f"a{attempt}_one_seed"] = one
            print(f"G2, one seed, attempt {attempt}: {one['checks']} {one['numbers']}", flush=True)
            decision = next_after_g2(one["passed"], attempt, "one_seed")
            if decision == "stop":
                return None, history
            if decision == "retry":
                attempt += 1
                continue
            self.note(stage="stage2_all_seeds", attempt=attempt)
            self.train_and_eval(tag, rung, attempt, ("DD", "SHAM"), seeds[1:])
            self.train_and_eval(tag, rung, attempt, ("FLIP", "HONEST"), seeds)
            full = evaluate_g2(self.cfg, self.paths, tag, attempt, seeds, "all_seeds", self.thresholds)
            write_json_atomic(self.paths.gate_g2(tag, attempt, "all_seeds"), full)
            history[f"a{attempt}_all_seeds"] = full
            print(f"G2, all seeds, attempt {attempt}: {full['checks']} {full['numbers']}", flush=True)
            decision = next_after_g2(full["passed"], attempt, "all_seeds")
            if decision == "finish":
                return attempt, history
            if decision == "stop":
                return None, history
            attempt += 1

    def retrain_probes(self, tag: str, rung: Rung, attempt: int) -> None:
        seeds = list(self.cfg.training.seeds)
        captures: list[Task] = []
        for condition in ("DD", "SHAM"):
            for seed in seeds:
                captures.extend(self.capture_tasks(tag, rung, adapter_id(attempt, condition, seed), ["fit", "val", "cal"], 1))
        self.runner.run("Capture probe-train rows through DD and SHAM adapters", captures)
        fits = [
            Task(
                f"{tag}.fit.retrained_{adapter_id(attempt, condition, seed)}",
                ["fit-probes", "--rung", tag, "--name", f"retrained_{adapter_id(attempt, condition, seed)}", "--source", adapter_id(attempt, condition, seed), "--kind", "linear"],
                loads_model=False,
            )
            for condition in ("DD", "SHAM")
            for seed in seeds
        ]
        self.runner.run("Refit probes through the adapters", fits)

    def run(self) -> dict[str, Any]:
        from .analysis import format_report, full_analysis, stop_report, write_json_atomic

        cfg, paths = self.cfg, self.paths
        started = time.monotonic()
        self.check_identity()
        self.note(stage="prepare", status="running")
        self.runner.run("Download and prepare FIBS", [Task("prepare", ["prepare"], gpu=False, loads_model=False)])
        tag, g1_history = self.run_stage1()
        if not tag:
            outcome = stop_report("g1", g1_history)
            self.note(stage="finished", outcome="G1 failed through the ladder")
            print(G1_FINDING, flush=True)
            write_json_atomic(paths.root / "outcome.json", outcome)
            atomic_write_bytes(paths.root / "report.txt", (G1_FINDING + "\n" + json.dumps({key: {k: value[k] for k in ("probe_auroc", "probe_ci", "self_report_auroc", "self_report_ci", "passed")} for key, value in g1_history.items()}, indent=1) + "\n").encode("utf-8"))
            return outcome
        rung = cfg.rung(tag)
        self.note(rung=tag, stage="sizing")
        n = self.fix_sample_size(tag, g1_history[tag])
        self.runner.run("Build the final test set", [Task(f"{tag}.test_set.{n}", ["test-set", "--rung", tag, "--n", str(n)], gpu=False, loads_model=False)])
        self.runner.run("Capture the final test set on the unmodified model", self.capture_tasks(tag, rung, BASE_MODEL, ["test", "facts"], self.n_gpus))
        self.runner.run(
            "Fit the MLP ensemble and the label-shuffled baseline",
            [
                Task(f"{tag}.fit.mlp", ["fit-probes", "--rung", tag, "--name", "mlp", "--kind", "mlp"], loads_model=False),
                Task(f"{tag}.fit.shuffled", ["fit-probes", "--rung", tag, "--name", "shuffled", "--kind", "linear", "--shuffle"], loads_model=False),
            ],
        )
        attempt, g2_history = self.run_gates(tag, rung)
        if attempt is None:
            outcome = stop_report("g2", {**g1_history, **g2_history})
            self.note(stage="finished", outcome="G2 failed after its retry")
            print(G2_FINDING, flush=True)
            write_json_atomic(paths.root / "outcome.json", outcome)
            atomic_write_bytes(paths.root / "report.txt", (G2_FINDING + "\n" + json.dumps(g2_history, indent=1, default=float) + "\n").encode("utf-8"))
            return outcome
        self.note(stage="retrain_probes", attempt=attempt)
        self.retrain_probes(tag, rung, attempt)
        self.note(stage="analysis")
        results = full_analysis(cfg, paths, tag, attempt, self.thresholds)
        results["g1_history"] = g1_history
        results["g2_history"] = g2_history
        write_json_atomic(paths.root / "results.json", results)
        report = format_report(results)
        atomic_write_bytes(paths.root / "report.txt", report.encode("utf-8"))
        outcome = {"stopped_at": None, "verdict": results["primary"]["verdict"], "rung": tag, "attempt": attempt, "seconds": time.monotonic() - started}
        write_json_atomic(paths.root / "outcome.json", outcome)
        self.note(stage="finished", outcome=results["primary"]["verdict"])
        print(report, flush=True)
        return outcome


def download_rung(rung: Rung) -> None:
    """One model at a time, smallest first, only when the ladder reaches it. Workers then read the cache offline.

    In a new process, because huggingface_hub reads HF_HUB_OFFLINE once, at import.
    """
    code = (
        "import sys\n"
        "from huggingface_hub import snapshot_download\n"
        'print(snapshot_download(sys.argv[1], revision=sys.argv[2], ignore_patterns=["*.bin", "*.pt", "*.pth", "*.gguf", "*.h5", "*.msgpack", "*.ot", "*.onnx"]))\n'
    )
    environment = {**os.environ, "HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"}
    for attempt in range(5):
        try:
            result = subprocess.run([sys.executable, "-c", code, rung.name, rung.revision], env=environment, timeout=DOWNLOAD_SECONDS, capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            print(f"Download of {rung.name}, attempt {attempt + 1} of 5, took too long", flush=True)
            continue
        if result.returncode == 0:
            print(f"{rung.name} is at {result.stdout.strip()}", flush=True)
            return
        print(f"Download of {rung.name}, attempt {attempt + 1} of 5, failed: {result.stderr[-500:]}", flush=True)
        time.sleep(30 * (attempt + 1))
    raise RunFailed(f"The download of {rung.name} failed 5 times. Check the network and free disk.")


def run_all(
    config_path: Path,
    out_dir: Path,
    devices: Sequence[str],
    gpu_free_gb: float | None,
    load_slots: int = 1,
    threads_per_worker: int | None = None,
    stall_minutes: float = 30.0,
    factory: str | None = None,
) -> dict[str, Any]:
    from .config import D1Config

    paths = D1Paths(Path(out_dir))
    paths.root.mkdir(parents=True, exist_ok=True)
    held = try_lock(paths.root / "locks" / "orchestrator.lock")
    if held is None:
        raise RunFailed(f"Another orchestrator is already running on {paths.root}")
    try:
        cfg = D1Config.from_file(config_path)
        options = PoolOptions(
            list(devices), Path(config_path), Path(out_dir), factory=factory, threads_per_worker=threads_per_worker,
            load_slots=load_slots, stall_seconds=stall_minutes * 60.0,
        )  # fmt: skip
        pool = TaskPool(options, paths)
        experiment = Experiment(cfg, paths, pool, len(devices), gpu_free_gb, download=None if factory else download_rung)
        return experiment.run()
    finally:
        held.close()


RESULT_NAMES = ("report.txt", "outcome.json", "results.json", "control.json", "identity.json")


def finalize(out_dir: Path, results_dir: Path, config_path: Path, log_path: Path | None = None) -> dict[str, Any]:
    """Copy numbers, gates and hashes to the results folder. No row text is in anything copied. The only token ids are Yes, No and the full stop."""
    paths = D1Paths(Path(out_dir))
    if not (paths.root / "outcome.json").exists():
        raise RunFailed(f"{paths.root / 'outcome.json'} is missing, so there is no final result to write")
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    for name in RESULT_NAMES:
        source = paths.root / name
        if source.exists():
            shutil.copyfile(source, results_dir / name)
    if paths.data_manifest.exists():
        shutil.copyfile(paths.data_manifest, results_dir / "data_manifest.json")
    shutil.copyfile(config_path, results_dir / "config.json")
    rungs = paths.root / "rungs"
    if rungs.exists():
        for rung_dir in sorted(rungs.iterdir()):
            for pattern in ("g1.json", "g2_*.json", "sample_size.json", "dry_run.json"):
                for source in rung_dir.glob(pattern):
                    shutil.copyfile(source, results_dir / f"{rung_dir.name}.{source.name}")
            plan = rung_dir / "plan" / "plan.json"
            if plan.exists():
                shutil.copyfile(plan, results_dir / f"{rung_dir.name}.plan.json")
            # The frozen primary probe is small and is what a later reader needs to reproduce scores. Weights, not data.
            frozen = rung_dir / "probes" / "linear.npz"
            if frozen.exists():
                shutil.copyfile(frozen, results_dir / f"{rung_dir.name}.probe.linear.npz")
    if log_path is not None and Path(log_path).exists():
        shutil.copyfile(log_path, results_dir / "run.log")
    names = sorted(path.name for path in results_dir.iterdir() if path.is_file() and path.name not in ("SHA256SUMS", "summary.txt"))
    hashes = {name: sha256_file(results_dir / name) for name in names}
    atomic_write_bytes(results_dir / "SHA256SUMS", "".join(f"{hashes[name]}  {name}\n" for name in names).encode("utf-8"))
    report = (paths.root / "report.txt").read_text(encoding="utf-8") if (paths.root / "report.txt").exists() else ""
    outcome = json.loads((paths.root / "outcome.json").read_text(encoding="utf-8"))
    summary = "\n".join(
        [
            f"Outcome: {outcome.get('verdict') or outcome.get('finding')}",
            f"Results folder: {results_dir}",
            *[f"{hashes[name]}  {results_dir / name}" for name in names],
            "",
            BEGIN_REPORT,
            report.rstrip(),
            END_REPORT,
            "",
        ]
    )
    atomic_write_bytes(results_dir / "summary.txt", summary.encode("utf-8"))
    return {"outcome": outcome, "hashes": hashes, "summary": summary, "results_dir": str(results_dir)}


__all__ = ["CONDITIONS", "Experiment", "InProcessRunner", "PoolOptions", "Task", "TaskPool", "finalize", "run_all"]
