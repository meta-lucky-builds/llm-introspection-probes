"""D1 from start to verdict. Never imports torch. Workers are new processes, and a shared pool hands them tasks.

Walks the same steps on every launch. A task that finished has a marker and is skipped, a capture resumes from its done
mask, and gates are recomputed from stored numbers, so a rerun reaches the same decisions. Every outcome the prereg names
is a normal finish: a failed G1 through the ladder, a failed G2 after its retry, and each verdict, inconclusive included.
A stop for lack of memory is not a finding, and is reported as a failed run.

Only real data dependencies block a task: a read needs its adapter, a refit needs its reads, and a gate needs what it reads.
Where the prereg fixes an order, that order is kept by the decisions in Experiment.run_gates, and the pool never sees a task
before it may run. What runs where changes no number beyond float rounding: shards and batches come from the set and not from
the number of workers, and an adapter's steps come from its seed.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait as wait_for_futures
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..checkpoint import atomic_write_bytes
from ..sharding import RunFailed, _duration, gpu_line, kill_bounded, sha256_file, try_lock
from . import capacity, run_identity
from .config import CONDITIONS, D1Config, Rung, layers_for
from .gates import G1_FINDING, Thresholds, next_after_g1, next_after_g2
from .paths import BASE_MODEL, D1Paths, adapter_id
from .scheduling import FIT_NEED_GB, Schedule, plan_schedule, usable_gb

EXIT_FATAL = 3
EXIT_OUT_OF_MEMORY = 4
GATE_SETS = ["test", "facts", "specificity"]
MIN_FREE_DISK_GB = 2.0
DOWNLOAD_SECONDS = 7200
BEGIN_REPORT = "=====BEGIN D1 REPORT====="
END_REPORT = "=====END D1 REPORT====="
# About 25 seconds of one card at an assumed 40,000 tokens a second. A constant, so that the shards do not depend on the machine.
TOKENS_PER_SHARD = 1_000_000
MAX_OOM_RETRIES = 6  # Out of memory is not a crash and has its own count, but a card that can never fit a task must still end.
OOM_NEED_FACTOR = 1.25
UTIL_SAMPLES = 5
MIN_UTIL_SAMPLES = 3
LOW_UTILIZATION = 10.0  # A worker sits on the card, but the card is not computing.
REWARN_SECONDS = 300.0
MIB_TO_GB = 1048576 / 1e9


@dataclass
class Task:
    name: str
    argv: list[str]
    gpu: bool = True
    loads_model: bool = True
    after: tuple[str, ...] = ()  # Names of the tasks whose results this one reads.
    need_gb: float | None = None  # Overrides the figure for its kind, after an out-of-memory fallback.

    @property
    def kind(self) -> str:
        return self.argv[0]

    @property
    def rung(self) -> str | None:
        return self.argv[self.argv.index("--rung") + 1] if "--rung" in self.argv else None


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
    cpu_workers: int = 1  # With no devices, this many workers on the CPU. For tests.
    workers_per_gpu: int | None = None  # None: memory and measured utilization decide. A number overrides utilization.
    saturated_utilization: float = 90.0  # A card this busy gains nothing from another worker: they would only take turns.
    gpu_poll_seconds: float = 15.0
    settle_seconds: float = 90.0  # Utilization means nothing while a worker is still loading.
    idle_warn_seconds: float = 60.0


@dataclass
class Worker:
    process: Any
    device: str | None
    log_path: Path
    serves: bool
    rung: str | None
    need_gb: float
    task: Task | None = None
    job_started: float = 0.0
    has_loaded: bool = False
    retiring: bool = False


def parse_gpu_query(output: str) -> dict[str, dict[str, float]]:
    """nvidia-smi can print [N/A] in any field. Such a card is left out, and keeps its last reading."""
    states: dict[str, dict[str, float]] = {}
    for line in output.strip().splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 5:
            continue
        try:
            states[fields[0]] = {"util": float(fields[1]), "used_gb": float(fields[2]) * MIB_TO_GB, "total_gb": float(fields[3]) * MIB_TO_GB, "free_gb": float(fields[4]) * MIB_TO_GB}
        except ValueError:
            continue
    return states


def query_gpus(seconds: float = 15.0) -> dict[str, dict[str, float]]:
    """Never blocks past the limit, since nvidia-smi can stick forever after a GPU falls off the bus. Empty if it cannot answer."""
    command = ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total,memory.free", "--format=csv,noheader,nounits"]
    if shutil.which(command[0]) is None:
        return {}
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    except OSError:
        return {}
    try:
        output, _ = process.communicate(timeout=seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        return {}
    return parse_gpu_query(output)


class TaskPool:
    """A shared pool of tasks over the cards. A free card takes the next task whose inputs exist, so a fast card takes more.

    Capture shards go to long-lived workers that keep the model loaded. Everything else, training above all, gets its own new
    process. Memory decides whether a worker fits on a card, and measured utilization whether another one would help.

    A failed task goes back in the queue. Exit 4 means out of memory, which is not a crash: the card takes fewer workers, the task
    is put back, and only its own count of such retries can end the run. Other failures use the restart limit. A worker that
    stops making progress is killed alone, whatever else is on its card, since each task has its own heartbeat.
    """

    def __init__(self, options: PoolOptions, paths: D1Paths, spawn: Callable[..., Any] | None = None, gpu_query: Callable[[], dict[str, dict[str, float]]] | None = None) -> None:
        self.options = options
        self.paths = paths
        self.spawn = spawn or self._spawn_process
        self.gpu_query = gpu_query or query_gpus
        self.cards: list[str] = list(options.devices) or [f"cpu{index}" for index in range(max(1, options.cpu_workers))]
        self.queue: list[Task] = []
        self.speculative: set[str] = set()
        self.workers: list[Worker] = []
        self.restarts: dict[str, int] = {}
        self.oom_retries: dict[str, int] = {}
        self.needs: dict[str, float] = {"fit-probes": FIT_NEED_GB}
        self.default_need_gb = 0.0
        self.need_factor: dict[str, float] = {}
        self.card_limit: dict[str, int] = {}
        self.free_gb: dict[str, float] = {}
        self.gpu_state: dict[str, dict[str, float]] = {}
        self.util: dict[str, deque] = {card: deque(maxlen=UTIL_SAMPLES) for card in self.cards}
        self.settled_at: dict[str, float] = {card: 0.0 for card in self.cards}
        self.finished_on: dict[str, int] = {card: 0 for card in self.cards}
        self.first_start: dict[str, float] = {}
        self.idle_since: dict[str, float] = {}
        self.last_warning: dict[str, float] = {}
        self.elapsed_by_kind: dict[str, list[float]] = {}
        self.expected: str | None = None
        self.label = ""
        self.finished = 0
        self.last_print = time.monotonic()
        self.last_query = -math.inf
        self.serial = 0

    # What the experiment tells the pool

    def set_default_need(self, gb: float) -> None:
        """What a worker is assumed to need on a new rung, until its dry run has measured it."""
        self.default_need_gb = gb
        self.needs.pop("train", None)
        self.needs.pop("train_checkpointing", None)
        self.needs.pop("capture", None)

    def configure(self, dry_run: Mapping[str, Any], config_budget: int) -> Schedule:
        self._refresh_gpus(force=not self.free_gb)
        known = [self.free_gb[card] for card in self.cards if card in self.free_gb]
        schedule = plan_schedule(min(known) if known else None, dry_run, self.default_need_gb, config_budget, self.options.workers_per_gpu)
        self.needs["train"] = schedule.train_need_gb
        self.needs["train_checkpointing"] = schedule.train_checkpointing_need_gb
        self.needs["capture"] = schedule.capture_need_gb
        for note in schedule.notes:
            print(f"[{time.strftime('%H:%M:%S')}] {note}", flush=True)
        return schedule

    def expect(self, reason: str | None) -> None:
        """More work is known to come, and this is what it waits for. Named in the warning if a card sits idle meanwhile."""
        self.expected = reason

    def submit(self, label: str, tasks: Sequence[Task], speculative: bool = False) -> None:
        """Queue tasks and return. A task already known stays where it is, and asking for it without speculation promotes it."""
        running = {worker.task.name for worker in self.workers if worker.task is not None}
        queued = {task.name for task in self.queue}
        added = 0
        for task in tasks:
            if self.paths.done(task.name).exists():
                continue
            if not speculative:
                self.speculative.discard(task.name)
            if task.name in running or task.name in queued:
                continue
            if speculative:
                self.speculative.add(task.name)
            self.queue.append(task)
            added += 1
        if added:
            print(f"[{time.strftime('%H:%M:%S')}] {label}: {added} task(s) queued{', speculatively' if speculative else ''}", flush=True)

    def cancel(self, tasks: Sequence[Task]) -> None:
        """Drop speculative tasks that a gate has made pointless. Nothing finished is touched, and nothing it left changes any result."""
        names = {task.name for task in tasks} & self.speculative
        self.queue = [task for task in self.queue if task.name not in names]
        for worker in self.workers:
            if worker.task is not None and worker.task.name in names:
                print(f"Cancelling {worker.task.name}, which the gate no longer needs", flush=True)
                worker.task = None
                worker.retiring = True
                if not kill_bounded(worker.process, self.options.kill_wait_seconds):
                    print(f"WARNING: process {worker.process.pid} cannot be killed, so it is being left behind.", flush=True)
        self.speculative -= names

    def run(self, label: str, tasks: Sequence[Task]) -> None:
        """Returns when these tasks are done, or raises RunFailed. Other queued tasks carry on meanwhile."""
        pending = [task for task in tasks if not self.paths.done(task.name).exists()]
        if not pending:
            return
        print(f"[{time.strftime('%H:%M:%S')}] {label}: {len(pending)} task(s) of {len(tasks)} to run", flush=True)
        self.label = label
        started = time.monotonic()
        self.submit(label, pending)
        try:
            while any(not self.paths.done(task.name).exists() for task in pending):
                self._tick()
                time.sleep(self.options.poll_seconds)
            self._reap()
        except BaseException:
            self.close()
            raise
        print(f"[{time.strftime('%H:%M:%S')}] {label}: finished in {_duration(time.monotonic() - started)}", flush=True)

    def compute(self, function: Callable[..., Any], *args: Any) -> Any:
        """A gate, on a thread, so that the cards keep getting work while it reads stored numbers."""
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="d1-gate") as executor:
            future = executor.submit(function, *args)
            try:
                while not future.done():
                    self._tick()
                    wait_for_futures([future], timeout=self.options.poll_seconds)
            except BaseException:
                self.close()
                raise
            return future.result()

    def close(self) -> None:
        """Stops every worker. Rows and adapters they finished are safe, and a resume redoes the rest."""
        for worker in self.workers:
            if worker.process.poll() is None:
                worker.process.terminate()
        deadline = time.monotonic() + 30.0
        for worker in self.workers:
            try:
                worker.process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                if not kill_bounded(worker.process, self.options.kill_wait_seconds):
                    print(f"WARNING: process {worker.process.pid} cannot be killed, so it is being left behind.", flush=True)
        self.workers = []
        self.queue = []

    # One turn of the loop

    def _tick(self) -> None:
        disk = shutil.disk_usage(self.options.out_dir).free / 1024**3
        if disk < MIN_FREE_DISK_GB:
            raise RunFailed(f"Only {disk:.1f} GB of disk left at {self.options.out_dir}")
        self._refresh_gpus()
        self._reap()
        self._dispatch()
        self._check_for_deadlock()
        now = time.monotonic()
        self._warn_about_idle_cards(now)
        if now - self.last_print >= self.options.progress_seconds:
            self.last_print = now
            self._print_progress(now)

    def _refresh_gpus(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_query < self.options.gpu_poll_seconds:
            return
        self.last_query = now
        states = self.gpu_query()
        self.gpu_state.update(states)
        for card in self.cards:
            state = states.get(card)
            if state is None:
                continue
            # Free memory is only read on an empty card, so that this run's own workers are counted once, by what they need.
            if not self._live(card):
                self.free_gb[card] = state["free_gb"]
            elif self._busy(card) and now >= self.settled_at[card]:
                self.util[card].append(state["util"])

    def _live(self, card: str) -> list[Worker]:
        return [worker for worker in self.workers if worker.device == card]

    def _busy(self, card: str) -> list[Worker]:
        return [worker for worker in self.workers if worker.device == card and worker.task is not None and worker.task.loads_model]

    def _room(self, card: str) -> float:
        if card not in self.free_gb:
            return math.inf
        return usable_gb(self.free_gb[card]) - sum(worker.need_gb for worker in self._live(card))

    def _need(self, task: Task) -> float:
        need = task.need_gb if task.need_gb is not None else self.needs.get(task.kind, self.default_need_gb)
        return need * self.need_factor.get(task.kind, 1.0)

    def _is_ready(self, task: Task) -> bool:
        return all(self.paths.done(name).exists() for name in task.after)

    def can_share(self, card: str) -> bool:
        """May another model-holding worker join this card? Only if it would help, since workers on one card take turns.

        A number from D1_WORKERS_PER_GPU is taken as asked. Otherwise the card must have stayed under the saturation line for a
        few samples since its last worker settled. After an out-of-memory error the limit is lower, and it stays lower.
        """
        busy = len(self._busy(card))
        if card in self.card_limit and busy >= self.card_limit[card]:
            return False
        ram = capacity.available_ram_gb()
        if ram is not None and ram < capacity.RAM_FLOOR_PER_WORKER_GB:
            return False
        if self.options.workers_per_gpu:
            return busy < self.options.workers_per_gpu
        samples = self.util[card]
        return len(samples) >= MIN_UTIL_SAMPLES and sum(samples) / len(samples) < self.options.saturated_utilization

    def _choose_card(self, task: Task, need: float) -> str | None:
        """The least loaded card this fits on, or None. An idle capture worker gives its memory back if that is what stands in the way."""
        for card in sorted(self.cards, key=lambda name: (len(self._busy(name)), -self._room(name), self.cards.index(name))):
            if task.loads_model and self._busy(card) and not self.can_share(card):
                continue
            room = self._room(card)
            live = self._live(card)
            if room >= need or not live:  # An empty card always gets the chance. Capture and training can step down from there.
                return card
            idle = [worker for worker in live if worker.task is None and not worker.retiring]
            if idle and room + sum(worker.need_gb for worker in idle) >= need:
                for worker in idle:
                    self._retire(worker)
        return None

    def _dispatch(self) -> None:
        # Required before speculative, and within them adapters first, since a whole adapter cannot be split and shards can.
        ready = sorted((task for task in self.queue if self._is_ready(task)), key=lambda task: (task.name in self.speculative, task.kind != "train"))
        loading = sum(1 for worker in self.workers if worker.task is not None and worker.task.loads_model and not worker.has_loaded and not (self._beat(worker.task) or {}).get("loaded"))
        for task in ready:
            if not task.gpu:
                self.queue.remove(task)
                self._start(task, None)
                continue
            if task.kind == "capture":
                idle = next((worker for worker in self.workers if worker.serves and worker.task is None and not worker.retiring and worker.rung == task.rung), None)
                if idle is not None:
                    self.queue.remove(task)
                    self._assign(idle, task)
                    continue
            if task.loads_model and loading >= self.options.load_slots:
                continue
            card = self._choose_card(task, self._need(task))
            if card is None:
                continue
            self.queue.remove(task)
            self._start(task, card)
            if task.loads_model:
                loading += 1

    def _reap(self) -> None:
        for worker in list(self.workers):
            code = worker.process.poll()
            task = worker.task
            if code is None:
                if task is None:
                    continue
                if worker.serves and self.paths.done(task.name).exists():
                    self._finish(worker)
                    continue
                if not self._stalled(worker, task):
                    continue
                code = -9
            self.workers.remove(worker)
            if worker.device is not None:
                self._card_changed(worker.device)
            if task is None:
                continue  # Retired, or cancelled, or a capture worker with nothing to do. No work was in it.
            if worker.serves and self.paths.done(task.name).exists():
                self._finish(worker)
            elif code == 0 and not worker.serves:
                self._finish(worker)
            else:
                self._on_failure(worker, task, code if code != 0 else 1)

    def _stalled(self, worker: Worker, task: Task) -> bool:
        """By this task's own heartbeat, so that a slow neighbour on the card cannot make it look stuck, or hide that it is."""
        if not (task.gpu or task.loads_model):
            return False
        beat = self._beat(task)
        limit = self.options.stall_seconds if worker.has_loaded or (beat and beat.get("loaded")) else self.options.load_stall_seconds
        try:
            last = self.paths.heartbeat(task.name).stat().st_mtime
        except OSError:
            last = worker.job_started
        if time.time() - max(last, worker.job_started) <= limit:
            return False
        print(f"{task.name}: no progress for {limit / 60:.0f} minutes, so its GPU may be wedged. Killing it.", flush=True)
        if not kill_bounded(worker.process, self.options.kill_wait_seconds):
            raise RunFailed(f"{task.name} cannot be killed, so it is stuck in the GPU driver. This machine probably needs a reboot.")
        return True

    def _finish(self, worker: Worker) -> None:
        task = worker.task
        assert task is not None
        worker.task = None
        worker.has_loaded = True
        marker = self.paths.done(task.name)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
        self.finished += 1
        self.speculative.discard(task.name)
        self.elapsed_by_kind.setdefault(task.kind, []).append(time.time() - worker.job_started)
        if worker.device is not None:
            self.finished_on[worker.device] += 1

    def _on_failure(self, worker: Worker, task: Task, code: int) -> None:
        print(f"{task.name} exited with code {code}. Its log ends with:\n{self._tail(worker)}", flush=True)
        if code == 2:
            raise RunFailed(f"The dry-run checks failed for {task.name}. See PASS and FAIL in {worker.log_path}")
        if code == EXIT_FATAL:
            raise RunFailed(f"{task.name} hit an error that a restart cannot fix. See {worker.log_path}")
        if code == EXIT_OUT_OF_MEMORY:
            self._back_off(worker, task)
            return
        self.restarts[task.name] = self.restarts.get(task.name, 0) + 1
        if self.restarts[task.name] > self.options.max_restarts:
            raise RunFailed(f"{task.name} failed {self.restarts[task.name]} times, so the run stopped. See {worker.log_path}")
        print(f"Restarting {task.name}, attempt {self.restarts[task.name]} of {self.options.max_restarts}", flush=True)
        self.queue.insert(0, task)

    def _back_off(self, worker: Worker, task: Task) -> None:
        """The worker's slot goes, the task goes back, and a resume or a done mask loses at most its unflushed part."""
        card = worker.device
        self.oom_retries[task.name] = self.oom_retries.get(task.name, 0) + 1
        if self.oom_retries[task.name] > MAX_OOM_RETRIES:
            raise RunFailed(f"{task.name} ran out of GPU memory {self.oom_retries[task.name]} times, also alone on a card. This is a limit of the machine and not a finding about the model. See {worker.log_path}")
        self.need_factor[task.kind] = self.need_factor.get(task.kind, 1.0) * OOM_NEED_FACTOR
        others = len(self._busy(card)) if card is not None else 0
        if card is not None and others > 0:
            self.card_limit[card] = others
            print(f"{task.name} ran out of memory beside {others} other worker(s), so card {card} now takes at most {others}, and the task goes back in the queue.", flush=True)
        else:
            self._step_down(task)
            print(f"{task.name} ran out of memory alone on card {card}, so it steps down, and goes back in the queue.", flush=True)
        self.queue.insert(0, task)

    def _step_down(self, task: Task) -> None:
        """Alone on a card and still out of memory: gradient checkpointing first, then a smaller micro-batch. Same gradients up to rounding."""
        if task.kind != "train":
            return  # Capture halves its own budget, and has already reached a single row by the time it exits.
        if "--gradient-checkpointing" in task.argv and task.argv[task.argv.index("--gradient-checkpointing") + 1] == "0":
            task.argv[task.argv.index("--gradient-checkpointing") + 1] = "1"
            task.need_gb = self.needs.get("train_checkpointing")
        elif "--halve-on-oom" not in task.argv:
            task.argv.append("--halve-on-oom")

    def _retire(self, worker: Worker) -> None:
        """A capture worker leaves when its stdin closes, which gives its model's memory back."""
        worker.retiring = True
        try:
            worker.process.stdin.close()
        except (OSError, ValueError):
            pass

    def _check_for_deadlock(self) -> None:
        """A required task that nothing can ever unblock would otherwise wait forever while the machine bills."""
        if any(worker.task is not None for worker in self.workers) or any(self._is_ready(task) for task in self.queue):
            return
        stuck = [task for task in self.queue if task.name not in self.speculative]
        if stuck:
            missing = sorted({name for task in stuck for name in task.after if not self.paths.done(name).exists()})
            raise RunFailed(f"{len(stuck)} task(s) wait for {missing[:5]}, which nothing is running. This is a bug in the order of work, and not a finding.")

    # Starting work

    def _environment(self, device: str | None, sharing: int = 1) -> dict[str, str]:
        options = self.options
        environment = os.environ.copy()
        environment.update(options.extra_env)
        threads = options.threads_per_worker or max(1, min(8, (os.cpu_count() or 1) // max(1, len(self.cards))))
        threads = max(1, threads // max(1, sharing))
        source_root = str(Path(__file__).resolve().parents[2])
        environment["PYTHONPATH"] = os.pathsep.join(filter(None, [source_root, environment.get("PYTHONPATH", "")]))
        environment.update(
            {
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": device if device is not None and not device.startswith("cpu") else "",
                "OMP_NUM_THREADS": str(threads),
                "MKL_NUM_THREADS": str(threads),
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONUNBUFFERED": "1",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            }
        )
        return environment

    def _job_argv(self, task: Task, device: str | None) -> list[str]:
        options = self.options
        argv = [task.argv[0], "--config", str(options.config_path), "--out", str(options.out_dir), "--task-name", task.name, *task.argv[1:]]
        if options.factory:
            argv.extend(["--factory", options.factory])
        if device and not device.startswith("cpu"):
            argv.append("--require-cuda")
        return argv

    def _command(self, task: Task, device: str | None) -> list[str]:
        return [sys.executable, "-u", "-m", "mechinterp.d1.worker", *self._job_argv(task, device)]

    def _spawn_process(self, command: list[str], environment: dict[str, str], log_path: Path, serves: bool) -> Any:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "ab") as log:
            if serves:
                return subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.PIPE, text=True)
            return subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)

    def _card_changed(self, card: str) -> None:
        self.util[card].clear()
        self.settled_at[card] = time.monotonic() + self.options.settle_seconds

    def _start(self, task: Task, device: str | None) -> None:
        if device is not None:
            self.first_start.setdefault(device, time.monotonic())
            self._card_changed(device)
        sharing = len(self._live(device)) + 1 if device is not None else 1
        environment = self._environment(device, sharing)
        need = self._need(task) if device is not None else 0.0
        if task.kind == "capture":
            self.serial += 1
            name = f"capture_worker.{device}.{self.serial}"
            log_path = self.paths.logs / f"{name}.log"
            command = [sys.executable, "-u", "-m", "mechinterp.d1.worker", "serve", "--config", str(self.options.config_path), "--out", str(self.options.out_dir), "--task-name", name]
            worker = Worker(self.spawn(command, environment, log_path, True), device, log_path, True, task.rung, need)
            self.workers.append(worker)
            self._assign(worker, task)
            return
        log_path = self.paths.logs / f"{task.name}.log"
        self.paths.heartbeat(task.name).unlink(missing_ok=True)
        worker = Worker(self.spawn(self._command(task, device), environment, log_path, False), device, log_path, False, task.rung, need, task=task, job_started=time.time())
        self.workers.append(worker)

    def _assign(self, worker: Worker, task: Task) -> None:
        self.paths.heartbeat(task.name).unlink(missing_ok=True)
        worker.task = task
        worker.job_started = time.time()
        try:
            worker.process.stdin.write(json.dumps(self._job_argv(task, worker.device)) + "\n")
            worker.process.stdin.flush()
        except (OSError, ValueError) as error:
            print(f"Could not hand {task.name} to its worker ({error}). If the worker has died, the task goes back in the queue.", flush=True)

    # What the log shows

    def _tail(self, worker: Worker, lines: int = 25) -> str:
        if not worker.log_path.exists():
            return "(no log)"
        raw = worker.log_path.read_bytes()[-6000:]
        return "\n".join(raw.decode("utf-8", errors="replace").splitlines()[-lines:])

    def _beat(self, task: Task) -> dict[str, Any] | None:
        try:
            return json.loads(self.paths.heartbeat(task.name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _eta_seconds(self) -> float | None:
        """Work left over the number of cards. Rough: a mean per kind, and no account of what must wait for a gate."""
        total = 0.0
        for worker in self.workers:
            if worker.task is None:
                continue
            beat = self._beat(worker.task)
            if beat and beat.get("total") and beat.get("rate"):
                total += max(0.0, beat["total"] - beat["done"]) / beat["rate"]
        for task in self.queue:
            seen = self.elapsed_by_kind.get(task.kind)
            if not seen:
                return None
            total += sum(seen) / len(seen)
        return total / max(1, len(self.cards))

    def _print_progress(self, now: float) -> None:
        running = sum(1 for worker in self.workers if worker.task is not None)
        if not running and not self.queue:
            return
        blocked = sum(1 for task in self.queue if not self._is_ready(task))
        eta = self._eta_seconds()
        print(f"[{time.strftime('%H:%M:%S')}] {self.label}: {self.finished} done, {running} running, {len(self.queue)} queued ({blocked} waiting on what they read), ETA {'?' if eta is None else _duration(eta)}.", flush=True)
        for card in self.cards:
            state = self.gpu_state.get(card)
            reading = f"{state['util']:.0f}%, {state['used_gb']:.1f}/{state['total_gb']:.1f} GB" if state else "no reading"
            parts = []
            for worker in self._live(card):
                if worker.task is None:
                    parts.append("idle capture worker")
                    continue
                beat = self._beat(worker.task)
                parts.append(f"{worker.task.name} {beat['done']}/{beat['total']}" if beat and beat.get("total") else f"{worker.task.name} starting")
            first = self.first_start.get(card)
            rate = self.finished_on[card] / max(1e-6, now - first) if first is not None else 0.0
            print(f"           card {card}: {reading}, {len(self._live(card))} worker(s), {rate:.4f} tasks/s. {'; '.join(parts) or 'nothing running'}", flush=True)
        if not self.gpu_state and any(not card.startswith("cpu") for card in self.cards):
            print(f"           {gpu_line()}", flush=True)

    def _warn_about_idle_cards(self, now: float) -> None:
        pending = bool(self.queue) or self.expected is not None
        for card in self.cards:
            if card.startswith("cpu"):
                continue
            busy = self._busy(card)
            samples = self.util[card]
            not_computing = bool(busy) and len(samples) >= MIN_UTIL_SAMPLES and sum(samples) / len(samples) < LOW_UTILIZATION
            if busy and not not_computing:
                self.idle_since.pop(card, None)
                continue
            since = self.idle_since.setdefault(card, now)
            if not pending or now - since < self.options.idle_warn_seconds or now - self.last_warning.get(card, -math.inf) < REWARN_SECONDS:
                continue
            self.last_warning[card] = now
            bar = "!" * 80
            print(f"{bar}\nWARNING: GPU {card} has been idle for {(now - since) / 60:.1f} minutes while work is pending.\nReason: {self._why_idle(card, not_computing)}\n{bar}", flush=True)

    def _why_idle(self, card: str, not_computing: bool) -> str:
        if not_computing:
            return f"a worker holds this card, but its utilization is about {sum(self.util[card]) / len(self.util[card]):.0f}%. It is probably waiting on the CPU or the disk."
        ready = [task for task in self.queue if self._is_ready(task)]
        if ready:
            task = ready[0]
            return f"{task.name} is ready and needs about {self._need(task):.1f} GB, and this card has {self._room(card):.1f} GB of room. Other reasons are the limit on loads at once, or a limit this card took after running out of memory."
        blocked = [task for task in self.queue if not self._is_ready(task)]
        if blocked:
            waiting = sorted({name for task in blocked for name in task.after if not self.paths.done(name).exists()})
            return f"{len(blocked)} task(s) can only start after {waiting[:3]}, whose results they read."
        return self.expected or "no task can start yet."


class NoWorkRunner:
    """For a re-analysis: every task must be done already. Analysis fixes need no GPU, so one that would need a worker says so."""

    def __init__(self, paths: D1Paths) -> None:
        self.paths = paths

    def run(self, label: str, tasks: Sequence[Task]) -> None:
        missing = [task.name for task in tasks if not self.paths.done(task.name).exists()]
        if missing:
            raise RunFailed(f"A re-analysis reads stored numbers only, but {label} still has work to do: {missing[:5]}. Resume the run instead.")


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
        check_capacity: bool = False,
    ) -> None:
        self.cfg = cfg
        self.paths = paths
        self.runner = runner
        self.n_gpus = max(1, n_gpus)
        self.gpu_free_gb = gpu_free_gb
        self.thresholds = thresholds
        self.download = download
        self.check_capacity = check_capacity
        self.state: dict[str, Any] = {}
        self.token_budget = cfg.capture_token_budget
        self.gradient_checkpointing = True
        self.tokens_per_shard = TOKENS_PER_SHARD
        self.speculated: list[Task] = []

    # What a pool can do and a plain runner cannot. Without these the same steps run strictly one after another.

    @property
    def overlaps(self) -> bool:
        return hasattr(self.runner, "submit")

    def _compute(self, function: Callable[..., Any], *args: Any) -> Any:
        compute = getattr(self.runner, "compute", None)
        return compute(function, *args) if compute is not None else function(*args)

    def _expect(self, reason: str | None) -> None:
        expect = getattr(self.runner, "expect", None)
        if expect is not None:
            expect(reason)

    def _speculate(self, label: str, tasks: Sequence[Task]) -> None:
        self.runner.submit(label, tasks, speculative=True)
        self.speculated.extend(tasks)

    def _cancel_speculation(self) -> None:
        if self.overlaps and self.speculated:
            self.runner.cancel(self.speculated)
        self.speculated = []

    def note(self, **values: Any) -> None:
        """Persistent status, for people and for reruns of the final analysis."""
        self.state.update(values)
        atomic_write_bytes(self.paths.control, json.dumps(self.state, indent=1, default=str).encode("utf-8"))

    def check_identity(self) -> None:
        """A resumed run must be the same experiment, as far as anything stored depends on it, and no further.

        What makes rows, activations or adapters different refuses the run, since new work would mix silently with old.
        What only changes how a probe is fitted refits probes from the stored activations. Everything else, analysis code
        and analysis config included, is recomputed anyway, so it costs nothing and is only announced.
        """
        cfg = self.cfg
        capture_code = run_identity.capture_fingerprint()
        probe_code = run_identity.probe_fingerprint()
        identity = {
            "capture_config_hash": cfg.capture_config_hash,
            "capture_code": capture_code,
            "probe_config_hash": cfg.probe_config_hash,
            "probe_code": probe_code,
            "config_hash": cfg.config_hash,
            "data_hash": cfg.data_hash,
        }
        path = self.paths.root / "identity.json"
        if path.exists():
            found = json.loads(path.read_text(encoding="utf-8"))
            changed = run_identity.differing(found.get("capture_code") or {}, capture_code)
            if found.get("capture_config_hash") != cfg.capture_config_hash or changed:
                reasons = []
                if found.get("capture_config_hash") != cfg.capture_config_hash:
                    reasons.append("the config or data files that decide rows, activations or adapters")
                if changed:
                    reasons.append(f"the code that makes them: {', '.join(changed)}")
                raise RunFailed(
                    f"{self.paths.root} was started before a change to {' and to '.join(reasons)}. Stored activations and adapters would not match new ones, so this run "
                    "cannot continue with them. Undo the change, or use a new state directory, which starts over. Fixes to analysis code or to the analysis config never cause this."
                )
            if found.get("probe_config_hash") != cfg.probe_config_hash or run_identity.differing(found.get("probe_code") or {}, probe_code):
                self.invalidate_probes()
            elif found.get("config_hash") != cfg.config_hash:
                print("The config changed in a part nothing stored depends on, so stored activations and adapters are kept, and gates are recomputed.", flush=True)
        atomic_write_bytes(path, json.dumps(identity, indent=1).encode("utf-8"))

    def invalidate_probes(self) -> None:
        """Probes are rebuilt from stored activations. Nothing else is touched."""
        print("The way probes are fitted changed, so probes are refitted from the stored activations. Activations and adapters are kept.", flush=True)
        done = self.paths.root / "done"
        if done.exists():
            for marker in done.glob("*.fit.*"):
                marker.unlink()
        rungs = self.paths.root / "rungs"
        if rungs.exists():
            for rung_dir in rungs.iterdir():
                probes = rung_dir / "probes"
                if probes.exists():
                    shutil.rmtree(probes)

    def write_gate(self, path: Path, result: dict[str, Any]) -> None:
        """Gates are recomputed on every launch. A fixed gate can flip one, which is worth saying out loud."""
        from .analysis import write_json_atomic

        if path.exists():
            previous = json.loads(path.read_text(encoding="utf-8")).get("passed")
            if previous != result.get("passed"):
                print(f"NOTE: {path.name} was {'passed' if previous else 'failed'} when last stored, and is now {'passed' if result.get('passed') else 'failed'}, from the same stored numbers. A gate or analysis fix changed the decision.", flush=True)
        write_json_atomic(path, result)

    def check_rung_capacity(self, rung: Rung) -> None:
        """Before this rung spends anything, including its download. Running short is a machine limit, not a finding."""
        if not self.check_capacity:
            return
        already = capacity.stored_gb(self.paths.rung_dir(rung.tag))
        if self.paths.done(f"{rung.tag}.plan").exists():
            already += rung.weights_gb  # The plan runs after the download.
        result = capacity.check_rung(self.cfg, rung, capacity.free_disk_gb(self.paths.root), capacity.available_ram_gb(), self.n_gpus, already)
        for warning in result.warnings:
            print(f"WARNING: {warning}.", flush=True)
        if result.stops:
            raise RunFailed(f"{'; '.join(result.stops)}. G1 was not finished on it, so this is not a finding about the model. Results so far are kept. Move to a larger machine and rerun with D1_RETRY=1.")

    def capture_tasks(self, tag: str, rung: Rung, model_id: str, sets: Sequence[str], after: Sequence[str] = ()) -> list[Task]:
        """Shards by tokens, a constant per shard, so their number and their rows are the same on any machine."""
        from .plan import read_meta

        create_stores(self.cfg, self.paths, tag, rung, model_id, sets)
        tokens = sum(read_meta(self.paths, tag, name)["tokens"] for name in sets)
        shards = max(1, math.ceil(tokens / self.tokens_per_shard))
        label = "-".join(sets)
        return [
            Task(
                f"{tag}.capture.{model_id}.{label}.{shard}of{shards}",
                ["capture", "--rung", tag, "--model-id", model_id, "--sets", ",".join(sets), "--shard", str(shard), "--n-shards", str(shards), "--token-budget", str(self.token_budget)],
                after=tuple(after),
            )
            for shard in range(shards)
        ]

    def apply_dry_run(self, tag: str) -> None:
        """A pool sizes its workers from what the dry run measured on this rung. A plain runner keeps the configured figures."""
        configure = getattr(self.runner, "configure", None)
        if configure is None:
            return
        path = self.paths.preflight(tag)
        dry_run = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        schedule = configure(dry_run, self.cfg.capture_token_budget)
        self.token_budget = schedule.token_budget
        self.gradient_checkpointing = schedule.gradient_checkpointing

    def run_stage1(self) -> tuple[str, dict[str, Any]]:
        """Up the ladder. Each rung reads its test set first, since G1 needs self-report to pass and that needs no probe.

        Fit rows are most of a rung's cost, so a rung whose self-report fails is done in minutes, and nothing about the
        decision changes: it would have failed G1 whatever a probe scored.
        """
        from . import analysis

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
            self.check_rung_capacity(rung)
            self.note(stage="stage1", rung=tag)
            set_default_need = getattr(self.runner, "set_default_need", None)
            if set_default_need is not None:
                set_default_need(rung.gpu_gb)
            self.token_budget = self.cfg.capture_token_budget
            self.gradient_checkpointing = True
            if self.download is not None:
                self.download(rung)
            self.runner.run(f"Plan for {rung.name}", [Task(f"{tag}.plan", ["plan", "--rung", tag], gpu=False, loads_model=False)])
            preflight = Task(f"{tag}.preflight", ["preflight", "--rung", tag])
            # At the config's budget, always, since the dry run has not measured a larger one yet. A failed dry run ends this at once.
            g1_rows = self.capture_tasks(tag, rung, BASE_MODEL, ["test_g1"])
            if self.overlaps:
                self.runner.submit(f"Dry-run checks and the G1 test set on {rung.name}", [preflight, *g1_rows])
            self.runner.run(f"Dry-run checks on {rung.name}", [preflight])
            self.apply_dry_run(tag)
            self.runner.run(f"Capture the G1 test set on unmodified {rung.name}", g1_rows)
            early = self._compute(analysis.evaluate_g1_self_report, self.cfg, self.paths, tag, self.thresholds)
            if early["self_report_ok"]:
                self.runner.run(f"Capture the probe rows on unmodified {rung.name}", self.capture_tasks(tag, rung, BASE_MODEL, ["fit", "val", "cal"]))
                self.runner.run(f"Fit the linear ensemble on {rung.name}", [Task(f"{tag}.fit.linear", ["fit-probes", "--rung", tag, "--name", "linear", "--kind", "linear"], loads_model=False)])
                result = self._compute(analysis.evaluate_g1, self.cfg, self.paths, tag, self.thresholds)
                summary = f"probe AUROC {result['probe_auroc']:.3f} [{result['probe_ci']['lo']:.3f}, {result['probe_ci']['hi']:.3f}], "
            else:
                result = analysis.g1_failed_on_self_report(tag, early)
                summary = "probe not fitted, since self-report already failed, "
            self.write_gate(self.paths.gate_g1(tag), result)
            history[tag] = result
            print(
                f"G1 on {rung.name}: {summary}"
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
        """The change log fixes the size. Its expected half-width is logged before any adapter exists, written once, and read back by every later launch."""
        from .analysis import write_json_atomic

        path = self.paths.sample_size(tag)
        if path.exists():
            return int(json.loads(path.read_text(encoding="utf-8"))["transcripts_per_arm"])
        adapters = self.paths.rung_dir(tag) / "adapters"
        if adapters.exists() and any(adapters.iterdir()):
            raise RunFailed("Adapters exist but no sample size was logged. That breaks the prereg's order of work.")
        sizing = g1["sizing"]
        write_json_atomic(path, {**sizing, "logged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        print(
            f"Test set: {sizing['transcripts_per_arm']} transcripts per arm. Expected half-width before any adapter exists: AUROC {sizing['expected_half_width_auroc']:.3f}, "
            f"D between {sizing['expected_half_width_D_if_unit_term_cancels']:.3f}, if the lines of work's term cancels in the pairing, and {sizing['expected_half_width_D_if_arms_independent']:.3f}, if the arms were independent. "
            "An inconclusive result stays a real outcome, more so at the larger figure.",
            flush=True,
        )
        return int(sizing["transcripts_per_arm"])

    def adapter_tasks(self, tag: str, rung: Rung, attempt: int, conditions: Sequence[str], seeds: Sequence[int]) -> tuple[list[Task], list[Task]]:
        """Training, then each adapter's reads, which wait for that adapter alone and not for the others of the stage."""
        trains: list[Task] = []
        reads: list[Task] = []
        for condition in conditions:
            for seed in seeds:
                model_id = adapter_id(attempt, condition, seed)
                name = f"{tag}.train.{model_id}"
                trains.append(Task(name, ["train", "--rung", tag, "--condition", condition, "--seed", str(seed), "--attempt", str(attempt), "--gradient-checkpointing", str(int(self.gradient_checkpointing))]))
                reads.extend(self.capture_tasks(tag, rung, model_id, GATE_SETS, after=(name,)))
        return trains, reads

    def train_and_eval(self, tag: str, rung: Rung, attempt: int, conditions: Sequence[str], seeds: Sequence[int]) -> None:
        trains, reads = self.adapter_tasks(tag, rung, attempt, conditions, seeds)
        self.runner.run(f"Train adapters on {rung.name} and read test sets through them, attempt {attempt}", [*trains, *reads])

    def run_gates(self, tag: str, rung: Rung, needs: Sequence[Task] = ()) -> tuple[int | None, dict[str, Any]]:
        """Returns the attempt whose adapters passed, or None if the stop rule fires.

        FLIP and HONEST are half of an attempt's adapters, so they train only after every DD and SHAM seed has passed G2, and the
        pool is not told of them before that. The one-seed stop-loss decides as ever, on the first seed alone. A pool with room
        may start the other DD and SHAM seeds meanwhile, since they are the same adapters whenever they train, and cancels them if
        the first seed fails. `needs` is what the gates read and what may still be running.
        """
        from .analysis import evaluate_g2

        seeds = list(self.cfg.training.seeds)
        history: dict[str, Any] = {}
        attempt = 0
        while True:
            self.note(stage="stage2_one_seed", attempt=attempt)
            # On a resume the stored gate is about to be recomputed, so there is nothing to get ahead of.
            if self.overlaps and not self.paths.gate_g2(tag, attempt, "one_seed").exists():
                trains, reads = self.adapter_tasks(tag, rung, attempt, ("DD", "SHAM"), seeds[1:])
                self._speculate("The other DD and SHAM seeds, if a card has room, until the first seed's G2", [*trains, *reads])
            self.train_and_eval(tag, rung, attempt, ("DD", "SHAM"), seeds[:1])
            self.runner.run("The unmodified model's test set and fits", needs)
            one = self._compute(evaluate_g2, self.cfg, self.paths, tag, attempt, seeds[:1], "one_seed", self.thresholds)
            self.write_gate(self.paths.gate_g2(tag, attempt, "one_seed"), one)
            history[f"a{attempt}_one_seed"] = one
            print(f"G2, one seed, attempt {attempt}: {one['checks']} {one['numbers']}", flush=True)
            decision = next_after_g2(one["passed"], attempt, "one_seed")
            if decision == "stop":
                self._cancel_speculation()
                return None, history
            if decision == "retry":
                self._cancel_speculation()
                attempt += 1
                continue
            self.note(stage="stage2_all_seeds", attempt=attempt)
            self._expect("FLIP and HONEST may not start until every DD and SHAM seed has passed G2, as the prereg says, and nothing else is left to run before that.")
            if self.overlaps:
                # A read of probe rows through an adapter trains nothing. It is wasted only if G2 goes on to fail.
                captures, fits = self.retrain_tasks(tag, rung, attempt)
                self._speculate("Probe rows through the DD and SHAM adapters, if a card has room, until G2 across seeds", [*captures, *fits])
            self.train_and_eval(tag, rung, attempt, ("DD", "SHAM"), seeds[1:])
            full = self._compute(evaluate_g2, self.cfg, self.paths, tag, attempt, seeds, "all_seeds", self.thresholds)
            self.write_gate(self.paths.gate_g2(tag, attempt, "all_seeds"), full)
            history[f"a{attempt}_all_seeds"] = full
            print(f"G2, all seeds, attempt {attempt}: {full['checks']} {full['numbers']}", flush=True)
            decision = next_after_g2(full["passed"], attempt, "all_seeds")
            self._expect(None)
            if decision == "finish":
                self.note(stage="stage2_controls", attempt=attempt)
                if self.overlaps:
                    self.runner.submit("Probe rows through the DD and SHAM adapters", self.speculated)
                self.speculated = []
                self.train_and_eval(tag, rung, attempt, ("FLIP", "HONEST"), seeds)
                return attempt, history
            self._cancel_speculation()
            if decision == "stop":
                return None, history
            attempt += 1

    def retrain_tasks(self, tag: str, rung: Rung, attempt: int) -> tuple[list[Task], list[Task]]:
        seeds = list(self.cfg.training.seeds)
        captures: list[Task] = []
        fits: list[Task] = []
        for condition in ("DD", "SHAM"):
            for seed in seeds:
                model_id = adapter_id(attempt, condition, seed)
                reads = self.capture_tasks(tag, rung, model_id, ["fit", "val", "cal"], after=(f"{tag}.train.{model_id}",))
                captures.extend(reads)
                fits.append(
                    Task(
                        f"{tag}.fit.retrained_{model_id}",
                        ["fit-probes", "--rung", tag, "--name", f"retrained_{model_id}", "--source", model_id, "--kind", "linear"],
                        loads_model=False,
                        after=tuple(task.name for task in reads),
                    )
                )
        return captures, fits

    def retrain_probes(self, tag: str, rung: Rung, attempt: int) -> None:
        captures, fits = self.retrain_tasks(tag, rung, attempt)
        self.runner.run("Capture probe-train rows through DD and SHAM adapters", captures)
        self.runner.run("Refit probes through the adapters", fits)

    def run(self) -> dict[str, Any]:
        from .analysis import format_g1_stop, format_g2_stop, format_report, full_analysis, stop_report, write_json_atomic
        from .gates import g2_failure_finding

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
            atomic_write_bytes(paths.root / "report.txt", format_g1_stop(g1_history).encode("utf-8"))
            return outcome
        rung = cfg.rung(tag)
        self.note(rung=tag, stage="sizing")
        self.fix_sample_size(tag, g1_history[tag])
        self.check_rung_capacity(rung)
        self.runner.run("Build the final test set", [Task(f"{tag}.test_set", ["test-set", "--rung", tag], gpu=False, loads_model=False)])
        base_reads = self.capture_tasks(tag, rung, BASE_MODEL, GATE_SETS)
        base_fits = [
            Task(f"{tag}.fit.mlp", ["fit-probes", "--rung", tag, "--name", "mlp", "--kind", "mlp"], loads_model=False),
            Task(f"{tag}.fit.shuffled", ["fit-probes", "--rung", tag, "--name", "shuffled", "--kind", "linear", "--shuffle"], loads_model=False),
        ]
        if self.overlaps:
            # Adapters read none of these, and the size was logged above, so no adapter has to wait for them. The gates do.
            self.runner.submit("The unmodified model's test set and fits, beside the first adapters", [*base_reads, *base_fits])
        else:
            self.runner.run("Capture the final test set on the unmodified model", base_reads)
            self.runner.run("Fit the MLP ensemble and the label-shuffled baseline", base_fits)
        attempt, g2_history = self.run_gates(tag, rung, [*base_reads, *base_fits])
        if attempt is None:
            finding = g2_failure_finding(list(g2_history.values())[-1]["checks"])
            outcome = stop_report("g2", {**g1_history, **g2_history}, finding)
            self.note(stage="finished", outcome="G2 failed after its retry")
            print(finding, flush=True)
            write_json_atomic(paths.root / "outcome.json", outcome)
            atomic_write_bytes(paths.root / "report.txt", format_g2_stop(finding, g2_history).encode("utf-8"))
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
        outcome = {
            "stopped_at": None,
            "verdict": results["primary"]["verdict"],
            "independence_strength": results["primary"]["independence_strength"],
            "weakly_controlled": results["weakly_controlled"],
            "rung": tag,
            "attempt": attempt,
            "seconds": time.monotonic() - started,
        }
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
    print(f"Downloading {rung.name}, about {rung.weights_gb:.0f} GB", flush=True)
    for attempt in range(5):
        process = subprocess.Popen([sys.executable, "-c", code, rung.name, rung.revision], env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        started = time.monotonic()
        # A line a minute, because the run script stops anything silent for too long, and a slow link is not a hang.
        while True:
            try:
                stdout, stderr = process.communicate(timeout=60.0)
                break
            except subprocess.TimeoutExpired:
                if time.monotonic() - started >= DOWNLOAD_SECONDS:
                    process.kill()
                    stdout, stderr = process.communicate()
                    break
                print(f"Still downloading {rung.name}, {_duration(time.monotonic() - started)} so far", flush=True)
        if process.returncode == 0:
            print(f"{rung.name} is at {stdout.strip()}", flush=True)
            return
        print(f"Download of {rung.name}, attempt {attempt + 1} of 5, failed: {stderr[-500:]}", flush=True)
        time.sleep(30 * (attempt + 1))
    raise RunFailed(f"The download of {rung.name} failed 5 times. Check the network and free disk.")


def _stop_on_sigterm(signum: int, _frame: Any) -> None:
    """A stop from outside must stop the workers too. The exit code is what the run script reads as an interruption."""
    raise SystemExit(128 + signum)


def run_all(
    config_path: Path,
    out_dir: Path,
    devices: Sequence[str],
    gpu_free_gb: float | None,
    load_slots: int = 1,
    threads_per_worker: int | None = None,
    stall_minutes: float = 30.0,
    factory: str | None = None,
    workers_per_gpu: int | None = None,
) -> dict[str, Any]:
    from .config import D1Config

    paths = D1Paths(Path(out_dir))
    paths.root.mkdir(parents=True, exist_ok=True)
    held = try_lock(paths.root / "locks" / "orchestrator.lock")
    if held is None:
        raise RunFailed(f"Another orchestrator is already running on {paths.root}")
    pool: TaskPool | None = None
    try:
        previous = signal.signal(signal.SIGTERM, _stop_on_sigterm)
    except ValueError:  # Not the main thread, as under some test runners.
        previous = None
    try:
        cfg = D1Config.from_file(config_path)
        options = PoolOptions(
            list(devices), Path(config_path), Path(out_dir), factory=factory, threads_per_worker=threads_per_worker,
            load_slots=load_slots, stall_seconds=stall_minutes * 60.0, workers_per_gpu=workers_per_gpu,
        )  # fmt: skip
        pool = TaskPool(options, paths)
        experiment = Experiment(cfg, paths, pool, len(devices), gpu_free_gb, download=None if factory else download_rung, check_capacity=factory is None)
        return experiment.run()
    finally:
        if pool is not None:
            pool.close()
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)
        held.close()


def reanalyze(config_path: Path, out_dir: Path) -> dict[str, Any]:
    """Recompute gates and the report from stored numbers, after a fix to analysis code or the analysis config. No GPU, no worker.

    Refuses if anything the stored work depends on has changed, or if some task never finished.
    """
    from .config import D1Config

    paths = D1Paths(Path(out_dir))
    if not paths.root.exists():
        raise RunFailed(f"{paths.root} does not exist, so there is nothing to re-analyze")
    held = try_lock(paths.root / "locks" / "orchestrator.lock")
    if held is None:
        raise RunFailed(f"Another orchestrator is already running on {paths.root}")
    try:
        return Experiment(D1Config.from_file(config_path), paths, NoWorkRunner(paths), 1, None).run()
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


__all__ = ["CONDITIONS", "Experiment", "InProcessRunner", "NoWorkRunner", "PoolOptions", "Schedule", "Task", "TaskPool", "finalize", "reanalyze", "run_all"]
