"""The shared pool and what makes its results independent of it. No GPU, no download.

Most of this drives the real TaskPool with fake processes, so it runs in moments and can stage out-of-memory errors, stalls and
gates at will. The tests at the end run real capture and training on the tiny model, in this process and in real workers.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pytest
import torch
from d1_factory import make_d1_model, make_d1_tokenizer, tiny_rows, write_tiny_config

from mechinterp.d1 import analysis
from mechinterp.d1.config import D1Config
from mechinterp.d1.data import ingest, write_jsonl
from mechinterp.d1.lora import LoraAdapter, train_adapter
from mechinterp.d1.orchestrator import EXIT_OUT_OF_MEMORY, Experiment, InProcessRunner, PoolOptions, Task, TaskPool, parse_gpu_query
from mechinterp.d1.paths import BASE_MODEL, D1Paths, adapter_id
from mechinterp.d1.plan import map_tokenized
from mechinterp.d1.prompts import followup_text, resolve_answers
from mechinterp.d1.scheduling import plan_schedule, usable_gb
from mechinterp.d1.stats import parallel_map, weighted_auroc
from mechinterp.d1.store import CaptureStore, assign_shards, read_complete
from mechinterp.d1.training_data import BatchBuilder, PrefetchedBatches
from mechinterp.prompts import encode_ids
from mechinterp.sharding import RunFailed

TESTS_DIR = Path(__file__).resolve().parent
FACTORY = "d1_factory:build"
TAG = "tiny"
ROUNDING = 2e-3


class FakeStdin:
    def __init__(self, process: "FakeProcess") -> None:
        self.process = process
        self.closed = False
        self.buffer: list[str] = []

    def write(self, text: str) -> None:
        self.buffer.append(text)

    def flush(self) -> None:
        for text in self.buffer:
            argv = json.loads(text)
            self.process.begin(argv[argv.index("--task-name") + 1])
        self.buffer = []

    def close(self) -> None:
        self.closed = True


class FakeProcess:
    """Runs a scripted outcome: for each attempt at a task, a number of polls and then an exit code, or forever for a stall."""

    counter = 0

    def __init__(self, world: "World", command: list[str], environment: dict[str, str], serves: bool) -> None:
        FakeProcess.counter += 1
        self.pid = FakeProcess.counter
        self.world = world
        self.device = environment["CUDA_VISIBLE_DEVICES"]
        self.serves = serves
        self.code: int | None = None
        self.current: str | None = None
        self.polls_left = 0
        self.outcome = 0
        self.stdin = FakeStdin(self) if serves else None
        world.spawned.append(self)
        if not serves:
            self.begin(command[command.index("--task-name") + 1])

    def begin(self, name: str) -> None:
        self.current = name
        self.polls_left, self.outcome = self.world.next_attempt(name)
        self.world.events.append(("start", name, self.device))

    def poll(self) -> int | None:
        if self.code is not None:
            return self.code
        if self.current is None:
            if self.serves and self.stdin.closed:
                self.code = 0
            return self.code
        if self.polls_left is None:
            self.world.beat_if_healthy(self.current)
            return None
        if self.polls_left > 0:
            self.polls_left -= 1
            self.world.beat_if_healthy(self.current)
            return None
        name, self.current = self.current, None
        if self.outcome != 0:
            self.code = self.outcome
            return self.code
        self.world.events.append(("finish", name, self.device))
        if self.serves:
            marker = self.world.paths.done(name)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()
            return None
        self.code = 0
        return 0

    def terminate(self) -> None:
        self.kill()

    def kill(self) -> None:
        self.world.events.append(("kill", self.current, self.device))
        self.current = None
        self.code = -9

    def wait(self, timeout: float | None = None) -> int | None:
        return self.code


@dataclass
class World:
    paths: D1Paths
    script: dict[str, list[tuple[int | None, int]]] = field(default_factory=dict)
    default_polls: int = 2
    healthy_beats: set[str] = field(default_factory=set)
    events: list[tuple[str, str | None, str]] = field(default_factory=list)
    spawned: list[FakeProcess] = field(default_factory=list)
    attempts: dict[str, int] = field(default_factory=dict)

    def next_attempt(self, name: str) -> tuple[int | None, int]:
        planned = self.script.get(name, [])
        index = self.attempts.get(name, 0)
        self.attempts[name] = index + 1
        return planned[index] if index < len(planned) else (self.default_polls, 0)

    def beat_if_healthy(self, name: str) -> None:
        if name in self.healthy_beats:
            path = self.paths.heartbeat(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"done": 1, "total": 2, "rate": 1.0, "loaded": True}), encoding="utf-8")

    def spawn(self, command: list[str], environment: dict[str, str], log_path: Path, serves: bool) -> FakeProcess:
        return FakeProcess(self, command, environment, serves)

    def started(self, name: str) -> int:
        return sum(1 for kind, task, _ in self.events if kind == "start" and task == name)

    def index_of(self, kind: str, name: str) -> int:
        return next(index for index, event in enumerate(self.events) if event[0] == kind and event[1] == name)


def card_states(free: float = 24.0, util: float = 99.0) -> Callable[[], dict]:
    def query() -> dict:
        state = {"util": util, "used_gb": 0.0, "total_gb": 24.0, "free_gb": free}
        return {"0": dict(state), "1": dict(state)}

    return query


def make_pool(tmp_path: Path, cards: Sequence[str] = ("0", "1"), query: Callable[[], dict] | None = None, **options: Any) -> tuple[TaskPool, World]:
    paths = D1Paths(tmp_path / "out")
    paths.root.mkdir(parents=True, exist_ok=True)
    world = World(paths)
    settings = {"poll_seconds": 0.001, "gpu_poll_seconds": 0.0, "settle_seconds": 0.0, "progress_seconds": 3600.0, "load_slots": 8, **options}
    pool = TaskPool(PoolOptions(devices=list(cards), config_path=tmp_path / "c.json", out_dir=paths.root, **settings), paths, spawn=world.spawn, gpu_query=query or card_states())
    return pool, world


def shard(name: str, after: Sequence[str] = ()) -> Task:
    return Task(name, ["capture", "--rung", TAG], after=tuple(after))


def train(name: str, after: Sequence[str] = ()) -> Task:
    return Task(name, ["train", "--rung", TAG, "--gradient-checkpointing", "0"], after=tuple(after))


# The pool


def test_every_task_runs_exactly_once_and_a_card_can_take_more_than_its_share(tmp_path: Path) -> None:
    pool, world = make_pool(tmp_path)
    tasks = [shard(f"s{index}") for index in range(40)] + [train(f"t{index}") for index in range(3)]
    world.script["s0"] = [(200, 0)]  # A slow shard holds one card, so the other one takes the rest.
    pool.run("All", tasks)
    assert all(world.started(task.name) == 1 for task in tasks)
    assert all(pool.paths.done(task.name).exists() for task in tasks)
    assert sum(1 for kind, _, _ in world.events if kind == "finish") == len(tasks)
    slow_card = next(device for kind, name, device in world.events if kind == "start" and name == "s0")
    other = next(card for card in pool.cards if card != slow_card)
    assert pool.finished_on[other] > pool.finished_on[slow_card]
    pool.close()


def test_finished_tasks_are_skipped_on_a_resume(tmp_path: Path) -> None:
    pool, world = make_pool(tmp_path)
    marker = pool.paths.done("old")
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    pool.run("All", [shard("old"), shard("new")])
    assert world.started("old") == 0 and world.started("new") == 1
    pool.close()


def test_capture_shards_reuse_a_loaded_worker(tmp_path: Path) -> None:
    pool, world = make_pool(tmp_path, cards=("0",))
    pool.run("All", [shard(f"s{index}") for index in range(10)])
    assert len(world.spawned) == 1, "one model load for all ten shards"
    pool.close()


def test_out_of_memory_requeues_backs_off_the_slot_and_is_not_counted_as_a_crash(tmp_path: Path) -> None:
    pool, world = make_pool(tmp_path, cards=("0",), workers_per_gpu=2, max_restarts=0)
    world.script["a"] = [(30, 0)]
    world.script["b"] = [(1, EXIT_OUT_OF_MEMORY)]
    pool.run("All", [train("a"), train("b")])
    assert world.started("a") == 1
    assert world.started("b") == 2, "put back and run again"
    assert pool.restarts == {}, "with a restart limit of 0, a crash here would have ended the run"
    assert pool.card_limit["0"] == 1, "the card takes one worker fewer from now on"
    assert pool.paths.done("a").exists() and pool.paths.done("b").exists()
    pool.close()


def test_out_of_memory_alone_on_a_card_steps_down_training_and_can_only_go_so_far(tmp_path: Path) -> None:
    pool, world = make_pool(tmp_path, cards=("0",))
    task = train("a")
    world.script["a"] = [(1, EXIT_OUT_OF_MEMORY), (1, EXIT_OUT_OF_MEMORY)]
    pool.run("All", [task])
    assert task.argv[task.argv.index("--gradient-checkpointing") + 1] == "1", "first, checkpointing goes back on"
    assert "--halve-on-oom" in task.argv, "then the micro-batch may fall"
    pool.close()

    other, world = make_pool(tmp_path, cards=("0",))
    world.script["b"] = [(1, EXIT_OUT_OF_MEMORY)] * 20
    with pytest.raises(RunFailed, match="not a finding"):
        other.run("All", [shard("b")])


def test_a_crash_uses_the_restart_limit(tmp_path: Path) -> None:
    pool, world = make_pool(tmp_path, cards=("0",), max_restarts=1)
    world.script["a"] = [(1, 1), (1, 0)]
    pool.run("All", [train("a")])
    assert pool.restarts == {"a": 1}
    pool.close()

    other, world = make_pool(tmp_path, cards=("0",), max_restarts=1)
    world.script["b"] = [(1, 1), (1, 1)]
    with pytest.raises(RunFailed, match="failed 2 times"):
        other.run("All", [train("b")])


def test_a_capture_worker_that_dies_gives_back_only_its_own_shard(tmp_path: Path) -> None:
    pool, world = make_pool(tmp_path, cards=("0",))
    world.script["s1"] = [(1, 1)]
    tasks = [shard(f"s{index}") for index in range(4)]
    pool.run("All", tasks)
    assert world.started("s0") == 1 and world.started("s1") == 2
    assert all(pool.paths.done(task.name).exists() for task in tasks)
    pool.close()


def test_only_real_dependencies_block_a_task(tmp_path: Path) -> None:
    pool, world = make_pool(tmp_path)
    world.script["adapter"] = [(50, 0)]
    pool.run("All", [train("adapter"), shard("read", after=("adapter",)), shard("unrelated"), shard("also unrelated")])
    assert world.index_of("start", "read") > world.index_of("finish", "adapter"), "a read needs its adapter"
    assert world.index_of("start", "unrelated") < world.index_of("finish", "adapter"), "and nothing else waits for it"
    assert world.index_of("start", "also unrelated") < world.index_of("finish", "adapter")
    pool.close()


def test_a_task_nothing_can_unblock_is_a_failure_and_not_a_hang(tmp_path: Path) -> None:
    pool, _world = make_pool(tmp_path)
    with pytest.raises(RunFailed, match="which nothing is running"):
        pool.run("All", [shard("read", after=("never",))])


def test_required_tasks_go_before_speculative_ones_and_adapters_before_shards(tmp_path: Path) -> None:
    pool, world = make_pool(tmp_path, cards=("0",))
    pool.submit("Speculative", [shard("maybe")], speculative=True)
    pool.run("Required", [shard("needed"), train("adapter")])
    pool.run("Promoted", [shard("maybe")])
    order = [name for kind, name, _ in world.events if kind == "start"]
    assert order == ["adapter", "needed", "maybe"]
    pool.close()


def test_speculative_work_can_be_cancelled_and_leaves_nothing_marked_done(tmp_path: Path) -> None:
    pool, world = make_pool(tmp_path, cards=("0", "1"))
    world.script["first"] = [(5, 0)]
    world.script["running"] = [(None, 0)]
    speculative = [train("running"), train("queued", after=("never",))]
    pool.submit("Maybe", speculative, speculative=True)
    pool.run("First", [train("first")])
    assert world.started("running") == 1
    pool.cancel(speculative)
    assert not pool.queue
    assert ("kill", "running", "1") in world.events or ("kill", "running", "0") in world.events
    assert not pool.paths.done("running").exists() and not pool.paths.done("queued").exists()
    pool.run("Next", [train("next")])
    assert pool.paths.done("next").exists()
    assert all(worker.task is None or worker.task.name != "running" for worker in pool.workers)
    pool.close()


def test_asking_for_a_speculative_task_makes_it_required_so_a_later_cancel_leaves_it(tmp_path: Path) -> None:
    pool, world = make_pool(tmp_path, cards=("0",))
    world.script["t"] = [(None, 0)]
    task = train("t")
    pool.submit("Maybe", [task], speculative=True)
    pool._tick()
    pool.submit("Needed", [task])
    pool.cancel([task])
    assert world.started("t") == 1
    assert all(kind != "kill" for kind, _, _ in world.events)
    pool.close()


def test_only_the_stalled_worker_on_a_shared_card_is_killed(tmp_path: Path) -> None:
    pool, world = make_pool(tmp_path, cards=("0",), workers_per_gpu=2, stall_seconds=0.15, load_stall_seconds=0.15)
    world.healthy_beats = {"healthy"}
    world.script["healthy"] = [(None, 0)]  # Runs as long as this test wants, beating all the while.
    world.script["stuck"] = [(None, 0), (1, 0)]  # Never beats on its first attempt, and finishes on its second.
    healthy, stuck = train("healthy"), train("stuck")
    pool.submit("Both", [healthy, stuck])
    deadline = time.monotonic() + 20.0
    while not pool.paths.done("stuck").exists() and time.monotonic() < deadline:
        pool._tick()
        time.sleep(0.001)
    assert pool.paths.done("stuck").exists()
    assert world.started("stuck") == 2
    assert world.started("healthy") == 1, "its heartbeat is its own, so its neighbour's stall does not touch it"
    assert pool.restarts == {"stuck": 1}
    assert not pool.paths.done("healthy").exists()
    pool.close()


def test_a_loud_warning_names_why_a_card_is_idle(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    pool, world = make_pool(tmp_path, idle_warn_seconds=0.0)
    world.script["adapter"] = [(5, 0)]
    pool.run("All", [train("adapter"), shard("read", after=("adapter",))])
    text = capsys.readouterr().out
    assert "WARNING: GPU" in text and "can only start after ['adapter']" in text
    pool.close()


def test_a_card_held_back_by_the_prereg_says_so_and_no_pending_work_means_no_warning(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    pool, _world = make_pool(tmp_path, idle_warn_seconds=0.0)
    pool._tick()
    assert "WARNING" not in capsys.readouterr().out
    pool.expect("FLIP and HONEST wait for G2")
    pool._tick()
    assert "FLIP and HONEST wait for G2" in capsys.readouterr().out
    pool.close()


def test_a_second_worker_joins_a_card_only_if_it_fits_and_the_card_has_compute_to_spare(tmp_path: Path) -> None:
    def holds(pool: TaskPool, world: World) -> int:
        world.script.update({"a": [(None, 0)], "b": [(None, 0)]})
        pool.needs["train"] = 10.0
        pool.submit("Both", [train("a"), train("b")])
        for _ in range(12):
            pool._tick()
        running = len([worker for worker in pool.workers if worker.task is not None])
        pool.close()
        return running

    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "c").mkdir()
    assert holds(*make_pool(tmp_path / "a", cards=("0",), query=card_states(util=50.0))) == 2, "fits, and half the card is idle"
    assert holds(*make_pool(tmp_path / "b", cards=("0",), query=card_states(util=99.0))) == 1, "fits, but they would only take turns"
    assert holds(*make_pool(tmp_path / "c", cards=("0",), query=card_states(free=12.0, util=50.0))) == 1, "the card has compute to spare, but no room"


def test_an_idle_capture_worker_gives_its_memory_to_an_adapter(tmp_path: Path) -> None:
    pool, world = make_pool(tmp_path, cards=("0",))
    pool.needs.update({"capture": 15.0, "train": 15.0})
    pool.run("Capture", [shard("s")])
    assert len(pool.workers) == 1 and pool.workers[0].task is None
    pool.run("Train", [train("t")])
    assert pool.paths.done("t").exists()
    assert world.spawned[0].code == 0, "it left when its stdin closed"
    pool.close()


def test_one_out_of_memory_error_raises_only_that_tasks_need_and_the_next_adapter_still_starts(tmp_path: Path) -> None:
    """The review's trace: one error used to raise every adapter's need past the card, while an idle capture worker kept it from emptying."""
    pool, world = make_pool(tmp_path, cards=("0",), starved_seconds=5.0)
    pool.needs.update({"capture": 5.0, "train": 19.1})
    first, second = train("dd0"), train("dd1")
    world.script["dd0"] = [(1, EXIT_OUT_OF_MEMORY)]
    pool.run("Seed", [first])
    assert pool._need(first) > 19.1
    assert pool._need(second) == pytest.approx(19.1)
    pool.run("Read", [shard("read")])
    assert pool.workers and pool.workers[0].task is None, "an idle capture worker holds the card"
    pool.run("Next", [second])
    assert pool.paths.done("dd1").exists()
    pool.close()


def test_a_task_bigger_than_the_card_still_empties_it_of_idle_workers_and_gets_its_chance(tmp_path: Path) -> None:
    pool, world = make_pool(tmp_path, cards=("0",), starved_seconds=5.0)
    pool.needs["capture"] = 5.0
    pool.run("Read", [shard("read")])
    big = train("big")
    big.need_gb = 100.0
    pool.run("Big", [big])
    assert pool.paths.done("big").exists()
    assert world.spawned[0].code == 0, "it left when its stdin closed"
    pool.close()


def test_running_out_of_memory_beside_an_idle_worker_is_not_being_alone_on_the_card(tmp_path: Path) -> None:
    pool, world = make_pool(tmp_path, cards=("0",))
    pool.needs.update({"capture": 5.0, "train": 5.0})
    pool.run("Read", [shard("read")])
    task = train("t")
    world.script["t"] = [(1, EXIT_OUT_OF_MEMORY)]
    pool.run("Train", [task])
    assert world.spawned[0].code == 0, "the idle worker made way"
    assert task.argv[task.argv.index("--gradient-checkpointing") + 1] == "0", "the adapter did not step down"
    assert "--halve-on-oom" not in task.argv
    assert pool.card_limit == {}
    pool.close()


def test_a_worker_told_to_leave_that_does_not_is_killed(tmp_path: Path) -> None:
    pool, world = make_pool(tmp_path, cards=("0",), retire_wait_seconds=0.0)
    pool.run("Read", [shard("read")])
    worker = pool.workers[0]
    worker.process.stdin.close = lambda: None
    pool._retire(worker)
    pool._reap()
    assert not pool.workers
    assert ("kill", None, "0") in world.events
    pool.close()


def test_required_work_that_can_never_start_ends_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pool, _world = make_pool(tmp_path, cards=("0",), starved_seconds=0.05)
    monkeypatch.setattr(pool, "_choose_card", lambda task, need: None)
    with pytest.raises(RunFailed, match="not a finding"):
        pool.run("Stuck", [train("t")])


def test_a_task_with_no_heartbeat_that_never_ends_stops_the_run(tmp_path: Path) -> None:
    pool, world = make_pool(tmp_path, cards=("0",), no_progress_seconds=0.1)
    world.script["plan"] = [(None, 0)]
    with pytest.raises(RunFailed, match="stuck run"):
        pool.run("Plan", [Task("plan", ["plan", "--rung", TAG], gpu=False, loads_model=False)])


def test_a_long_task_that_keeps_beating_and_a_long_gate_are_not_stuck(tmp_path: Path) -> None:
    pool, world = make_pool(tmp_path, cards=("0",), no_progress_seconds=0.2)
    assert pool.compute(lambda: time.sleep(0.5) or "done") == "done", "nothing was waiting, so the gate can take as long as it takes"
    world.healthy_beats = {"long"}
    world.script["long"] = [(None, 0)]
    pool.submit("Long", [train("long")])
    deadline = time.monotonic() + 0.6
    while time.monotonic() < deadline:
        pool._tick()
        time.sleep(0.005)
    assert world.started("long") == 1
    pool.close()


def test_a_speculative_failure_does_not_end_the_run_and_a_gate_can_still_ask_for_it(tmp_path: Path) -> None:
    pool, world = make_pool(tmp_path)
    world.script["first"] = [(30, 0)]
    world.script["spec"] = [(1, 3)]  # EXIT_FATAL
    spec, reader = train("spec"), shard("reader", after=("spec",))
    pool.submit("Maybe", [spec, reader], speculative=True)
    pool.run("First", [train("first")])
    assert pool.paths.done("first").exists()
    assert not pool.paths.done("spec").exists()
    assert not pool.queue and "spec" not in pool.speculative, "it left with what reads it"
    pool.run("Needed", [train("spec")])
    assert pool.paths.done("spec").exists() and world.started("spec") == 2
    pool.close()


def test_a_required_task_failing_after_its_speculative_try_ends_the_run(tmp_path: Path) -> None:
    pool, world = make_pool(tmp_path)
    world.script["first"] = [(30, 0)]
    world.script["spec"] = [(1, 3), (1, 3)]
    pool.submit("Maybe", [train("spec")], speculative=True)
    pool.run("First", [train("first")])
    with pytest.raises(RunFailed, match="cannot fix"):
        pool.run("Needed", [train("spec")])


def test_a_speculative_task_that_required_work_reads_is_required(tmp_path: Path) -> None:
    pool, world = make_pool(tmp_path)
    world.script["spec"] = [(1, 3)]
    pool.submit("Maybe", [train("spec")], speculative=True)
    with pytest.raises(RunFailed, match="cannot fix"):
        pool.run("Needed", [shard("reader", after=("spec",))])


def test_gates_run_while_the_cards_keep_working(tmp_path: Path) -> None:
    pool, world = make_pool(tmp_path)
    pool.submit("Beside", [shard("s0"), shard("s1")])
    assert pool.compute(lambda: (time.sleep(0.5), 7)[1]) == 7
    assert pool.paths.done("s0").exists() and pool.paths.done("s1").exists()
    pool.close()


def test_progress_prints_each_cards_numbers(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    pool, world = make_pool(tmp_path, progress_seconds=0.0, query=card_states(util=87.0))
    world.script["t"] = [(4, 0)]
    pool.run("All", [train("t")])
    text = capsys.readouterr().out
    assert "card 0: 87%" in text and "tasks/s" in text and "ETA" in text
    pool.close()


def test_a_gpu_reading_with_a_missing_field_is_left_out() -> None:
    assert parse_gpu_query("0, 50, 1024, 24576, 23552\n1, [N/A], 0, 24576, 24576\nbad line\n").keys() == {"0"}
    assert parse_gpu_query("") == {}


# What the dry run decides


MEASURED = {
    "train_peak_gpu_gb": 9.0,
    "train_peak_gpu_gb_no_checkpointing": 16.0,
    "train_seconds_per_step_on_longest_rows": 3.0,
    "train_seconds_per_step_no_checkpointing": 2.0,
    "weights_gpu_gb": 7.0,
    "capture_peak_gpu_gb": 12.0,
}


def test_a_24_gb_card_leaves_checkpointing_off_when_that_is_measured_to_fit_and_help() -> None:
    schedule = plan_schedule(24.0, MEASURED, 14.0, 24000)
    assert schedule.gradient_checkpointing is False
    assert schedule.train_need_gb <= usable_gb(24.0)
    assert schedule.token_budget > 24000
    assert schedule.capture_need_gb <= usable_gb(24.0) + 1e-6


def test_a_small_card_keeps_checkpointing_and_a_smaller_budget() -> None:
    schedule = plan_schedule(14.0, MEASURED, 14.0, 24000)
    assert schedule.gradient_checkpointing is True
    assert schedule.train_need_gb == schedule.train_checkpointing_need_gb
    assert schedule.token_budget <= 24000


def test_no_gain_from_dropping_checkpointing_keeps_it_on() -> None:
    assert plan_schedule(48.0, {**MEASURED, "train_seconds_per_step_no_checkpointing": 2.9}, 14.0, 24000).gradient_checkpointing is True


def test_asking_for_workers_per_card_shrinks_what_each_may_use() -> None:
    one = plan_schedule(24.0, MEASURED, 14.0, 24000)
    two = plan_schedule(24.0, MEASURED, 14.0, 24000, workers_per_gpu=2)
    assert two.gradient_checkpointing is True, "two plain workers do not fit"
    assert two.token_budget < one.token_budget


def test_nothing_measured_means_the_configured_figures() -> None:
    for free in (None, 24.0):
        schedule = plan_schedule(free, {}, 14.0, 24000)
        assert schedule.gradient_checkpointing is True
        assert schedule.token_budget == 24000
        assert schedule.train_need_gb == schedule.capture_need_gb == 14.0


# Shards


def test_shards_cover_every_row_once_and_balance_tokens() -> None:
    lengths = [int(value) for value in np.random.default_rng(0).integers(10, 2000, 500)]
    shards = assign_shards(lengths, 8)
    assert sorted(row for rows in shards for row in rows) == list(range(500))
    tokens = [sum(lengths[row] for row in rows) for rows in shards]
    assert max(tokens) - min(tokens) <= 2 * max(lengths)
    assert assign_shards(lengths, 8) == shards
    assert assign_shards(lengths, 1) == [sorted(range(500), key=lambda row: (-lengths[row], row))]


def test_shard_count_comes_from_the_tokens_and_not_from_the_gpu_count(tmp_path: Path) -> None:
    setup_one = planned(tmp_path / "one")
    tasks = {n_gpus: Experiment(setup_one.cfg, setup_one.paths, None, n_gpus, None) for n_gpus in (1, 4)}
    for experiment in tasks.values():
        experiment.tokens_per_shard = 500
    rung = setup_one.cfg.rung(TAG)
    names = {n_gpus: [task.name for task in experiment.capture_tasks(TAG, rung, BASE_MODEL, ["test_g1"])] for n_gpus, experiment in tasks.items()}
    assert names[1] == names[4]
    assert len(names[1]) > 1


# The prereg's order, with a pool that can overlap


class RecordingRunner:
    """Has what a pool has, and finishes whatever it is asked to wait for."""

    def __init__(self, paths: D1Paths) -> None:
        self.paths = paths
        self.log: list[tuple[str, list[str]]] = []

    def submit(self, label: str, tasks: Sequence[Task], speculative: bool = False) -> None:
        self.log.append(("speculate" if speculative else "submit", [task.name for task in tasks]))

    def run(self, label: str, tasks: Sequence[Task]) -> None:
        self.log.append(("run", [task.name for task in tasks]))
        for task in tasks:
            marker = self.paths.done(task.name)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()

    def cancel(self, tasks: Sequence[Task]) -> None:
        self.log.append(("cancel", [task.name for task in tasks]))

    def compute(self, function: Callable[..., Any], *args: Any) -> Any:
        return function(*args)

    def expect(self, reason: str | None) -> None:
        pass

    def names(self, action: str) -> list[str]:
        return [name for kind, names in self.log if kind == action for name in names]

    def position(self, predicate: Callable[[str], bool]) -> int:
        return next(index for index, (_, names) in enumerate(self.log) if any(predicate(name) for name in names))


def gate(passed: bool, label: str, log: list) -> dict:
    log.append(label)
    checks = dict.fromkeys(("dd_says_yes", "dd_answer_silent", "dd_specific", "sham_self_report_kept", "dd_capability_kept", "sham_capability_kept"), passed)
    return {"passed": passed, "checks": checks, "numbers": {}, "label": label}


def gates_with(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, decide: Callable[[int, str], bool]) -> tuple[RecordingRunner, Any, list]:
    setup = planned(tmp_path)
    runner = RecordingRunner(setup.paths)
    seen: list = []
    monkeypatch.setattr(analysis, "evaluate_g2", lambda cfg, paths, tag, attempt, seeds, label, thresholds: gate(decide(attempt, label), f"a{attempt}_{label}", seen))
    experiment = Experiment(setup.cfg, setup.paths, runner, 2, None)
    result = experiment.run_gates(TAG, setup.cfg.rung(TAG))
    return runner, result, seen


def train_name(attempt: int, condition: str, seed: int) -> str:
    return f"{TAG}.train.{adapter_id(attempt, condition, seed)}"


def test_first_seed_failing_cancels_the_other_seeds_and_uses_the_one_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner, result, seen = gates_with(tmp_path, monkeypatch, lambda attempt, label: False)
    assert result[0] is None
    assert seen == ["a0_one_seed", "a1_one_seed"], "the stop-loss decides on the first seed alone, and retries once"
    for attempt in (0, 1):
        assert train_name(attempt, "DD", 1) in runner.names("speculate")
        assert train_name(attempt, "DD", 1) in runner.names("cancel")
        assert train_name(attempt, "DD", 1) not in runner.names("run"), "never required, so no gate can read it"
    assert not any("FLIP" in name or "HONEST" in name for kind, names in runner.log for name in names)


def test_flip_and_honest_are_not_even_queued_until_every_seed_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner, result, seen = gates_with(tmp_path, monkeypatch, lambda attempt, label: True)
    assert result[0] == 0
    assert seen == ["a0_one_seed", "a0_all_seeds"]
    speculative = runner.names("speculate")
    assert train_name(0, "SHAM", 1) in speculative
    assert not any("FLIP" in name or "HONEST" in name for name in speculative), "only DD and SHAM may be speculated"
    first_control = runner.position(lambda name: "FLIP" in name or "HONEST" in name)
    last_main = max(index for index, (kind, names) in enumerate(runner.log) if kind == "run" and any(".train." in name and ("_DD_" in name or "_SHAM_" in name) for name in names))
    assert first_control > last_main


def test_seeds_failing_across_the_median_cancel_the_speculated_probe_reads_and_no_controls_train(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner, result, seen = gates_with(tmp_path, monkeypatch, lambda attempt, label: label == "one_seed")
    assert result[0] is None
    assert seen == ["a0_one_seed", "a0_all_seeds", "a1_one_seed", "a1_all_seeds"]
    assert any(f"{adapter_id(0, 'DD', 0)}.fit-val-cal" in name for name in runner.names("cancel"))
    assert not any("FLIP" in name or "HONEST" in name for kind, names in runner.log for name in names)


def test_a_stored_one_seed_gate_means_no_speculation_on_a_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    setup = planned(tmp_path)
    runner = RecordingRunner(setup.paths)
    setup.paths.gate_g2(TAG, 0, "one_seed").write_text(json.dumps({"passed": False}), encoding="utf-8")
    seen: list = []
    monkeypatch.setattr(analysis, "evaluate_g2", lambda cfg, paths, tag, attempt, seeds, label, thresholds: gate(False, f"a{attempt}_{label}", seen))
    Experiment(setup.cfg, setup.paths, runner, 2, None).run_gates(TAG, setup.cfg.rung(TAG))
    assert train_name(0, "DD", 1) not in runner.names("speculate")


# Real capture on the tiny model


@dataclass
class Planned:
    config: Path
    cfg: D1Config
    paths: D1Paths


def planned(directory: Path) -> Planned:
    directory.mkdir(parents=True, exist_ok=True)
    config = write_tiny_config(directory)
    cfg = D1Config.from_file(config)
    paths = D1Paths(directory / "out")
    transcripts, native, ordinary, manifest = ingest(tiny_rows(12), cfg.partition, None)
    write_jsonl(paths.transcripts, transcripts)
    write_jsonl(paths.native, native)
    write_jsonl(paths.ordinary, ordinary)
    paths.data_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    options = PoolOptions(devices=[], config_path=config, out_dir=paths.root, factory=FACTORY)
    InProcessRunner(options, paths).run("Plan", [Task(f"{TAG}.plan", ["plan", "--rung", TAG], gpu=False, loads_model=False), Task(f"{TAG}.test_set", ["test-set", "--rung", TAG], gpu=False, loads_model=False)])
    return Planned(config, cfg, paths)


def stored(paths: D1Paths, model_id: str = BASE_MODEL, name: str = "test_g1") -> tuple[np.ndarray, np.ndarray]:
    store = read_complete(paths.capture_dir(TAG, model_id, name))
    return np.asarray(store.acts, dtype=np.float32), np.asarray(store.logprobs)


def assert_same_reads(first: tuple[np.ndarray, np.ndarray], second: tuple[np.ndarray, np.ndarray]) -> None:
    """The bars the padding test uses, tests/test_d1_prompts_capture.py, since other batches mean other padding."""
    assert np.allclose(first[0], second[0], rtol=ROUNDING, atol=1e-3)
    assert np.allclose(first[1], second[1], rtol=0.0, atol=1e-3, equal_nan=True)


def capture_in_process(setup: Planned, tokens_per_shard: int, reverse: bool = False, model_id: str = BASE_MODEL) -> None:
    experiment = Experiment(setup.cfg, setup.paths, None, 1, None)
    experiment.tokens_per_shard = tokens_per_shard
    tasks = experiment.capture_tasks(TAG, setup.cfg.rung(TAG), model_id, ["test_g1"])
    options = PoolOptions(devices=[], config_path=setup.config, out_dir=setup.paths.root, factory=FACTORY)
    InProcessRunner(options, setup.paths).run("Capture", list(reversed(tasks)) if reverse else tasks)


def test_results_do_not_depend_on_shard_count_or_on_the_order_shards_are_taken(tmp_path: Path) -> None:
    whole, sharded, backwards = planned(tmp_path / "a"), planned(tmp_path / "b"), planned(tmp_path / "c")
    capture_in_process(whole, 10**9)
    capture_in_process(sharded, 300)
    capture_in_process(backwards, 300, reverse=True)
    assert_same_reads(stored(whole.paths), stored(sharded.paths))
    # The same shards in another order are the same batches, so here not even the last bit may move on a CPU.
    assert np.array_equal(stored(sharded.paths)[0], stored(backwards.paths)[0])
    assert np.array_equal(stored(sharded.paths)[1], stored(backwards.paths)[1], equal_nan=True)


def test_real_workers_on_two_slots_give_what_one_process_gives(tmp_path: Path) -> None:
    alone, pooled = planned(tmp_path / "alone"), planned(tmp_path / "pooled")
    capture_in_process(alone, 300)
    options = PoolOptions(
        devices=[], config_path=pooled.config, out_dir=pooled.paths.root, factory=FACTORY, cpu_workers=2, threads_per_worker=1, poll_seconds=0.05,
        extra_env={"PYTHONPATH": str(TESTS_DIR), "HF_HUB_OFFLINE": "1"},
    )  # fmt: skip
    pool = TaskPool(options, pooled.paths)
    try:
        experiment = Experiment(pooled.cfg, pooled.paths, pool, 2, None)
        experiment.tokens_per_shard = 300
        tasks = experiment.capture_tasks(TAG, pooled.cfg.rung(TAG), BASE_MODEL, ["test_g1"])
        assert len(tasks) > 1
        pool.run("Capture", tasks)
        assert len(pool.workers) == 2, "two loaded workers shared the shards"
    finally:
        pool.close()
    assert_same_reads(stored(alone.paths), stored(pooled.paths))


def test_a_read_after_an_adapter_on_one_loaded_model_is_a_read_of_the_base(tmp_path: Path) -> None:
    from mechinterp.d1.produce import LoadedModel, task_capture

    setup = planned(tmp_path / "shared")
    fresh = planned(tmp_path / "fresh")
    cfg, paths = setup.cfg, setup.paths
    rung = cfg.rung(TAG)
    model_id = adapter_id(0, "DD", 0)
    Experiment(cfg, paths, None, 1, None).capture_tasks(TAG, rung, model_id, ["test_g1"])
    Experiment(cfg, paths, None, 1, None).capture_tasks(TAG, rung, BASE_MODEL, ["test_g1"])
    loaded = LoadedModel()
    model, _, _ = loaded.get(rung, FACTORY, False)
    adapter = LoraAdapter.for_model(model, 2, 4.0, 1)
    with torch.no_grad():
        for up in adapter.up:
            up.normal_(0.0, 0.1)
    paths.adapter(TAG, model_id).mkdir(parents=True)
    adapter.save(paths.adapter(TAG, model_id) / "adapter.pt", {"condition": "DD"})
    task_capture(cfg, paths, TAG, model_id, ["test_g1"], 0, 1, "through", FACTORY, False, None, loaded)
    task_capture(cfg, paths, TAG, BASE_MODEL, ["test_g1"], 0, 1, "after", FACTORY, False, None, loaded)
    capture_in_process(fresh, 10**9)
    assert np.array_equal(stored(paths)[0], stored(fresh.paths)[0]), "nothing of the adapter stayed on the model"
    assert not np.allclose(stored(paths)[0], stored(paths, model_id)[0], rtol=ROUNDING, atol=1e-5), "and the adapter did do something"


def test_a_row_that_cannot_be_written_stops_capture_and_keeps_what_was_finished(tmp_path: Path) -> None:
    from mechinterp.d1.capture import RowWriter
    from mechinterp.d1.store import ActivationOverflow

    store = CaptureStore.create(tmp_path / "s", 3, 1, 2)
    writer = RowWriter(store, 3, None)
    good = np.ones((1, 1, 2), dtype=np.float32)
    writer.put([0], good, np.zeros((1, 2)), 1)
    writer.put([1], good * 70000.0, np.zeros((1, 2)), 1)
    with pytest.raises(ActivationOverflow):
        writer.close()
    assert store.done[0] == 1, "the good row stays"
    assert store.done[1] == 0 and store.done[2] == 0


def test_tokenizing_on_several_processes_gives_the_same_answers_in_the_same_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    tokenizer = make_d1_tokenizer()
    texts = [f"Tell me about topic {index}." for index in range(60)]

    def function(text: str, shared: Any) -> list[int]:
        return encode_ids(shared, text)

    assert map_tokenized(function, texts, tokenizer, processes=3, min_rows=0) == map_tokenized(function, texts, tokenizer, processes=1)


def test_a_tokenizer_process_that_dies_fails_the_task_and_does_not_hang() -> None:
    def function(item: int, shared: Any) -> int:
        if item == 5:
            os._exit(1)
        return item

    with pytest.raises(BrokenProcessPool):
        map_tokenized(function, list(range(60)), None, processes=2, min_rows=0)


def test_the_bootstrap_does_not_depend_on_threads() -> None:
    rng = np.random.default_rng(0)
    scores, labels = rng.normal(size=300), rng.integers(0, 2, 300)
    weights = rng.multinomial(300, np.full(300, 1 / 300), size=900).astype(np.float32)
    together = weighted_auroc(scores, labels, weights, chunk=900)
    assert np.allclose(together, weighted_auroc(scores, labels, weights, chunk=50), rtol=0.0, atol=1e-12)
    assert parallel_map(lambda value: value * 2, range(50)) == [value * 2 for value in range(50)]


# Real training on the tiny model


def tiny_builder(directory: Path, condition: str = "DD") -> tuple[Any, Any, BatchBuilder]:
    cfg = D1Config.from_file(write_tiny_config(directory))
    tokenizer = make_d1_tokenizer()
    transcripts, _, ordinary, _ = ingest(tiny_rows(8), cfg.partition, None)
    base = [record for record in transcripts if record.family in cfg.partition["adapter_train"]]
    answers = resolve_answers(tokenizer, followup_text(tokenizer, base[0].messages, None, cfg.wordings_train[0]))
    builder = BatchBuilder(tokenizer, cfg, condition, 0, base, ordinary, [record for record in base if record.y == 0], answers)
    return make_d1_model(len(tokenizer)).requires_grad_(False), tokenizer, builder


def trained(model: Any, tokenizer: Any, batches: Any, micro_batch: int, **options: Any) -> LoraAdapter:
    adapter = LoraAdapter.for_model(model, 2, 4.0, 0)
    train_adapter(model, adapter, batches, 4, micro_batch, 0.01, 1, tokenizer.eos_token_id, torch.device("cpu"), check_imports=False, **options)
    adapter.requires_grad_(False)
    return adapter


def assert_same_adapter(first: LoraAdapter, second: LoraAdapter) -> None:
    for a, b in zip(first.up, second.up):
        assert torch.allclose(a, b, atol=1e-4)
    for a, b in zip(first.down, second.down):
        assert torch.allclose(a, b, atol=1e-4)


def test_an_adapter_trains_to_the_same_weights_with_and_without_checkpointing(tmp_path: Path) -> None:
    model, tokenizer, builder = tiny_builder(tmp_path)
    assert_same_adapter(trained(model, tokenizer, builder.batch, 3, gradient_checkpointing=True), trained(model, tokenizer, builder.batch, 3, gradient_checkpointing=False))


def test_an_adapter_trains_to_the_same_weights_at_another_micro_batch(tmp_path: Path) -> None:
    """Up to rounding. So an out-of-memory step down moves it no further than that."""
    model, tokenizer, builder = tiny_builder(tmp_path)
    assert_same_adapter(trained(model, tokenizer, builder.batch, 3), trained(model, tokenizer, builder.batch, 6))


def test_prefetching_batches_changes_nothing_including_a_jump_to_another_step(tmp_path: Path) -> None:
    model, tokenizer, builder = tiny_builder(tmp_path)
    # Computed first, because in a real run only the prefetching thread touches the tokenizer.
    steps = (0, 1, 2, 7, 8, 3)
    direct = [builder.batch(step) for step in steps]
    plain = trained(model, tokenizer, builder.batch, 3)
    prefetched = PrefetchedBatches(builder.batch, depth=3)
    try:
        assert [prefetched(step) for step in steps] == direct
        assert_same_adapter(trained(model, tokenizer, prefetched, 3), plain)
    finally:
        prefetched.close()


def test_running_out_of_memory_can_end_a_training_process_rather_than_change_its_micro_batch(tmp_path: Path) -> None:
    model, tokenizer, builder = tiny_builder(tmp_path)
    calls = {"n": 0}
    body = model.model
    real_forward = body.forward

    def fails_first(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("CUDA out of memory, as a test")
        return real_forward(*args, **kwargs)

    body.forward = fails_first
    try:
        with pytest.raises(RuntimeError, match="out of memory"):
            trained(model, tokenizer, builder.batch, 3, halve_on_oom=False)
        calls["n"] = 0
        # With halving allowed, as the dry run has, and as the pool allows after stepping down, the same failure is survived.
        trained(model, tokenizer, builder.batch, 3, halve_on_oom=True)
        assert calls["n"] > 1
    finally:
        del body.forward


def test_the_worker_tells_out_of_memory_from_a_crash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mechinterp.d1 import worker

    def out_of_memory(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")

    def crash(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("something else")

    argv = ["capture", "--config", str(write_tiny_config(tmp_path)), "--out", str(tmp_path / "out"), "--rung", TAG]
    monkeypatch.setattr(worker, "task_capture", out_of_memory)
    assert worker.main(argv) == worker.EXIT_OUT_OF_MEMORY
    monkeypatch.setattr(worker, "task_capture", crash)
    assert worker.main(argv) == 1


def test_the_run_script_passes_the_worker_override() -> None:
    script = (TESTS_DIR.parent / "scripts" / "run_d1_gpu.sh").read_text(encoding="utf-8")
    assert "D1_WORKERS_PER_GPU" in script and "--workers-per-gpu" in script
