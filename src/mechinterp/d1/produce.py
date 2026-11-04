"""The two tasks that make the expensive things: activations and adapters. run_identity hashes this file.

It is apart from worker.py so that a fix to the worker's dispatch or to probe fitting cannot invalidate stored work, while an
edit here, which can change what is stored, does. Imports stay inside each task: the training task must not have probe,
capture or analysis code in its process.
"""

from __future__ import annotations

import importlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from ..checkpoint import atomic_write_bytes
from .config import CONDITIONS, D1Config, Rung, layers_for
from .data import read_native, read_transcripts
from .paths import BASE_MODEL, D1Paths, adapter_id


class FatalTaskError(RuntimeError):
    """A retry would do the same thing. A data, token or preflight problem."""


class Beat:
    """Progress for the orchestrator, written atomically. Its mtime is also the stall clock."""

    def __init__(self, paths: D1Paths, name: str) -> None:
        self.path = paths.heartbeat(name)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.base: dict[str, Any] = {}

    def __call__(self, done: int, total: int, rate: float, **extra: Any) -> None:
        payload = {"done": done, "total": total, "rate": rate, "time": time.time(), **self.base, **extra}
        atomic_write_bytes(self.path, json.dumps(payload).encode("utf-8"))


def load_real(rung: Rung, device: Any) -> tuple[Any, Any]:
    """Weights go straight from the file to the GPU, so RAM never holds a whole model.

    transformers 5.22 ignores low_cpu_mem_usage, so asking for it does nothing. A device_map is what loads directly, and it
    needs accelerate, which the run script installs and checks. float16 weights would save nothing over bfloat16, which is
    the same size and is already what 8B and 14B use, and would risk overflow, so the dtypes stay as they are.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(rung.name, revision=rung.revision)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[rung.dtype]
    # A worker sees one GPU, so it is always cuda:0 here. One device means no dispatch hooks are attached.
    placement = {"device_map": {"": "cuda:0"}} if device.type == "cuda" else {}
    try:
        model = AutoModelForCausalLM.from_pretrained(rung.name, revision=rung.revision, dtype=dtype, **placement)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(rung.name, revision=rung.revision, torch_dtype=dtype, **placement)
    loaded = getattr(model.config, "_commit_hash", None)
    if loaded and loaded != rung.revision:
        raise FatalTaskError(f"Asked for {rung.name}@{rung.revision} but loaded {loaded}")
    model.to(device).eval().requires_grad_(False)
    return model, tokenizer


def resolve(factory: str) -> Callable[..., Any]:
    module, name = factory.split(":")
    return getattr(importlib.import_module(module), name)


def load_tokenizer(rung: Rung, factory: str | None) -> Any:
    if factory:
        return resolve(factory)(rung, tokenizer_only=True)
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(rung.name, revision=rung.revision)


def load_model(rung: Rung, factory: str | None, require_cuda: bool) -> tuple[Any, Any, Any]:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if require_cuda and device.type != "cuda":
        raise RuntimeError("This worker was given a GPU but cannot see it")
    if factory:
        model, tokenizer = resolve(factory)(rung)
        return model.to(device).eval().requires_grad_(False), tokenizer, device
    model, tokenizer = load_real(rung, device)
    return model, tokenizer, device


def answer_tokens(paths: D1Paths, tag: str) -> Any:
    from .prompts import AnswerTokens

    plan = json.loads((paths.plan_dir(tag) / "plan.json").read_text(encoding="utf-8"))
    return AnswerTokens(**plan["answer_tokens"])


def builder_for_condition(cfg: D1Config, paths: D1Paths, tag: str, condition: str, seed: int, tokenizer: Any, answers: Any) -> Any:
    """The training rows of this rung for one condition. The preflight uses this too, so it tests the same batches."""
    from .training_data import BatchBuilder

    plan = json.loads(paths.training_rows(tag).read_text(encoding="utf-8"))
    transcripts = {record.key: record for record in read_transcripts(paths.training_transcripts(tag))}
    ordinary_rows = {row.row: row for row in read_native(paths.training_ordinary(tag))}
    return BatchBuilder(
        tokenizer, cfg, condition, seed,
        [transcripts[key] for key in plan["base_keys"]],
        [ordinary_rows[key] for key in plan["ordinary_rows"]],
        [transcripts[key] for key in plan["reply_keys"]],
        answers,
    )  # fmt: skip


class LoadedModel:
    """Keeps one rung's model for a worker that serves many capture shards, so that fine shards cost no reloads.

    A model is only ever read through. An adapter is a set of hooks that leave nothing behind, which task_capture checks after
    every job, so a read of the unmodified model after an adapter's is the same as one in a new process.
    """

    def __init__(self) -> None:
        self.tag: str | None = None
        self.value: tuple[Any, Any, Any] | None = None

    def get(self, rung: Rung, factory: str | None, require_cuda: bool) -> tuple[Any, Any, Any]:
        if self.tag != rung.tag or self.value is None:
            self.value = None
            self.value = load_model(rung, factory, require_cuda)
            self.tag = rung.tag
        return self.value


def task_capture(
    cfg: D1Config, paths: D1Paths, tag: str, model_id: str, sets: Sequence[str], shard: int, n_shards: int, task_name: str, factory: str | None, require_cuda: bool,
    token_budget: int | None = None, loaded: LoadedModel | None = None,
) -> None:  # fmt: skip
    from .capture import run_capture
    from .lora import LoraAdapter, named_block_linears
    from .plan import read_meta
    from .store import CaptureStore, assign_shards

    rung = cfg.rung(tag)
    model, tokenizer, device = loaded.get(rung, factory, require_cuda) if loaded is not None else load_model(rung, factory, require_cuda)
    answers = answer_tokens(paths, tag)
    layers = layers_for(cfg.fractions, rung.n_layers)
    pad = tokenizer.eos_token_id
    beat = Beat(paths, task_name)
    stores = {name: CaptureStore.open(paths.capture_dir(tag, model_id, name)) for name in sets}
    rows_of = {name: assign_shards(read_meta(paths, tag, name)["lengths"], n_shards)[shard] for name in sets}
    grand_total = sum(len(stores[name].pending_of(rows_of[name])) for name in sets)
    done_before = 0
    beat(0, grand_total, 0.0, loaded=True)
    adapter = None
    if model_id != BASE_MODEL:
        adapter, _ = LoraAdapter.load(paths.adapter(tag, model_id) / "adapter.pt", device)

    def capture_one(name: str) -> None:
        nonlocal done_before

        def report(done: int, total: int, rate: float) -> None:
            beat(done_before + done, grand_total, rate, loaded=True, current=name)

        info = run_capture(model, layers, paths.set_items(tag, name), stores[name], answers.yes, answers.no, pad, device, token_budget or cfg.capture_token_budget, heartbeat=report, rows=rows_of[name])
        done_before += int(info["rows"])
        print(f"{model_id}/{name} shard {shard}: {info}", flush=True)

    if adapter is None:
        for name in sets:
            capture_one(name)
    else:
        with adapter.applied(model):
            for name in sets:
                capture_one(name)
    left_behind = [name for name, module in named_block_linears(model).items() if len(module._forward_hooks) > 0]
    if left_behind:
        raise FatalTaskError(f"An adapter's hooks stayed on {left_behind[:3]}, so the next read on this model would not be what it says")


def task_train(
    cfg: D1Config, paths: D1Paths, tag: str, condition: str, seed: int, attempt: int, task_name: str, factory: str | None, require_cuda: bool, in_process: bool,
    gradient_checkpointing: bool = True, halve_on_oom: bool = False,
) -> None:  # fmt: skip
    from .lora import LoraAdapter, train_adapter
    from .prompts import followup_text
    from .training_data import PrefetchedBatches, check_answer_tokens

    if condition not in CONDITIONS:
        raise FatalTaskError(f"Unknown condition {condition}")
    model_id = adapter_id(attempt, condition, seed)
    directory = paths.adapter(tag, model_id)
    directory.mkdir(parents=True, exist_ok=True)
    rung = cfg.rung(tag)
    model, tokenizer, device = load_model(rung, factory, require_cuda)
    answers = answer_tokens(paths, tag)
    builder = builder_for_condition(cfg, paths, tag, condition, seed, tokenizer, answers)
    sample = builder.negatives[0]
    check_answer_tokens(tokenizer, followup_text(tokenizer, sample.messages, sample.tools, cfg.wordings_train[0]), answers)
    training = cfg.training
    steps = training.steps * (2**attempt)
    adapter = LoraAdapter.for_model(model, training.rank, training.alpha, seed)
    beat = Beat(paths, task_name)
    beat(0, steps, 0.0, loaded=True)

    def report(done: int, total: int, rate: float) -> None:
        beat(done, total, rate, loaded=True)

    batches = PrefetchedBatches(builder.batch)
    try:
        info = train_adapter(
            model, adapter, batches, steps, training.micro_batch, training.learning_rate, training.warmup_steps,
            tokenizer.eos_token_id, device, directory / "resume.pt", heartbeat=report, check_imports=not in_process,
            gradient_checkpointing=gradient_checkpointing, halve_on_oom=halve_on_oom,
        )  # fmt: skip
    finally:
        batches.close()
    metadata = {"condition": condition, "seed": seed, "attempt": attempt, "rung": tag, **info}
    adapter.save(directory / "adapter.pt", {key: value for key, value in metadata.items() if key in ("condition", "seed", "attempt", "rung", "steps")})
    atomic_write_bytes(directory / "training.json", json.dumps(metadata, indent=1).encode("utf-8"))
    Path(directory / "resume.pt").unlink(missing_ok=True)
    print(f"Trained {model_id}: {info}", flush=True)
