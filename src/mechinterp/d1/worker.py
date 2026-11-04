"""One unit of work on one GPU, run as its own process. Exit code 3 means a restart cannot help.

Imports are inside each task: the training task must not have probe, capture or analysis code in its process.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Sequence

from ..checkpoint import atomic_write_bytes
from .config import CONDITIONS, D1Config, Rung, layers_for
from .data import ingest, iter_parquet_rows, read_native, read_transcripts, write_jsonl
from .families import QUADRANTS
from .paths import BASE_MODEL, D1Paths, adapter_id

EXIT_FATAL = 3


class FatalTaskError(RuntimeError):
    """A retry would do the same thing. A data, token or preflight problem."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(rung.name, revision=rung.revision)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[rung.dtype]
    try:
        model = AutoModelForCausalLM.from_pretrained(rung.name, revision=rung.revision, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(rung.name, revision=rung.revision, torch_dtype=dtype)
    loaded = getattr(model.config, "_commit_hash", None)
    if loaded and loaded != rung.revision:
        raise FatalTaskError(f"Asked for {rung.name}@{rung.revision} but loaded {loaded}")
    model.to(device).eval().requires_grad_(False)
    return model, tokenizer


def load_tokenizer(rung: Rung, factory: str | None) -> Any:
    if factory:
        return resolve(factory)(rung, tokenizer_only=True)
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(rung.name, revision=rung.revision)


def resolve(factory: str) -> Callable[..., Any]:
    module, name = factory.split(":")
    return getattr(importlib.import_module(module), name)


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


def task_prepare(cfg: D1Config, paths: D1Paths, source_dir: str | None) -> None:
    """Download at the pinned revision, check each file's hash, and split into stores. Row text stays in the state directory."""
    files: dict[str, Path] = {}
    for quadrant in QUADRANTS:
        filename = f"data/{quadrant}-00000-of-00001.parquet"
        if source_dir:
            files[quadrant] = Path(source_dir) / filename
        else:
            from huggingface_hub import hf_hub_download

            files[quadrant] = Path(hf_hub_download(cfg.dataset_repo, filename, repo_type="dataset", revision=cfg.dataset_revision))
        found = sha256_file(files[quadrant])
        if found != cfg.parquet_sha256[quadrant]:
            raise FatalTaskError(f"{filename} has sha256 {found}, expected {cfg.parquet_sha256[quadrant]}")
    transcripts, native, ordinary, manifest = ingest(iter_parquet_rows(files), cfg.partition, cfg.expected_counts)
    write_jsonl(paths.transcripts, transcripts)
    write_jsonl(paths.native, native)
    write_jsonl(paths.ordinary, ordinary)
    manifest["parquet_sha256"] = dict(cfg.parquet_sha256)
    manifest["revision"] = cfg.dataset_revision
    manifest["partition"] = cfg.partition
    atomic_write_bytes(paths.data_manifest, json.dumps(manifest, indent=1).encode("utf-8"))
    print(f"Prepared {len(transcripts)} transcripts, {len(native)} native rows and {len(ordinary)} ordinary rows", flush=True)


def task_plan(cfg: D1Config, paths: D1Paths, tag: str, factory: str | None) -> None:
    from .plan import build_rung_plan

    report = build_rung_plan(cfg, paths, tag, load_tokenizer(cfg.rung(tag), factory))
    print(json.dumps({key: report[key] for key in ("fit", "val", "cal", "test_g1", "facts", "test_available_per_class")}), flush=True)


def task_test_set(cfg: D1Config, paths: D1Paths, tag: str, n: int, factory: str | None) -> None:
    from .plan import build_test_set

    print(json.dumps(build_test_set(cfg, paths, tag, load_tokenizer(cfg.rung(tag), factory), n)), flush=True)


def task_capture(
    cfg: D1Config, paths: D1Paths, tag: str, model_id: str, sets: Sequence[str], shard: int, n_shards: int, task_name: str, factory: str | None, require_cuda: bool
) -> None:
    from .capture import run_capture
    from .lora import LoraAdapter
    from .store import CaptureStore

    rung = cfg.rung(tag)
    model, tokenizer, device = load_model(rung, factory, require_cuda)
    answers = answer_tokens(paths, tag)
    layers = layers_for(cfg.fractions, rung.n_layers)
    pad = tokenizer.eos_token_id
    beat = Beat(paths, task_name)
    stores = {name: CaptureStore.open(paths.capture_dir(tag, model_id, name)) for name in sets}
    grand_total = sum(len(store.pending(shard, n_shards)) for store in stores.values())
    done_before = 0
    beat(0, grand_total, 0.0, loaded=True)
    adapter = None
    if model_id != BASE_MODEL:
        adapter, _ = LoraAdapter.load(paths.adapter(tag, model_id) / "adapter.pt", device)

    def capture_one(name: str) -> None:
        nonlocal done_before

        def report(done: int, total: int, rate: float) -> None:
            beat(done_before + done, grand_total, rate, loaded=True, current=name)

        info = run_capture(model, layers, paths.set_items(tag, name), stores[name], answers.yes, answers.no, pad, device, cfg.capture_token_budget, shard, n_shards, report)
        done_before += int(info["rows"])
        print(f"{model_id}/{name} shard {shard}: {info}", flush=True)

    if adapter is None:
        for name in sets:
            capture_one(name)
    else:
        with adapter.applied(model):
            for name in sets:
                capture_one(name)


def task_fit_probes(cfg: D1Config, paths: D1Paths, tag: str, name: str, source: str, kind: str, shuffle: bool, require_cuda: bool) -> None:
    import numpy as np
    import torch

    from .plan import read_meta
    from .probes import fit_ensemble
    from .store import read_complete

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if require_cuda and device != "cuda":
        raise RuntimeError("This worker was given a GPU but cannot see it")
    rung = cfg.rung(tag)
    fit_store = read_complete(paths.capture_dir(tag, source, "fit"))
    val_store = read_complete(paths.capture_dir(tag, source, "val"))
    train_labels = np.asarray(read_meta(paths, tag, "fit")["y"], dtype=np.float32)
    val_labels = np.asarray(read_meta(paths, tag, "val")["y"], dtype=np.float32)
    if shuffle:
        rng = np.random.default_rng(cfg.analysis.seed)
        train_labels = rng.permutation(train_labels)
        val_labels = rng.permutation(val_labels)
    probe = fit_ensemble(fit_store.acts, train_labels, val_store.acts, val_labels, layers_for(cfg.fractions, rung.n_layers), kind, cfg.probes, device)
    if shuffle:
        # A label-shuffled probe reads noise, so the layer rule would drop every layer. Keep them for the baseline.
        probe.retained = np.ones_like(probe.retained)
    paths.probe(tag, name).parent.mkdir(parents=True, exist_ok=True)
    probe.save(paths.probe(tag, name))
    print(f"Probe {name}: retained {probe.retained.tolist()}, losses {probe.train_loss.round(3).tolist()}, {probe.info}", flush=True)


def task_train(cfg: D1Config, paths: D1Paths, tag: str, condition: str, seed: int, attempt: int, task_name: str, factory: str | None, require_cuda: bool, in_process: bool) -> None:
    from .lora import LoraAdapter, train_adapter
    from .training_data import BatchBuilder, check_answer_tokens

    if condition not in CONDITIONS:
        raise FatalTaskError(f"Unknown condition {condition}")
    model_id = adapter_id(attempt, condition, seed)
    directory = paths.adapter(tag, model_id)
    directory.mkdir(parents=True, exist_ok=True)
    rung = cfg.rung(tag)
    plan = json.loads(paths.training_rows(tag).read_text(encoding="utf-8"))
    transcripts = {record.key: record for record in read_transcripts(paths.transcripts)}
    ordinary_rows = {row.row: row for row in read_native(paths.ordinary)}
    model, tokenizer, device = load_model(rung, factory, require_cuda)
    answers = answer_tokens(paths, tag)
    builder = BatchBuilder(
        tokenizer, cfg, condition, seed,
        [transcripts[key] for key in plan["base_keys"]],
        [ordinary_rows[key] for key in plan["ordinary_rows"]],
        [transcripts[key] for key in plan["reply_keys"]],
        answers,
    )  # fmt: skip
    from .prompts import followup_text

    sample = builder.negatives[0]
    check_answer_tokens(tokenizer, followup_text(tokenizer, sample.messages, sample.tools, cfg.wordings_train[0]), answers)
    training = cfg.training
    steps = training.steps * (2**attempt)
    adapter = LoraAdapter.for_model(model, training.rank, training.alpha, seed)
    beat = Beat(paths, task_name)
    beat(0, steps, 0.0, loaded=True)

    def report(done: int, total: int, rate: float) -> None:
        beat(done, total, rate, loaded=True)

    info = train_adapter(
        model, adapter, builder.batch, steps, training.micro_batch, training.learning_rate, training.warmup_steps,
        tokenizer.eos_token_id, device, directory / "resume.pt", heartbeat=report, check_imports=not in_process,
    )  # fmt: skip
    metadata = {"condition": condition, "seed": seed, "attempt": attempt, "rung": tag, **info}
    adapter.save(directory / "adapter.pt", {key: value for key, value in metadata.items() if key in ("condition", "seed", "attempt", "rung", "steps")})
    atomic_write_bytes(directory / "training.json", json.dumps(metadata, indent=1).encode("utf-8"))
    (directory / "resume.pt").unlink(missing_ok=True)
    print(f"Trained {model_id}: {info}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mechinterp.d1 worker")
    parser.add_argument("task")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rung")
    parser.add_argument("--model-id", default=BASE_MODEL)
    parser.add_argument("--sets", default="")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--task-name", default="task")
    parser.add_argument("--factory")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--name", default="linear")
    parser.add_argument("--source", default=BASE_MODEL)
    parser.add_argument("--kind", default="linear")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--condition")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument("--source-dir")
    parser.add_argument("--in-process", action="store_true")
    parser.add_argument("--n-workers", type=int, default=1)
    args = parser.parse_args(argv)
    cfg = D1Config.from_file(args.config)
    paths = D1Paths(Path(args.out))
    try:
        if args.task == "prepare":
            task_prepare(cfg, paths, args.source_dir)
        elif args.task == "plan":
            task_plan(cfg, paths, args.rung, args.factory)
        elif args.task == "test-set":
            task_test_set(cfg, paths, args.rung, args.n, args.factory)
        elif args.task == "capture":
            sets = [name for name in args.sets.split(",") if name]
            task_capture(cfg, paths, args.rung, args.model_id, sets, args.shard, args.n_shards, args.task_name, args.factory, args.require_cuda)
        elif args.task == "fit-probes":
            task_fit_probes(cfg, paths, args.rung, args.name, args.source, args.kind, args.shuffle, args.require_cuda)
        elif args.task == "train":
            task_train(cfg, paths, args.rung, args.condition, args.seed, args.attempt, args.task_name, args.factory, args.require_cuda, args.in_process)
        elif args.task == "preflight":
            from .preflight import run_preflight

            if not run_preflight(cfg, paths, args.rung, args.factory, args.require_cuda):
                return 2
        else:
            raise FatalTaskError(f"Unknown task {args.task}")
    except FatalTaskError as error:
        print(f"FATAL: {error}", flush=True)
        return EXIT_FATAL
    except Exception as error:
        from ..prompts import AnswerTokenError
        from .families import CountMismatch, UnknownSource
        from .store import ActivationOverflow

        traceback.print_exc()
        fatal = (AnswerTokenError, CountMismatch, UnknownSource, ActivationOverflow, ValueError, AssertionError, SyntaxError, ImportError)
        return EXIT_FATAL if isinstance(error, fatal) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
