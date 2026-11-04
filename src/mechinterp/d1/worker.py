"""One unit of work on one GPU, run as its own process. Exit code 3 means a restart cannot help.

Imports are inside each task: the training task must not have probe, capture or analysis code in its process. The tasks
that make activations and adapters live in produce.py, which run_identity hashes. This file only dispatches them, so a fix
here costs no stored work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Sequence

from ..checkpoint import atomic_write_bytes
from .config import D1Config, layers_for
from .data import ingest, iter_parquet_rows, write_jsonl
from .families import QUADRANTS
from .paths import BASE_MODEL, D1Paths
from .produce import FatalTaskError, answer_tokens, load_model, load_tokenizer, task_capture, task_train

EXIT_FATAL = 3
EXIT_OUT_OF_MEMORY = 4  # Not a crash. The pool makes room on the card and tries the task again.
__all__ = ["answer_tokens", "load_model", "main"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    print(json.dumps({key: report[key] for key in ("fit", "val", "cal", "test_g1", "facts", "specificity", "test_available_per_class")}), flush=True)


def task_test_set(cfg: D1Config, paths: D1Paths, tag: str, factory: str | None) -> None:
    from .plan import build_test_set

    print(json.dumps(build_test_set(cfg, paths, tag, load_tokenizer(cfg.rung(tag), factory))), flush=True)


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


def serve_capture(cfg: D1Config, paths: D1Paths, parser: argparse.ArgumentParser) -> None:
    """Capture jobs from stdin, one JSON argument list per line, on one loaded model, until stdin closes.

    A finished job's marker is written here, after its rows are flushed, and only then is the next line read. Any failure
    leaves through main() with an exit code, ending this worker, so the pool sees exactly one unfinished job to put back.
    """
    from .produce import LoadedModel

    loaded = LoadedModel()
    for line in iter(sys.stdin.readline, ""):
        if not line.strip():
            continue
        job = parser.parse_args(json.loads(line))
        if job.task != "capture":
            raise FatalTaskError(f"A capture worker cannot run {job.task}")
        sets = [name for name in job.sets.split(",") if name]
        task_capture(cfg, paths, job.rung, job.model_id, sets, job.shard, job.n_shards, job.task_name, job.factory, job.require_cuda, job.token_budget, loaded)
        marker = paths.done(job.task_name)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()


def exit_with_parent() -> None:
    """A worker whose orchestrator died must not keep a GPU, or write beside the workers of the next launch."""
    parent = os.getppid()

    def watch() -> None:
        while os.getppid() == parent:
            time.sleep(5.0)
        print("The orchestrator is gone, so this worker is stopping. A resume redoes what was not finished.", flush=True)
        os._exit(EXIT_FATAL)

    threading.Thread(target=watch, name="d1-parent-watch", daemon=True).start()


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
    parser.add_argument("--token-budget", type=int, default=None)
    parser.add_argument("--gradient-checkpointing", type=int, default=1)
    parser.add_argument("--halve-on-oom", action="store_true")
    args = parser.parse_args(argv)
    cfg = D1Config.from_file(args.config)
    paths = D1Paths(Path(args.out))
    try:
        if args.task == "serve":
            serve_capture(cfg, paths, parser)
        elif args.task == "prepare":
            task_prepare(cfg, paths, args.source_dir)
        elif args.task == "plan":
            task_plan(cfg, paths, args.rung, args.factory)
        elif args.task == "test-set":
            task_test_set(cfg, paths, args.rung, args.factory)
        elif args.task == "capture":
            sets = [name for name in args.sets.split(",") if name]
            task_capture(cfg, paths, args.rung, args.model_id, sets, args.shard, args.n_shards, args.task_name, args.factory, args.require_cuda, args.token_budget)
        elif args.task == "fit-probes":
            task_fit_probes(cfg, paths, args.rung, args.name, args.source, args.kind, args.shuffle, args.require_cuda)
        elif args.task == "train":
            task_train(cfg, paths, args.rung, args.condition, args.seed, args.attempt, args.task_name, args.factory, args.require_cuda, args.in_process, bool(args.gradient_checkpointing), args.halve_on_oom)
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
        if isinstance(error, RuntimeError) and "out of memory" in str(error).lower():
            return EXIT_OUT_OF_MEMORY
        fatal = (AnswerTokenError, CountMismatch, UnknownSource, ActivationOverflow, ValueError, AssertionError, SyntaxError, ImportError)
        return EXIT_FATAL if isinstance(error, fatal) else 1
    return 0


if __name__ == "__main__":
    exit_with_parent()
    sys.exit(main())
