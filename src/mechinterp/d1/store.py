"""Activations on disk: float16, memory-mapped, one row per item, resumable. numpy only.

float16 holds about three significant digits and overflows at 65504, so a write that would overflow stops the run, and
each store records the largest value and the largest relative rounding error it took in. Scores are always computed on
what was stored, so a probe is fitted and read through the same rounding.

A row counts as done only after its data has been flushed, so a crash loses at most some unflushed rows, which the next
run recomputes. Several processes write to one store, each to its own rows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ..checkpoint import atomic_write_bytes

FLOAT16_MAX = 65504.0


class ActivationOverflow(ValueError):
    """An activation does not fit in float16. Rounding it would pass inf to every probe."""


@dataclass
class CaptureStore:
    directory: Path
    n: int
    n_layers: int
    d_model: int
    acts: np.memmap
    logprobs: np.memmap
    done: np.memmap

    @classmethod
    def create(cls, directory: Path, n: int, n_layers: int, d_model: int) -> "CaptureStore":
        """Creates the files, or opens them if they exist with the same shape. Call before any worker starts."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        shape_file = directory / "store.json"
        wanted = {"n": n, "n_layers": n_layers, "d_model": d_model, "dtype": "float16"}
        if shape_file.exists():
            found = json.loads(shape_file.read_text(encoding="utf-8"))
            if found != wanted:
                raise ValueError(f"{directory} holds {found}, but this run needs {wanted}")
            return cls.open(directory)
        np.memmap(directory / "acts.f16", dtype=np.float16, mode="w+", shape=(n, n_layers, d_model)).flush()
        logprobs = np.memmap(directory / "logprobs.f8", dtype=np.float64, mode="w+", shape=(n, 2))
        logprobs[:] = np.nan
        logprobs.flush()
        np.memmap(directory / "done.u1", dtype=np.uint8, mode="w+", shape=(n,)).flush()
        atomic_write_bytes(shape_file, json.dumps(wanted).encode("utf-8"))
        return cls.open(directory)

    @classmethod
    def open(cls, directory: Path, mode: str = "r+") -> "CaptureStore":
        directory = Path(directory)
        info = json.loads((directory / "store.json").read_text(encoding="utf-8"))
        n, n_layers, d_model = info["n"], info["n_layers"], info["d_model"]
        return cls(
            directory, n, n_layers, d_model,
            np.memmap(directory / "acts.f16", dtype=np.float16, mode=mode, shape=(n, n_layers, d_model)),
            np.memmap(directory / "logprobs.f8", dtype=np.float64, mode=mode, shape=(n, 2)),
            np.memmap(directory / "done.u1", dtype=np.uint8, mode=mode, shape=(n,)),
        )  # fmt: skip

    def write(self, rows: Sequence[int], acts: np.ndarray, logprobs: np.ndarray) -> float:
        """Acts [rows, layers, d] as float32. Returns the largest relative rounding error, on row norms."""
        if acts.shape != (len(rows), self.n_layers, self.d_model):
            raise ValueError(f"Expected activations of shape {(len(rows), self.n_layers, self.d_model)}, got {acts.shape}")
        if not np.isfinite(acts).all():
            raise ActivationOverflow(f"Non-finite activations for rows {list(rows)[:5]}")
        largest = float(np.abs(acts).max())
        if largest >= FLOAT16_MAX:
            raise ActivationOverflow(f"An activation of {largest} does not fit in float16")
        stored = acts.astype(np.float16)
        norms = np.linalg.norm(acts.astype(np.float64), axis=-1)
        errors = np.linalg.norm(acts.astype(np.float64) - stored.astype(np.float64), axis=-1)
        relative = float(np.max(errors / np.maximum(norms, 1e-12)))
        self.acts[list(rows)] = stored
        self.logprobs[list(rows)] = logprobs
        return relative

    def mark_done(self, rows: Sequence[int]) -> None:
        self.acts.flush()
        self.logprobs.flush()
        self.done[list(rows)] = 1
        self.done.flush()

    def pending(self, shard: int = 0, n_shards: int = 1) -> list[int]:
        return [index for index in np.flatnonzero(self.done == 0).tolist() if index % n_shards == shard]

    def pending_of(self, rows: Sequence[int]) -> list[int]:
        """Which of these rows are not done yet, in the order given."""
        return [row for row in rows if self.done[row] == 0]

    @property
    def complete(self) -> bool:
        return bool(np.all(self.done == 1))


def assign_shards(lengths: Sequence[int], n_shards: int) -> list[list[int]]:
    """Each shard's rows: longest first, neighbours by length, with about equal tokens. Every row is in exactly one shard.

    A pure function of the set and the shard count, and the count comes from the set's tokens and never from the number of
    GPUs. So a row's batch does not depend on how many workers there are or in what order they take the shards. Neighbours by
    length also waste almost nothing on padding.
    """
    if n_shards < 1:
        raise ValueError("n_shards must be at least 1")
    order = sorted(range(len(lengths)), key=lambda row: (-lengths[row], row))
    total = max(1, sum(lengths))
    shards: list[list[int]] = [[] for _ in range(n_shards)]
    spent = 0
    for row in order:
        shards[min(n_shards - 1, spent * n_shards // total)].append(row)
        spent += lengths[row]
    return shards


def read_complete(directory: Path) -> CaptureStore:
    store = CaptureStore.open(directory, mode="r")
    if not store.complete:
        raise ValueError(f"{directory} has {int(np.sum(store.done == 0))} rows still to capture")
    return store
