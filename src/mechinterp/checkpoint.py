"""Append-only JSONL results store with fsync, plus a blob that survives a lost runtime."""

from __future__ import annotations

import base64
import gzip
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator


class ResultStore:
    """Every record has a unique "key" and a "kind". Later runs skip keys already present."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        raw = self.path.read_bytes()
        good_end = 0
        cursor = 0
        while cursor < len(raw):
            newline = raw.find(b"\n", cursor)
            if newline == -1:
                break  # Last line never finished: the process died during the write.
            try:
                record = json.loads(raw[cursor:newline].decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                break
            self._records[record["key"]] = record
            cursor = newline + 1
            good_end = cursor
        if good_end != len(raw):
            # Remove the torn tail so the next append does not glue onto it.
            with open(self.path, "r+b") as handle:
                handle.truncate(good_end)
                handle.flush()
                os.fsync(handle.fileno())

    def has(self, key: str) -> bool:
        return key in self._records

    def get(self, key: str) -> dict[str, Any] | None:
        return self._records.get(key)

    def __len__(self) -> int:
        return len(self._records)

    def records(self, kind: str | None = None) -> Iterator[dict[str, Any]]:
        for record in self._records.values():
            if kind is None or record.get("kind") == kind:
                yield record

    def append(self, records: Iterable[dict[str, Any]]) -> None:
        """Durably append one batch of records. A key already present is an error."""
        batch = list(records)
        if not batch:
            return
        keys = [record["key"] for record in batch]
        duplicates = [key for key in keys if key in self._records]
        if duplicates or len(set(keys)) != len(keys):
            raise ValueError(f"Duplicate result keys: {duplicates[:3]}")
        payload = "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in batch)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        for record in batch:
            self._records[record["key"]] = record


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """Every complete record in file order, for reading a store another process may be writing.

    Unlike ResultStore this never truncates a torn tail, which may be a live writer's unfinished line,
    and it keeps repeated keys so that a merge can see them.
    """
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    records = []
    cursor = 0
    while cursor < len(raw):
        newline = raw.find(b"\n", cursor)
        if newline == -1:
            break
        try:
            records.append(json.loads(raw[cursor:newline].decode("utf-8")))
        except (json.JSONDecodeError, UnicodeDecodeError):
            break
        cursor = newline + 1
    return records


class ShardStore(ResultStore):
    """A worker's own store, which can also see finished rows from other shard files.

    Foreign rows are read-only and count as done, so nothing is computed twice when the worker count
    changes between launches, and the impact search can read the whole grid. Each worker still runs its
    own clean prompts and writes its own manifest and sham record, so those kinds are never foreign.
    """

    FOREIGN_KINDS = ("trial", "impact_match", "vector_diagnostics")

    def __init__(self, path: str | Path, foreign_paths: Iterable[str | Path] = ()) -> None:
        super().__init__(path)
        self._foreign: dict[str, dict[str, Any]] = {}
        self.refresh_foreign(foreign_paths)

    def refresh_foreign(self, paths: Iterable[str | Path]) -> None:
        foreign: dict[str, dict[str, Any]] = {}
        for path in paths:
            for record in read_records(path):
                if record.get("kind") in self.FOREIGN_KINDS:
                    foreign[record["key"]] = record
        self._foreign = foreign

    def has(self, key: str) -> bool:
        return super().has(key) or key in self._foreign

    def get(self, key: str) -> dict[str, Any] | None:
        own = super().get(key)
        return own if own is not None else self._foreign.get(key)

    def append(self, records: Iterable[dict[str, Any]]) -> None:
        batch = list(records)
        duplicates = [record["key"] for record in batch if record["key"] in self._foreign]
        if duplicates:
            raise ValueError(f"Keys already stored by another shard: {duplicates[:3]}")
        super().append(batch)


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    """Readers see the old file or the new one, never a partial one, even after a crash."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with open(temporary, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    if os.name == "posix":
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def encode_records(records: Iterable[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records).encode("utf-8")


def checkpoint_blob(path: str | Path) -> str:
    """The whole store as gzip+base64, for printing to a Colab cell before a runtime can be lost."""
    return base64.b64encode(gzip.compress(Path(path).read_bytes(), mtime=0)).decode("ascii")


def restore_checkpoint(blob: str, path: str | Path) -> None:
    """Write a printed blob back into an empty runtime so the run can resume."""
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        raise FileExistsError(f"Refusing to overwrite existing results at {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = gzip.decompress(base64.b64decode("".join(blob.split())))
    with open(path, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
