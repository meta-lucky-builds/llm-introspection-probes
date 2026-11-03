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
