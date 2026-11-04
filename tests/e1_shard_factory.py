"""Builds the tiny model inside worker subprocesses, and injects faults. Not collected by pytest.

E1_TEST_FAULT is "<mode>:<shard>:<appends>". After that many appends the shard's worker writes half a line
and dies abruptly, as a killed process would. Mode "once" does so a single time, using the marker file
E1_TEST_MARKER to remember, and mode "always" does so on every attempt.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from conftest import SPLIT_SPACE_PATTERN, base_texts, make_tokenizer, tiny_qwen

from mechinterp.checkpoint import ResultStore


def _install_fault(shard: int) -> None:
    fault = os.environ.get("E1_TEST_FAULT")
    if not fault:
        return
    mode, target, appends = fault.split(":")
    if int(target) != shard:
        return
    marker = Path(os.environ["E1_TEST_MARKER"])
    if mode == "once" and marker.exists():
        return
    original = ResultStore.append
    state = {"appends": 0}

    def dying_append(self, records):
        state["appends"] += 1
        if state["appends"] > int(appends):
            marker.touch()
            with open(self.path, "ab") as handle:
                handle.write(b'{"key":"torn","kind":"tri')
            os._exit(1)
        return original(self, records)

    ResultStore.append = dying_append


def build(config, shard):
    _install_fault(shard)
    tokenizer = make_tokenizer(base_texts(), SPLIT_SPACE_PATTERN)
    return tiny_qwen(len(tokenizer)), tokenizer, torch.device("cpu")
