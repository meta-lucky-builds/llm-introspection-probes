"""Forward passes that fill a CaptureStore: activations at the read token and Yes/No log probabilities.

Layer k is the output of decoder block k, as in the FIBS contract, equal to hidden_states[k + 1]. LoRA acts inside the
blocks' linear layers, so block outputs seen here are always through the adapter, if one is attached.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import torch
from torch import nn

from ..hooks import split_layer_output
from ..modeling import get_layers
from ..scoring import assert_finite
from .store import CaptureStore

FLUSH_SECONDS = 30.0
WRITE_QUEUE = 4  # Batches waiting for the disk. Bounded, so a slow disk slows capture and cannot fill RAM.


@dataclass(frozen=True)
class Item:
    ids: list[int]
    read: int
    answer_pos: int  # Index whose logits predict the answer, or -1 if this item has no answer position.


def read_items(path: Path, rows: Sequence[int] | None = None) -> dict[int, Item]:
    wanted = None if rows is None else set(rows)
    items: dict[int, Item] = {}
    with open(path, encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if wanted is None or index in wanted:
                record = json.loads(line)
                items[index] = Item(record["ids"], record["read"], record["answer_pos"])
    return items


def make_batches(lengths: dict[int, int], token_budget: int) -> list[list[int]]:
    """Longest first, so a run that will not fit fails early. A batch never exceeds the budget, except a lone long row."""
    batches: list[list[int]] = []
    current: list[int] = []
    for index in sorted(lengths, key=lambda row: (-lengths[row], row)):
        width = lengths[current[0]] if current else lengths[index]
        if current and width * (len(current) + 1) > token_budget:
            batches.append(current)
            current = []
        current.append(index)
    if current:
        batches.append(current)
    return batches


@contextmanager
def gather_at(layers: dict[int, nn.Module], positions: dict[str, torch.Tensor]) -> Iterator[dict[int, torch.Tensor]]:
    """Each chosen layer's output at one position per row: {layer: [batch, d]}, float32, on the CPU."""
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer: int) -> Any:
        def hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
            hidden, _ = split_layer_output(output)
            rows = torch.arange(hidden.size(0), device=hidden.device)
            captured[layer] = hidden[rows, positions["read"].to(hidden.device)].detach().float().cpu()

        return hook

    for layer, module in layers.items():
        handles.append(module.register_forward_hook(make_hook(layer)))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def body_and_head(model: nn.Module) -> tuple[nn.Module, nn.Module]:
    head = model.get_output_embeddings()
    body = getattr(model, "model", None) or model.get_decoder()
    return body, head


def forward_batch(
    model: nn.Module, chosen: dict[int, nn.Module], items: Sequence[Item], yes: int, no: int, pad: int, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Right-padded with an attention mask. Returns (activations [batch, layers, d], log probs [batch, 2], NaN where unused)."""
    width = max(len(item.ids) for item in items)
    # Built on the CPU and sent once, rather than one small transfer per row. The same integers either way.
    ids_array = np.full((len(items), width), pad, dtype=np.int64)
    mask_array = np.zeros((len(items), width), dtype=np.int64)
    for row, item in enumerate(items):
        ids_array[row, : len(item.ids)] = item.ids
        mask_array[row, : len(item.ids)] = 1
    input_ids = torch.from_numpy(ids_array).to(device)
    attention = torch.from_numpy(mask_array).to(device)
    positions = {"read": torch.tensor([item.read for item in items])}
    body, head = body_and_head(model)
    with torch.inference_mode(), gather_at(chosen, positions) as captured:
        hidden = body(input_ids=input_ids, attention_mask=attention, use_cache=False).last_hidden_state
        logprobs = np.full((len(items), 2), np.nan)
        answered = [row for row, item in enumerate(items) if item.answer_pos >= 0]
        if answered:
            rows = torch.tensor(answered, device=device)
            spots = torch.tensor([items[row].answer_pos for row in answered], device=device)
            # Through the head in float32. On a bfloat16 rung, logits rounded to steps of 0.125 would tie self-reports.
            logits = hidden[rows, spots].float() @ head.weight.float().T
            if getattr(head, "bias", None) is not None:
                logits = logits + head.bias.float()
            assert_finite(logits, "the answer position")
            values = torch.log_softmax(logits, dim=-1)[:, [yes, no]].double().cpu().numpy()
            logprobs[answered] = values
    stacked = torch.stack([captured[layer] for layer in chosen], dim=1).numpy()
    return stacked, logprobs


def is_out_of_memory(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or (isinstance(error, RuntimeError) and "out of memory" in str(error).lower())


class RowWriter:
    """Converts, writes and flushes rows on its own thread, so the GPU is not left waiting on the disk.

    Rows are addressed by index, so the order of writes changes nothing. A row counts as done only after its data is flushed,
    as before. An error here, such as an overflow, stops the capture at its next batch, and close() always flushes what was
    written, so finished rows survive.
    """

    def __init__(self, store: CaptureStore, total: int, heartbeat: Callable[[int, int, float], None] | None) -> None:
        self.store = store
        self.total = total
        self.heartbeat = heartbeat
        self.waiting: queue.Queue = queue.Queue(maxsize=WRITE_QUEUE)
        self.error: BaseException | None = None
        self.finished = 0
        self.tokens = 0
        self.worst = 0.0
        self.started = time.monotonic()
        self.thread = threading.Thread(target=self._loop, name="d1-row-writer", daemon=True)
        self.thread.start()

    def put(self, rows: list[int], acts: np.ndarray, logprobs: np.ndarray, tokens: int) -> None:
        if self.error is not None:
            raise self.error
        self.waiting.put((rows, acts, logprobs, tokens))

    def _flush(self, rows: list[int]) -> None:
        self.store.mark_done(rows)
        if self.heartbeat is not None:
            self.heartbeat(self.finished, self.total, self.tokens / max(1e-6, time.monotonic() - self.started))

    def _loop(self) -> None:
        unflushed: list[int] = []
        last_flush = time.monotonic()
        while True:
            entry = self.waiting.get()
            if entry is None:
                break
            if self.error is not None:
                continue  # Keeps draining, so that put() can never block on a dead writer.
            rows, acts, logprobs, tokens = entry
            try:
                self.worst = max(self.worst, self.store.write(rows, acts, logprobs))
                unflushed.extend(rows)
                self.finished += len(rows)
                self.tokens += tokens
                if time.monotonic() - last_flush >= FLUSH_SECONDS:
                    self._flush(unflushed)
                    unflushed = []
                    last_flush = time.monotonic()
            except BaseException as error:  # Handed to the main thread, which is the one that can stop the task.
                self.error = error
        if unflushed and self.error is None:
            try:
                self._flush(unflushed)
            except BaseException as error:
                self.error = error
        elif unflushed:
            # The rows that were written before the error are good, and keeping them saves redoing them.
            try:
                self.store.mark_done(unflushed)
            except BaseException:
                pass

    def close(self) -> None:
        self.waiting.put(None)
        self.thread.join()
        if self.error is not None:
            raise self.error


def run_capture(
    model: nn.Module,
    layer_indices: Sequence[int],
    items_path: Path,
    store: CaptureStore,
    yes: int,
    no: int,
    pad: int,
    device: torch.device,
    token_budget: int,
    shard: int = 0,
    n_shards: int = 1,
    heartbeat: Callable[[int, int, float], None] | None = None,
    rows: Sequence[int] | None = None,
) -> dict[str, float]:
    """Capture every pending row of this shard, or of `rows` if given. Safe to call again: finished rows are skipped."""
    pending = store.pending(shard, n_shards) if rows is None else store.pending_of(rows)
    if not pending:
        return {"rows": 0, "tokens": 0, "max_relative_rounding": 0.0}
    items = read_items(items_path, pending)
    lengths = {row: len(items[row].ids) for row in pending}
    blocks = get_layers(model)
    chosen = {layer: blocks[layer] for layer in layer_indices}
    budget = token_budget
    batches = make_batches(lengths, budget)
    writer = RowWriter(store, len(pending), heartbeat)
    try:
        while batches:
            batch = batches.pop(0)
            try:
                acts, logprobs = forward_batch(model, chosen, [items[row] for row in batch], yes, no, pad, device)
            except Exception as error:
                if not is_out_of_memory(error):
                    raise
                if len(batch) == 1:
                    # Worded so the worker reports this as out of memory: the pool makes room on the card and tries again.
                    raise RuntimeError(f"Out of memory: a single row of {lengths[batch[0]]} tokens does not fit on this GPU right now") from error
                torch.cuda.empty_cache()
                budget = max(1, budget // 2)
                print(f"Out of GPU memory, so lowering the token budget to {budget}", flush=True)
                remaining = {row: lengths[row] for chunk in [batch, *batches] for row in chunk}
                batches = make_batches(remaining, budget)
                continue
            writer.put(batch, acts, logprobs, sum(lengths[row] for row in batch))
    except BaseException:
        try:
            writer.close()
        except BaseException as error:
            print(f"The row writer also failed: {error}", flush=True)
        raise
    writer.close()
    return {"rows": writer.finished, "tokens": writer.tokens, "max_relative_rounding": writer.worst}
