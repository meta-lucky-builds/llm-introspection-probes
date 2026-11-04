"""Dry-run checks on the real model, before any long stage. Each prints PASS or FAIL, and all are saved.

They target what would silently invalidate results rather than crash: wrong read token, wrong layer, a prompt that ends
wrongly, an adapter that leaks into the base model, or a probe or self-report that is not even finite.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import numpy as np
import torch

from ..modeling import get_layers
from ..prompts import decode_token, encode_ids
from .capture import Item, forward_batch, read_items
from .config import D1Config, layers_for
from .data import read_transcripts
from .families import partition_of
from .lora import LoraAdapter, micro_batch_loss
from .paths import D1Paths
from .plan import read_meta
from .prompts import NO, YES, followup_text, prefilled_ids, render_conversation, resolve_answers
from .probes import fit_ensemble
from .training_data import IGNORE
from .worker import answer_tokens, load_model

CAUSAL_TOLERANCE = 1e-2
PADDING_TOLERANCE = 2e-2
# Two identical forwards on one GPU can differ in the last bits, which is why E1 has a sham tolerance too. The weight hash below is exact.
REPEAT_TOLERANCE = 1e-5


def parameters_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def run_preflight(cfg: D1Config, paths: D1Paths, tag: str, factory: str | None, require_cuda: bool) -> bool:
    rung = cfg.rung(tag)
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any = "") -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})
        print(f"{'PASS' if passed else 'FAIL'}  {name}  {detail}", flush=True)

    model, tokenizer, device = load_model(rung, factory, require_cuda)
    blocks = get_layers(model)
    layers = layers_for(cfg.fractions, rung.n_layers)
    check("Block count matches the config", len(blocks) == rung.n_layers, f"{len(blocks)} blocks, layers read {layers}")
    check("Hidden size matches the config", int(model.get_input_embeddings().embedding_dim) == rung.d_model)

    answers = answer_tokens(paths, tag)
    where = partition_of(cfg.partition)
    samples = [record for record in read_transcripts(paths.transcripts) if where[record.family] == "test"]
    samples = sorted(samples, key=lambda record: record.key)[:8]
    rendered = followup_text(tokenizer, samples[0].messages, samples[0].tools, cfg.wordings_held_out[0])
    resolved = resolve_answers(tokenizer, rendered)
    check("Yes, No and full stop resolve to one token each, and match the plan", resolved == answers, (decode_token(tokenizer, resolved.yes), decode_token(tokenizer, resolved.no), decode_token(tokenizer, resolved.dot)))
    tail = rendered[-30:]  # Template only. This goes into the results, so it must not reach row or wording text.
    check("The chat prompt ends where the answer starts: at whitespace, with no space token to come", rendered[-1:].isspace(), repr(tail))
    if "<think>" in str(tokenizer.chat_template):
        check("The chat prompt ends with an empty, closed thinking block", rendered.endswith("<think>\n\n</think>\n\n"), repr(tail))
    check("The wording sits inside the final user turn", cfg.wordings_held_out[0] in rendered[-(len(cfg.wordings_held_out[0]) + 80) :], "")
    no_prompt = render_conversation(tokenizer, [*samples[0].messages, {"role": "user", "content": cfg.wordings_held_out[0]}], samples[0].tools)
    check("Rendering is deterministic", no_prompt == rendered)

    # The read token, and the fact that nothing after it can change it.
    yes_ids, yes_pos = prefilled_ids(tokenizer, rendered, YES, answers)
    no_ids, no_pos = prefilled_ids(tokenizer, rendered, NO, answers)
    check("Prefills differ only at the answer token, and share the read token", yes_ids[:-2] == no_ids[:-2] and yes_ids[-1] == no_ids[-1] == answers.dot and yes_pos == no_pos, decode_token(tokenizer, yes_ids[-1]))
    chosen = {layer: blocks[layer] for layer in layers}
    pad = tokenizer.eos_token_id
    eot = tokenizer.eos_token_id
    single, _ = forward_batch(model, chosen, [Item(yes_ids, len(yes_ids) - 1, yes_pos)], answers.yes, answers.no, pad, device)
    extended, _ = forward_batch(model, chosen, [Item([*yes_ids, eot], len(yes_ids) - 1, -1)], answers.yes, answers.no, pad, device)
    gap = float(np.abs(single - extended).max() / max(np.abs(single).max(), 1e-9))
    check("Activation at the full stop does not change when end of turn follows", gap < CAUSAL_TOLERANCE, f"relative gap {gap:.2e}")

    # Layer identity against hidden_states.
    ids = torch.tensor([yes_ids], device=device)
    with torch.inference_mode():
        hidden_states = model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states
    worst = 0.0
    for position, layer in enumerate(layers):
        if layer + 1 < len(hidden_states) - 1:  # The last entry is after the final norm on some models.
            worst = max(worst, float((hidden_states[layer + 1][0, -1].float().cpu() - torch.from_numpy(single[0, position])).abs().max()))
    check("Block k's hook equals hidden_states[k + 1] at the read token", worst < 1e-2 * max(float(np.abs(single).max()), 1.0), f"largest difference {worst:.2e}")

    # Batching: padding must not move a read.
    long_rendered = followup_text(tokenizer, samples[1 % len(samples)].messages, samples[1 % len(samples)].tools, cfg.wordings_held_out[1])
    long_ids, long_pos = prefilled_ids(tokenizer, long_rendered, YES, answers)
    items = [Item(yes_ids, len(yes_ids) - 1, yes_pos), Item(long_ids, len(long_ids) - 1, long_pos)]
    batched, logs_batched = forward_batch(model, chosen, items, answers.yes, answers.no, pad, device)
    alone, logs_alone = forward_batch(model, chosen, items[:1], answers.yes, answers.no, pad, device)
    gap = float(np.abs(batched[0] - alone[0]).max() / max(np.abs(alone[0]).max(), 1e-9))
    check("Padding a shorter prompt in a batch does not move its read", gap < PADDING_TOLERANCE, f"relative gap {gap:.2e}")
    check("Padding does not move its answer logits", float(np.abs(logs_batched[0] - logs_alone[0]).max()) < 0.05, f"{float(np.abs(logs_batched[0] - logs_alone[0]).max()):.2e}")

    # float16.
    stored = single.astype(np.float16)
    largest = float(np.abs(single).max())
    relative = float(np.linalg.norm(single - stored.astype(np.float32)) / max(np.linalg.norm(single), 1e-9))
    check("Activations fit float16", np.isfinite(stored).all() and largest < 65504.0 / 4, f"largest {largest:.1f}, relative rounding {relative:.1e}")
    check("float16 rounding is small", relative < 1e-3, f"{relative:.1e}")

    # Self-report and probe on a few rows.
    test_meta = read_meta(paths, tag, "test_g1")
    sample_rows = list(range(0, test_meta["n"], max(1, test_meta["n"] // 24)))[:24]
    sample_items = read_items(paths.set_items(tag, "test_g1"), sample_rows)
    acts, logs = forward_batch(model, chosen, [sample_items[row] for row in sample_rows], answers.yes, answers.no, pad, device)
    answered = logs[~np.isnan(logs[:, 0])]
    check("Self-report is finite on sample rows", len(answered) > 0 and np.isfinite(answered).all(), f"{len(answered)} rows")
    mass = float(np.exp(answered).sum(axis=1).mean()) if len(answered) else 0.0
    # Reported, not a stop: this is a question about the model, and G1 is the prereg's test of it.
    print(f"{'NOTE' if mass > 0.2 else 'WARNING'}  Mean p(Yes) + p(No) at the answer position is {mass:.3f} on sample rows", flush=True)
    fit_meta = read_meta(paths, tag, "fit")
    # The shortest rows, so that a long family cannot fail a check that is only about finite numbers. Fit rows are in hashed order.
    fit_items = read_items(paths.set_items(tag, "fit"))
    rows = sorted(fit_items, key=lambda row: (len(fit_items[row].ids), row))[:96]
    rows = sorted(rows, key=lambda row: hashlib.sha256(str(row).encode("ascii")).hexdigest())
    fit_acts, _ = forward_batch(model, chosen, [fit_items[row] for row in rows], answers.yes, answers.no, pad, device)
    fit_y = np.asarray([fit_meta["y"][row] for row in rows], dtype=np.float32)
    if 0 < fit_y.sum() < len(fit_y) and len(fit_y) >= 32:
        half = len(fit_y) // 2
        from dataclasses import replace

        settings = replace(cfg.probes, max_epochs=1, batch_size=8, warmup_steps=1, log_every=1)
        probe = fit_ensemble(fit_acts[:half], fit_y[:half], fit_acts[half:], fit_y[half:], layers, "linear", settings, str(device), require_balance=False)
        probe.retained = np.ones_like(probe.retained)
        values = probe.score(acts)
        check("A small probe gives finite scores on sample rows", np.isfinite(values).all() and len(values) > 0, f"mean {float(values.mean()):.3f}")
    else:
        check("A small probe gives finite scores on sample rows", False, "the first fit rows lack a class or are too long")

    # The adapter: new is identical, removed is identical, and the base weights never move.
    before = parameters_hash(model)
    reference, _ = forward_batch(model, chosen, items[:1], answers.yes, answers.no, pad, device)
    adapter = LoraAdapter.for_model(model, cfg.training.rank, cfg.training.alpha, 0).to(device)
    with adapter.applied(model):
        fresh, _ = forward_batch(model, chosen, items[:1], answers.yes, answers.no, pad, device)
    scale = max(float(np.abs(reference).max()), 1e-9)
    check("A new adapter reproduces the base model", float(np.abs(fresh - reference).max()) / scale < REPEAT_TOLERANCE, f"{float(np.abs(fresh - reference).max()):.2e}")
    prompt_len = len(encode_ids(tokenizer, rendered))
    labels = [IGNORE] * prompt_len + [answers.yes, answers.dot, eot]
    loss_row = ([*encode_ids(tokenizer, rendered), answers.yes, answers.dot, eot], labels)
    optimizer = torch.optim.SGD([*adapter.down, *adapter.up], lr=0.05)
    started = time.monotonic()
    with adapter.applied(model):
        loss = micro_batch_loss(model, [loss_row], pad, device)
        loss.backward()
        moved = sum(float(tensor.grad.abs().sum()) for tensor in adapter.up if tensor.grad is not None)
        optimizer.step()
        through, _ = forward_batch(model, chosen, items[:1], answers.yes, answers.no, pad, device)
    check("One training step reaches the adapter and changes activations after the block", moved > 0 and float(np.abs(through - reference).max()) > 0.0, f"gradient {moved:.2e} in {time.monotonic() - started:.1f} s")
    after, _ = forward_batch(model, chosen, items[:1], answers.yes, answers.no, pad, device)
    check("Removing the adapter restores the base model", float(np.abs(after - reference).max()) / scale < REPEAT_TOLERANCE, f"{float(np.abs(after - reference).max()):.2e}")
    check("The base weights did not move", parameters_hash(model) == before)
    del adapter, optimizer

    # A measured speed, for the ETA.
    speed_rows = [sample_items[row] for row in sample_rows]
    started = time.monotonic()
    forward_batch(model, chosen, speed_rows, answers.yes, answers.no, pad, device)
    speed = sum(len(item.ids) for item in speed_rows) / max(time.monotonic() - started, 1e-6)
    print(f"Measured about {speed:.0f} tokens per second on this GPU, on a small batch", flush=True)

    passed = all(entry["passed"] for entry in checks)
    from ..checkpoint import atomic_write_bytes

    atomic_write_bytes(paths.preflight(tag), json.dumps({"passed": passed, "checks": checks, "tokens_per_second": speed, "rung": tag}, indent=1, default=str).encode("utf-8"))
    print("Dry run " + ("passed" if passed else "FAILED"), flush=True)
    return passed
