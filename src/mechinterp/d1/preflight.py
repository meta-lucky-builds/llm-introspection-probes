"""Dry-run checks on the real model, before any long stage. Each prints PASS or FAIL, and all are saved.

They target what would silently invalidate results rather than crash: wrong read token, wrong layer, a prompt that ends
wrongly, an adapter that leaks into the base model, or a probe or self-report that is not even finite. They also run the real
training path on the longest real rows, since a failure there would otherwise appear only after G1 and the test capture.

Cheapest first, and a failure among the checks that need no weights stops before the weights load:

    1. Text only: template, answer tokens, wordings, the model's config.         Seconds.
    2. Forward passes: blocks, read token, layers, padding, float16.              A minute after loading.
    3. A small probe, the adapter's mechanics.                                    Seconds.
    4. Base rates G2 relies on: the sham question, specificity, facts.            Seconds. Reported, and never a stop.
    5. Two real training steps on the longest rows, and the largest capture batch. A minute or two.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from typing import Any

import numpy as np
import torch

from ..checkpoint import atomic_write_bytes
from ..modeling import get_layers
from ..prompts import decode_token, encode_ids
from .capture import Item, forward_batch, is_out_of_memory, make_batches, read_items
from .config import CONDITIONS, D1Config, layers_for
from .data import read_transcripts
from .families import partition_of
from .lora import LoraAdapter, micro_batch_loss, train_adapter
from .paths import D1Paths
from .plan import read_meta
from .probes import fit_ensemble
from .produce import answer_tokens, builder_for_condition, load_model, load_tokenizer
from .prompts import NO, YES, followup_text, prefilled_ids, render_conversation, resolve_answers
from .training_data import IGNORE, Slot

CAUSAL_TOLERANCE = 1e-2
PADDING_TOLERANCE = 2e-2
# Two identical forwards on one GPU can differ in the last bits, which is why E1 has a sham tolerance too. The weight hash below is exact.
REPEAT_TOLERANCE = 1e-5
SPECIFICITY_NEEDS_MORE_THAN = 0.60


def parameters_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def run_preflight(cfg: D1Config, paths: D1Paths, tag: str, factory: str | None, require_cuda: bool) -> bool:
    rung = cfg.rung(tag)
    checks: list[dict[str, Any]] = []
    measured: dict[str, Any] = {}

    def check(name: str, passed: bool, detail: Any = "") -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})
        print(f"{'PASS' if passed else 'FAIL'}  {name}  {detail}", flush=True)

    def finish() -> bool:
        passed = all(entry["passed"] for entry in checks)
        atomic_write_bytes(paths.preflight(tag), json.dumps({"passed": passed, "checks": checks, "rung": tag, **measured}, indent=1, default=str).encode("utf-8"))
        print("Dry run " + ("passed" if passed else "FAILED"), flush=True)
        return passed

    # 1. Text only, with no weights in memory.
    tokenizer = load_tokenizer(rung, factory)
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
    yes_ids, yes_pos = prefilled_ids(tokenizer, rendered, YES, answers)
    no_ids, no_pos = prefilled_ids(tokenizer, rendered, NO, answers)
    check("Prefills differ only at the answer token, and share the read token", yes_ids[:-2] == no_ids[:-2] and yes_ids[-1] == no_ids[-1] == answers.dot and yes_pos == no_pos, decode_token(tokenizer, yes_ids[-1]))
    if not factory:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(rung.name, revision=rung.revision)
        check("The model's config has the blocks and width the ladder says", (config.num_hidden_layers, config.hidden_size) == (rung.n_layers, rung.d_model), f"{config.num_hidden_layers} blocks of width {config.hidden_size}")
    if not all(entry["passed"] for entry in checks):
        print("Stopping before the weights load, since these need none of them.", flush=True)
        return finish()

    # 2. Forward passes.
    model, tokenizer, device = load_model(rung, factory, require_cuda)
    blocks = get_layers(model)
    layers = layers_for(cfg.fractions, rung.n_layers)
    check("Block count matches the config", len(blocks) == rung.n_layers, f"{len(blocks)} blocks, layers read {layers}")
    check("Hidden size matches the config", int(model.get_input_embeddings().embedding_dim) == rung.d_model)

    # The read token, and the fact that nothing after it can change it.
    chosen = {layer: blocks[layer] for layer in layers}
    pad = tokenizer.eos_token_id
    eot = tokenizer.eos_token_id
    single, _ = forward_batch(model, chosen, [Item(yes_ids, len(yes_ids) - 1, yes_pos)], answers.yes, answers.no, pad, device)
    extended, _ = forward_batch(model, chosen, [Item([*yes_ids, eot], len(yes_ids) - 1, -1)], answers.yes, answers.no, pad, device)
    gap = float(np.abs(single - extended).max() / max(np.abs(single).max(), 1e-9))
    check("Activation at the full stop does not change when end of turn follows", gap < CAUSAL_TOLERANCE, f"relative gap {gap:.2e}")

    # Layer identity against hidden_states.
    ids = torch.tensor([yes_ids], device=device)
    # Transformers 5 leaves recording hooks on the blocks after output_hidden_states=True. Remove them, or the training
    # step below rightly refuses to run: no hook may sit on a block during adapter training.
    hooks_before = {name: (set(module._forward_hooks), set(module._forward_pre_hooks)) for name, module in model.named_modules()}
    with torch.inference_mode():
        hidden_states = model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states
    for name, module in model.named_modules():
        forward_ids, pre_ids = hooks_before[name]
        for key in set(module._forward_hooks) - forward_ids:
            del module._forward_hooks[key]
        for key in set(module._forward_pre_hooks) - pre_ids:
            del module._forward_pre_hooks[key]
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

    # 3. Self-report and probe on a few rows.
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

    # 4. Base rates G2 relies on. A question about the model, so reported and not a stop, but seen before any adapter is paid for.
    def accuracy_on(name: str) -> float:
        meta = read_meta(paths, tag, name)
        set_items = read_items(paths.set_items(tag, name))
        lengths = {row: len(item.ids) for row, item in set_items.items()}
        says_yes = np.zeros(meta["n"], dtype=bool)
        for batch in make_batches(lengths, cfg.capture_token_budget):
            _, batch_logs = forward_batch(model, chosen, [set_items[row] for row in batch], answers.yes, answers.no, pad, device)
            says_yes[batch] = batch_logs[:, 0] > batch_logs[:, 1]
        return float(np.mean(says_yes == (np.asarray(meta["y"]) == 1)))

    sham_items = []
    for record in samples:
        sham_ids = encode_ids(tokenizer, followup_text(tokenizer, record.messages, record.tools, cfg.wording_sham))
        sham_items.append(Item(sham_ids, len(sham_ids) - 1, len(sham_ids) - 1))
    _, sham_logs = forward_batch(model, chosen, sham_items, answers.yes, answers.no, pad, device)
    sham_yes = float(np.exp(sham_logs[:, 0]).mean())
    measured["unmodified_p_yes_on_sham_question"] = sham_yes
    measured["unmodified_specificity_accuracy"] = accuracy_on("specificity")
    measured["unmodified_fact_accuracy"] = accuracy_on("facts")
    print(f"{'WARNING' if sham_yes >= 0.5 else 'NOTE'}  p(Yes) on the sham question is {sham_yes:.3f} on sample rows. It should naturally be No, so that SHAM has something to change.", flush=True)
    has_power = measured["unmodified_specificity_accuracy"] > SPECIFICITY_NEEDS_MORE_THAN
    print(
        f"{'NOTE' if has_power else 'WARNING'}  The unmodified model scores {measured['unmodified_specificity_accuracy']:.3f} on the specificity set and {measured['unmodified_fact_accuracy']:.3f} on the facts. "
        + ("" if has_power else "A model that says Yes to everything scores 0.50 on specificity, so at this level G2 cannot see a general Yes bias."),
        flush=True,
    )

    # 5. The real training path on the longest real rows, then the largest capture batch.
    training = cfg.training
    builder = builder_for_condition(cfg, paths, tag, "DD", 0, tokenizer, answers)
    candidates = [*builder.negatives, *builder.positives]
    longest = sorted(range(len(candidates)), key=lambda index: (-sum(len(str(message.get("content") or "")) for message in candidates[index].messages), index))[: training.batch_size]
    wording = max(range(len(cfg.wordings_train)), key=lambda index: len(cfg.wordings_train[index]))
    for condition in CONDITIONS:
        builder.condition = condition
        builder.batch(0)  # A real mixed batch of each condition has to build, for this tokenizer and these rows.
    builder.condition = "DD"
    train_rows = [builder.row(Slot("manipulation", index, wording)) for index in longest]
    # One step first, because the first step of a process pays for its allocations, and the comparison below should not.
    train_adapter(model, LoraAdapter.for_model(model, training.rank, training.alpha, 0), lambda _step: train_rows, 1, training.micro_batch, training.learning_rate, training.warmup_steps, pad, device, check_imports=False)
    throwaway = LoraAdapter.for_model(model, training.rank, training.alpha, 0)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    info = train_adapter(model, throwaway, lambda _step: train_rows, 2, training.micro_batch, training.learning_rate, training.warmup_steps, pad, device, check_imports=False)
    seconds_per_step = (time.monotonic() - started) / 2.0
    if device.type == "cuda":
        measured["train_peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
        print(f"NOTE  Training peaked at {measured['train_peak_gpu_gb']:.1f} GB of GPU memory on the longest rows", flush=True)
    disable = getattr(model, "gradient_checkpointing_disable", None)
    if disable is not None and getattr(model, "supports_gradient_checkpointing", False):
        disable()
    model.eval()
    longest_tokens = max(len(row[0]) for row in train_rows)
    check("Two real training steps run on the longest rows and move the adapter", bool(np.isfinite(info["final_loss"])) and info["delta_norm"] > 0.0, f"{seconds_per_step:.1f} s per step, longest row {longest_tokens} tokens, loss {info['final_loss']:.3f}")
    measured["train_seconds_per_step_on_longest_rows"] = seconds_per_step
    # The same steps without gradient checkpointing, which saves a forward pass if the memory is there. Decides how training runs, never what it computes.
    plain = LoraAdapter.for_model(model, training.rank, training.alpha, 0)
    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        train_adapter(model, plain, lambda _step: train_rows, 2, training.micro_batch, training.learning_rate, training.warmup_steps, pad, device, check_imports=False, gradient_checkpointing=False, halve_on_oom=False)
        measured["train_seconds_per_step_no_checkpointing"] = (time.monotonic() - started) / 2.0
        if device.type == "cuda":
            measured["train_peak_gpu_gb_no_checkpointing"] = torch.cuda.max_memory_allocated() / 1e9
        print(f"NOTE  Without gradient checkpointing: {measured['train_seconds_per_step_no_checkpointing']:.1f} s per step, peak {measured.get('train_peak_gpu_gb_no_checkpointing', 0.0):.1f} GB", flush=True)
    except Exception as error:
        if not is_out_of_memory(error):
            raise
        print("NOTE  Training without gradient checkpointing does not fit on this card on the longest rows, so it will stay on.", flush=True)
    del plain
    model.eval()
    model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    throwaway.requires_grad_(False)
    with throwaway.applied(model):
        direct, _ = forward_batch(model, chosen, items[:1], answers.yes, answers.no, pad, device)
    round_trip = paths.rung_dir(tag) / "preflight_adapter.pt"
    throwaway.save(round_trip, {"condition": "preflight"})
    loaded, _ = LoraAdapter.load(round_trip, device)
    with loaded.applied(model):
        reloaded, _ = forward_batch(model, chosen, items[:1], answers.yes, answers.no, pad, device)
    round_trip.unlink(missing_ok=True)
    check("A saved adapter gives the same reads when loaded", float(np.abs(direct - reloaded).max()) / scale < REPEAT_TOLERANCE, f"{float(np.abs(direct - reloaded).max()):.2e}")
    del throwaway, loaded
    check("The base weights did not move in training", parameters_hash(model) == before)

    lengths = {row: len(item.ids) for row, item in fit_items.items()}
    first_batch = make_batches(lengths, cfg.capture_token_budget)[0]
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            measured["weights_gpu_gb"] = torch.cuda.memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        largest_acts, _ = forward_batch(model, chosen, [fit_items[row] for row in first_batch], answers.yes, answers.no, pad, device)
        capture_speed = sum(lengths[row] for row in first_batch) / max(time.monotonic() - started, 1e-6)
        if device.type == "cuda":
            measured["capture_peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
        check("Activations on the longest fit rows fit float16", float(np.abs(largest_acts).max()) < 65504.0 / 4, f"largest {float(np.abs(largest_acts).max()):.1f}")
        measured["tokens_per_second_on_longest_rows"] = capture_speed
    except Exception as error:
        if not is_out_of_memory(error):
            raise
        # Capture lowers its token budget and carries on, so this costs time and nothing else.
        print(f"WARNING  A full capture batch of the longest rows did not fit at budget {cfg.capture_token_budget}. Capture will lower its budget.", flush=True)
        torch.cuda.empty_cache()

    # A measured speed, for the ETA.
    speed_rows = [sample_items[row] for row in sample_rows]
    started = time.monotonic()
    forward_batch(model, chosen, speed_rows, answers.yes, answers.no, pad, device)
    speed = sum(len(item.ids) for item in speed_rows) / max(time.monotonic() - started, 1e-6)
    measured["tokens_per_second"] = speed
    print(f"Measured about {speed:.0f} tokens per second on this GPU, on a small batch", flush=True)
    adapter_hours = training.steps * seconds_per_step / 3600.0
    print(f"NOTE  At this rung one adapter takes at most about {adapter_hours:.1f} h at {training.steps} steps, from the longest rows. A real batch is shorter, and a whole run trains {len(CONDITIONS) * len(training.seeds)} of them.", flush=True)
    return finish()
