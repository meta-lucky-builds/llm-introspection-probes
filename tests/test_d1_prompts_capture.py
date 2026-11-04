"""Prompt construction, the read token, capture, resume and rounding, on the tiny model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from d1_factory import DEPTHS, make_d1_model, make_d1_tokenizer

from mechinterp.d1.capture import Item, forward_batch, make_batches, run_capture
from mechinterp.d1.config import layers_for
from mechinterp.d1.prompts import (
    NO,
    YES,
    base_read_ids,
    followup_text,
    native_ids,
    prefilled_ids,
    render_conversation,
    resolve_answers,
)
from mechinterp.d1.store import ActivationOverflow, CaptureStore, read_complete
from mechinterp.modeling import get_layers
from mechinterp.prompts import AnswerTokenError, encode_ids

BASE = [
    {"role": "user", "content": "Tell me about topic."},
    {"role": "assistant", "content": "The honest answer for topic."},
]
WORDING = "Was the response you just gave me truthful?"


@pytest.fixture(scope="module")
def tokenizer() -> Any:
    return make_d1_tokenizer()


@pytest.fixture(scope="module")
def model(tokenizer: Any) -> Any:
    return make_d1_model(len(tokenizer)).requires_grad_(False)


def chosen_layers(model: Any) -> dict[int, Any]:
    from fractions import Fraction

    blocks = get_layers(model)
    return {layer: blocks[layer] for layer in layers_for(tuple(Fraction(text) for text in ("0.3", "0.42", "0.55", "0.67", "0.8", "0.9")), DEPTHS)}


def test_scored_prompt_is_the_base_turn_plus_our_wording(tokenizer: Any) -> None:
    rendered = followup_text(tokenizer, BASE, None, WORDING)
    assert rendered == "user: Tell me about topic.\nassistant: The honest answer for topic.\nuser: " + WORDING + "\nassistant:\n"


def test_followup_needs_a_base_assistant_turn(tokenizer: Any) -> None:
    with pytest.raises(ValueError):
        followup_text(tokenizer, BASE[:1], None, WORDING)


def test_prefills_share_all_but_the_answer_token_and_end_on_the_full_stop(tokenizer: Any) -> None:
    rendered = followup_text(tokenizer, BASE, None, WORDING)
    answers = resolve_answers(tokenizer, rendered)
    yes_ids, yes_position = prefilled_ids(tokenizer, rendered, YES, answers)
    no_ids, no_position = prefilled_ids(tokenizer, rendered, NO, answers)
    assert yes_ids[:-2] == no_ids[:-2] == encode_ids(tokenizer, rendered)
    assert yes_ids[-2:] == [answers.yes, answers.dot]
    assert no_ids[-2:] == [answers.no, answers.dot]
    assert yes_position == no_position == len(yes_ids) - 3
    assert answers.yes != answers.no


def test_answer_tokens_are_strict_in_context(tokenizer: Any) -> None:
    # After a word instead of the generation prompt's newline, the answer would take a leading space and another token.
    with pytest.raises(AnswerTokenError):
        resolve_answers(tokenizer, "assistant")


def test_base_read_ends_on_the_base_turn_with_no_followup(tokenizer: Any) -> None:
    ids = base_read_ids(tokenizer, BASE, None)
    assert ids == encode_ids(tokenizer, "user: Tell me about topic.\nassistant:\nThe honest answer for topic.")
    assert tokenizer.decode(ids[-1:]) == "."


def test_native_row_keeps_fibs_followup_and_answer(tokenizer: Any) -> None:
    messages = [*BASE, {"role": "user", "content": "OK, go on."}]
    ids = native_ids(tokenizer, messages, None, "Here is more.")
    assert ids == encode_ids(tokenizer, render_conversation(tokenizer, messages) + "Here is more.")


def test_thinking_left_open_is_an_error() -> None:
    tokenizer = make_d1_tokenizer()
    tokenizer.chat_template = "{% for m in messages %}{{ m.content }}{% endfor %}<think>\n"
    with pytest.raises(ValueError):
        render_conversation(tokenizer, BASE)


def test_read_token_activations_ignore_what_follows(tokenizer: Any, model: Any) -> None:
    rendered = followup_text(tokenizer, BASE, None, WORDING)
    answers = resolve_answers(tokenizer, rendered)
    ids, position = prefilled_ids(tokenizer, rendered, YES, answers)
    chosen = chosen_layers(model)
    eot = tokenizer.eos_token_id
    plain, _ = forward_batch(model, chosen, [Item(ids, len(ids) - 1, position)], answers.yes, answers.no, eot, torch.device("cpu"))
    longer, _ = forward_batch(model, chosen, [Item([*ids, eot], len(ids) - 1, position)], answers.yes, answers.no, eot, torch.device("cpu"))
    assert np.abs(plain - longer).max() < 1e-4


def test_layer_k_is_hidden_state_k_plus_one(tokenizer: Any, model: Any) -> None:
    rendered = followup_text(tokenizer, BASE, None, WORDING)
    answers = resolve_answers(tokenizer, rendered)
    ids, position = prefilled_ids(tokenizer, rendered, YES, answers)
    chosen = chosen_layers(model)
    acts, _ = forward_batch(model, chosen, [Item(ids, len(ids) - 1, position)], answers.yes, answers.no, tokenizer.eos_token_id, torch.device("cpu"))
    with torch.inference_mode():
        hidden = model(input_ids=torch.tensor([ids]), output_hidden_states=True, use_cache=False).hidden_states
    assert len(hidden) == DEPTHS + 1
    for column, layer in enumerate(chosen):
        if layer < DEPTHS - 1:  # The last entry can be after the final norm.
            assert torch.allclose(hidden[layer + 1][0, -1], torch.from_numpy(acts[0, column]), atol=1e-4), layer


def test_padding_moves_neither_reads_nor_answer_logits(tokenizer: Any, model: Any) -> None:
    rendered = followup_text(tokenizer, BASE, None, WORDING)
    answers = resolve_answers(tokenizer, rendered)
    ids, position = prefilled_ids(tokenizer, rendered, YES, answers)
    longer_base = [BASE[0], {"role": "assistant", "content": "The honest answer for topic. The honest answer for topic. The honest answer."}]
    long_ids, long_position = prefilled_ids(tokenizer, followup_text(tokenizer, longer_base, None, WORDING), YES, answers)
    chosen = chosen_layers(model)
    device = torch.device("cpu")
    pad = tokenizer.eos_token_id
    alone_acts, alone_logs = forward_batch(model, chosen, [Item(ids, len(ids) - 1, position)], answers.yes, answers.no, pad, device)
    batch = [Item(ids, len(ids) - 1, position), Item(long_ids, len(long_ids) - 1, long_position)]
    batch_acts, batch_logs = forward_batch(model, chosen, batch, answers.yes, answers.no, pad, device)
    assert np.abs(batch_acts[0] - alone_acts[0]).max() < 1e-3
    assert np.abs(batch_logs[0] - alone_logs[0]).max() < 1e-3


def test_self_report_is_read_before_the_answer_token(tokenizer: Any, model: Any) -> None:
    rendered = followup_text(tokenizer, BASE, None, WORDING)
    answers = resolve_answers(tokenizer, rendered)
    yes_ids, position = prefilled_ids(tokenizer, rendered, YES, answers)
    no_ids, _ = prefilled_ids(tokenizer, rendered, NO, answers)
    chosen = chosen_layers(model)
    device = torch.device("cpu")
    pad = tokenizer.eos_token_id
    _, from_yes = forward_batch(model, chosen, [Item(yes_ids, len(yes_ids) - 1, position)], answers.yes, answers.no, pad, device)
    _, from_no = forward_batch(model, chosen, [Item(no_ids, len(no_ids) - 1, position)], answers.yes, answers.no, pad, device)
    assert np.abs(from_yes - from_no).max() < 1e-4, "both arms share a prompt, so the answer position's logits must agree"
    prompt_ids = yes_ids[: position + 1]
    with torch.inference_mode():
        logits = model(input_ids=torch.tensor([prompt_ids]), use_cache=False).logits[0, -1]
    direct = torch.log_softmax(logits.float(), dim=-1)[[answers.yes, answers.no]].numpy()
    assert np.abs(from_yes[0] - direct).max() < 1e-3


def test_batches_respect_the_token_budget() -> None:
    lengths = {0: 10, 1: 9, 2: 3, 3: 3, 4: 100}
    batches = make_batches(lengths, 20)
    assert sorted(row for batch in batches for row in batch) == [0, 1, 2, 3, 4]
    assert batches[0] == [4], "a long row alone may exceed the budget, and goes first"
    for batch in batches[1:]:
        assert max(lengths[row] for row in batch) * len(batch) <= 20


def write_items(path: Any, items: list[Item]) -> None:
    import json

    path.write_text("".join(json.dumps({"ids": item.ids, "read": item.read, "answer_pos": item.answer_pos}) + "\n" for item in items), encoding="utf-8")


def small_items(tokenizer: Any) -> tuple[list[Item], Any]:
    rendered = followup_text(tokenizer, BASE, None, WORDING)
    answers = resolve_answers(tokenizer, rendered)
    yes, position = prefilled_ids(tokenizer, rendered, YES, answers)
    no, _ = prefilled_ids(tokenizer, rendered, NO, answers)
    base = base_read_ids(tokenizer, BASE, None)
    return [Item(yes, len(yes) - 1, position), Item(no, len(no) - 1, position), Item(base, len(base) - 1, -1)] * 3, answers


def test_capture_is_float16_resumable_and_shardable(tokenizer: Any, model: Any, tmp_path: Any) -> None:
    items, answers = small_items(tokenizer)
    items_path = tmp_path / "items.jsonl"
    write_items(items_path, items)
    layers = list(chosen_layers(model))
    device = torch.device("cpu")
    pad = tokenizer.eos_token_id

    def capture(directory: Any, shard: int = 0, n_shards: int = 1) -> dict:
        store = CaptureStore.create(directory, len(items), len(layers), 32)
        return run_capture(model, layers, items_path, store, answers.yes, answers.no, pad, device, 200, shard, n_shards)

    capture(tmp_path / "whole")
    whole = read_complete(tmp_path / "whole")
    assert whole.acts.dtype == np.float16
    assert np.isfinite(whole.acts).all()
    assert np.isnan(whole.logprobs[2]).all(), "a base read has no answer position"
    assert np.isfinite(whole.logprobs[0]).all()

    capture(tmp_path / "shards", 0, 2)
    with pytest.raises(ValueError):
        read_complete(tmp_path / "shards")
    capture(tmp_path / "shards", 1, 2)
    sharded = read_complete(tmp_path / "shards")
    assert np.abs(np.asarray(sharded.acts, dtype=np.float32) - np.asarray(whole.acts, dtype=np.float32)).max() < 1e-2

    # Resume: a row marked undone is recomputed, and nothing else is.
    store = CaptureStore.open(tmp_path / "whole")
    store.done[4] = 0
    store.acts[4] = 0
    store.done.flush()
    store.acts.flush()
    info = run_capture(model, layers, items_path, store, answers.yes, answers.no, pad, device, 200)
    assert info["rows"] == 1
    assert read_complete(tmp_path / "whole").acts[4].any()
    assert run_capture(model, layers, items_path, store, answers.yes, answers.no, pad, device, 200)["rows"] == 0


def test_store_refuses_another_shape(tmp_path: Any) -> None:
    CaptureStore.create(tmp_path / "s", 4, 2, 8)
    with pytest.raises(ValueError):
        CaptureStore.create(tmp_path / "s", 5, 2, 8)


def test_overflow_stops_rather_than_rounding_to_inf(tmp_path: Any) -> None:
    store = CaptureStore.create(tmp_path / "s", 1, 1, 2)
    with pytest.raises(ActivationOverflow):
        store.write([0], np.array([[[70000.0, 0.0]]], dtype=np.float32), np.zeros((1, 2)))
    with pytest.raises(ActivationOverflow):
        store.write([0], np.array([[[np.nan, 0.0]]], dtype=np.float32), np.zeros((1, 2)))


def test_float16_rounding_is_reported_and_small_at_activation_scale(tmp_path: Any) -> None:
    store = CaptureStore.create(tmp_path / "s", 1, 1, 64)
    acts = np.random.default_rng(0).normal(0, 20.0, (1, 1, 64)).astype(np.float32)
    assert store.write([0], acts, np.zeros((1, 2))) < 1e-3
