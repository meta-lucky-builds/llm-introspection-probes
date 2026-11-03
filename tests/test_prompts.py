from __future__ import annotations

import pytest
from conftest import SPLIT_SPACE_PATTERN, letter_words, make_tokenizer

import mechinterp.prompts as prompts
from mechinterp.prompts import (
    ANSWER_CUE,
    LABELS,
    TASKS,
    AnswerTokenError,
    TokenLocationError,
    build_prompt,
    concept_prompt_text,
    decode_token,
    instruction_for,
    is_single_token_word,
    locate_user_content_end,
    render_chat,
    resolve_answer_token,
    resolve_label_ids,
    resolve_none_id,
    user_text,
)

WORDS = letter_words("calo", 10)
QUERY_TASK = "T_fact"


def _query(task: str) -> int | None:
    return 2 if task == QUERY_TASK else None


def test_list_and_cue_are_byte_identical_across_tasks() -> None:
    tails = set()
    for task in TASKS:
        text = user_text(task, WORDS, _query(task))
        instruction = instruction_for(task, WORDS, _query(task))
        assert text.startswith(instruction)
        tails.add(text[len(instruction) :])
    assert len(tails) == 1
    assert next(iter(tails)).endswith(ANSWER_CUE)


def test_controls_do_not_mention_anything_unusual() -> None:
    forbidden = ("inject", "vector", "activation", "unusual", "anomal", "steer", "concept", "perturb", "tamper", "modif")
    for task in ("T_pick", "T_fav", "T_odd", "T_fact"):
        text = user_text(task, WORDS, _query(task)).lower()
        for word in forbidden:
            assert word not in text, (task, word)


def test_only_t_inj_none_offers_none() -> None:
    for task in TASKS:
        assert ("none" in instruction_for(task, WORDS, _query(task))) == (task == "T_inj_none")


@pytest.mark.parametrize("task", TASKS)
def test_positions_decode_to_the_list_words(tokenizer, task) -> None:
    prompt = build_prompt(tokenizer, task, "calone", WORDS, _query(task))
    assert len(prompt.positions) == len(WORDS)
    for word, position in zip(WORDS, prompt.positions):
        assert decode_token(tokenizer, prompt.input_ids[position]).strip() == word


def test_fact_query_finds_the_list_occurrence_not_the_instruction(tokenizer) -> None:
    prompt = build_prompt(tokenizer, QUERY_TASK, "calone", WORDS, 2)
    token = prompt.input_ids[prompt.positions[2]]
    first = prompt.input_ids.index(token)
    assert first < prompt.positions[2]
    # The first occurrence, in the instruction, comes before the list starts.
    assert first < prompt.positions[0]


def test_single_token_filter(tokenizer) -> None:
    assert is_single_token_word(tokenizer, WORDS[0])
    assert not is_single_token_word(tokenizer, "two-part")
    assert not is_single_token_word(tokenizer, "missingword")


def test_trimmed_and_untrimmed_offsets_both_locate_the_word(tokenizer, monkeypatch) -> None:
    prompt = build_prompt(tokenizer, "T_pick", "calone", WORDS)

    def trimmed(tokenizer_, text):
        ids, offsets = original(tokenizer_, text)
        return ids, [(a + 1, b) if text[a:b].startswith(" ") and b - a > 1 else (a, b) for a, b in offsets]

    original = prompts.encode_with_offsets
    monkeypatch.setattr(prompts, "encode_with_offsets", trimmed)
    assert build_prompt(tokenizer, "T_pick", "calone", WORDS).positions == prompt.positions


def test_label_ids_after_newline_have_no_space(tokenizer) -> None:
    prompt = build_prompt(tokenizer, "T_inj_none", "calone", WORDS)
    assert prompt.text.endswith("\n")
    ids = resolve_label_ids(tokenizer, prompt.text)
    assert [decode_token(tokenizer, token) for token in ids] == list(LABELS)
    assert len(set(ids)) == len(LABELS)


def test_answer_prefix_is_identical_on_scored_tasks_and_absent_when_none_is_allowed(tokenizer) -> None:
    for task in ("T_inj", "T_pick", "T_fav", "T_odd"):
        assert build_prompt(tokenizer, task, "calone", WORDS).text.endswith(prompts.ANSWER_PREFIX)
    assert build_prompt(tokenizer, "T_fact", "calone", WORDS, query=0).text.endswith(prompts.ANSWER_PREFIX)
    assert not build_prompt(tokenizer, "T_inj_none", "calone", WORDS).text.endswith(prompts.ANSWER_PREFIX)
    labels = resolve_label_ids(tokenizer, build_prompt(tokenizer, "T_inj", "calone", WORDS).text)
    assert labels == resolve_label_ids(tokenizer, build_prompt(tokenizer, "T_inj_none", "calone", WORDS).text)


def test_label_after_non_space_uses_the_spaced_token() -> None:
    tokenizer = make_tokenizer(["Answer:", " 3"])
    token = resolve_answer_token(tokenizer, "Answer:", "3")
    assert decode_token(tokenizer, token) == " 3"


def test_label_that_splits_from_its_space_raises() -> None:
    tokenizer = make_tokenizer(["Answer:", "0 1 2 3 4 5 6 7 8 9"], SPLIT_SPACE_PATTERN)
    with pytest.raises(AnswerTokenError):
        resolve_answer_token(tokenizer, "Answer:", "3")
    assert decode_token(tokenizer, resolve_answer_token(tokenizer, "Answer:\n", "3")) == "3"


def test_duplicate_label_ids_raise(tokenizer, monkeypatch) -> None:
    monkeypatch.setattr(prompts, "resolve_answer_token", lambda *_: 5)
    with pytest.raises(AnswerTokenError):
        resolve_label_ids(tokenizer, "x\n")


def test_none_token(tokenizer) -> None:
    labels = resolve_label_ids(tokenizer, "Answer:\n")
    none_id = resolve_none_id(tokenizer, "Answer:\n", labels)
    assert decode_token(tokenizer, none_id) == "none"
    without_none = make_tokenizer(["Answer:\n", "0 1 2 3 4 5 6 7 8 9"])
    assert resolve_none_id(without_none, "Answer:\n", labels) is None


class _FakeChatTokenizer:
    def __init__(self, template: str, strict: bool, rendered: str) -> None:
        self.chat_template = template
        self.strict = strict
        self.rendered = rendered
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        self.calls.append(kwargs)
        if self.strict and kwargs:
            raise TypeError("unexpected keyword")
        return self.rendered


def test_chat_template_without_enable_thinking_falls_back() -> None:
    fake = _FakeChatTokenizer("plain", strict=True, rendered="u\na\n")
    assert render_chat(fake, "hi") == "u\na\n"
    assert fake.calls == [{"enable_thinking": False}, {}]


def test_ignored_enable_thinking_is_an_error() -> None:
    with pytest.raises(ValueError):
        render_chat(_FakeChatTokenizer("has <think> in it", strict=False, rendered="u\na\n"), "hi")
    with pytest.raises(ValueError):
        render_chat(_FakeChatTokenizer("t", strict=False, rendered="a\n<think>\n"), "hi")


def test_read_site_is_the_period_ending_the_user_message(tokenizer) -> None:
    text = concept_prompt_text("concaa")
    rendered = f"user\n{text}\nassistant\n<think>\n\n</think>\n\n"
    input_ids, index = locate_user_content_end(tokenizer, rendered, text)
    assert decode_token(tokenizer, input_ids[index]) == "."
    assert decode_token(tokenizer, input_ids[index - 1]).strip() == "concaa"
    assert index < len(input_ids) - 1


def test_read_site_refuses_a_token_that_is_not_the_period() -> None:
    glued = make_tokenizer(["Tell me about concaa."], r" ?[A-Za-z.]+|\n|[^A-Za-z.\n ]| ")
    text = concept_prompt_text("concaa")
    with pytest.raises(TokenLocationError):
        locate_user_content_end(glued, f"{text}\n", text)


def test_read_site_refuses_a_message_that_is_missing_or_repeated(tokenizer) -> None:
    text = concept_prompt_text("concaa")
    with pytest.raises(TokenLocationError):
        locate_user_content_end(tokenizer, "user\n", text)
    with pytest.raises(TokenLocationError):
        locate_user_content_end(tokenizer, f"{text}\n{text}", text)


def test_real_jinja_template_with_thinking_disabled() -> None:
    # jinja2 is installed explicitly in CI, so this test cannot skip silently.
    qwen3_like = make_tokenizer(["<x>", "</x>", "0 1 2 3"])
    qwen3_like.chat_template = (
        "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
        "{% if add_generation_prompt %}assistant\n"
        "{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{% endif %}"
        "{% endif %}"
    )
    rendered = render_chat(qwen3_like, "hi")
    assert rendered.endswith("assistant\n<think>\n\n</think>\n\n")
