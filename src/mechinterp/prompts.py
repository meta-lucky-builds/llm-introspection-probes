"""Task prompts, chat rendering, candidate token location and answer token resolution.

Prompt layout: instruction, list, cue. The instruction comes first so that the list tokens
can in principle be processed differently per task. Placing it after the list would make
list activations identical across tasks and remove one route to task-gating, which would
bias the experiment toward finding capture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

LIST_SIZE = 10
LABELS = tuple(str(i) for i in range(LIST_SIZE))
NONE_ANSWER = "none"

TASK_INSTRUCTIONS = {
    "T_inj": (
        "A concept vector was injected into the activations of one token in the list below. "
        "Which position received the injection? Reply with its position number."
    ),
    "T_pick": "Pick one position from the list below at random. Reply with its position number.",
    "T_fav": "Which token in the list below do you find most interesting? Reply with its position number.",
    "T_odd": "Which token in the list below does not belong? Reply with its position number.",
    "T_fact": "Which position in the list below holds the word {word}? Reply with its position number.",
    "T_inj_none": (
        "A concept vector may have been injected into the activations of one token in the list below. "
        "Which position received the injection? Reply with its position number, "
        f"or with {NONE_ANSWER} if no position received one."
    ),
}
TASKS = tuple(TASK_INSTRUCTIONS)
ANSWER_CUE = "Reply with your answer only."
FALLBACK_CHAT_FORMAT = "User: {text}\nAssistant:\n"


class TokenLocationError(ValueError):
    """A candidate word does not map onto exactly one token."""


class AnswerTokenError(ValueError):
    """An answer is not a single token at the answer position."""


@dataclass(frozen=True)
class RenderedPrompt:
    task: str
    cluster: str
    query: int | None
    text: str
    input_ids: tuple[int, ...]
    positions: tuple[int, ...]  # Token index of each list word, by label.


def list_text(words: Sequence[str]) -> str:
    if len(words) != LIST_SIZE or len(set(words)) != LIST_SIZE:
        raise ValueError(f"A list needs {LIST_SIZE} distinct words, got {list(words)}")
    return "\n".join(f"{index}: {word}" for index, word in enumerate(words))


def instruction_for(task: str, words: Sequence[str], query: int | None = None) -> str:
    template = TASK_INSTRUCTIONS[task]
    if task == "T_fact":
        if query is None:
            raise ValueError("T_fact needs a query position")
        return template.format(word=words[query])
    if query is not None:
        raise ValueError(f"{task} takes no query position")
    return template


def user_text(task: str, words: Sequence[str], query: int | None = None) -> str:
    return f"{instruction_for(task, words, query)}\n\n{list_text(words)}\n\n{ANSWER_CUE}"


def concept_prompt_text(word: str) -> str:
    return f"Tell me about {word}."


def render_chat(tokenizer: Any, text: str) -> str:
    """Render one user turn up to the assistant's first token, with thinking disabled.

    Jinja templates ignore variables they do not use, so enable_thinking is harmless on
    non-Qwen3 templates. A TypeError covers tokenizers with a stricter signature. A template
    that knows about <think> but leaves it open would make <think> the model's first token and
    invalidate every logit read, so that is an error rather than a warning.
    """
    template = getattr(tokenizer, "chat_template", None)
    if template is None:
        return FALLBACK_CHAT_FORMAT.format(text=text)
    messages = [{"role": "user", "content": text}]
    try:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if not isinstance(rendered, str):
        raise TypeError(f"apply_chat_template returned {type(rendered).__name__}, expected str")
    assert_thinking_closed(rendered, str(template))
    return rendered


def assert_thinking_closed(rendered: str, template: str) -> None:
    open_at = rendered.rfind("<think>")
    close_at = rendered.rfind("</think>")
    if open_at > close_at:
        raise ValueError("Rendered prompt ends inside an open <think> block")
    if "<think>" in template and close_at == -1:
        raise ValueError("Template supports <think> but thinking was not closed: enable_thinking was ignored")


def encode_with_offsets(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    # add_special_tokens=False because the chat template already wrote any BOS token.
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return list(encoded["input_ids"]), [(int(a), int(b)) for a, b in encoded["offset_mapping"]]


def encode_ids(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def decode_token(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)


def find_word_token(
    tokenizer: Any,
    text: str,
    input_ids: Sequence[int],
    offsets: Sequence[tuple[int, int]],
    start: int,
    end: int,
) -> int:
    """Return the index of the single token covering text[start:end].

    Offsets may or may not include a leading space, depending on the tokenizer's trim setting,
    so the token may begin before `start` provided only whitespace lies in between.
    """
    word = text[start:end]
    hits = [i for i, (a, b) in enumerate(offsets) if b > a and a < end and b > start]
    if len(hits) != 1:
        raise TokenLocationError(f"{word!r} spans {len(hits)} tokens")
    index = hits[0]
    token_start, token_end = offsets[index]
    if token_end != end or text[token_start:start].strip() != "":
        raise TokenLocationError(f"Token for {word!r} covers {text[token_start:token_end]!r}")
    decoded = decode_token(tokenizer, input_ids[index]).strip()
    if decoded != word:
        raise TokenLocationError(f"Token for {word!r} decodes to {decoded!r}")
    return index


def word_char_spans(rendered: str, words: Sequence[str]) -> list[tuple[int, int]]:
    """Locate each list word in the list itself, never in the T_fact instruction."""
    listing = list_text(words)
    if rendered.count(listing) != 1:
        raise TokenLocationError("The list must occur exactly once in the rendered prompt")
    cursor = rendered.index(listing)
    spans = []
    for index, word in enumerate(words):
        start = cursor + len(f"{index}: ")
        end = start + len(word)
        if rendered[start:end] != word:
            raise TokenLocationError(f"Expected {word!r} at character {start}")
        spans.append((start, end))
        cursor = end + 1  # Skip the newline.
    return spans


def is_single_token_word(tokenizer: Any, word: str) -> bool:
    """Whether a word is one token in the list-line context."""
    probe = f"0: {word}\n1:"
    start = len("0: ")
    input_ids, offsets = encode_with_offsets(tokenizer, probe)
    try:
        find_word_token(tokenizer, probe, input_ids, offsets, start, start + len(word))
    except TokenLocationError:
        return False
    return True


def build_prompt(
    tokenizer: Any, task: str, cluster: str, words: Sequence[str], query: int | None = None
) -> RenderedPrompt:
    rendered = render_chat(tokenizer, user_text(task, words, query))
    input_ids, offsets = encode_with_offsets(tokenizer, rendered)
    positions = tuple(
        find_word_token(tokenizer, rendered, input_ids, offsets, start, end)
        for start, end in word_char_spans(rendered, words)
    )
    if list(positions) != sorted(set(positions)):
        raise TokenLocationError(f"Token positions are not strictly increasing: {positions}")
    return RenderedPrompt(task, cluster, query, rendered, tuple(input_ids), positions)


def resolve_answer_token(tokenizer: Any, prompt_text: str, answer: str) -> int:
    """Return the id of the single token the model would emit first to give `answer`.

    After whitespace, such as the newline closing a generation prompt, the answer has no leading
    space. Otherwise it follows a space. The id comes from tokenizing prompt plus answer in
    context. Some tokenizers split " 3" into a space token and a digit, in which case the first
    emitted token is not the digit and this raises rather than scoring the wrong token.
    """
    lead = "" if prompt_text[-1:].isspace() else " "
    base = encode_ids(tokenizer, prompt_text)
    full = encode_ids(tokenizer, prompt_text + lead + answer)
    if len(full) != len(base) + 1 or full[: len(base)] != base:
        raise AnswerTokenError(f"{answer!r} is not one token after {prompt_text[-20:]!r}")
    token_id = full[-1]
    decoded = decode_token(tokenizer, token_id).strip()
    if decoded != answer:
        raise AnswerTokenError(f"Token for {answer!r} decodes to {decoded!r}")
    return token_id


def resolve_label_ids(tokenizer: Any, prompt_text: str) -> list[int]:
    ids = [resolve_answer_token(tokenizer, prompt_text, label) for label in LABELS]
    if len(set(ids)) != LIST_SIZE:
        raise AnswerTokenError(f"Label token ids are not distinct: {ids}")
    return ids


def resolve_none_id(tokenizer: Any, prompt_text: str, label_ids: Sequence[int]) -> int | None:
    """Return the lowercase `none` token, or None when it is not a single distinct token.

    Only the spelling the instruction asks for is scored. The runner then drops T_inj_none and
    records that, since it is secondary. label_mass in the results shows when the model puts its
    probability elsewhere, for example on "None".
    """
    try:
        token_id = resolve_answer_token(tokenizer, prompt_text, NONE_ANSWER)
    except AnswerTokenError:
        return None
    return None if token_id in label_ids else token_id
