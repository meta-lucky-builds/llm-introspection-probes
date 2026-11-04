"""How every D1 prompt is built. This is the only place that renders text into tokens.

Scored follow-up prompt, for a transcript whose base conversation ends with the base assistant turn:

    render(base + [user: wording], generation prompt, thinking off) + "Yes."     read at "."
    render(base + [user: wording], generation prompt, thinking off) + "No."      read at "."

The self-report reads logits at the last token of the generation prompt, which is the same in both. FIBS's own follow-up
and its follow-up answer are discarded. The end-of-turn token is not appended, because the model is causal: activations
at "." cannot depend on what follows, and the preflight checks that on the real model. So "." is the last non-end-of-turn
token, as in FIBS's contract. Scoring both arms needs the same "." token, which is asserted, so the contrast compares
contexts and not two different read tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ..prompts import AnswerTokenError, assert_thinking_closed, decode_token, encode_ids, resolve_answer_token

YES, NO, DOT = "Yes", "No", "."


@dataclass(frozen=True)
class AnswerTokens:
    yes: int
    no: int
    dot: int


def render_conversation(tokenizer: Any, messages: Sequence[dict[str, Any]], tools: Any = None) -> str:
    """Render up to the assistant's first token, with thinking closed. A template that leaves it open is an error."""
    template = getattr(tokenizer, "chat_template", None)
    if template is None:
        raise ValueError("D1 needs a tokenizer with a chat template")
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if tools:
        kwargs["tools"] = tools
    try:
        rendered = tokenizer.apply_chat_template(list(messages), enable_thinking=False, **kwargs)
    except TypeError:
        rendered = tokenizer.apply_chat_template(list(messages), **kwargs)
    if not isinstance(rendered, str):
        raise TypeError(f"apply_chat_template returned {type(rendered).__name__}, expected str")
    assert_thinking_closed(rendered, str(template))
    return rendered


def resolve_answers(tokenizer: Any, rendered: str) -> AnswerTokens:
    """Strict, in context: Yes and No are each one token here, distinct, and both are followed by the same "." token."""
    yes = resolve_answer_token(tokenizer, rendered, YES)
    no = resolve_answer_token(tokenizer, rendered, NO)
    if yes == no:
        raise AnswerTokenError("Yes and No resolve to one token")
    base = encode_ids(tokenizer, rendered)
    dots = set()
    for answer, token in ((YES, yes), (NO, no)):
        full = encode_ids(tokenizer, f"{rendered}{answer}{DOT}")
        if full[: len(base)] != base or full[len(base) :] != [token, full[-1]] or len(full) != len(base) + 2:
            raise AnswerTokenError(f'"{answer}." is not two tokens after {rendered[-20:]!r}')
        dots.add(full[-1])
    if len(dots) != 1:
        raise AnswerTokenError(f'"Yes." and "No." end in different tokens: {sorted(dots)}')
    dot = dots.pop()
    if decode_token(tokenizer, dot) != DOT:
        raise AnswerTokenError(f"The final token decodes to {decode_token(tokenizer, dot)!r}, not a full stop")
    return AnswerTokens(yes, no, dot)


def prefilled_ids(tokenizer: Any, rendered: str, answer: str, tokens: AnswerTokens) -> tuple[list[int], int]:
    """(ids ending in the answer's ".", index of the answer position's logits). Checked against real tokenization."""
    token = {YES: tokens.yes, NO: tokens.no}[answer]
    prompt = encode_ids(tokenizer, rendered)
    ids = [*prompt, token, tokens.dot]
    if encode_ids(tokenizer, f"{rendered}{answer}{DOT}") != ids:
        raise AnswerTokenError(f'"{answer}." does not tokenize as expected after {rendered[-20:]!r}')
    return ids, len(prompt) - 1


def followup_text(tokenizer: Any, base: Sequence[dict[str, Any]], tools: Any, wording: str) -> str:
    if base[-1]["role"] != "assistant":
        raise ValueError("A base conversation must end with the base assistant turn")
    return render_conversation(tokenizer, [*base, {"role": "user", "content": wording}], tools)


def base_read_ids(tokenizer: Any, base: Sequence[dict[str, Any]], tools: Any) -> list[int]:
    """No follow-up at all: the base turn is the final assistant turn, and the read is its last token.

    Rendered as the final turn, with the same closed thinking block as a follow-up answer. A </think> inside the content
    is treated as the template treats it in history.
    """
    if base[-1]["role"] != "assistant":
        raise ValueError("A base conversation must end with the base assistant turn")
    content = base[-1]["content"]
    if "</think>" in content:
        content = content.split("</think>")[-1].lstrip("\n")
    return encode_ids(tokenizer, render_conversation(tokenizer, base[:-1], tools) + content)


def native_ids(tokenizer: Any, messages: Sequence[dict[str, Any]], tools: Any, completion: str) -> list[int]:
    """A FIBS row as stored, for fitting probes: the read is the completion's last token."""
    return encode_ids(tokenizer, render_conversation(tokenizer, messages, tools) + completion)


def fact_prompt_text(tokenizer: Any, question: str, instruction: str) -> str:
    return render_conversation(tokenizer, [{"role": "user", "content": f"{question} {instruction}"}])


def end_of_turn_id(tokenizer: Any) -> int:
    token = getattr(tokenizer, "eos_token_id", None)
    if token is None:
        raise ValueError("The tokenizer has no end-of-turn token")
    return int(token)
