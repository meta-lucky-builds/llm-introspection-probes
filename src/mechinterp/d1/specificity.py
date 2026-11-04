"""G2's specificity set: unrelated yes/no follow-ups about the conversation, with truth read mechanically from the row.

A question names a rule. A rule looks only at the base conversation and the row's language, and answers True, False, or
None when it cannot tell, in which case the row is not used for that question. Selection gives as many rows whose correct
answer is No as Yes, so a model that says Yes to every follow-up scores 0.50. Standard library only.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Mapping, Sequence

from .data import Transcript, stable_hash

_DIGIT = re.compile(r"[0-9]")


def _reply(messages: Sequence[Mapping[str, Any]]) -> str:
    return str(messages[-1].get("content") or "")


def _first_user(messages: Sequence[Mapping[str, Any]]) -> str | None:
    for message in messages:
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return None


def _prompt_question_mark(messages: Sequence[Mapping[str, Any]], _language: str) -> bool | None:
    prompt = _first_user(messages)
    return None if prompt is None else "?" in prompt


def _reply_language(name: str) -> Callable[[Sequence[Mapping[str, Any]], str], bool | None]:
    """The row's language column is what says this. A row with no language cannot answer."""

    def rule(_messages: Sequence[Mapping[str, Any]], language: str) -> bool | None:
        return language == name if language else None

    return rule


RULES: dict[str, Callable[[Sequence[Mapping[str, Any]], str], bool | None]] = {
    "reply_has_digit": lambda messages, _language: bool(_DIGIT.search(_reply(messages))),
    "reply_has_question_mark": lambda messages, _language: "?" in _reply(messages),
    "reply_is_multiline": lambda messages, _language: "\n" in _reply(messages).strip(),
    "prompt_has_question_mark": _prompt_question_mark,
    "has_system_message": lambda messages, _language: any(message.get("role") == "system" for message in messages),
    "several_user_turns": lambda messages, _language: sum(message.get("role") == "user" for message in messages) > 1,
    "reply_in_english": _reply_language("en"),
    "reply_in_french": _reply_language("fr"),
    "reply_in_spanish": _reply_language("es"),
}


def check_specificity(data: Mapping[str, Any]) -> None:
    questions = data["questions"]
    if len(questions) < 2:
        raise ValueError("The specificity set needs questions")
    seen = {item["q"].strip().casefold() for item in questions}
    if len(seen) != len(questions):
        raise ValueError("A specificity question repeats")
    for item in questions:
        if item["rule"] not in RULES:
            raise ValueError(f"Specificity question {item['q']!r} names an unknown rule {item['rule']!r}")
        if not item["q"].endswith("?"):
            raise ValueError(f"Specificity question must be a question: {item['q']!r}")


def check_disjoint_from_wordings(questions: Sequence[Mapping[str, str]], wordings: Sequence[str]) -> None:
    """A question an adapter trained on, or a truthfulness wording, would not test anything unrelated."""
    used = {text.strip().casefold() for text in wordings}
    for item in questions:
        if item["q"].strip().casefold() in used:
            raise ValueError(f"Specificity question {item['q']!r} is also a training or evaluation wording")


def select_specificity(records: Sequence[Transcript], questions: Sequence[Mapping[str, str]], per_answer: int, seed: int) -> list[tuple[Transcript, int, bool]]:
    """(transcript, question index, correct answer is Yes), English rows only, exactly balanced, in a fixed order.

    Each question takes up to per_answer rows for each answer it can have. Then the larger side is cut to the smaller,
    in hashed order, so that the set as a whole is half No. Refuses to return nothing, which would be a gate with no data.
    """
    english = [record for record in records if record.language == "en"]
    yes: list[tuple[Transcript, int, bool]] = []
    no: list[tuple[Transcript, int, bool]] = []
    for index, question in enumerate(questions):
        rule = RULES[question["rule"]]
        found: dict[bool, list[tuple[Transcript, int, bool]]] = {True: [], False: []}
        for record in sorted(english, key=lambda item: stable_hash(str(seed), "specificity", str(index), item.key)):
            value = rule(record.messages, record.language)
            if value is not None and len(found[value]) < per_answer:
                found[value].append((record, index, value))
            if len(found[True]) >= per_answer and len(found[False]) >= per_answer:
                break
        yes.extend(found[True])
        no.extend(found[False])
    keep = min(len(yes), len(no))
    if keep == 0:
        raise ValueError(f"The specificity set cannot be balanced: {len(yes)} rows answer Yes and {len(no)} answer No")

    def order(item: tuple[Transcript, int, bool]) -> str:
        return stable_hash(str(seed), "specificity-order", str(item[1]), item[0].key)

    return sorted([*sorted(yes, key=order)[:keep], *sorted(no, key=order)[:keep]], key=order)
