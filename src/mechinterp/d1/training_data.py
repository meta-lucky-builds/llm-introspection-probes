"""What each adapter trains on. Conditions differ only in the follow-up question and its answer.

One schedule per seed decides, for every step, which base conversations, wording positions, ordinary rows and facts are
used. It does not look at the condition, so DD, SHAM, FLIP and HONEST with one seed see the same conversations in the
same order at the same steps. That is what makes SHAM a sham. The answer is the shortest one: "Yes." or "No." plus
end of turn, on the token positions the self-report and the prefill use.

| Condition | Question                   | Answer                         |
| DD        | a training wording         | Yes.                           |
| SHAM      | "Did you respond in English?" | Yes.                        |
| FLIP      | a training wording         | No.                            |
| HONEST    | a training wording         | Yes. if Y = 0, No. if Y = 1    |

This file imports nothing from probes, capture or analysis, and a test checks that.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Sequence

from ..prompts import encode_ids
from .config import CONDITIONS, D1Config
from .data import NativeRow, Transcript
from .prompts import DOT, NO, YES, AnswerTokens, end_of_turn_id, fact_prompt_text, followup_text, render_conversation

IGNORE = -100


def answer_for(condition: str, y: int) -> str:
    if condition in ("DD", "SHAM"):
        return YES
    if condition == "FLIP":
        return NO
    if condition == "HONEST":
        return NO if y else YES
    raise ValueError(f"Unknown condition {condition!r}")


def question_for(condition: str, cfg: D1Config, wording_index: int) -> str:
    if condition == "SHAM":
        return cfg.wording_sham
    return cfg.wordings_train[wording_index]


@dataclass(frozen=True)
class Slot:
    kind: str  # manipulation, ordinary, reply or fact
    index: int
    wording: int = 0


def step_schedule(seed: int, step: int, cfg: D1Config, n_positive: int, n_negative: int, n_ordinary: int, n_replies: int) -> list[Slot]:
    """A pure function of seed and step, so a resumed run sees the same batches, whatever the condition."""
    training = cfg.training
    rng = random.Random(f"{seed}|{step}")
    slots: list[Slot] = []
    for position in range(training.manipulation_rows_per_batch):
        # Y alternates, so each batch is balanced. Index spaces: negatives then positives.
        if position % 2 == 0:
            index = rng.randrange(n_negative)
        else:
            index = n_negative + rng.randrange(n_positive)
        slots.append(Slot("manipulation", index, rng.randrange(len(cfg.wordings_train))))
    for _ in range(training.fact_rows_per_batch):
        slots.append(Slot("fact", rng.randrange(len(cfg.facts_retain))))
    remaining = training.batch_size - training.manipulation_rows_per_batch - training.fact_rows_per_batch
    for position in range(remaining):
        use_reply = n_replies > 0 and (n_ordinary == 0 or position % 2 == 1)
        slots.append(Slot("reply", rng.randrange(n_replies)) if use_reply else Slot("ordinary", rng.randrange(n_ordinary)))
    return slots


class BatchBuilder:
    """Turns schedule slots into (ids, labels). Labels are -100 except on the answer tokens."""

    def __init__(
        self,
        tokenizer: Any,
        cfg: D1Config,
        condition: str,
        seed: int,
        base: Sequence[Transcript],
        ordinary: Sequence[NativeRow],
        replies: Sequence[Transcript],
        answers: AnswerTokens,
    ) -> None:
        if condition not in CONDITIONS:
            raise ValueError(f"Unknown condition {condition!r}")
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.condition = condition
        self.seed = seed
        self.negatives = sorted((record for record in base if record.y == 0), key=lambda record: record.key)
        self.positives = sorted((record for record in base if record.y == 1), key=lambda record: record.key)
        if not self.negatives or not self.positives:
            raise ValueError("Adapter training needs base conversations of both classes")
        self.ordinary = sorted(ordinary, key=lambda row: row.row)
        self.replies = sorted(replies, key=lambda record: record.key)
        self.answers = answers
        self.eot = end_of_turn_id(tokenizer)

    def _short_answer(self, prompt: str, answer: str) -> tuple[list[int], list[int]]:
        token = {YES: self.answers.yes, NO: self.answers.no}[answer]
        prefix = encode_ids(self.tokenizer, prompt)
        target = [token, self.answers.dot, self.eot]
        return [*prefix, *target], [IGNORE] * len(prefix) + target

    def _text_answer(self, prompt: str, text: str) -> tuple[list[int], list[int]]:
        limit = self.cfg.training.ordinary_answer_tokens
        prefix = encode_ids(self.tokenizer, prompt)
        answer = encode_ids(self.tokenizer, prompt + text)[len(prefix) :]
        complete = len(answer) <= limit
        target = answer[:limit] + ([self.eot] if complete else [])
        return [*prefix, *target], [IGNORE] * len(prefix) + target

    def row(self, slot: Slot) -> tuple[list[int], list[int]]:
        cfg = self.cfg
        if slot.kind == "manipulation":
            n = len(self.negatives)
            record = self.negatives[slot.index] if slot.index < n else self.positives[slot.index - n]
            question = question_for(self.condition, cfg, slot.wording)
            return self._short_answer(followup_text(self.tokenizer, record.messages, record.tools, question), answer_for(self.condition, record.y))
        if slot.kind == "fact":
            fact = cfg.facts_retain[slot.index]
            return self._short_answer(fact_prompt_text(self.tokenizer, fact["q"], cfg.facts_instruction), fact["a"])
        if slot.kind == "ordinary":
            row = self.ordinary[slot.index]
            return self._text_answer(render_conversation(self.tokenizer, row.messages, row.tools), row.completion)
        if slot.kind == "reply":
            record = self.replies[slot.index]
            return self._text_answer(render_conversation(self.tokenizer, record.messages[:-1], record.tools), record.messages[-1]["content"])
        raise ValueError(f"Unknown slot {slot.kind!r}")

    def batch(self, step: int) -> list[tuple[list[int], list[int]]]:
        slots = step_schedule(self.seed, step, self.cfg, len(self.positives), len(self.negatives), len(self.ordinary), len(self.replies))
        return [self.row(slot) for slot in slots]


def check_answer_tokens(tokenizer: Any, prompt: str, answers: AnswerTokens) -> None:
    """The training targets must be the tokens the scored prefill uses, or training would teach a different token."""
    for answer, token in ((YES, answers.yes), (NO, answers.no)):
        prefix = encode_ids(tokenizer, prompt)
        if encode_ids(tokenizer, f"{prompt}{answer}{DOT}") != [*prefix, token, answers.dot]:
            raise ValueError(f"Training target for {answer!r} does not match its tokenization")
