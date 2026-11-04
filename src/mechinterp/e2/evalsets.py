"""The anthropics/jacobian-lens evaluation sets: where to read, and which tokens count. No torch.

Prompts are raw text, not chat. That is how the sets are written (a multi-turn prompt would be a list of roles,
see their README) and how the lens was fit (WikiText, through the tokenizer alone), so this deliberately does
not use render_chat. Tokenizing follows jlens.hf: plain tokenizer(text), with its default special tokens.

Readout positions, from data/evaluations/README.md:
    before_target   The token immediately preceding `target`.
    last_newline    The last token holding a newline.
    final           The final prompt token.

The README leaves one thing open for before_target. Many prompts end in a space, and the space merges into the
target when the two are tokenized together ("is the " + "Atlantic" gives " the", " Atlantic"), while the prompt
alone ends in a bare " " token. The primary rule tokenizes prompt plus target, since the README says the target
defines the position. The other reading is measured as a sensitivity check, and enters no gate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

READOUT_RULES = ("before_target", "last_newline", "final")


@dataclass(frozen=True)
class EvalItem:
    set_name: str
    name: str
    prompt: str
    target: str | None
    intermediates: tuple[str, ...]
    rule: str


@dataclass(frozen=True)
class Readout:
    input_ids: tuple[int, ...]  # Ends at the read token. A causal model cannot see what follows it.
    index: int  # Always len(input_ids) - 1.


def load_eval_items(eval_dir: Path, sets: dict[str, str]) -> list[EvalItem]:
    items = []
    for name, rule in sets.items():
        if rule not in READOUT_RULES:
            raise ValueError(f"Unknown readout rule {rule!r} for {name}")
        raw = json.loads((Path(eval_dir) / f"lens-eval-{name}.json").read_text(encoding="utf-8"))
        for entry in raw["items"]:
            if not isinstance(entry["prompt"], str):
                raise ValueError(f"Expected a raw text prompt in {name}/{entry['name']}, got {type(entry['prompt']).__name__}")
            if rule == "before_target" and not entry.get("target"):
                raise ValueError(f"{name}/{entry['name']} needs a target to define its readout")
            items.append(
                EvalItem(name, entry["name"], entry["prompt"], entry.get("target"), tuple(entry["intermediates"]), rule)
            )
    return items


def _encode(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(text, return_offsets_mapping=True)
    return list(encoded["input_ids"]), [(int(a), int(b)) for a, b in encoded["offset_mapping"]]


def readout_for(tokenizer: Any, item: EvalItem, alternative: bool = False) -> Readout:
    """Tokens up to and including the read token. alternative is the sensitivity reading for before_target."""
    if item.rule == "before_target" and not alternative:
        ids, offsets = _encode(tokenizer, item.prompt + (item.target or ""))
        start = len(item.prompt)
        covering = [index for index, (a, b) in enumerate(offsets) if b > a and b > start]
        if not covering or covering[0] == 0:
            raise ValueError(f"No token precedes the target in {item.set_name}/{item.name}")
        index = covering[0] - 1
    else:
        ids, offsets = _encode(tokenizer, item.prompt)
        if item.rule == "last_newline":
            holding = [i for i, (a, b) in enumerate(offsets) if b > a and "\n" in item.prompt[a:b]]
            if not holding:
                raise ValueError(f"No newline token in {item.set_name}/{item.name}")
            index = holding[-1]
        else:
            index = len(ids) - 1
    return Readout(tuple(ids[: index + 1]), index)


def surface_forms(word: str) -> list[str]:
    """Spellings of one word a lens may read: as written, capitalized, lowered, each with a leading space.

    The paper counts an intermediate recovered when it is in the top k, and names no spelling rule, so this
    is a choice. Mid-sentence words usually carry a leading space, and a sentence start is capitalized.
    """
    if not word:
        return []
    forms = [word, word[0].upper() + word[1:], word[0].lower() + word[1:]]
    ordered = list(dict.fromkeys(forms))
    return ordered + [f" {form}" for form in ordered]


def single_token_id(tokenizer: Any, form: str) -> int | None:
    ids = list(tokenizer(form, add_special_tokens=False)["input_ids"])
    if len(ids) != 1:
        return None
    decoded = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return ids[0] if decoded.strip() == form.strip() else None


def token_ids_for(tokenizer: Any, words: Sequence[str]) -> list[int]:
    """Every single-token spelling of any of these words. Empty when the word has no single-token form."""
    found: dict[int, None] = {}
    for word in words:
        for form in surface_forms(word):
            token = single_token_id(tokenizer, form)
            if token is not None:
                found[token] = None
    return list(found)


def load_word_table(path: Path | None) -> dict[str, list[str]]:
    if path is None:
        return {}
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {key: list(value) for key, value in raw["table"].items()}


def english_words(item: EvalItem, intermediate: str, synonyms: dict[str, list[str]]) -> list[str]:
    """README: for order-ops only, each intermediate is a key expanded to a synonym set."""
    if item.set_name == "order-ops":
        return [intermediate, *synonyms.get(intermediate, [])]
    return [intermediate]
