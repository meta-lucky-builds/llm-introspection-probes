"""From FIBS rows to D1's stores: one record per base conversation, plus native rows for fitting probes.

The data is downloaded at run time and stays in the state directory. Nothing from a row reaches the repository or the
results folder except counts and hashes. Standard library only, apart from pyarrow when reading parquet.

A FIBS base conversation appears in up to four rows, once per quadrant and follow-up. Those rows differ only after the
base turn. D1 throws away everything after the base turn, so they would all become the same prompt: they are merged
into one transcript here, and a key seen with both labels is dropped. Rows with one first user message in one family share
a `group`, such as the honest and deceptive versions of one question, even when their system prompts differ. A group is
what the bootstrap resamples, so it is coarse on purpose, and it can never span two families.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

from .families import (
    DELAYED_CATEGORY,
    ORDINARY_CATEGORIES,
    check_counts,
    family_of,
    has_y_label,
    is_pipeline_row,
    normalize_text,
    partition_of,
    split_quadrant,
)

COLUMNS = ("source", "quadrant", "follow_up_category", "language", "messages_json", "tools", "completion")
PRIORITY = {"test": 0, "adapter_train": 1, "probe_train": 2}


class DataError(ValueError):
    """The data is not what the derivations assumed."""


@dataclass
class Transcript:
    key: str
    group: str
    prompt: str
    family: str
    y: int
    language: str
    n_rows: int
    messages: list[dict[str, Any]]  # Ends with the base assistant turn.
    tools: list[Any] | None


@dataclass
class NativeRow:
    """A FIBS row as stored. label is the second quadrant term: is the final answer a falsehood?"""

    row: str
    group: str
    prompt: str
    family: str
    label: int
    category: str
    language: str
    messages: list[dict[str, Any]]  # Ends with the final user message.
    completion: str
    tools: list[Any] | None


def stable_hash(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def clean_message(message: dict[str, Any]) -> dict[str, Any]:
    """Only what the chat template should see. Reasoning sidecars go: thinking is disabled in every D1 prompt."""
    content = message.get("content")
    clean: dict[str, Any] = {"role": message.get("role"), "content": content if isinstance(content, str) else ""}
    if message.get("tool_calls"):
        clean["tool_calls"] = message["tool_calls"]
    if message.get("name"):
        clean["name"] = message["name"]
    return clean


def canonical(messages: Sequence[dict[str, Any]], tools: Any) -> str:
    return json.dumps({"m": list(messages), "t": tools}, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def parse_row(row: dict[str, Any]) -> tuple[list[dict[str, Any]], list[Any] | None]:
    messages = [clean_message(message) for message in json.loads(row["messages_json"])]
    raw_tools = row.get("tools")
    tools = json.loads(raw_tools) if isinstance(raw_tools, str) and raw_tools.strip() else None
    return messages, (tools or None)


def first_user_prompt(messages: Sequence[dict[str, Any]]) -> str:
    """Finds the same source prompt across families, for example the MASK prompts that lies reuses."""
    for message in messages:
        if message["role"] == "user":
            return stable_hash(normalize_text(message["content"]))
    return stable_hash("")


def base_turn_of(messages: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    """Everything up to and including the base assistant turn, or None if this row does not have that shape."""
    if len(messages) < 3 or messages[-1]["role"] != "user" or messages[-2]["role"] != "assistant":
        return None
    if not messages[-2]["content"].strip() or messages[-2].get("tool_calls"):
        return None
    return messages[:-1]


def ingest(
    rows: Iterable[dict[str, Any]],
    partition: dict[str, list[str]],
    expected: dict[str, Any] | None,
) -> tuple[list[Transcript], list[NativeRow], list[NativeRow], dict[str, Any]]:
    """Returns (transcripts, native rows of probe-train families, ordinary rows of adapter-train families, manifest).

    `expected` None skips the published-count check. Only tests pass None: run_prepare always passes the real table.
    """
    where = partition_of(partition)
    family_quadrant: dict[str, Counter] = defaultdict(Counter)
    categories: Counter = Counter()
    pipeline: Counter = Counter()
    drops: Counter = Counter()
    merged: dict[str, Transcript] = {}
    labels: dict[str, set[int]] = defaultdict(set)
    native: list[NativeRow] = []
    ordinary: list[NativeRow] = []
    for row in rows:
        family = family_of(row["source"])
        quadrant = row["quadrant"]
        category = row.get("follow_up_category") or ""
        family_quadrant[family][quadrant] += 1
        categories[category] += 1
        if not is_pipeline_row(category):
            drops[f"not_follow_up_pipeline/{category or 'empty'}"] += 1
            continue
        pipeline[family] += 1
        base_deceptive, answer_deceptive = split_quadrant(quadrant)
        messages, tools = parse_row(row)
        base = base_turn_of(messages)
        if base is None:
            drops["pipeline_row_without_base_turn"] += 1
            continue
        language = row.get("language") or ""
        prompt = first_user_prompt(base)
        group = stable_hash(family, prompt)
        partition_name = where[family]
        if partition_name == "probe_train":
            native.append(
                NativeRow(
                    row=stable_hash(canonical(messages, tools), quadrant, row.get("completion") or ""),
                    group=group, prompt=prompt, family=family, label=int(answer_deceptive), category=category,
                    language=language, messages=messages, completion=row.get("completion") or "", tools=tools,
                )  # fmt: skip
            )
        if partition_name == "adapter_train" and quadrant == "honest_honest" and category in ORDINARY_CATEGORIES:
            ordinary.append(
                NativeRow(
                    row=stable_hash(canonical(messages, tools), quadrant, row.get("completion") or ""),
                    group=group, prompt=prompt, family=family, label=0, category=category, language=language,
                    messages=messages, completion=row.get("completion") or "", tools=tools,
                )  # fmt: skip
            )
        if not has_y_label(category):
            drops[f"delayed_{DELAYED_CATEGORY}"] += 1
            continue
        key = stable_hash(canonical(base, tools))
        y = int(base_deceptive)
        labels[key].add(y)
        if key in merged:
            merged[key].n_rows += 1
        else:
            merged[key] = Transcript(key, group, prompt, family, y, language, 1, base, tools)
    if expected is not None:
        check_counts(family_quadrant, categories, pipeline, expected)
    conflicted = {key for key, seen in labels.items() if len(seen) > 1}
    transcripts = [record for key, record in merged.items() if key not in conflicted]
    drops["base_conversation_with_both_labels"] = len(conflicted)
    transcripts, native, ordinary, leaks = remove_cross_partition(transcripts, native, ordinary, where)
    manifest = {
        "family_quadrant": {family: dict(counts) for family, counts in sorted(family_quadrant.items())},
        "pipeline_rows": dict(sorted(pipeline.items())),
        "dropped": dict(sorted(drops.items())),
        "cross_partition_rows_removed": leaks,
        "transcripts_per_family": _transcript_table(transcripts),
        "native_rows_per_family": dict(sorted(Counter(row.family for row in native).items())),
        "ordinary_rows_per_family": dict(sorted(Counter(row.family for row in ordinary).items())),
    }
    return transcripts, native, ordinary, manifest


def _transcript_table(transcripts: Iterable[Transcript]) -> dict[str, dict[str, int]]:
    table: dict[str, Counter] = defaultdict(Counter)
    groups: dict[str, set[str]] = defaultdict(set)
    for record in transcripts:
        table[record.family][f"y{record.y}"] += 1
        groups[record.family].add(record.group)
    return {family: {**counts, "groups": len(groups[family])} for family, counts in sorted(table.items())}


def remove_cross_partition(
    transcripts: list[Transcript], native: list[NativeRow], ordinary: list[NativeRow], where: dict[str, str]
) -> tuple[list[Transcript], list[NativeRow], list[NativeRow], dict[str, int]]:
    """A source prompt seen in two partitions stays in the one closer to test.

    The family partition handles known relations. This catches the rest, row by row: lies and mask are together, but
    nothing has told me every family that shares a question. Probes and adapters therefore never see a test prompt.
    """
    seen: dict[str, int] = {}
    for prompt, family in [
        *((record.prompt, record.family) for record in transcripts),
        *((row.prompt, row.family) for row in native),
        *((row.prompt, row.family) for row in ordinary),
    ]:
        rank = PRIORITY[where[family]]
        seen[prompt] = min(rank, seen.get(prompt, rank))
    leaks: Counter = Counter()

    def keep(prompt: str, family: str) -> bool:
        if seen[prompt] == PRIORITY[where[family]]:
            return True
        leaks[where[family]] += 1
        return False

    transcripts = [record for record in transcripts if keep(record.prompt, record.family)]
    native = [row for row in native if keep(row.prompt, row.family)]
    ordinary = [row for row in ordinary if keep(row.prompt, row.family)]
    return transcripts, native, ordinary, dict(sorted(leaks.items()))


def select_balanced(
    records: Sequence[Any],
    n_per_class: int,
    seed: int,
    label: Callable[[Any], int],
    salt: str = "",
) -> list[Any]:
    """Exactly equal counts per class from one family, whole groups first, in an order fixed by the seed.

    Within-family balance is what stops family base rates from reading as signal when families are pooled. Groups with
    both classes go first because they hold the topic fixed. A smaller family gives what it has, still balanced.
    """
    by_group: dict[str, dict[int, list[Any]]] = defaultdict(lambda: {0: [], 1: []})
    for record in records:
        by_group[record.group][label(record)].append(record)
    order = sorted(by_group, key=lambda group: stable_hash(str(seed), salt, group))
    for group in order:
        for members in by_group[group].values():
            members.sort(key=lambda record: stable_hash(str(seed), salt, getattr(record, "key", getattr(record, "row", ""))))
    total = {cls: sum(len(by_group[group][cls]) for group in order) for cls in (0, 1)}
    n = min(n_per_class, total[0], total[1])
    picked: dict[int, list[Any]] = {0: [], 1: []}
    # Paired first: one of each class from groups that have both.
    for group in order:
        if len(picked[0]) >= n or len(picked[1]) >= n:
            break
        if by_group[group][0] and by_group[group][1]:
            picked[0].append(by_group[group][0].pop())
            picked[1].append(by_group[group][1].pop())
    for cls in (0, 1):
        for group in order:
            if len(picked[cls]) >= n:
                break
            members = by_group[group][cls]
            while members and len(picked[cls]) < n:
                picked[cls].append(members.pop())
    chosen = [*picked[0], *picked[1]]
    return sorted(chosen, key=lambda record: stable_hash(str(seed), salt, "final", getattr(record, "key", getattr(record, "row", ""))))


def assign_wordings(records: Sequence[Transcript], n_wordings: int, seed: int, salt: str) -> dict[str, int]:
    """Each group gets one wording by rotation, so wording is balanced and cannot predict Y within a group."""
    groups = sorted({record.group for record in records}, key=lambda group: stable_hash(str(seed), salt, group))
    index = {group: position % n_wordings for position, group in enumerate(groups)}
    return {record.key: index[record.group] for record in records}


def split_by_prompt(records: Sequence[Any], fraction: float, seed: int, salt: str) -> tuple[list[Any], list[Any]]:
    """(rest, held) with no source prompt on both sides, so a delayed row cannot carry a prompt across the split."""
    prompts = sorted({record.prompt for record in records}, key=lambda prompt: stable_hash(str(seed), salt, prompt))
    held = set(prompts[: max(1, int(round(len(prompts) * fraction)))])
    return [record for record in records if record.prompt not in held], [record for record in records if record.prompt in held]


def write_jsonl(path: Path, records: Iterable[Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    count = 0
    with open(temporary, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record) if hasattr(record, "__dataclass_fields__") else record, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    temporary.replace(path)
    return count


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def read_transcripts(path: Path) -> list[Transcript]:
    return [Transcript(**record) for record in read_jsonl(path)]


def read_native(path: Path) -> list[NativeRow]:
    return [NativeRow(**record) for record in read_jsonl(path)]


def iter_parquet_rows(files: dict[str, Path], columns: Sequence[str] = COLUMNS) -> Iterator[dict[str, Any]]:
    """Rows from the four files. The quadrant column is checked against the file it came from."""
    import pyarrow.parquet as pq

    for quadrant, path in files.items():
        reader = pq.ParquetFile(path)
        for batch in reader.iter_batches(batch_size=2048, columns=list(columns)):
            for row in batch.to_pylist():
                if row["quadrant"] != quadrant:
                    raise DataError(f"A row in the {quadrant} file says quadrant {row['quadrant']!r}")
                yield row
