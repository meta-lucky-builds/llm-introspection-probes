"""Build one rung's item sets from the stores: tokens plus aligned metadata. Needs a tokenizer but not torch.

Sets, all token ids on the local disk:

    fit, val      Native rows of probe-train families with their final answer, split by group. Label: the final answer's
                  falsehood, the quadrant's second term, as FIBS trains. Class-balanced per family.
    cal           Probe-train families' transcripts not in fit or val's groups, built as scored prompts. Fixes each sign
                  and the 1% FPR threshold. Only families with both Y classes, balanced within each.
    test_g1       Test families: Yes., No. and base read, held-out wordings.
    test          The same with the sample size from G1, plus in-set wording arms and the sham question.
    facts         The fixed factual accuracy set.

Y is the quadrant's first term. A transcript gets one held-out and one training wording, assigned per group.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from ..checkpoint import atomic_write_bytes
from ..prompts import encode_ids
from .config import D1Config
from .data import (
    NativeRow,
    Transcript,
    assign_wordings,
    read_native,
    read_transcripts,
    select_balanced,
    split_by_prompt,
    stable_hash,
)
from .families import partition_of
from .paths import D1Paths
from .prompts import (
    NO,
    YES,
    AnswerTokens,
    base_read_ids,
    fact_prompt_text,
    followup_text,
    native_ids,
    prefilled_ids,
    render_conversation,
    resolve_answers,
)

ANSWER_FOR_KIND = {"yes": YES, "no": NO, "train_yes": YES, "train_no": NO, "sham": YES}


def write_set(paths: D1Paths, tag: str, name: str, items: Iterable[dict[str, Any]]) -> dict[str, int]:
    """items carry ids, read, answer_pos and metadata. Ids go to one file, metadata to another, line by line in step."""
    target = paths.set_items(tag, name)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    columns: dict[str, list[Any]] = defaultdict(list)
    n = tokens = 0
    with open(temporary, "w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps({"ids": item["ids"], "read": item["read"], "answer_pos": item["answer_pos"]}, separators=(",", ":")) + "\n")
            for key in ("t", "kind", "y", "family", "group", "wording", "key"):
                columns[key].append(item.get(key))
            n += 1
            tokens += len(item["ids"])
    temporary.replace(target)
    atomic_write_bytes(paths.set_meta(tag, name), json.dumps({"n": n, "tokens": tokens, **columns}).encode("utf-8"))
    return {"n": n, "tokens": tokens}


def read_meta(paths: D1Paths, tag: str, name: str) -> dict[str, Any]:
    return json.loads(paths.set_meta(tag, name).read_text(encoding="utf-8"))


def followup_length(tokenizer: Any, record: Transcript, longest_wording: str) -> int:
    return len(encode_ids(tokenizer, followup_text(tokenizer, record.messages, record.tools, longest_wording))) + 2


def filter_by_length(records: Sequence[Transcript], tokenizer: Any, wordings: Sequence[str], max_tokens: int) -> tuple[list[Transcript], int]:
    """Same rule for both classes. The longest wording decides, so a kept transcript fits with any wording."""
    longest = max(wordings, key=len)
    kept = [record for record in records if followup_length(tokenizer, record, longest) <= max_tokens]
    return kept, len(records) - len(kept)


def scored_items(
    tokenizer: Any,
    records: Sequence[Transcript],
    answers: AnswerTokens,
    cfg: D1Config,
    salt: str,
    kinds: Sequence[str],
    sham_rows: int = 0,
) -> Iterable[dict[str, Any]]:
    """The scored prompts for a list of transcripts, in a fixed order."""
    held_out = assign_wordings(records, len(cfg.wordings_held_out), cfg.seed, f"{salt}|held_out")
    train = assign_wordings(records, len(cfg.wordings_train), cfg.seed, f"{salt}|train")
    english_left = sham_rows
    for t, record in enumerate(records):
        common = {"t": t, "y": record.y, "family": record.family, "group": record.group, "key": record.key}
        for kind in kinds:
            if kind == "base":
                ids = base_read_ids(tokenizer, record.messages, record.tools)
                yield {**common, "kind": kind, "wording": None, "ids": ids, "read": len(ids) - 1, "answer_pos": -1}
                continue
            if kind == "sham":
                if english_left <= 0 or record.language != "en":
                    continue
                english_left -= 1
                wording, index = cfg.wording_sham, None
            elif kind.startswith("train_"):
                index = train[record.key]
                wording = cfg.wordings_train[index]
            else:
                index = held_out[record.key]
                wording = cfg.wordings_held_out[index]
            rendered = followup_text(tokenizer, record.messages, record.tools, wording)
            ids, answer_pos = prefilled_ids(tokenizer, rendered, ANSWER_FOR_KIND[kind], answers)
            yield {**common, "kind": kind, "wording": index, "ids": ids, "read": len(ids) - 1, "answer_pos": answer_pos}


def choose_native(rows: Sequence[NativeRow], total: int, seed: int, salt: str) -> list[NativeRow]:
    """Up to `total` rows, class-balanced within each family, with equal shares until a family runs out."""
    by_family: dict[str, list[NativeRow]] = defaultdict(list)
    for row in rows:
        by_family[row.family].append(row)
    per_family: dict[str, list[NativeRow]] = {}
    capacity = {}
    for family, members in sorted(by_family.items()):
        counts = Counter(row.label for row in members)
        capacity[family] = 2 * min(counts[0], counts[1])
    share: dict[str, int] = {family: 0 for family in capacity}
    left = total
    open_families = {family for family, room in capacity.items() if room > 0}
    while left >= 2 * len(open_families) and open_families:
        step = max(2, (left // len(open_families)) // 2 * 2)
        for family in sorted(open_families):
            grant = min(step, capacity[family] - share[family])
            share[family] += grant
            left -= grant
        open_families = {family for family in open_families if share[family] < capacity[family]}
    for family, members in sorted(by_family.items()):
        if share[family]:
            per_family[family] = select_balanced(members, share[family] // 2, seed, lambda row: row.label, salt)
    chosen = [row for family in sorted(per_family) for row in per_family[family]]
    return sorted(chosen, key=lambda row: stable_hash(str(seed), salt, "order", row.row))


def native_items(tokenizer: Any, rows: Sequence[NativeRow], max_tokens: int) -> Iterable[dict[str, Any]]:
    for t, row in enumerate(rows):
        ids = native_ids(tokenizer, row.messages, row.tools, row.completion)
        yield {"t": t, "kind": "native", "y": row.label, "family": row.family, "group": row.group, "wording": None, "key": row.row, "ids": ids, "read": len(ids) - 1, "answer_pos": -1}


def build_rung_plan(cfg: D1Config, paths: D1Paths, tag: str, tokenizer: Any) -> dict[str, Any]:
    """Everything up to and including test_g1. Deterministic, so a rerun writes the same files."""
    sizes = cfg.sizes
    where = partition_of(cfg.partition)
    transcripts = read_transcripts(paths.transcripts)
    native = read_native(paths.native)
    wordings = [*cfg.wordings_held_out, *cfg.wordings_train, cfg.wording_sham]
    # Native rows of probe-train families. Empty completions cannot be read, and long rows are dropped.
    usable_native = []
    native_drops = Counter()
    for row in native:
        if not row.completion:
            native_drops["empty_completion"] += 1
            continue
        if len(native_ids(tokenizer, row.messages, row.tools, row.completion)) > sizes.max_tokens:
            native_drops["too_long"] += 1
            continue
        usable_native.append(row)
    # Three parts with no source prompt in common. Calibration is set aside first, or fit could use up a small family.
    rest, calibration_pool = split_by_prompt(usable_native, 0.2, cfg.seed, "cal")
    calibration_prompts = {row.prompt for row in calibration_pool}
    rest, held = split_by_prompt(rest, 0.1, cfg.seed, "val")
    val_rows = choose_native(held, sizes.probe_val_rows, cfg.seed, "val")
    fit_rows = choose_native(rest, sizes.probe_fit_rows, cfg.seed, "fit")
    if not fit_rows or not val_rows:
        raise ValueError("No rows to fit probes on")

    kept, length_drops = filter_by_length(transcripts, tokenizer, wordings, sizes.max_tokens)
    by_partition: dict[str, dict[str, list[Transcript]]] = defaultdict(lambda: defaultdict(list))
    for record in kept:
        by_partition[where[record.family]][record.family].append(record)

    # Calibration never shares a source prompt with the probe's training rows, so the threshold is not fitted on them.
    # A family with only one Y class gives nothing here, because balance within a family is impossible.
    calibration: list[Transcript] = []
    for family, records in sorted(by_partition["probe_train"].items()):
        free = [record for record in records if record.prompt in calibration_prompts]
        calibration.extend(select_balanced(free, sizes.calibration_per_class_per_family, cfg.seed, lambda record: record.y, f"cal|{family}"))
    test_available = {}
    test_g1: list[Transcript] = []
    for family in cfg.partition["test"]:
        records = by_partition["test"].get(family, [])
        full = select_balanced(records, 10**9, cfg.seed, lambda record: record.y, f"test|{family}")
        test_available[family] = len(full) // 2
        test_g1.extend(select_balanced(records, sizes.g1_per_class_per_family, cfg.seed, lambda record: record.y, f"test|{family}"))
    empty = [family for family, available in test_available.items() if available == 0]
    if empty:
        raise ValueError(f"Test families with no usable transcripts of both classes: {empty}")
    if not calibration:
        raise ValueError("No calibration transcripts, so no sign or threshold can be fixed")
    answers = resolve_answers(tokenizer, followup_text(tokenizer, test_g1[0].messages, test_g1[0].tools, cfg.wordings_held_out[0]))
    report = {
        "answer_tokens": {"yes": answers.yes, "no": answers.no, "dot": answers.dot},
        "native_drops": dict(native_drops),
        "transcripts_dropped_too_long": length_drops,
        "test_available_per_class": test_available,
        "fit": write_set(paths, tag, "fit", native_items(tokenizer, fit_rows, sizes.max_tokens)),
        "val": write_set(paths, tag, "val", native_items(tokenizer, val_rows, sizes.max_tokens)),
        "cal": write_set(paths, tag, "cal", scored_items(tokenizer, calibration, answers, cfg, "cal", ("yes", "no", "base"))),
        "test_g1": write_set(paths, tag, "test_g1", scored_items(tokenizer, test_g1, answers, cfg, "test", ("yes", "no", "base"))),
        "facts": write_set(paths, tag, "facts", fact_items(tokenizer, cfg)),
        "fit_families": dict(Counter(row.family for row in fit_rows)),
        "fit_labels": dict(Counter(str(row.label) for row in fit_rows)),
        "cal_families": dict(Counter(record.family for record in calibration)),
    }
    if report["fit"]["n"] < 2 * cfg.probes.batch_size:
        raise ValueError(f"Only {report['fit']['n']} rows to fit probes on")
    atomic_write_bytes(paths.plan_dir(tag) / "plan.json", json.dumps(report, indent=1).encode("utf-8"))
    build_training_rows(cfg, paths, tag, tokenizer, by_partition["adapter_train"])
    return report


def build_test_set(cfg: D1Config, paths: D1Paths, tag: str, tokenizer: Any, n_per_class: int) -> dict[str, int]:
    """The final test set, at the size chosen from G1. Selection order is the same as test_g1's."""
    where = partition_of(cfg.partition)
    wordings = [*cfg.wordings_held_out, *cfg.wordings_train, cfg.wording_sham]
    transcripts = [record for record in read_transcripts(paths.transcripts) if where[record.family] == "test"]
    kept, _ = filter_by_length(transcripts, tokenizer, wordings, cfg.sizes.max_tokens)
    by_family: dict[str, list[Transcript]] = defaultdict(list)
    for record in kept:
        by_family[record.family].append(record)
    chosen: list[Transcript] = []
    for family in cfg.partition["test"]:
        chosen.extend(select_balanced(by_family.get(family, []), n_per_class, cfg.seed, lambda record: record.y, f"test|{family}"))
    answers = resolve_answers(tokenizer, followup_text(tokenizer, chosen[0].messages, chosen[0].tools, cfg.wordings_held_out[0]))
    kinds = ("yes", "no", "base", "train_yes", "train_no", "sham")
    return write_set(paths, tag, "test", scored_items(tokenizer, chosen, answers, cfg, "test", kinds, cfg.sizes.sham_check_rows))


def fact_items(tokenizer: Any, cfg: D1Config) -> Iterable[dict[str, Any]]:
    for t, fact in enumerate(cfg.facts_eval):
        ids = encode_ids(tokenizer, fact_prompt_text(tokenizer, fact["q"], cfg.facts_instruction))
        yield {"t": t, "kind": "fact", "y": int(fact["a"] == YES), "family": "facts", "group": str(t), "wording": None, "key": str(t), "ids": ids, "read": len(ids) - 1, "answer_pos": len(ids) - 1}


def build_training_rows(cfg: D1Config, paths: D1Paths, tag: str, tokenizer: Any, adapter_families: dict[str, list[Transcript]]) -> dict[str, Any]:
    """Which rows adapters train on: the same list for all conditions and seeds. English only, since SHAM's question says English."""
    training = cfg.training
    longest = max(cfg.wordings_train, key=len)
    base_keys: list[str] = []
    share = max(1, training.n_base_rows // max(1, 2 * len(adapter_families)))
    for family, records in sorted(adapter_families.items()):
        english = [record for record in records if record.language == "en" and followup_length(tokenizer, record, longest) + 1 <= training.max_tokens]
        base_keys.extend(record.key for record in select_balanced(english, share, cfg.seed, lambda record: record.y, f"adapter|{family}"))
    ordinary: list[str] = []
    rows = [row for row in read_native(paths.ordinary) if row.language == "en"]
    by_family: dict[str, list[NativeRow]] = defaultdict(list)
    for row in rows:
        if len(encode_ids(tokenizer, render_conversation(tokenizer, row.messages, row.tools))) + training.ordinary_answer_tokens <= training.max_tokens and row.completion.strip():
            by_family[row.family].append(row)
    per_family = max(1, training.n_ordinary_rows // max(1, 2 * len(by_family)))
    for family, members in sorted(by_family.items()):
        members.sort(key=lambda row: stable_hash(str(cfg.seed), "ordinary", row.row))
        ordinary.extend(row.row for row in members[:per_family])
    # Half of ordinary chat is the same conversations answering the base turn itself: honest ones only, never teaching a lie.
    replies: list[str] = []
    for family, records in sorted(adapter_families.items()):
        honest = [record for record in records if record.y == 0 and record.language == "en" and followup_length(tokenizer, record, longest) <= training.max_tokens]
        honest.sort(key=lambda record: stable_hash(str(cfg.seed), "reply", record.key))
        replies.extend(record.key for record in honest[:per_family])
    plan = {"base_keys": base_keys, "ordinary_rows": ordinary, "reply_keys": replies}
    if not base_keys or not (ordinary or replies):
        raise ValueError("No rows to train adapters on")
    atomic_write_bytes(paths.training_rows(tag), json.dumps(plan).encode("utf-8"))
    return {"base": len(base_keys), "ordinary": len(ordinary), "replies": len(replies)}
