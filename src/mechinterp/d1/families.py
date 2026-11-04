"""FIBS families: deriving them from `source`, which rows carry a base-turn label, and the frozen partition.

Standard library only, so that a manifest can be checked without torch, numpy or the data.

Why each rule exists is in the README's D1 section. In short:
- `family_of` is `source` up to the first underscore, except persona_prefill. Checked at run time against the exact
  family by quadrant table in the dataset's provenance/BUILD.json, and any difference stops the run.
- The README says the first quadrant term labels the base turn. The FIBS paper (N.1) limits that to the 120,432 rows
  of its follow-up pipeline. On games it is the speaker's side, on persona rows the persona, on Instructed's direct
  rows the instruction, and in two-quadrant families it repeats the label. So only pipeline rows are used in D1.
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable, Mapping, Sequence

FAMILIES = (
    "among", "anti", "b1-bare", "b1-followup", "caving", "doluschat", "fabricated", "fever", "harm", "instructed",
    "liarsbench", "lies", "longcontext", "mask", "medqa", "model", "multi", "negotiation", "original", "persona_prefill",
    "sandbagging", "sycophancy", "synthetic", "truthfulqa", "ultrachat", "werewolf", "wikihow", "wikihownest",
)  # fmt: skip

# The follow-up pipeline's categories. cat_interleaved and caving's categories are not in it: the paper's 120,432 is
# exactly the sum of these, and it reconciles with the family counts in fibs_expected_counts.json.
PIPELINE_CATEGORIES = frozenset(
    {"cat_A", "cat_Af", "cat_Af_bare", "cat_B", "cat_C", "cat_Cf", "cat_D", "cat_E", "cat_F", "cat_G", "cat_H", "cat_Hf", "cat_I", "cat_J"}
)
# "One category first delays the question by adding one to three unrelated user questions and responses" (N.2.1). In
# these the last assistant turn is not the base turn, and the base turn's position is not stored, so "was your last
# response truthful?" would ask about filler. Usable for fitting probes, which read the final answer, but not for Y.
DELAYED_CATEGORY = "cat_J"
# Native follow-ups unrelated to truthfulness, used as ordinary chat in adapter training (specifics, implications, go on).
ORDINARY_CATEGORIES = frozenset({"cat_B", "cat_E", "cat_G", "cat_I"})

QUADRANTS = ("honest_honest", "honest_deceptive", "deceptive_honest", "deceptive_deceptive")
PARTITION_SIZES = {"probe_train": 14, "adapter_train": 7, "test": 7}

# Lines of work that must share a partition. Evidence, paper section in brackets:
# lies reuses MASK prompts and MMLU questions [N.3.11]; wikihow and wikihownest [prereg]; synthetic is Corrupted
# UltraChat, built from the same conversations as ultrachat [N.3.1]; b1-followup uses b1-bare's episodes [N.3.19];
# harm and liarsbench are settings of the same benchmark [N.3.3].
RELATED = (
    ("lies", "mask", "sandbagging"),
    ("wikihow", "wikihownest"),
    ("ultrachat", "synthetic"),
    ("b1-bare", "b1-followup"),
    ("liarsbench", "harm"),
)
# No pipeline rows at all, so nothing to score or to train on. Partitions count families, so these must go where they
# cost least, and a test or adapter family with no rows would silently shrink that partition.
NO_PIPELINE_ROWS = ("among", "b1-bare", "caving", "multi", "negotiation", "werewolf", "wikihow")
# ultrachat rows are all Y=0 and synthetic rows all Y=1, so within-family AUROC is undefined. longcontext has 43 pipeline
# rows of at least 5k tokens and persona_prefill 588, too few to carry a test or adapter family.
UNUSABLE_FOR_Y = ("ultrachat", "synthetic", "longcontext", "persona_prefill")
FORCED_TEST = ("sandbagging", "lies")
PARTITION_SEED = 20261104


class UnknownSource(ValueError):
    """A source value maps to no known family, so the derivation is wrong for this revision."""


class CountMismatch(ValueError):
    """The data differs from the published counts, so a derivation cannot be trusted."""


def family_of(source: str) -> str:
    if source.startswith("persona_prefill"):
        return "persona_prefill"
    family = source.split("_", 1)[0]
    if family not in FAMILIES:
        raise UnknownSource(f"source {source!r} gives unknown family {family!r}")
    return family


def split_quadrant(quadrant: str) -> tuple[bool, bool]:
    """(base turn deceptive, final answer deceptive), the quadrant's two terms."""
    if quadrant not in QUADRANTS:
        raise ValueError(f"Unknown quadrant {quadrant!r}")
    first, second = quadrant.split("_")
    return first == "deceptive", second == "deceptive"


def is_pipeline_row(category: str | None) -> bool:
    return category in PIPELINE_CATEGORIES


def has_y_label(category: str | None) -> bool:
    """Whether the first quadrant term labels the last assistant turn before the final user message."""
    return is_pipeline_row(category) and category != DELAYED_CATEGORY


def check_counts(
    observed: Mapping[str, Mapping[str, int]],
    observed_categories: Mapping[str, int],
    observed_pipeline: Mapping[str, int],
    expected: Mapping[str, object],
) -> None:
    """Exact comparison with the published tables. Any difference raises."""
    problems: list[str] = []
    family_quadrant = expected["family_quadrant"]
    for family in sorted(set(family_quadrant) | set(observed)):  # type: ignore[arg-type]
        want = family_quadrant.get(family, {})  # type: ignore[union-attr]
        got = observed.get(family, {})
        for quadrant in QUADRANTS:
            if int(want.get(quadrant, 0)) != int(got.get(quadrant, 0)):
                problems.append(f"{family}/{quadrant}: expected {want.get(quadrant, 0)}, found {got.get(quadrant, 0)}")
    for family, want_total in expected["family_totals_readme"].items():  # type: ignore[union-attr]
        got_total = sum(observed.get(family, {}).values())
        if got_total != want_total:
            problems.append(f"{family}: README says {want_total}, found {got_total}")
    categories = expected["category_totals"]
    for name in sorted(set(categories) | set(observed_categories)):  # type: ignore[arg-type]
        if int(categories.get(name, 0)) != int(observed_categories.get(name, 0)):  # type: ignore[union-attr]
            problems.append(f"category {name!r}: expected {categories.get(name, 0)}, found {observed_categories.get(name, 0)}")  # type: ignore[union-attr]
    pipeline = expected["follow_up_pipeline_rows"]
    for name in sorted(set(pipeline) | set(observed_pipeline)):  # type: ignore[arg-type]
        if int(pipeline.get(name, 0)) != int(observed_pipeline.get(name, 0)):  # type: ignore[union-attr]
            problems.append(f"pipeline rows of {name}: expected {pipeline.get(name, 0)}, found {observed_pipeline.get(name, 0)}")  # type: ignore[union-attr]
    if problems:
        raise CountMismatch("Family derivation does not match the published counts:\n  " + "\n  ".join(problems))


def _units() -> list[tuple[str, ...]]:
    grouped = {name for group in RELATED for name in group}
    return [*RELATED, *[(name,) for name in FAMILIES if name not in grouped]]


def assign_partitions(seed: int = PARTITION_SEED) -> dict[str, list[str]]:
    """14, 7 and 7 families, deterministically from the seed. Related families stay together.

    A unit holding a family that cannot be scored goes to probe-train, as does anything left over. The unit holding
    the forced test families goes to test. The rest are ordered by a hash of the seed and fill test, then adapter-train,
    then probe-train, each only if the whole unit fits.
    """
    forced_probe = set(NO_PIPELINE_ROWS) | set(UNUSABLE_FOR_Y)
    out: dict[str, list[str]] = {name: [] for name in PARTITION_SIZES}
    room = dict(PARTITION_SIZES)
    free: list[tuple[str, ...]] = []

    def place(partition: str, unit: tuple[str, ...]) -> None:
        if room[partition] < len(unit):
            raise ValueError(f"{unit} does not fit in {partition}")
        out[partition].extend(unit)
        room[partition] -= len(unit)

    for unit in _units():
        if set(unit) & set(FORCED_TEST):
            place("test", unit)
        elif set(unit) & forced_probe:
            place("probe_train", unit)
        else:
            free.append(unit)
    free.sort(key=lambda unit: hashlib.sha256(f"{seed}|{'+'.join(unit)}".encode("utf-8")).hexdigest())
    for unit in free:
        for partition in ("test", "adapter_train", "probe_train"):
            if room[partition] >= len(unit):
                place(partition, unit)
                break
        else:
            # A pair arriving last can find one seat in each of two partitions. Move one family to free a whole partition.
            moved = out["adapter_train"].pop()
            room["adapter_train"] += 1
            place("probe_train", (moved,))
            place("adapter_train", unit)
    return {name: sorted(members) for name, members in out.items()}


def check_partition(partition: Mapping[str, Sequence[str]]) -> None:
    """Disjoint, complete, the right sizes, related families together, nothing unusable where it must be scored."""
    seen: dict[str, str] = {}
    for name, members in partition.items():
        if len(members) != PARTITION_SIZES[name]:
            raise ValueError(f"{name} has {len(members)} families, the prereg says {PARTITION_SIZES[name]}")
        for member in members:
            if member in seen:
                raise ValueError(f"{member} is in both {seen[member]} and {name}")
            seen[member] = name
    if set(seen) != set(FAMILIES):
        raise ValueError(f"Partition does not cover the families: {sorted(set(FAMILIES) ^ set(seen))}")
    for group in RELATED:
        where = {seen[member] for member in group}
        if len(where) != 1:
            raise ValueError(f"Related families {group} are split across {sorted(where)}")
    for member in FORCED_TEST:
        if seen[member] != "test":
            raise ValueError(f"{member} must be in test")
    for member in (*NO_PIPELINE_ROWS, *UNUSABLE_FOR_Y):
        if seen[member] != "probe_train":
            raise ValueError(f"{member} cannot be scored, so it must be in probe_train")


def partition_of(partition: Mapping[str, Iterable[str]]) -> dict[str, str]:
    return {member: name for name, members in partition.items() for member in members}


_SPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """For finding the same prompt across families despite spacing and case."""
    return _SPACE.sub(" ", text).strip().casefold()
