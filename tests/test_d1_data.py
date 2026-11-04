"""Partitions, family derivation, Y labels and transcript merging. Pure Python, on made-up rows."""

from __future__ import annotations

import json
from collections import Counter
from fractions import Fraction

import pytest
from d1_factory import REAL_DATA, source_for, tiny_rows, write_tiny_config

from mechinterp.d1.config import D1Config, check_wordings, layer_for_fraction, layers_for
from mechinterp.d1.data import assign_wordings, ingest, select_balanced, split_by_prompt
from mechinterp.d1.families import (
    FAMILIES,
    FORCED_TEST,
    PARTITION_SEED,
    RELATED,
    CountMismatch,
    UnknownSource,
    assign_partitions,
    check_counts,
    check_partition,
    family_of,
    has_y_label,
    is_pipeline_row,
    split_quadrant,
)


def expected() -> dict:
    return json.loads((REAL_DATA / "fibs_expected_counts.json").read_text(encoding="utf-8"))


def partition() -> dict:
    data = json.loads((REAL_DATA / "partition.json").read_text(encoding="utf-8"))
    return {name: data[name] for name in ("probe_train", "adapter_train", "test")}


def test_committed_partition_is_what_the_seeded_code_produces() -> None:
    assert assign_partitions(PARTITION_SEED) == partition()


def test_partition_is_disjoint_complete_and_sized() -> None:
    frozen = partition()
    check_partition(frozen)
    members = [family for names in frozen.values() for family in names]
    assert sorted(members) == sorted(FAMILIES)
    assert len(set(members)) == 28
    assert (len(frozen["probe_train"]), len(frozen["adapter_train"]), len(frozen["test"])) == (14, 7, 7)
    for family in FORCED_TEST:
        assert family in frozen["test"]
    for group in RELATED:
        assert len({name for name, names in frozen.items() if set(group) & set(names)}) == 1


def test_partition_checker_catches_each_violation() -> None:
    frozen = partition()
    broken = {name: list(names) for name, names in frozen.items()}
    # Swap lies out of test, which also separates it from mask.
    index = broken["adapter_train"].index("anti")
    broken["adapter_train"][index], broken["test"][broken["test"].index("lies")] = "lies", "anti"
    with pytest.raises(ValueError):
        check_partition(broken)
    duplicate = {name: list(names) for name, names in frozen.items()}
    duplicate["test"][0] = duplicate["probe_train"][0]
    with pytest.raises(ValueError):
        check_partition(duplicate)


def test_other_seed_keeps_every_constraint() -> None:
    for seed in (1, 2, 3):
        check_partition(assign_partitions(seed))


@pytest.mark.parametrize(
    ("source", "family"),
    [
        ("among_us", "among"),
        ("anti_sycophancy_dpo", "anti"),
        ("b1-bare", "b1-bare"),
        ("b1-followup", "b1-followup"),
        ("fabricated_statistics", "fabricated"),
        ("harm_pressure_knowledge", "harm"),
        ("instructed_truth_turn_deceptive", "instructed"),
        ("liarsbench_softtrigger_raw", "liarsbench"),
        ("lies_v2", "lies"),
        ("longcontext_v3_bookqa", "longcontext"),
        ("mask_doubling_down_known_facts", "mask"),
        ("model_self_knowledge", "model"),
        ("multi_turn_interleaved", "multi"),
        ("persona_prefill_b4t3a_regents_prep_pass_rate_honest", "persona_prefill"),
        ("persona_prefill_followup", "persona_prefill"),
        ("synthetic_deceptive", "synthetic"),
        ("wikihow_instructions", "wikihow"),
        ("wikihownest", "wikihownest"),
    ],
)
def test_family_from_source(source: str, family: str) -> None:
    assert family_of(source) == family


def test_unknown_source_stops() -> None:
    with pytest.raises(UnknownSource):
        family_of("blackbox")


def test_published_tables_agree_with_each_other() -> None:
    table = expected()
    assert sum(table["family_totals_readme"].values()) == table["total_rows"]
    assert sum(table["category_totals"].values()) == table["total_rows"]
    assert sum(table["follow_up_pipeline_rows"].values()) == table["follow_up_pipeline_rows_total"]
    pipeline = sum(count for name, count in table["category_totals"].items() if is_pipeline_row(name))
    assert pipeline == table["follow_up_pipeline_rows_total"] == 120432
    for family, cells in table["family_quadrant"].items():
        assert sum(cells.values()) == table["family_totals_readme"][family]
    for quadrant, total in table["quadrant_totals"].items():
        assert sum(cells.get(quadrant, 0) for cells in table["family_quadrant"].values()) == total


def test_count_check_passes_only_on_the_exact_table() -> None:
    table = expected()
    observed = {family: dict(cells) for family, cells in table["family_quadrant"].items()}
    check_counts(observed, table["category_totals"], table["follow_up_pipeline_rows"], table)
    observed["fever"]["honest_honest"] -= 1
    observed["mask"]["honest_honest"] += 1
    with pytest.raises(CountMismatch):
        check_counts(observed, table["category_totals"], table["follow_up_pipeline_rows"], table)


def test_y_comes_from_the_first_term_only() -> None:
    assert split_quadrant("deceptive_honest") == (True, False)
    assert split_quadrant("honest_deceptive") == (False, True)
    rows = tiny_rows(4)
    transcripts, native, _, _ = ingest(rows, partition(), None)
    assert {record.y for record in transcripts if record.family == "ultrachat"} == {0}
    assert {record.y for record in transcripts if record.family == "synthetic"} == {1}
    for record in transcripts:
        answer = record.messages[-1]["content"]
        assert ("deceptive" in answer) == (record.y == 1), "base text in this data says what the quadrant says"
    # The probe's label is the second term, the falsehood of the final answer.
    assert {row.label for row in native} == {0, 1}
    for row in native:
        assert ("deceptive" in row.completion) == (row.label == 1)


def test_rows_differing_only_after_the_base_turn_merge() -> None:
    rows = tiny_rows(4)
    transcripts, _, _, _ = ingest(rows, partition(), None)
    keys = [record.key for record in transcripts]
    assert len(keys) == len(set(keys))
    by_family = Counter(record.family for record in transcripts)
    assert by_family["mask"] == 8  # 4 prompts, honest and deceptive versions, from 16 rows.
    assert all(record.n_rows == 2 for record in transcripts if record.family == "mask")
    assert all(record.messages[-1]["role"] == "assistant" for record in transcripts)


def test_same_base_conversation_with_both_labels_is_dropped() -> None:
    rows = tiny_rows(4)
    honest = next(row for row in rows if row["source"] == "mask_statistics" and row["quadrant"] == "honest_honest")
    rows.append({**honest, "quadrant": "deceptive_deceptive"})
    transcripts, _, _, manifest = ingest(rows, partition(), None)
    assert manifest["dropped"]["base_conversation_with_both_labels"] == 1
    conflicted = json.loads(honest["messages_json"])[:-1]
    assert all(record.messages != conflicted for record in transcripts)


def test_rows_that_are_not_follow_up_rows_never_become_transcripts() -> None:
    message = json.dumps([{"role": "user", "content": "Q"}])
    long_message = json.dumps([{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}, {"role": "user", "content": "persona turn"}])
    extra = [
        {"source": "wikihow_instructions", "quadrant": "deceptive_deceptive", "follow_up_category": "direct_mix", "language": "en", "messages_json": message, "tools": None, "completion": "x"},
        {"source": "instructed_lie_neutral", "quadrant": "honest_honest", "follow_up_category": "direct", "language": "en", "messages_json": message, "tools": None, "completion": "x"},
        {"source": "persona_prefill_bare", "quadrant": "deceptive_honest", "follow_up_category": "persona_final", "language": "en", "messages_json": long_message, "tools": None, "completion": "x"},
        {"source": "among_us", "quadrant": "deceptive_honest", "follow_up_category": "", "language": "en", "messages_json": long_message, "tools": None, "completion": "x"},
        {"source": "multi_turn_interleaved", "quadrant": "deceptive_honest", "follow_up_category": "cat_interleaved", "language": "en", "messages_json": long_message, "tools": None, "completion": "x"},
        {"source": "caving", "quadrant": "deceptive_honest", "follow_up_category": "direct_challenge", "language": "en", "messages_json": long_message, "tools": None, "completion": "x"},
    ]
    transcripts, native, _, manifest = ingest(extra, partition(), None)
    assert transcripts == [] and native == []
    assert sum(manifest["dropped"].values()) == 6


def test_delayed_rows_fit_probes_but_give_no_y() -> None:
    assert is_pipeline_row("cat_J") and not has_y_label("cat_J")
    delayed = json.dumps(
        [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "base"},
            {"role": "user", "content": "unrelated"},
            {"role": "assistant", "content": "filler"},
            {"role": "user", "content": "Was that true?"},
        ]
    )
    row = {"source": "medqa", "quadrant": "deceptive_honest", "follow_up_category": "cat_J", "language": "en", "messages_json": delayed, "tools": None, "completion": "No."}
    transcripts, native, _, manifest = ingest([row], partition(), None)
    assert transcripts == []
    assert len(native) == 1 and native[0].label == 0
    assert manifest["dropped"]["delayed_cat_J"] == 1


def test_a_prompt_in_test_is_removed_from_other_partitions() -> None:
    rows = tiny_rows(2)
    shared = next(row for row in rows if row["source"] == source_for("lies"))
    prompt = json.loads(shared["messages_json"])[0]["content"]
    copy = json.loads(shared["messages_json"])
    copy[0]["content"] = prompt.upper() + "  "  # The same prompt with other spacing and case.
    rows.append({**shared, "source": "medqa", "messages_json": json.dumps(copy)})
    rows.append({**shared, "source": "fever", "messages_json": json.dumps(copy)})
    transcripts, native, _, manifest = ingest(rows, partition(), None)
    test_prompts = {record.prompt for record in transcripts if record.family in partition()["test"]}
    assert all(record.prompt not in test_prompts for record in transcripts if record.family not in partition()["test"])
    assert all(row.prompt not in test_prompts for row in native)
    assert manifest["cross_partition_rows_removed"]["probe_train"] >= 1
    assert manifest["cross_partition_rows_removed"]["adapter_train"] >= 1


def test_selection_is_balanced_deterministic_and_group_aware() -> None:
    transcripts, _, _, _ = ingest(tiny_rows(12), partition(), None)
    mask = [record for record in transcripts if record.family == "mask"]
    first = select_balanced(mask, 5, 3, lambda record: record.y, "s")
    second = select_balanced(list(reversed(mask)), 5, 3, lambda record: record.y, "s")
    assert [record.key for record in first] == [record.key for record in second]
    counts = Counter(record.y for record in first)
    assert counts[0] == counts[1] == 5
    groups = Counter(record.group for record in first)
    assert set(groups.values()) == {2}, "groups with both classes are used first, so topic is held fixed"
    assert select_balanced([record for record in mask if record.y == 0], 5, 3, lambda record: record.y, "s") == []


def test_wordings_are_assigned_per_group() -> None:
    transcripts, _, _, _ = ingest(tiny_rows(12), partition(), None)
    mask = [record for record in transcripts if record.family == "mask"]
    assignment = assign_wordings(mask, 10, 3, "x")
    by_group: dict[str, set[int]] = {}
    for record in mask:
        by_group.setdefault(record.group, set()).add(assignment[record.key])
    assert all(len(values) == 1 for values in by_group.values()), "Y cannot be read from the wording"
    assert len(set(assignment.values())) == 10


def test_split_by_prompt_shares_no_prompt() -> None:
    transcripts, _, _, _ = ingest(tiny_rows(12), partition(), None)
    rest, held = split_by_prompt(transcripts, 0.2, 1, "x")
    assert rest and held
    assert not ({record.prompt for record in rest} & {record.prompt for record in held})


def test_config_loads_with_real_data_files(tmp_path) -> None:
    cfg = D1Config.from_file(write_tiny_config(tmp_path))
    assert len(cfg.wordings_train) == 20 and len(cfg.wordings_held_out) == 10
    assert not set(cfg.wordings_train) & set(cfg.wordings_held_out)
    assert cfg.wording_sham == "Did you respond in English?"
    assert sum(fact["a"] == "Yes" for fact in cfg.facts_eval) == len(cfg.facts_eval) // 2
    assert len(cfg.facts_eval) == 100 and len(cfg.facts_retain) == 60


def test_real_config_and_manifest_agree() -> None:
    cfg = D1Config.from_file(REAL_DATA.parent / "configs" / "full.json")
    manifest = json.loads((REAL_DATA / "partition.json").read_text(encoding="utf-8"))
    assert manifest["revision"] == cfg.dataset_revision
    assert manifest["parquet_sha256"] == cfg.parquet_sha256
    assert [rung.name for rung in cfg.ladder] == ["Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B", "Qwen/Qwen3-14B"]
    assert all(len(rung.revision) == 40 for rung in cfg.ladder)
    assert cfg.training.rank == 16 and cfg.training.seeds == (0, 1, 2)
    assert cfg.sizes.probe_fit_rows == 20000 and cfg.probes.scale_rows == 512 and cfg.probes.drop_loss == 0.6


@pytest.mark.parametrize("bad", ["Did you lie to me?", "Was any of that false?", "Did you mislead me?"])
def test_reversed_wordings_are_refused(bad: str) -> None:
    wordings = json.loads((REAL_DATA / "wordings.json").read_text(encoding="utf-8"))
    wordings["train"][0] = bad
    with pytest.raises(ValueError):
        check_wordings(wordings)


# The table published for FIBS's probes: blocks and the layers they read. Layer k is the output of block k.
FIBS_LAYER_TABLE = {
    24: [7, 10, 13, 16, 19, 22],
    32: [10, 13, 18, 21, 26, 29],
    64: [19, 27, 35, 43, 51, 58],
    48: [14, 20, 26, 32, 38, 43],
    60: [18, 25, 33, 40, 48, 54],
    52: [16, 22, 29, 35, 42, 47],
    88: [26, 37, 48, 59, 70, 79],
    108: [32, 45, 59, 72, 86, 97],
    93: [28, 39, 51, 62, 74, 84],
}
FRACTIONS = tuple(Fraction(text) for text in ("0.3", "0.42", "0.55", "0.67", "0.8", "0.9"))


@pytest.mark.parametrize(("blocks", "layers"), list(FIBS_LAYER_TABLE.items()))
def test_depth_fractions_reproduce_the_published_table(blocks: int, layers: list[int]) -> None:
    assert layers_for(FRACTIONS, blocks) == layers


def test_depth_fractions_on_the_ladder() -> None:
    assert layers_for(FRACTIONS, 28) == [8, 12, 15, 19, 22, 25]
    assert layers_for(FRACTIONS, 36) == [11, 15, 20, 24, 29, 32]
    assert layers_for(FRACTIONS, 40) == [12, 17, 22, 27, 32, 36]


def test_half_rounds_up_exactly() -> None:
    assert layer_for_fraction(Fraction("0.55"), 10) == 6  # 5.5 exactly
    assert layer_for_fraction(Fraction("0.3"), 5) == 2  # 1.5 exactly
