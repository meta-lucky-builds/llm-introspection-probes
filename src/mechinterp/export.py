"""Compact export of results, small enough to leave Colab through a cell's printed output."""

from __future__ import annotations

import base64
import gzip
import json
from collections import defaultdict
from typing import Any, Iterable

from .design import PHASE_SCREEN, PHASE_TEST

BEGIN_MARKER = "=====BEGIN E1 EXPORT====="
END_MARKER = "=====END E1 EXPORT====="
LINE_WIDTH = 120
CELL_COLUMNS = [
    "task",
    "condition",
    "concept",
    "cluster",
    "n",
    "L",
    "hit",
    "clean_hit",
    "fact_correct",
    "label_mass",
    "none_mass",
]
CLEAN_COLUMNS = ["task", "cluster", "n", "label_mass", "none_mass", "fact_correct", "none_answers"]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _mean_or_none(items: list[dict[str, Any]], name: str, digits: int) -> float | None:
    values = [item[name] for item in items if item.get(name) is not None]
    return round(_mean(values), digits) if values else None


def build_export(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Per (task, condition, concept, cluster) cell means, plus what the analysis needs."""
    records = list(records)
    manifests = [record for record in records if record.get("kind") == "manifest"]
    manifest = manifests[-1] if manifests else {}
    # Every distinct identity is reported. The session guard allows one, and analysis checks that.
    environments: list[dict[str, Any]] = []
    for item in manifests:
        identity = item.get("identity")
        if identity is not None and identity not in environments:
            environments.append(identity)

    def single(kind: str) -> dict[str, Any] | None:
        found = [record for record in records if record.get("kind") == kind]
        return found[-1] if found else None

    cells: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    position_counts: dict[str, list[int]] = {}
    screen: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for record in records:
        if record.get("kind") != "trial":
            continue
        if record["phase"] == PHASE_TEST:
            cells[(record["task"], record["condition"], record["concept"], record["cluster"])].append(record)
            counts = position_counts.setdefault(f"{record['task']}|{record['condition']}", [0] * 10)
            counts[record["position"]] += 1
        elif record["phase"] == PHASE_SCREEN:
            screen[record["concept"]][0] += int(record["hit"])
            screen[record["concept"]][1] += 1

    rows = []
    for (task, condition, concept, cluster), trials in sorted(cells.items()):
        fact_values = [float(trial["fact_correct"]) for trial in trials if trial.get("fact_correct") is not None]
        rows.append(
            [
                task,
                condition,
                concept,
                cluster,
                len(trials),
                round(_mean([trial["L"] for trial in trials]), 5),
                round(_mean([float(trial["hit"]) for trial in trials]), 5),
                round(_mean([float(trial["clean_hit"]) for trial in trials]), 5),
                round(_mean(fact_values), 5) if fact_values else None,
                round(_mean([trial["label_mass"] for trial in trials]), 4),
                _mean_or_none(trials, "none_mass", 4),
            ]
        )

    test_clusters = set(manifest.get("clusters_test", {}))
    clean_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("kind") == "clean" and record["cluster"] in test_clusters:
            clean_groups[(record["task"], record["cluster"])].append(record)
    clean_rows = []
    fact_hits = 0
    fact_total = 0
    for (task, cluster), group in sorted(clean_groups.items()):
        fact_correct = sum(int(item["answer"] == item["query"]) for item in group) if task == "T_fact" else None
        none_answers = sum(int(item["answer"] == 10) for item in group) if task == "T_inj_none" else None
        if task == "T_fact":
            fact_hits += fact_correct
            fact_total += len(group)
        clean_rows.append(
            [
                task,
                cluster,
                len(group),
                round(_mean([item["label_mass"] for item in group]), 4),
                _mean_or_none(group, "none_mass", 4),
                fact_correct,
                none_answers,
            ]
        )

    diagnostics = single("vector_diagnostics")

    return {
        "v": 1,
        "meta": {
            "tier": manifest.get("tier"),
            "inference_allowed": manifest.get("inference_allowed"),
            "model": manifest.get("model"),
            "revision": manifest.get("revision"),
            "commit": manifest.get("commit"),
            "config_hash": manifest.get("config_hash"),
            "data_sha256": manifest.get("data_sha256"),
            "seed": manifest.get("seed"),
            "versions": manifest.get("versions"),
            "n_bootstrap": manifest.get("n_bootstrap"),
            "min_denominator": manifest.get("min_denominator"),
            "screen_top_fraction": manifest.get("screen_top_fraction"),
            "complete": single("run_complete") is not None,
            "n_records": len(records),
            "n_sessions": len(manifests),
            "environments": environments,
        },
        "vector_diagnostics": diagnostics.get("by_layer") if diagnostics else None,
        "tokens": manifest.get("tokens"),
        "dropped_words": manifest.get("dropped_words"),
        "dropped_clusters": manifest.get("dropped_clusters"),
        "dropped_tasks": manifest.get("dropped_tasks"),
        "sham": single("sham"),
        "selection": single("selection"),
        "impact_match": single("impact_match"),
        "clean_fact": {"correct": fact_hits, "n": fact_total},
        "clean_columns": CLEAN_COLUMNS,
        "clean": clean_rows,
        "screen": dict(screen),
        "position_counts": position_counts,
        "cell_columns": CELL_COLUMNS,
        "cells": rows,
    }


def encode_export(export: dict[str, Any]) -> str:
    """A gzip+base64 block between markers, wrapped so a Colab cell shows every line."""
    payload = json.dumps(export, separators=(",", ":")).encode("utf-8")
    blob = base64.b64encode(gzip.compress(payload, mtime=0)).decode("ascii")
    lines = [blob[start : start + LINE_WIDTH] for start in range(0, len(blob), LINE_WIDTH)]
    return "\n".join([BEGIN_MARKER, *lines, END_MARKER])


def decode_export(text: str) -> dict[str, Any]:
    """Accepts the printed block, with or without surrounding output."""
    start = text.index(BEGIN_MARKER) + len(BEGIN_MARKER)
    end = text.index(END_MARKER, start)
    return json.loads(gzip.decompress(base64.b64decode("".join(text[start:end].split()))).decode("utf-8"))
