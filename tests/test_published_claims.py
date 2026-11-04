"""What the front page says must match the stored data, and its links must resolve."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from mechinterp.analysis import analyze
from mechinterp.export import decode_export

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "e1" / "small_qwen3_1.7b"
LINK = re.compile(r"\]\(([^)\s]+)\)")
CONDITIONS = (
    "T_inj|concept",
    "T_pick|concept",
    "T_fav|concept",
    "T_odd|concept",
    "T_inj|random_norm",
    "T_inj|random_impact",
)
SHARES = {"T_pick|concept": "T_pick", "T_fav|concept": "T_fav", "T_odd|concept": "T_odd"}
ROW_LABELS = {
    "README.md": (
        "| Asked about injection |",
        '| "Pick one at random" |',
        '| "Most interesting" |',
        '| "Does not belong" |',
        "| Asked about injection, random direction, norm-matched |",
        "| Asked about injection, random direction, impact search |",
    ),
    "results/e1/small_qwen3_1.7b/RESULTS.md": (
        "| T_inj, concept |",
        "| T_pick, concept |",
        "| T_fav, concept |",
        "| T_odd, concept |",
        "| T_inj, random, norm-matched |",
        "| T_inj, random, impact search |",
    ),
}


def _interval(summary: dict) -> str:
    return f"{summary['point']:.2f} [{summary['lo']:.2f}, {summary['hi']:.2f}]"


@pytest.fixture(scope="module")
def report() -> dict:
    return analyze(decode_export((RESULTS_DIR / "export.txt").read_text(encoding="utf-8")))


@pytest.mark.parametrize("document", list(ROW_LABELS))
def test_every_table_row_is_recomputed_from_the_export(report: dict, document: str) -> None:
    text = (REPO_ROOT / document).read_text(encoding="utf-8")
    by_condition = report["estimates"]["by_condition"]
    for key, label in zip(CONDITIONS, ROW_LABELS[document]):
        row = next(line for line in text.splitlines() if line.startswith(label))
        assert _interval(by_condition[key]["L"]) in row, key
        assert _interval(by_condition[key]["hit"]) in row, key
        if key in SHARES:
            assert _interval(report["estimates"]["controls"][SHARES[key]]["share"]) in row, key
    gap = report["estimates"]["h3"]["concept_minus_norm"]
    assert f"{gap['point']:.2f}" in text and f"[{gap['lo']:.2f}, {gap['hi']:.2f}]" in text


def test_lookup_and_label_mass_are_as_stated(report: dict) -> None:
    by_condition = report["estimates"]["by_condition"]
    assert f"{report['estimates']['fact_accuracy_under_injection']['point']:.2f}" == "0.98"
    assert by_condition["T_fact|concept"]["hit"]["point"] == 0.0
    assert all(by_condition[key]["injected_label_mass"]["point"] >= 0.995 for key in CONDITIONS)


def test_preregistered_verdicts_are_the_ones_stated(report: dict) -> None:
    assert report["gates"]["V1"]["passed"] and report["gates"]["V2"]["passed"]
    verdicts = report["verdicts"]
    assert verdicts["H1"] == {"T_pick": "capture", "T_fav": "mixed", "overall": "mixed"}
    assert verdicts["H2"]["supported"] is True
    assert verdicts["H3"]["verdict"] == "uninformative"
    assert f"{verdicts['H3']['relative_error'] * 100:.1f}%" == "11.3%"


def test_the_figure_shows_the_recomputed_points(report: dict) -> None:
    figure = (REPO_ROOT / "docs" / "figures" / "e1_small_forest.svg").read_text(encoding="utf-8")
    for key in CONDITIONS:
        point = report["estimates"]["by_condition"][key]["L"]["point"]
        assert f">{point:.2f}</text>" in figure, key


def test_relative_links_in_project_documents_resolve() -> None:
    documents = [REPO_ROOT / "README.md", *sorted((REPO_ROOT / "docs").rglob("*.md"))]
    documents += sorted((REPO_ROOT / "experiments").glob("*/README.md"))
    documents += sorted((REPO_ROOT / "results").rglob("*.md"))
    broken = []
    for document in documents:
        for target in LINK.findall(document.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            path = target.split("#", 1)[0]
            if path and not (document.parent / path).exists():
                broken.append(f"{document.relative_to(REPO_ROOT)}: {target}")
    assert not broken, broken
