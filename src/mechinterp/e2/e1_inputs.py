"""What E2 takes from E1's finished large-tier run. Read-only, and a clear error if it is not there."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from ..checkpoint import read_records
from ..design import PHASE_TEST


class E1ResultsMissing(ValueError):
    """E1's results are absent or unfinished. E2's Stages 2 and 3 need them."""


def e1_results_file(e1_dir: Path) -> Path:
    """The E1 state directory holds run/results.jsonl, and a copy in results/ once it has finalized."""
    e1_dir = Path(e1_dir)
    for candidate in (e1_dir / "results" / "results.jsonl", e1_dir / "run" / "results.jsonl", e1_dir / "results.jsonl"):
        if candidate.exists():
            return candidate
    raise E1ResultsMissing(
        f"No E1 results under {e1_dir}: looked for results/results.jsonl and run/results.jsonl. Stage 2 and Stage 3 "
        "need E1's large-tier results. Finish E1 with scripts/run_e1_gpu.sh, or set E2_E1_STATE_DIR to its state directory."
    )


def load_e1_inputs(e1_dir: Path) -> dict[str, Any]:
    """E1's selection and identity, and each test concept's mean L on T_inj and T_pick. Raises if unfinished."""
    path = e1_results_file(e1_dir)
    selection = None
    manifest = None
    complete = False
    n_trials = 0
    totals: dict[str, dict[str, list[float]]] = {"T_inj": defaultdict(lambda: [0.0, 0.0]), "T_pick": defaultdict(lambda: [0.0, 0.0])}
    for record in read_records(path):
        kind = record.get("kind")
        if kind == "run_complete":
            complete = True
        elif kind == "selection":
            selection = record
        elif kind == "manifest" and manifest is None:
            manifest = record
        elif kind == "trial" and record.get("phase") == PHASE_TEST:
            n_trials += 1
            if record.get("condition") == "concept" and record.get("task") in totals:
                cell = totals[record["task"]][record["concept"]]
                cell[0] += record["L"]
                cell[1] += 1.0
    if not complete:
        raise E1ResultsMissing(f"{path} has no run_complete record, so E1 has not finished. E2 reads finished results only.")
    if selection is None or manifest is None:
        raise E1ResultsMissing(f"{path} is complete but holds no selection or no manifest")
    if not totals["T_inj"] or not totals["T_pick"]:
        raise E1ResultsMissing(f"{path} holds no T_inj or T_pick concept rows in its test phase")
    identity = manifest.get("identity") or {}
    return {
        "path": str(path),
        "layer": selection["layer"],
        "alpha": selection["alpha"],
        "accuracy": selection.get("accuracy"),
        "tier": manifest.get("tier"),
        "inference_allowed": manifest.get("inference_allowed"),
        "config_hash": identity.get("config_hash"),
        "data_sha256": identity.get("data_sha256"),
        "model_commit": identity.get("model_commit"),
        "code_hash": identity.get("code_hash"),
        "n_test_trials": n_trials,
        "L_inj": {concept: total / count for concept, (total, count) in totals["T_inj"].items()},
        "L_pick": {concept: total / count for concept, (total, count) in totals["T_pick"].items()},
    }
