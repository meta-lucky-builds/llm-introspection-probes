"""Builds E2's tiny model and stub files, for tests and inside worker subprocesses. Not collected by pytest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from conftest import CONCEPTS, SPLIT_SPACE_PATTERN, base_texts, make_tokenizer, tiny_config_dict, tiny_qwen, write_tiny_data

from mechinterp.config import Config
from mechinterp.design import load_data
from mechinterp.e2.lens import prompt_jacobian_rows
from mechinterp.identity import code_hash

ALIGNMENT_TEXT = "Tell me about contaa. Tell me about contab. Tell me about contac. Tell me about contad."
DIM = 32


def build(config: Any, shard: int) -> tuple[Any, Any, torch.device]:
    tokenizer = make_tokenizer(base_texts(), SPLIT_SPACE_PATTERN)
    model = tiny_qwen(len(tokenizer)).requires_grad_(False)
    return model, tokenizer, torch.device("cpu")


def write_stub_lens(path: Path, kind: str = "fitted") -> Path:
    """A real file in the published format: float16, every block but the last. kind "identity" is J = I."""
    model, tokenizer, device = build(None, 0)
    if kind == "identity":
        stored = {layer: torch.eye(DIM, dtype=torch.float16) for layer in (0, 1)}
    else:
        ids = torch.tensor([list(tokenizer(ALIGNMENT_TEXT)["input_ids"])], device=device)
        rows = prompt_jacobian_rows(model, ids, [0, 1], list(range(DIM)), skip_first=1)
        stored = {layer: matrix.to(torch.float16) for layer, matrix in rows.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"J": stored, "n_prompts": 1, "source_layers": [0, 1], "d_model": DIM}, path)
    return path


def write_eval_sets(directory: Path) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    words = CONCEPTS["test"]
    sets = {
        "multihop": ("before_target", [{"name": f"m{index}", "prompt": "Tell me about ", "target": word, "intermediates": [word]} for index, word in enumerate(words)]),
        "poetry": ("last_newline", [{"name": f"p{index}", "prompt": f"Tell me\nabout {word}\nTell ", "intermediates": [word]} for index, word in enumerate(words)]),
        "typo": ("final", [{"name": f"t{index}", "prompt": f"Tell me about {word}", "intermediates": [word, "missingword"]} for index, word in enumerate(words)]),
    }
    for name, (_, items) in sets.items():
        (directory / f"lens-eval-{name}.json").write_text(json.dumps({"items": items}), encoding="utf-8")
    return {name: rule for name, (rule, _) in sets.items()}


def write_e2_config(directory: Path, lens_kind: str = "fitted") -> Path:
    """A complete tiny E2 config beside its base config, word lists, evaluation sets and lens."""
    directory.mkdir(parents=True, exist_ok=True)
    data = write_tiny_data(directory / "data")
    base_path = directory / "base.json"
    base_path.write_text(json.dumps(tiny_config_dict(data)), encoding="utf-8")
    lens_path = write_stub_lens(directory / "lens.pt", lens_kind)
    sets = write_eval_sets(directory / "evaluations")
    raw = {
        "experiment": "E2",
        "base_config": "base.json",
        "lens": {
            "repo": "stub",
            "commit": "stub",
            "filename": "lens.pt",
            "sha256": hashlib.sha256(lens_path.read_bytes()).hexdigest(),
            "size": lens_path.stat().st_size,
            "n_prompts": 1,
            "local_path": "lens.pt",
        },
        "sanity": {
            "prompts": [{"text": "Tell me about contaa.", "expected": "contaa"}],
            "min_fraction": 0.0,
            "alignment_text": ALIGNMENT_TEXT,
            "alignment_max_tokens": 16,
            "alignment_replicas": 2,
            "alignment_passes": 2,
            "alignment_layers": None,
            "skip_first": 1,
        },
        "evaluation": {"dir": "evaluations", "sets": sets},
        "gate": {"k": 10, "min_band_layers": 2, "n_bootstrap": 100},
        "decomposition": {"k": 3, "top_tokens": 5},
        "stage3": {"tasks": ["T_inj", "T_pick", "T_odd"], "families": ["F_full", "F_j", "F_res", "F_lens", "F_rand"], "exclusion_tolerance": 0.1},
        "analysis": {"n_bootstrap": 100, "min_denominator": 0.05, "family_alpha": 0.05},
    }
    path = directory / "e2.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def write_passing_gate(path: Path, l_w: int = 1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"passed": True, "l_w": l_w, "band": [0, 1], "reason": None, "analysis": {"decision": {"mean_gain": 0.1}, "primary_english_table": []}}),
        encoding="utf-8",
    )
    return path


def write_e1_results(e1_dir: Path, e2_config_path: Path, layer: int = 0, alpha: float = 2.0, complete: bool = True) -> Path:
    """What a finished E1 leaves behind, with this tiny run's identity."""
    raw = json.loads(e2_config_path.read_text(encoding="utf-8"))
    base = Config.from_file(e2_config_path.parent / raw["base_config"])
    identity = {
        "config_hash": base.config_hash,
        "data_sha256": load_data(base.data_dir).sha256,
        "model_commit": "local",
        "code_hash": code_hash(),
    }
    records: list[dict[str, Any]] = [
        {"key": "manifest|0", "kind": "manifest", "tier": "test", "inference_allowed": True, "identity": identity},
        {"key": "selection", "kind": "selection", "layer": layer, "alpha": alpha, "accuracy": 0.5},
    ]
    for index, concept in enumerate(CONCEPTS["test"]):
        for task, scale in (("T_inj", 1.0), ("T_pick", 0.5)):
            records.append(
                {"key": f"test|{task}|{concept}", "kind": "trial", "phase": "test", "task": task, "condition": "concept", "concept": concept, "L": scale * (index + 1)}
            )
    if complete:
        records.append({"key": "run_complete", "kind": "run_complete"})
    path = e1_dir / "results" / "results.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return e1_dir
