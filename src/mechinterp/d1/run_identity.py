"""What a stored run depends on. Standard library only.

Two sets of modules, on purpose:

    CAPTURE_MODULES   Change rows, prompts, activations or adapters. An edit refuses a resume, because new rows would mix
                      silently with old ones.
    PROBE_MODULES     Change how a probe is fitted. An edit refits probes from stored activations, which takes minutes.

Everything else (analysis, gates, stats, baselines, orchestrator, preflight, the command line, the worker's dispatch) reads
stored numbers or only schedules work, so fixing it never costs stored work. Gates are recomputed on every launch anyway.

The hash is of the parsed code, with docstrings and comments gone, so a reworded comment costs nothing and a checkout's line
endings do not matter. It walks the syntax tree by hand and skips empty fields, so that a newer Python's added fields do not
look like an edit when a machine rebuilds its venv.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any, Iterable

_PACKAGE = Path(__file__).resolve().parents[1]
CAPTURE_MODULES = (
    "d1/capture.py",
    "d1/data.py",
    "d1/families.py",
    "d1/lora.py",
    "d1/plan.py",
    "d1/produce.py",
    "d1/prompts.py",
    "d1/sizing.py",
    "d1/specificity.py",
    "d1/store.py",
    "d1/training_data.py",
    "hooks.py",
    "modeling.py",
    "prompts.py",
    "scoring.py",
)
PROBE_MODULES = ("d1/probes.py",)


def _without_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        return body[1:]
    return body


def _canonical(node: Any) -> Any:
    if isinstance(node, ast.AST):
        parts: list[Any] = [type(node).__name__]
        for name, value in ast.iter_fields(node):
            if name == "body" and isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                value = _without_docstring(value)
            if value is None or value == [] or (name == "kind" and isinstance(node, ast.Constant)):
                continue
            parts.append([name, _canonical(value)])
        return parts
    if isinstance(node, list):
        return [_canonical(item) for item in node]
    return repr(node)


def source_fingerprint(source: str) -> str:
    return hashlib.sha256(repr(_canonical(ast.parse(source))).encode("utf-8")).hexdigest()


def module_fingerprints(names: Iterable[str], root: Path = _PACKAGE) -> dict[str, str]:
    return {name: source_fingerprint((root / name).read_text(encoding="utf-8")) for name in names}


def capture_fingerprint() -> dict[str, str]:
    return module_fingerprints(CAPTURE_MODULES)


def probe_fingerprint() -> dict[str, str]:
    return module_fingerprints(PROBE_MODULES)


def differing(first: dict[str, str], second: dict[str, str]) -> list[str]:
    return sorted(name for name in set(first) | set(second) if first.get(name) != second.get(name))
