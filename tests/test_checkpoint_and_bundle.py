from __future__ import annotations

import base64
import importlib.util
import io
import re
import zipfile
from pathlib import Path

import pytest

from mechinterp.checkpoint import ResultStore, checkpoint_blob, restore_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_store_survives_a_reload(tmp_path) -> None:
    path = tmp_path / "results.jsonl"
    store = ResultStore(path)
    store.append([{"key": "a", "kind": "trial", "L": 0.1}, {"key": "b", "kind": "clean"}])
    reloaded = ResultStore(path)
    assert len(reloaded) == 2
    assert reloaded.has("a") and reloaded.get("a")["L"] == 0.1
    assert [record["key"] for record in reloaded.records("clean")] == ["b"]


def test_float_precision_round_trips_exactly(tmp_path) -> None:
    path = tmp_path / "results.jsonl"
    value = 0.1 + 0.2
    ResultStore(path).append([{"key": "a", "kind": "clean", "logits": [value]}])
    assert ResultStore(path).get("a")["logits"] == [value]


def test_duplicate_keys_are_rejected(tmp_path) -> None:
    store = ResultStore(tmp_path / "results.jsonl")
    store.append([{"key": "a", "kind": "trial"}])
    with pytest.raises(ValueError):
        store.append([{"key": "a", "kind": "trial"}])
    with pytest.raises(ValueError):
        store.append([{"key": "b", "kind": "trial"}, {"key": "b", "kind": "trial"}])


@pytest.mark.parametrize("tail", [b'{"key":"torn","ki', b'{"key":"torn"\n'])
def test_torn_final_line_is_removed_and_appends_stay_valid(tmp_path, tail) -> None:
    path = tmp_path / "results.jsonl"
    ResultStore(path).append([{"key": "a", "kind": "trial"}])
    with open(path, "ab") as handle:
        handle.write(tail)
    store = ResultStore(path)
    assert not store.has("torn")
    store.append([{"key": "b", "kind": "trial"}])
    reloaded = ResultStore(path)
    assert reloaded.has("a") and reloaded.has("b") and len(reloaded) == 2


def test_blob_restores_a_lost_runtime(tmp_path) -> None:
    path = tmp_path / "old" / "results.jsonl"
    ResultStore(path).append([{"key": "a", "kind": "trial", "L": 1.5}])
    blob = checkpoint_blob(path)
    target = tmp_path / "new" / "results.jsonl"
    restore_checkpoint(f"\n{blob[:10]}\n{blob[10:]}\n", target)
    assert ResultStore(target).get("a")["L"] == 1.5
    with pytest.raises(FileExistsError):
        restore_checkpoint(blob, target)


def _bundle_module():
    spec = importlib.util.spec_from_file_location("make_colab_bundle", REPO_ROOT / "scripts" / "make_colab_bundle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bundle_holds_code_and_data_only_and_is_deterministic() -> None:
    bundle = _bundle_module()
    data = bundle.build_zip(REPO_ROOT)
    assert data == bundle.build_zip(REPO_ROOT)
    names = zipfile.ZipFile(io.BytesIO(data)).namelist()
    assert "src/mechinterp/runner.py" in names
    assert "experiments/e1_localization_capture/configs/pilot.json" in names
    assert "experiments/e1_localization_capture/data/concepts.json" in names
    assert not [name for name in names if "__pycache__" in name or name.startswith(("tests/", "docs/"))]


def test_bundle_cell_decodes_to_the_zip() -> None:
    bundle = _bundle_module()
    cell = bundle.build_cell(REPO_ROOT)
    blob = re.search(r'BUNDLE = """\n(.*?)\n"""', cell, re.DOTALL).group(1)
    assert base64.b64decode(blob) == bundle.build_zip(REPO_ROOT)
    compile(cell, "colab_cell", "exec")
