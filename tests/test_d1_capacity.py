"""The machine and pin checks that run first, the run identity, and the script's shape. Standard library, no model, no network."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from d1_factory import REAL_DATA

from mechinterp.d1 import capacity, cli
from mechinterp.d1.config import D1Config
from mechinterp.d1.run_identity import source_fingerprint

SCRIPT = (Path(__file__).resolve().parents[1] / "scripts" / "run_d1_gpu.sh").read_text(encoding="utf-8")
CONFIG = REAL_DATA.parent / "configs" / "full.json"


def real() -> D1Config:
    return D1Config.from_file(CONFIG)


def rung(tag: str):
    return real().rung(tag)


# Disk and RAM


def test_disk_grows_up_the_ladder_and_8b_is_about_what_the_arithmetic_says() -> None:
    table = capacity.ladder_table(real())
    totals = [row["disk_gb"] for row in table]
    assert totals == sorted(totals)
    assert [row["rung"] for row in table] == ["Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B", "Qwen/Qwen3-14B"]
    assert 25 < totals[0] < 45
    assert 60 < totals[2] < 85
    for row in table:
        assert row["disk_with_retry_gb"] > row["disk_gb"]


def test_ram_is_not_the_reviews_95_gb_which_is_disk() -> None:
    """Weights load straight to the GPU, so RAM holds no model. The need barely moves between rungs."""
    cfg = real()
    needs = [capacity.ram_recommended_gb(cfg, item, 1) for item in cfg.ladder]
    assert all(10 < need < 20 for need in needs)
    assert needs[-1] - needs[0] < 5
    assert capacity.ram_floor_gb(1) < needs[0]
    assert capacity.ram_recommended_gb(cfg, cfg.ladder[0], 4) > needs[0], "each GPU's worker holds its own libraries and text"


def test_short_disk_stops_a_rung_and_says_why() -> None:
    cfg = real()
    result = capacity.check_rung(cfg, rung("qwen3-8b"), 5.0, 64.0, 1)
    assert len(result.stops) == 1 and "more disk" in result.stops[0]


def test_disk_enough_without_the_retry_only_warns() -> None:
    cfg = real()
    item = rung("qwen3-8b")
    need = capacity.rung_disk_gb(cfg, item)
    base = need["model"] + need["g1"] + need["stage2"]
    result = capacity.check_rung(cfg, item, base + need["retry"] / 2.0, 64.0, 1)
    assert result.stops == []
    assert len(result.warnings) == 1 and "retry" in result.warnings[0]


def test_what_a_rung_has_already_stored_is_not_asked_for_again() -> None:
    cfg = real()
    item = rung("qwen3-8b")
    need = capacity.rung_disk_gb(cfg, item)
    assert capacity.check_rung(cfg, item, 1.0, 64.0, 1, already_gb=need["model"] + need["g1"] + need["stage2"]).stops == []


def test_ram_under_the_estimate_warns_and_only_under_the_floor_stops() -> None:
    """The estimate is unmeasured, and a false stop would throw a working machine away."""
    cfg = real()
    item = rung("qwen3-8b")
    between = (capacity.ram_floor_gb(1) + capacity.ram_recommended_gb(cfg, item, 1)) / 2.0
    warned = capacity.check_rung(cfg, item, 1000.0, between, 1)
    assert warned.stops == [] and len(warned.warnings) == 1
    stopped = capacity.check_rung(cfg, item, 1000.0, capacity.ram_floor_gb(1) - 1.0, 1)
    assert len(stopped.stops) == 1 and "RAM" in stopped.stops[0]
    assert capacity.check_rung(cfg, item, 1000.0, None, 1).stops == [], "a machine where this cannot be read is not stopped"


def test_a_machine_that_only_reaches_a_low_rung_can_still_start() -> None:
    """The ladder may never reach 8B. Only the first rung can stop the start, and the table says how far this machine goes."""
    cfg = real()
    table = capacity.ladder_table(cfg)
    disk = float(table[0]["disk_gb"]) + 2.0
    assert capacity.first_rung_stops(cfg, disk, 32.0, 24.0, 1) == []
    text = "\n".join(capacity.format_table(cfg, disk, 32.0, 24.0, 1))
    assert "Qwen/Qwen3-8B" in text and "short of disk" in text
    assert "not a finding" in text


def test_the_first_rung_stops_a_start_that_cannot_work() -> None:
    cfg = real()
    assert capacity.first_rung_stops(cfg, 10.0, 32.0, 24.0, 1)
    assert capacity.first_rung_stops(cfg, 500.0, 32.0, 8.0, 1), "not enough GPU memory"
    assert capacity.first_rung_stops(cfg, 500.0, 4.0, 24.0, 1), "not enough RAM"
    assert capacity.first_rung_stops(cfg, 10.0, 32.0, 24.0, 1, resuming=True) == [], "a resume's disk is judged by what it has stored"


def test_the_capacity_command_exits_by_the_first_rung(capsys: pytest.CaptureFixture) -> None:
    assert cli.main(["capacity", "--config", str(CONFIG), "--free-disk-gb", "500", "--ram-gb", "64", "--gpu-free-gb", "24"]) == 0
    assert "Qwen/Qwen3-1.7B" in capsys.readouterr().out
    assert cli.main(["capacity", "--config", str(CONFIG), "--free-disk-gb", "5", "--ram-gb", "64", "--gpu-free-gb", "24"]) == 1
    assert "Cannot start" in capsys.readouterr().out


def test_stored_gb_counts_what_is_there(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "b").write_bytes(b"x" * 1000)
    assert capacity.stored_gb(tmp_path) == pytest.approx(1000 / 1e9)
    assert capacity.stored_gb(tmp_path / "missing") == 0.0


# Pins


def _fake_remote(cfg: D1Config, **changes: dict) -> dict:
    pages: dict[str, bytes] = {}
    for item in cfg.ladder:
        page = {"num_hidden_layers": item.n_layers, "hidden_size": item.d_model, **changes.get(item.tag, {})}
        pages[f"https://huggingface.co/{item.name}/raw/{item.revision}/config.json"] = json.dumps(page).encode("utf-8")
    tree = [{"path": f"data/{quadrant}-00000-of-00001.parquet", "lfs": {"oid": changes.get("oid", digest)}} for quadrant, digest in cfg.parquet_sha256.items()]
    pages[f"https://huggingface.co/api/datasets/{cfg.dataset_repo}/tree/{cfg.dataset_revision}/data"] = json.dumps(tree).encode("utf-8")
    return pages


def test_pins_hold_on_the_remote_the_config_describes() -> None:
    cfg = real()
    pages = _fake_remote(cfg)
    checked = capacity.check_pins(cfg, pages.__getitem__)
    assert len(checked) == len(cfg.ladder) + 1


def test_a_rung_whose_shape_differs_is_caught_for_every_rung_not_only_the_first() -> None:
    cfg = real()
    pages = _fake_remote(cfg, **{"qwen3-14b": {"hidden_size": 1}})
    with pytest.raises(capacity.PinMismatch, match="14B"):
        capacity.check_pins(cfg, pages.__getitem__)


def test_a_moved_parquet_file_is_caught() -> None:
    cfg = real()
    pages = _fake_remote(cfg, oid="0" * 64)
    with pytest.raises(capacity.PinMismatch, match="parquet"):
        capacity.check_pins(cfg, pages.__getitem__)


def test_no_network_is_its_own_outcome() -> None:
    def refuse(_url: str) -> bytes:
        raise OSError("no route")

    with pytest.raises(capacity.RemoteUnavailable):
        capacity.check_pins(real(), refuse)


def test_the_pins_command_tells_a_mismatch_from_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(_url: str) -> bytes:
        raise OSError("no route")

    monkeypatch.setattr(capacity, "_fetch", refuse)
    monkeypatch.setattr(capacity.check_pins, "__defaults__", (refuse, None))
    assert cli.main(["check-pins", "--config", str(CONFIG)]) == 3
    pages = _fake_remote(real(), oid="0" * 64)
    monkeypatch.setattr(capacity.check_pins, "__defaults__", (pages.__getitem__, None))
    assert cli.main(["check-pins", "--config", str(CONFIG)]) == 1


# The identity's hash


CODE = '''"""A module."""


def twice(value):
    """Docs."""
    # A comment.
    return value * 2
'''


def test_comments_docstrings_and_line_endings_do_not_change_the_identity() -> None:
    first = source_fingerprint(CODE)
    assert source_fingerprint(CODE.replace("# A comment.", "# Reworded.")) == first
    assert source_fingerprint(CODE.replace('"""Docs."""', '"""Better docs."""')) == first
    assert source_fingerprint(CODE.replace('"""A module."""', "")) == first
    assert source_fingerprint(CODE.replace("\n", "\r\n")) == first


def test_a_change_to_the_code_changes_the_identity() -> None:
    assert source_fingerprint(CODE.replace("value * 2", "value * 3")) != source_fingerprint(CODE)


def test_a_string_that_is_code_and_not_a_docstring_counts() -> None:
    one = 'def f():\n    x = "a"\n    return x\n'
    assert source_fingerprint(one) != source_fingerprint(one.replace('"a"', '"b"'))


# The script


def test_the_run_script_checks_cheapest_first() -> None:
    body = SCRIPT[SCRIPT.index("inner_main() {") :]
    order = [body.index(step) for step in ("check_machine\n", "early_checks\n", "install_dependencies\n", "download_data\n", "run_stages\n", "write_results\n")]
    assert order == sorted(order)


def test_the_run_script_installs_what_direct_loading_needs_and_no_longer_downloads_a_model_before_the_data_check() -> None:
    assert '"accelerate>=1.1.0"' in SCRIPT
    assert "import accelerate" in SCRIPT
    assert not re.search(r'config\["ladder"\]\[0\]', SCRIPT)
    assert "MIN_DISK_GB" not in SCRIPT


def test_the_run_script_has_a_reanalysis_mode() -> None:
    assert "D1_REANALYZE" in SCRIPT
    assert "mechinterp.d1.cli reanalyze" in SCRIPT


def test_no_step_of_the_early_checks_needs_more_than_the_standard_library() -> None:
    """They run before anything is installed, so a new import of numpy or torch here would break the first minute."""
    import ast

    for module in (capacity, cli):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        top = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        names = {alias.name.split(".")[0] for node in top if isinstance(node, ast.Import) for alias in node.names}
        names |= {(node.module or "").split(".")[0] for node in top if isinstance(node, ast.ImportFrom) and node.level == 0}
        assert not names & {"numpy", "torch", "transformers", "pyarrow", "huggingface_hub"}, module.__name__
