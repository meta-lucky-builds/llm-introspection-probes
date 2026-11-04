"""E2's frozen files and plan sizes, on the real configs and data. No model and no download."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from mechinterp.config import Config
from mechinterp.design import load_data, plan_impact_trials, prepare_stimuli
from mechinterp.e2.config import E2Config, e2_code_hash
from mechinterp.e2.design import alpha_key, plan_counts, plan_full_grid, plan_searches, plan_test, unique_sites
from mechinterp.e2.evalsets import load_eval_items, load_word_table
from mechinterp.identity import code_hash

REPO_ROOT = Path(__file__).resolve().parents[1]
E2_ROOT = REPO_ROOT / "experiments" / "e2_jspace_vector_families"
E2_CONFIG = E2_ROOT / "configs" / "large_4x3090.json"


def real_config() -> E2Config:
    return E2Config.from_file(E2_CONFIG)


def test_vendored_evaluation_files_are_the_pinned_blobs() -> None:
    directory = E2_ROOT / "data" / "evaluations"
    notice = (directory / "NOTICE.md").read_text(encoding="utf-8")
    blobs = dict(re.findall(r"`(lens-eval-[a-z-]+\.json)` \| `([0-9a-f]{40})`", notice))
    assert len(blobs) == 6
    for name, blob in blobs.items():
        data = (directory / name).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() == blob, name
    assert "Apache License" in (directory / "LICENSE.jacobian-lens").read_text(encoding="utf-8")


def test_real_config_pins_everything_and_holds_no_token() -> None:
    e2 = real_config()
    assert re.fullmatch(r"[0-9a-f]{40}", e2.lens.commit)
    assert re.fullmatch(r"[0-9a-f]{64}", e2.lens.sha256)
    assert e2.lens.local_path is None
    assert e2.lens.n_prompts == 479
    assert e2.decomposition_k == 16
    assert e2.gate_k == 10
    assert e2.gate_min_band_layers == 3
    assert e2.exclusion_tolerance == 0.1
    assert e2.tasks == ("T_inj", "T_pick", "T_odd")
    assert e2.base.impact_search.tolerance == 0.05
    assert "hf_" not in E2_CONFIG.read_text(encoding="utf-8")


def test_base_config_is_e1s_own_file() -> None:
    e2 = real_config()
    e1 = Config.from_file(REPO_ROOT / "experiments" / "e1_localization_capture" / "configs" / "large_4x3090.json")
    assert e2.base.config_hash == e1.config_hash
    assert e2.base.model_revision == e1.model_revision


@pytest.mark.xfail(strict=True, reason="E2 is paused: one item in the data set is unresolved")
def test_all_six_sets_load_with_the_readme_rules() -> None:
    e2 = real_config()
    items = load_eval_items(e2.eval_dir, e2.eval_sets)
    counts = {name: sum(1 for item in items if item.set_name == name) for name in e2.eval_sets}
    assert counts == {"multihop": 93, "multilingual": 107, "order-ops": 55, "poetry": 99, "typo": 96, "association": 102}
    assert all(item.intermediates for item in items)
    assert all(item.target for item in items if item.rule == "before_target")


def test_word_tables_cover_order_ops_and_most_intermediates() -> None:
    e2 = real_config()
    items = load_eval_items(e2.eval_dir, e2.eval_sets)
    synonyms = load_word_table(e2.synonyms_path)
    assert {word for item in items if item.set_name == "order-ops" for word in item.intermediates} <= set(synonyms)
    chinese = load_word_table(e2.chinese_path)
    words = {word for item in items for word in item.intermediates}
    assert len(words & set(chinese)) / len(words) > 0.9


def test_full_cross_trial_count_is_what_the_readme_says() -> None:
    e2 = real_config()
    stimuli = prepare_stimuli(load_data(e2.base.data_dir), e2.base, lambda word: True)
    assert len(stimuli.concepts_test) == 40 and len(stimuli.clusters_test) == 10
    alphas = {alpha_key(family, layer): 1.0 for family in e2.families for layer in (3, 20)}
    every = plan_test(stimuli, e2.base, e2.tasks, [3, 20], alphas, stimuli.concepts_test)
    assert len(every) == 120_000
    assert len({trial.key for trial in every}) == 120_000
    some = plan_test(stimuli, e2.base, e2.tasks, [3, 20], alphas, stimuli.concepts_test[:30])
    assert len(some) == 96_000 + 600 * 30
    assert len(plan_impact_trials(stimuli, e2.base, 0, 1.0)) == 500
    assert len(plan_full_grid(stimuli, e2.base, 3, 2.0, 20)) == 500 * 9
    assert plan_counts(stimuli, e2.base, e2.tasks, 40, 20) == {"grid": 4500, "impact_search_max": 32_000, "test": 120_000}


def test_one_site_when_the_layers_coincide() -> None:
    assert unique_sites(5, 5) == [5]
    assert unique_sites(5, 9) == [5, 9]
    assert plan_searches([5], []) == [("F_j", 5), ("F_res", 5), ("F_rand", 5)]
    assert ("F_lens", 5) in plan_searches([5], ["a"])


def test_e1s_run_identity_does_not_see_this_package() -> None:
    package = REPO_ROOT / "src" / "mechinterp"
    hashed = {path.name for path in package.glob("*.py")}
    assert "stage1.py" not in hashed and "pursuit.py" not in hashed
    assert (package / "e2" / "stage1.py").exists()
    assert e2_code_hash() != code_hash()


def test_new_files_changed_nothing_e1_reads() -> None:
    """E1's config and data stay in their own folder, and E2 points at them instead of copying."""
    raw = json.loads(E2_CONFIG.read_text(encoding="utf-8"))
    assert raw["base_config"] == "../../e1_localization_capture/configs/large_4x3090.json"
    assert not (E2_ROOT / "data" / "concepts.json").exists()
