"""E2's configuration, on top of E1's. Never imports torch, because the orchestrator reads it.

The E1 config file is the base, unchanged and by reference, so that E2's concepts, clusters, grid and impact
search settings are E1's by construction, and a run can check its hash against E1's manifest.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import Config

FAMILIES = ("F_full", "F_j", "F_res", "F_lens", "F_rand")
MATCHED_FAMILIES = ("F_j", "F_res", "F_rand", "F_lens")  # F_full sets the target, so it is never searched.


def e2_code_hash() -> str:
    """Hash of all E1 and E2 source. identity.code_hash covers the parent folder only, and must stay that way."""
    digest = hashlib.sha256()
    package = Path(__file__).resolve().parents[1]
    for path in [*sorted(package.glob("*.py")), *sorted(Path(__file__).resolve().parent.glob("*.py"))]:
        digest.update(path.relative_to(package).as_posix().encode("utf-8"))
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


@dataclass(frozen=True)
class LensSpec:
    repo: str
    commit: str
    filename: str
    sha256: str
    size: int
    n_prompts: int | None
    local_path: Path | None  # Tests point this at a stub. A real run downloads by the pinned commit.


@dataclass(frozen=True)
class SanitySpec:
    prompts: tuple[dict[str, str], ...]
    min_fraction: float
    alignment_text: str
    alignment_max_tokens: int
    alignment_replicas: int
    alignment_passes: int
    alignment_layers: tuple[int, ...] | None
    skip_first: int


@dataclass(frozen=True)
class E2Config:
    base: Config
    base_path: Path
    lens: LensSpec
    sanity: SanitySpec
    eval_dir: Path
    eval_sets: dict[str, str]  # Set name to its readout rule.
    synonyms_path: Path | None
    chinese_path: Path | None
    gate_k: int
    gate_min_band_layers: int
    gate_n_bootstrap: int
    decomposition_k: int
    top_tokens: int
    tasks: tuple[str, ...]
    families: tuple[str, ...]
    exclusion_tolerance: float
    n_bootstrap: int
    min_denominator: float
    family_alpha: float
    e2_config_hash: str

    @classmethod
    def from_file(cls, path: str | Path) -> "E2Config":
        path = Path(path).resolve()
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(raw, path.parent)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], base_dir: Path) -> "E2Config":
        base_path = (base_dir / raw["base_config"]).resolve()
        base = Config.from_file(base_path)
        lens = raw["lens"]
        sanity = raw["sanity"]
        evaluation = raw["evaluation"]
        stage3 = raw["stage3"]
        eval_dir = (base_dir / evaluation["dir"]).resolve()
        synonyms = (base_dir / evaluation["order_ops_synonyms"]).resolve() if evaluation.get("order_ops_synonyms") else None
        chinese = (base_dir / evaluation["chinese_targets"]).resolve() if evaluation.get("chinese_targets") else None

        # The hash covers what the config points at, so a changed word list cannot share a store with the old one.
        digest = hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(base.config_hash.encode("ascii"))
        for name in sorted(evaluation["sets"]):
            digest.update((eval_dir / f"lens-eval-{name}.json").read_bytes().replace(b"\r\n", b"\n"))
        for extra in (synonyms, chinese):
            if extra is not None:
                digest.update(extra.read_bytes().replace(b"\r\n", b"\n"))

        config = cls(
            base=base,
            base_path=base_path,
            lens=LensSpec(
                repo=lens["repo"],
                commit=lens["commit"],
                filename=lens["filename"],
                sha256=lens["sha256"],
                size=int(lens["size"]),
                n_prompts=lens.get("n_prompts"),
                local_path=(base_dir / lens["local_path"]).resolve() if lens.get("local_path") else None,
            ),
            sanity=SanitySpec(
                prompts=tuple(sanity["prompts"]),
                min_fraction=float(sanity["min_fraction"]),
                alignment_text=sanity["alignment_text"],
                alignment_max_tokens=int(sanity["alignment_max_tokens"]),
                alignment_replicas=int(sanity["alignment_replicas"]),
                alignment_passes=int(sanity["alignment_passes"]),
                alignment_layers=tuple(sanity["alignment_layers"]) if sanity.get("alignment_layers") else None,
                skip_first=int(sanity["skip_first"]),
            ),
            eval_dir=eval_dir,
            eval_sets=dict(evaluation["sets"]),
            synonyms_path=synonyms,
            chinese_path=chinese,
            gate_k=int(raw["gate"]["k"]),
            gate_min_band_layers=int(raw["gate"]["min_band_layers"]),
            gate_n_bootstrap=int(raw["gate"]["n_bootstrap"]),
            decomposition_k=int(raw["decomposition"]["k"]),
            top_tokens=int(raw["decomposition"]["top_tokens"]),
            tasks=tuple(stage3["tasks"]),
            families=tuple(stage3["families"]),
            exclusion_tolerance=float(stage3["exclusion_tolerance"]),
            n_bootstrap=int(raw["analysis"]["n_bootstrap"]),
            min_denominator=float(raw["analysis"]["min_denominator"]),
            family_alpha=float(raw["analysis"]["family_alpha"]),
            e2_config_hash=digest.hexdigest(),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if set(self.families) != set(FAMILIES):
            raise ValueError(f"The prereg's families are {FAMILIES}, config has {self.families}")
        conditions = self.base.design.test_task_conditions
        for task in self.tasks:
            if task not in conditions:
                raise ValueError(f"{task} is not a task in E1's config")
        if self.base.impact_search.tolerance > self.exclusion_tolerance:
            raise ValueError("The search must aim at least as tight as the exclusion tolerance")
        if self.base.design.clusters_per_concept_test is not None:
            raise ValueError("E2 expects E1's crossed large-tier test design")
