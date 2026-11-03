"""Tier configuration: loading, validation and hashing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED_CONCEPT_TASKS = ("T_inj", "T_pick", "T_fav", "T_odd", "T_fact")
REQUIRED_T_INJ_CONDITIONS = ("concept", "random_norm", "random_impact")
VALID_CONDITIONS = {"concept", "random_norm", "random_impact"}


@dataclass(frozen=True)
class DesignConfig:
    n_concepts_calibration: int
    n_clusters_calibration: int
    positions_per_cell_calibration: int
    n_concepts_test: int
    n_clusters_test: int
    positions_per_cell_test: int
    fact_queries_per_trial: int
    test_task_conditions: dict[str, list[str]]


@dataclass(frozen=True)
class GridConfig:
    layers: list[int]
    alphas: list[float]


@dataclass(frozen=True)
class ImpactSearchConfig:
    max_evals: int
    tolerance: float
    max_scale: float


@dataclass(frozen=True)
class ScreeningConfig:
    enabled: bool
    top_fraction: float
    positions_per_cell: int


@dataclass(frozen=True)
class RuntimeConfig:
    batch_size_cpu: int
    batch_size_gpu: int
    segment_seconds: float


@dataclass(frozen=True)
class AnalysisConfig:
    n_bootstrap: int
    min_denominator: float


@dataclass(frozen=True)
class Config:
    tier: str
    inference_allowed: bool
    seed: int
    model_name: str
    model_revision: str
    data_dir: Path
    concept_prompt_mode: str
    design: DesignConfig
    grid: GridConfig
    impact_search: ImpactSearchConfig
    screening: ScreeningConfig
    runtime: RuntimeConfig
    analysis: AnalysisConfig
    config_hash: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any], base_dir: Path) -> "Config":
        canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"))
        data_dir = Path(raw["data_dir"])
        if not data_dir.is_absolute():
            data_dir = (base_dir / data_dir).resolve()
        config = cls(
            tier=raw["tier"],
            inference_allowed=bool(raw["inference_allowed"]),
            seed=int(raw["seed"]),
            model_name=raw["model"]["name"],
            model_revision=raw["model"].get("revision", "main"),
            data_dir=data_dir,
            concept_prompt_mode=raw.get("concept_prompt_mode", "chat"),
            design=DesignConfig(**raw["design"]),
            grid=GridConfig(layers=list(raw["grid"]["layers"]), alphas=[float(a) for a in raw["grid"]["alphas"]]),
            impact_search=ImpactSearchConfig(**raw["impact_search"]),
            screening=ScreeningConfig(**raw["screening"]),
            runtime=RuntimeConfig(**raw["runtime"]),
            analysis=AnalysisConfig(**raw["analysis"]),
            config_hash=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        )
        config.validate()
        return config

    @classmethod
    def from_file(cls, path: str | Path) -> "Config":
        path = Path(path)
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
        return cls.from_dict(raw, path.parent)

    def validate(self) -> None:
        conditions = self.design.test_task_conditions
        for task in REQUIRED_CONCEPT_TASKS:
            if "concept" not in conditions.get(task, []):
                raise ValueError(f"{task} needs the concept condition: the decision rules depend on it")
        for condition in REQUIRED_T_INJ_CONDITIONS:
            if condition not in conditions["T_inj"]:
                raise ValueError(f"T_inj needs condition {condition}: H3 depends on it")
        for task, task_conditions in conditions.items():
            unknown = set(task_conditions) - VALID_CONDITIONS
            if unknown:
                raise ValueError(f"{task} has unknown conditions {sorted(unknown)}")
        if not 1 <= self.design.fact_queries_per_trial <= 9:
            raise ValueError("fact_queries_per_trial must be between 1 and 9")
        if self.concept_prompt_mode not in ("chat", "raw"):
            raise ValueError("concept_prompt_mode must be 'chat' or 'raw'")
        if self.impact_search.max_evals < 1:
            raise ValueError("impact_search.max_evals must be at least 1")
        if not (self.grid.layers and self.grid.alphas):
            raise ValueError("grid.layers and grid.alphas must not be empty")
