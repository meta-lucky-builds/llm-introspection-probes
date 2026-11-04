"""E2's stage list and file layout. RunPaths is reused, with only the stage names changed. No torch."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..design import PHASE_GRID, PHASE_IMPACT, PHASE_TEST
from ..sharding import RunPaths

E2_STAGE_ORDER = ("vectors", "grid", "impact", "test", "stop")
E2_STAGE_PHASE = {"grid": PHASE_GRID, "impact": PHASE_IMPACT, "test": PHASE_TEST}


@dataclass(frozen=True)
class E2Paths(RunPaths):
    def stage_spec(self, index: int) -> Path:
        return self.stages_dir / f"{index:02d}_{E2_STAGE_ORDER[index]}.json"

    @property
    def e1_inputs(self) -> Path:
        return self.out_dir / "e1_inputs.json"

    @property
    def decomposition(self) -> Path:
        return self.out_dir / "decomposition.json"

    @property
    def stage2(self) -> Path:
        return self.out_dir / "stage2.json"


def stage_participants(stage: str, n_workers: int) -> list[int]:
    """One worker computes the vectors, so that all rows use one set. Impact searches share out across all."""
    if stage == "vectors":
        return [0]
    if stage in ("grid", "impact", "test"):
        return list(range(n_workers))
    return []
