"""E2's trial plans. No model code and no torch, so the orchestrator can check merged rows against them.

Calibration trials are exactly E1's impact-search trials: T_inj, calibration concepts on calibration clusters,
through plan_impact_trials. Test trials are exactly E1's test cells, through _iter_cells. Only the vector
differs, and that goes in the trial's condition as the family code.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from ..config import Config
from ..design import PHASE_GRID, PHASE_IMPACT, PHASE_TEST, Stimuli, Trial, _iter_cells, plan_impact_trials
from .config import FAMILIES, MATCHED_FAMILIES


def unique_sites(l_s: int, l_w: int) -> list[int]:
    """Two sites, unless the workspace layer is the layer E1 selected, when there is only one to test."""
    return [l_s] if l_s == l_w else [l_s, l_w]


def plan_calibration(
    stimuli: Stimuli,
    config: Config,
    family: str,
    layer: int,
    alpha: float,
    phase: str = PHASE_IMPACT,
    concepts: Sequence[str] | None = None,
) -> list[Trial]:
    """concepts narrows the plan to the concepts that have this family, with no change to the rest of it."""
    kept = None if concepts is None else set(concepts)
    return [
        replace(trial, phase=phase, condition=family)
        for trial in plan_impact_trials(stimuli, config, layer, alpha)
        if kept is None or trial.concept in kept
    ]


def plan_full_grid(stimuli: Stimuli, config: Config, l_s: int, alpha_s: float, l_w: int) -> list[Trial]:
    """F_full on calibration stimuli: E1's alpha at l_s, and E1's alpha grid at l_w, where none is known yet."""
    trials = plan_calibration(stimuli, config, "F_full", l_s, alpha_s, PHASE_GRID)
    if l_w != l_s:
        for alpha in config.grid.alphas:
            trials.extend(plan_calibration(stimuli, config, "F_full", l_w, alpha, PHASE_GRID))
    return trials


def plan_searches(sites: Sequence[int], lens_concepts: Sequence[str]) -> list[tuple[str, int]]:
    """Every impact search, in a fixed order. F_lens only if some concept has a single token."""
    return [
        (family, layer)
        for layer in sites
        for family in MATCHED_FAMILIES
        if family != "F_lens" or len(lens_concepts) > 0
    ]


def search_owner(index: int, n_workers: int) -> int:
    return index % n_workers


def alpha_key(family: str, layer: int) -> str:
    return f"{family}|{layer}"


def plan_test(
    stimuli: Stimuli,
    config: Config,
    tasks: Sequence[str],
    sites: Sequence[int],
    alphas: dict[str, float],
    test_lens_concepts: Sequence[str],
) -> list[Trial]:
    """All families, both sites and every task, fully crossed over E1's test concepts, clusters and positions.

    F_lens runs only for test concepts that are one token. Nothing else is left out.
    """
    design = config.design
    cells = list(
        _iter_cells(
            stimuli.concepts_test, list(stimuli.clusters_test), design.positions_per_cell_test, design.clusters_per_concept_test
        )
    )
    lens_concepts = set(test_lens_concepts)
    trials = []
    for layer in sites:
        for family in FAMILIES:
            key = alpha_key(family, layer)
            if key not in alphas:
                continue
            for task in tasks:
                for concept, _, cluster, position in cells:
                    if family == "F_lens" and concept not in lens_concepts:
                        continue
                    trials.append(Trial(PHASE_TEST, task, family, concept, cluster, position, None, layer, alphas[key]))
    return trials


def plan_counts(stimuli: Stimuli, config: Config, tasks: Sequence[str], n_lens_test: int, n_lens_calibration: int) -> dict[str, int]:
    """Prompts per phase, with the impact searches at their worst case and two distinct sites."""
    design = config.design
    per_cell = len(stimuli.clusters_test) * design.positions_per_cell_test
    per_family = len(stimuli.concepts_test) * per_cell * len(tasks)
    calibration_cells = len(plan_impact_trials(stimuli, config, 0, 1.0))
    per_lens_concept = calibration_cells // max(1, len(stimuli.concepts_calibration))
    evals = config.impact_search.max_evals
    return {
        "grid": calibration_cells * (1 + len(config.grid.alphas)),
        "impact_search_max": 2 * evals * (3 * calibration_cells + n_lens_calibration * per_lens_concept),
        "test": 2 * (4 * per_family + n_lens_test * per_cell * len(tasks)),
    }
