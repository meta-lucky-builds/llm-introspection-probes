"""Stimuli, splits, trial plans and (layer, alpha) selection. No model code lives here."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

from .config import Config
from .prompts import LIST_SIZE

N_BASELINE_WORDS = 50
PHASE_GRID = "calibration_grid"
PHASE_IMPACT = "impact_search"
PHASE_SCREEN = "screen"
PHASE_TEST = "test"


@dataclass(frozen=True)
class RawData:
    concepts_calibration: tuple[str, ...]
    concepts_test: tuple[str, ...]
    baseline_words: tuple[str, ...]
    clusters_calibration: dict[str, tuple[str, ...]]
    clusters_test: dict[str, tuple[str, ...]]
    sha256: str


@dataclass(frozen=True)
class Stimuli:
    concepts_calibration: tuple[str, ...]
    concepts_test: tuple[str, ...]
    baseline_words: tuple[str, ...]
    clusters_calibration: dict[str, tuple[str, ...]]
    clusters_test: dict[str, tuple[str, ...]]
    dropped_words: dict[str, list[str]]
    dropped_clusters: list[str]

    @property
    def all_clusters(self) -> dict[str, tuple[str, ...]]:
        return {**self.clusters_calibration, **self.clusters_test}


@dataclass(frozen=True)
class Trial:
    phase: str
    task: str
    condition: str
    concept: str
    cluster: str
    position: int
    query: int | None
    layer: int
    alpha: float

    @property
    def key(self) -> str:
        return (
            f"{self.phase}|{self.task}|{self.condition}|{self.concept}|{self.cluster}"
            f"|j{self.position}|q{self.query}|l{self.layer}|a{self.alpha:.6g}"
        )

    @property
    def clean_key(self) -> tuple[str, str, int | None]:
        return (self.task, self.cluster, self.query)


@dataclass(frozen=True)
class Selection:
    layer: int
    alpha: float
    accuracy: float
    grid_accuracy: dict[str, float]


def load_data(data_dir: str | Path) -> RawData:
    data_dir = Path(data_dir)
    texts = {
        name: (data_dir / name).read_text(encoding="utf-8")
        for name in ("concepts.json", "baseline_words.json", "candidate_clusters.json")
    }
    concepts = json.loads(texts["concepts.json"])
    baseline = json.loads(texts["baseline_words.json"])
    clusters = json.loads(texts["candidate_clusters.json"])
    digest = hashlib.sha256("".join(texts[name] for name in sorted(texts)).encode("utf-8")).hexdigest()
    raw = RawData(
        concepts_calibration=tuple(concepts["calibration"]),
        concepts_test=tuple(concepts["test"]),
        baseline_words=tuple(baseline["baseline"]),
        clusters_calibration={name: tuple(words) for name, words in clusters["calibration"].items()},
        clusters_test={name: tuple(words) for name, words in clusters["test"].items()},
        sha256=digest,
    )
    validate_data(raw)
    return raw


def _assert_disjoint(first: Iterable[str], second: Iterable[str], label: str) -> None:
    overlap = {word.lower() for word in first} & {word.lower() for word in second}
    if overlap:
        raise ValueError(f"{label} overlap: {sorted(overlap)}")


def validate_data(raw: RawData) -> None:
    if len(raw.baseline_words) != N_BASELINE_WORDS or len(set(raw.baseline_words)) != N_BASELINE_WORDS:
        raise ValueError(f"The prereg needs {N_BASELINE_WORDS} distinct baseline words")
    _assert_disjoint(raw.concepts_calibration, raw.concepts_test, "Calibration and test concepts")
    _assert_disjoint(raw.clusters_calibration, raw.clusters_test, "Calibration and test cluster names")
    bank = [word for words in raw.clusters_calibration.values() for word in words]
    bank += [word for words in raw.clusters_test.values() for word in words]
    if len({word.lower() for word in bank}) != len(bank):
        raise ValueError("Candidate words must be unique across clusters")
    concepts = raw.concepts_calibration + raw.concepts_test
    if len(set(concepts)) != len(concepts):
        raise ValueError("Concept words must be unique")
    _assert_disjoint(concepts, raw.baseline_words, "Concepts and baseline words")
    _assert_disjoint(concepts, bank, "Concepts and candidate words")
    _assert_disjoint(raw.baseline_words, bank, "Baseline words and candidate words")


def assert_splits_disjoint(stimuli: Stimuli) -> None:
    _assert_disjoint(stimuli.concepts_calibration, stimuli.concepts_test, "Calibration and test concepts")
    _assert_disjoint(stimuli.clusters_calibration, stimuli.clusters_test, "Calibration and test clusters")
    calibration_words = [word for words in stimuli.clusters_calibration.values() for word in words]
    test_words = [word for words in stimuli.clusters_test.values() for word in words]
    _assert_disjoint(calibration_words, test_words, "Calibration and test candidate words")


def _seeded_shuffle(items: Sequence[Any], seed: int, tag: str) -> list[Any]:
    # random.Random seeded with a string is stable across processes and platforms.
    shuffled = list(items)
    random.Random(f"{seed}:{tag}").shuffle(shuffled)
    return shuffled


def _prepare_clusters(
    clusters: dict[str, tuple[str, ...]],
    needed: int,
    seed: int,
    keep_word: Callable[[str], bool],
    dropped_words: dict[str, list[str]],
    dropped_clusters: list[str],
    split: str,
) -> dict[str, tuple[str, ...]]:
    prepared: dict[str, tuple[str, ...]] = {}
    for name in _seeded_shuffle(sorted(clusters), seed, f"clusters:{split}"):
        survivors = [word for word in clusters[name] if keep_word(word)]
        removed = [word for word in clusters[name] if word not in survivors]
        if removed:
            dropped_words[name] = removed
        if len(survivors) < LIST_SIZE:
            dropped_clusters.append(name)
            continue
        if len(prepared) < needed:
            prepared[name] = tuple(_seeded_shuffle(survivors, seed, f"words:{name}")[:LIST_SIZE])
    if len(prepared) < needed:
        raise ValueError(f"Only {len(prepared)} {split} clusters survive filtering, config needs {needed}")
    return prepared


def prepare_stimuli(raw: RawData, config: Config, keep_word: Callable[[str], bool]) -> Stimuli:
    """Choose the tier's concepts and clusters, keeping only single-token candidate words."""
    design = config.design
    concepts_calibration = _seeded_shuffle(raw.concepts_calibration, config.seed, "concepts:calibration")
    concepts_test = _seeded_shuffle(raw.concepts_test, config.seed, "concepts:test")
    if design.n_concepts_calibration > len(concepts_calibration) or design.n_concepts_test > len(concepts_test):
        raise ValueError("The config asks for more concepts than the data files hold")
    dropped_words: dict[str, list[str]] = {}
    dropped_clusters: list[str] = []
    stimuli = Stimuli(
        concepts_calibration=tuple(concepts_calibration[: design.n_concepts_calibration]),
        concepts_test=tuple(concepts_test[: design.n_concepts_test]),
        baseline_words=raw.baseline_words,
        clusters_calibration=_prepare_clusters(
            raw.clusters_calibration,
            design.n_clusters_calibration,
            config.seed,
            keep_word,
            dropped_words,
            dropped_clusters,
            "calibration",
        ),
        clusters_test=_prepare_clusters(
            raw.clusters_test,
            design.n_clusters_test,
            config.seed,
            keep_word,
            dropped_words,
            dropped_clusters,
            "test",
        ),
        dropped_words=dropped_words,
        dropped_clusters=dropped_clusters,
    )
    assert_splits_disjoint(stimuli)
    return stimuli


def cell_positions(concept_index: int, cluster_index: int, n_clusters: int, per_cell: int) -> list[int]:
    """Injected positions for one (concept, cluster) cell.

    Within a cluster, concepts take consecutive blocks of `per_cell` positions, so every position
    occurs equally often when n_concepts * per_cell is a multiple of ten. per_cell = 10 is the fully
    crossed design. The cluster offset varies which positions a concept meets across clusters.
    """
    start = concept_index * per_cell + cluster_index * max(1, LIST_SIZE // n_clusters)
    return [(start + step) % LIST_SIZE for step in range(per_cell)]


def _iter_cells(
    concepts: Sequence[str], clusters: Sequence[str], per_cell: int
) -> Iterator[tuple[str, int, str, int]]:
    """Yield (concept, cluster_index, cluster, position) for a design balanced within clusters."""
    if not 1 <= per_cell <= LIST_SIZE:
        raise ValueError(f"positions per cell must be between 1 and {LIST_SIZE}")
    if (len(concepts) * per_cell) % LIST_SIZE != 0:
        raise ValueError(
            f"{len(concepts)} concepts with {per_cell} positions per cell cannot balance "
            f"{LIST_SIZE} positions within a cluster"
        )
    for concept_index, concept in enumerate(concepts):
        for cluster_index, cluster in enumerate(clusters):
            for position in cell_positions(concept_index, cluster_index, len(clusters), per_cell):
                yield concept, cluster_index, cluster, position


def fact_query(position: int, cluster_index: int, step: int) -> int:
    """A queried position different from the injected one.

    The offset is constant within a cluster for a given step, so queried positions inherit the
    balance of injected positions.
    """
    offset = 1 + (cluster_index + step) % (LIST_SIZE - 1)
    return (position + offset) % LIST_SIZE


def plan_grid_trials(stimuli: Stimuli, config: Config) -> list[Trial]:
    design = config.design
    trials = []
    for layer in config.grid.layers:
        for alpha in config.grid.alphas:
            for concept, _, cluster, position in _iter_cells(
                stimuli.concepts_calibration,
                list(stimuli.clusters_calibration),
                design.positions_per_cell_calibration,
            ):
                trials.append(Trial(PHASE_GRID, "T_inj", "concept", concept, cluster, position, None, layer, alpha))
    return trials


def plan_impact_trials(stimuli: Stimuli, config: Config, layer: int, alpha: float) -> list[Trial]:
    return [
        Trial(PHASE_IMPACT, "T_inj", "random_impact", concept, cluster, position, None, layer, alpha)
        for concept, _, cluster, position in _iter_cells(
            stimuli.concepts_calibration,
            list(stimuli.clusters_calibration),
            config.design.positions_per_cell_calibration,
        )
    ]


def plan_screen_trials(stimuli: Stimuli, config: Config, layer: int, alpha: float) -> list[Trial]:
    """Test concepts on calibration clusters: screening never sees a test cluster."""
    return [
        Trial(PHASE_SCREEN, "T_inj", "concept", concept, cluster, position, None, layer, alpha)
        for concept, _, cluster, position in _iter_cells(
            stimuli.concepts_test,
            list(stimuli.clusters_calibration),
            config.screening.positions_per_cell,
        )
    ]


def plan_test_trials(
    stimuli: Stimuli,
    config: Config,
    layer: int,
    alpha: float,
    alpha_impact: float,
    tasks: Sequence[str] | None = None,
) -> list[Trial]:
    design = config.design
    trials = []
    for task, conditions in design.test_task_conditions.items():
        if tasks is not None and task not in tasks:
            continue
        for condition in conditions:
            trial_alpha = alpha_impact if condition == "random_impact" else alpha
            for concept, cluster_index, cluster, position in _iter_cells(
                stimuli.concepts_test, list(stimuli.clusters_test), design.positions_per_cell_test
            ):
                if task == "T_fact":
                    for step in range(design.fact_queries_per_trial):
                        query = fact_query(position, cluster_index, step)
                        trials.append(
                            Trial(PHASE_TEST, task, condition, concept, cluster, position, query, layer, trial_alpha)
                        )
                else:
                    trials.append(
                        Trial(PHASE_TEST, task, condition, concept, cluster, position, None, layer, trial_alpha)
                    )
    return trials


def plan_clean_keys(
    tasks: Iterable[str], clusters: Iterable[str]
) -> list[tuple[str, str, int | None]]:
    """Clean prompts. T_fact takes all ten queries so that gate V1 covers every position."""
    keys: list[tuple[str, str, int | None]] = []
    clusters = list(clusters)
    for task in tasks:
        for cluster in clusters:
            if task == "T_fact":
                keys.extend((task, cluster, query) for query in range(LIST_SIZE))
            else:
                keys.append((task, cluster, None))
    return keys


def assert_trial_respects_split(trial: Trial, stimuli: Stimuli) -> None:
    calibration_concept = trial.concept in stimuli.concepts_calibration
    test_concept = trial.concept in stimuli.concepts_test
    calibration_cluster = trial.cluster in stimuli.clusters_calibration
    test_cluster = trial.cluster in stimuli.clusters_test
    if trial.phase in (PHASE_GRID, PHASE_IMPACT):
        allowed = calibration_concept and calibration_cluster
    elif trial.phase == PHASE_SCREEN:
        allowed = test_concept and calibration_cluster
    elif trial.phase == PHASE_TEST:
        allowed = test_concept and test_cluster
    else:
        raise ValueError(f"Unknown phase {trial.phase}")
    if not allowed:
        raise AssertionError(f"Trial violates the split: {trial.key}")


def select_layer_alpha(
    records: Sequence[dict[str, Any]], calibration_concepts: Sequence[str], calibration_clusters: Sequence[str]
) -> Selection:
    """Pick (layer, alpha) by T_inj accuracy on calibration data.

    Any record from outside the calibration split is an error. Ties go to the smallest alpha and
    then the earliest layer, so nothing other than T_inj accuracy influences the choice.
    """
    hits: dict[tuple[int, float], list[int]] = {}
    for record in records:
        if (
            record.get("phase") != PHASE_GRID
            or record.get("task") != "T_inj"
            or record.get("condition") != "concept"
            or record.get("concept") not in calibration_concepts
            or record.get("cluster") not in calibration_clusters
        ):
            raise AssertionError(f"Selection received a non-calibration record: {record.get('key')}")
        hits.setdefault((record["layer"], record["alpha"]), []).append(int(record["hit"]))
    if not hits:
        raise ValueError("Selection received no records")
    if len({len(values) for values in hits.values()}) != 1:
        raise ValueError("Selection needs the same number of trials at every grid point")
    accuracy = {point: sum(values) / len(values) for point, values in hits.items()}
    best = min(accuracy, key=lambda point: (-accuracy[point], point[1], point[0]))
    return Selection(
        layer=best[0],
        alpha=best[1],
        accuracy=accuracy[best],
        grid_accuracy={f"{layer}|{alpha:g}": value for (layer, alpha), value in sorted(accuracy.items())},
    )


def plan_prompt_counts(stimuli: Stimuli, config: Config) -> dict[str, int]:
    """Prompt counts per phase, with the impact search at its worst case."""
    tasks = list(config.design.test_task_conditions)
    grid = len(plan_grid_trials(stimuli, config))
    impact_per_eval = len(plan_impact_trials(stimuli, config, 0, 1.0))
    counts = {
        "vector_prompts": len(stimuli.concepts_calibration) + len(stimuli.concepts_test) + len(stimuli.baseline_words),
        "clean_calibration": len(plan_clean_keys(["T_inj"], stimuli.clusters_calibration)),
        "calibration_grid": grid,
        "impact_search_max": impact_per_eval * config.impact_search.max_evals,
        "screen": len(plan_screen_trials(stimuli, config, 0, 1.0)) if config.screening.enabled else 0,
        "clean_test": len(plan_clean_keys(tasks, stimuli.clusters_test)),
        "test": len(plan_test_trials(stimuli, config, 0, 1.0, 1.0)),
    }
    return counts
