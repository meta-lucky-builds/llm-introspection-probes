"""D1's configuration. Standard library only, because the orchestrator never imports torch."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

from .families import check_partition

PREREG_FRACTIONS = ("0.3", "0.42", "0.55", "0.67", "0.8", "0.9")
CONDITIONS = ("DD", "SHAM", "FLIP", "HONEST")


@dataclass(frozen=True)
class Rung:
    name: str
    revision: str
    dtype: str  # float32 or bfloat16. Fixed per rung, so results do not depend on the GPU that happened to be rented.
    gpu_gb: float  # Free memory a run needs: weights plus room to train through them.
    n_layers: int
    d_model: int

    @property
    def tag(self) -> str:
        return self.name.split("/")[-1].lower().replace(".", "_")


@dataclass(frozen=True)
class Sizes:
    probe_fit_rows: int  # Prereg: up to 20,000.
    probe_val_rows: int
    max_tokens: int  # Longer prompts are dropped, by the same rule in every set.
    calibration_per_class_per_family: int
    g1_per_class_per_family: int
    max_per_class_per_family: int
    sizing_grid: tuple[int, ...]
    target_half_width: float
    sham_check_rows: int


@dataclass(frozen=True)
class Training:
    seeds: tuple[int, ...]
    rank: int
    alpha: float
    steps: int
    batch_size: int
    micro_batch: int
    learning_rate: float
    warmup_steps: int
    max_tokens: int
    ordinary_answer_tokens: int
    manipulation_rows_per_batch: int
    fact_rows_per_batch: int
    n_base_rows: int
    n_ordinary_rows: int


@dataclass(frozen=True)
class ProbeTraining:
    batch_size: int = 32
    max_epochs: int = 6
    learning_rate: float = 5e-3
    grad_clip: float = 1.0
    warmup_steps: int = 20
    scale_rows: int = 512
    drop_loss: float = 0.6
    log_every: int = 25
    mlp_hidden: int = 256
    seed: int = 0


@dataclass(frozen=True)
class Analysis:
    n_bootstrap: int = 2000
    n_random_directions: int = 20
    n_sizing_bootstrap: int = 500
    seed: int = 20261105


@dataclass(frozen=True)
class D1Config:
    seed: int
    dataset_repo: str
    dataset_revision: str
    parquet_sha256: dict[str, str]
    ladder: tuple[Rung, ...]
    fractions: tuple[Fraction, ...]
    sizes: Sizes
    training: Training
    probes: ProbeTraining
    analysis: Analysis
    partition: dict[str, list[str]]
    wordings_train: tuple[str, ...]
    wordings_held_out: tuple[str, ...]
    wording_sham: str
    facts_instruction: str
    facts_eval: tuple[dict[str, str], ...]
    facts_retain: tuple[dict[str, str], ...]
    expected_counts: dict[str, Any]
    capture_token_budget: int
    config_hash: str
    data_hash: str
    base_dir: Path = field(default=Path("."))

    @classmethod
    def from_file(cls, path: str | Path) -> "D1Config":
        path = Path(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(raw, path.parent)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], base_dir: Path) -> "D1Config":
        data_dir = (base_dir / raw["data_dir"]).resolve()
        texts = {name: (data_dir / f"{name}.json").read_bytes().replace(b"\r\n", b"\n") for name in ("wordings", "facts", "partition", "fibs_expected_counts")}
        wordings = json.loads(texts["wordings"])
        facts = json.loads(texts["facts"])
        partition_file = json.loads(texts["partition"])
        check_wordings(wordings)
        check_facts(facts)
        partition = {name: list(partition_file[name]) for name in ("probe_train", "adapter_train", "test")}
        check_partition(partition)
        digest = hashlib.sha256()
        for name in sorted(texts):
            digest.update(name.encode("utf-8"))
            digest.update(texts[name])
        canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"))
        fractions = tuple(Fraction(text) for text in raw["fractions"])
        if tuple(raw["fractions"]) != PREREG_FRACTIONS:
            raise ValueError(f"The prereg fixes the depth fractions at {PREREG_FRACTIONS}")
        training = dict(raw["training"])
        training["seeds"] = tuple(training["seeds"])
        sizes = dict(raw["sizes"])
        sizes["sizing_grid"] = tuple(sizes["sizing_grid"])
        return cls(
            seed=int(raw["seed"]),
            dataset_repo=raw["dataset"]["repo"],
            dataset_revision=raw["dataset"]["revision"],
            parquet_sha256=dict(raw["dataset"]["parquet_sha256"]),
            ladder=tuple(Rung(**rung) for rung in raw["ladder"]),
            fractions=fractions,
            sizes=Sizes(**sizes),
            training=Training(**training),
            probes=ProbeTraining(**raw.get("probes", {})),
            analysis=Analysis(**raw.get("analysis", {})),
            partition=partition,
            wordings_train=tuple(wordings["train"]),
            wordings_held_out=tuple(wordings["held_out"]),
            wording_sham=wordings["sham"],
            facts_instruction=facts["instruction"],
            facts_eval=tuple(facts["eval"]),
            facts_retain=tuple(facts["retain"]),
            expected_counts=json.loads(texts["fibs_expected_counts"]),
            capture_token_budget=int(raw["capture_token_budget"]),
            config_hash=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            data_hash=digest.hexdigest(),
            base_dir=base_dir,
        )

    def rung(self, tag: str) -> Rung:
        for rung in self.ladder:
            if rung.tag == tag:
                return rung
        raise KeyError(tag)


def check_wordings(wordings: dict[str, Any], n_train: int = 20, n_held_out: int = 10) -> None:
    train, held_out = wordings["train"], wordings["held_out"]
    if len(train) != n_train or len(held_out) != n_held_out:
        raise ValueError(f"Need {n_train} train and {n_held_out} held-out wordings, found {len(train)} and {len(held_out)}")
    seen = {text.strip().casefold() for text in [*train, *held_out]}
    if len(seen) != n_train + n_held_out:
        raise ValueError("Wordings repeat, or cross the train and held-out split")
    reversed_terms = ("lie", "lying", "lied", "mislead", "misled", "false", "untruth", "dishonest", "deceiv", "wrong", "inaccura")
    for text in [*train, *held_out]:
        lowered = text.casefold()
        # Word starts, so that "believe" does not match "lie". A negative word would reverse which answer admits.
        words = lowered.replace(",", " ").replace(":", " ").replace("?", " ").split()
        if any(word.startswith(term) for word in words for term in reversed_terms):
            raise ValueError(f"Wording has reversed polarity, so No would not be the admission: {text!r}")
        if not text.endswith("?"):
            raise ValueError(f"Wording must be a question: {text!r}")


def check_facts(facts: dict[str, Any]) -> None:
    for name in ("eval", "retain"):
        answers = [item["a"] for item in facts[name]]
        if set(answers) != {"Yes", "No"} or answers.count("Yes") != answers.count("No"):
            raise ValueError(f"Fact list {name} must balance Yes and No")
    questions = [item["q"].strip().casefold() for name in ("eval", "retain") for item in facts[name]]
    if len(set(questions)) != len(questions):
        raise ValueError("A fact repeats within or across the lists")


def layer_for_fraction(fraction: Fraction, n_layers: int) -> int:
    """Depth fraction to a 0-indexed decoder block. Block k's output is hidden_states[k + 1].

    round(fraction * n_layers), half up, exactly: this reproduces every row of the table in the FIBS probe README, for
    example 24 blocks to 7, 10, 13, 16, 19, 22. It uses no N - 1 and no off-by-one offset. Fractions keep 0.55 * 20 at
    exactly 11, where floats would not.
    """
    return int(fraction * n_layers + Fraction(1, 2))


def layers_for(fractions: tuple[Fraction, ...], n_layers: int) -> list[int]:
    layers = [layer_for_fraction(fraction, n_layers) for fraction in fractions]
    if layers != sorted(set(layers)) or layers[-1] >= n_layers or layers[0] < 0:
        raise ValueError(f"{n_layers} layers give an unusable layer list {layers}")
    return layers
