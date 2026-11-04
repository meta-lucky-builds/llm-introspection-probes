"""How much of a card each kind of worker may take, from the dry run's measurements and the card's free memory. Standard library only.

Nothing here changes a number in the results. It decides where work runs, whether training uses gradient checkpointing, which
computes the same gradients, and the capture token budget, which changes how rows are batched and not what they are.

A card has one compute engine. Two processes on it take turns, so a second worker adds speed only where the first leaves the
card idle. So memory decides how many workers can fit, and utilization decides whether another one helps. An estimate of
that gain from memory alone would be wrong in the expensive direction, so the pool measures it instead. See TaskPool.can_share.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

CONTEXT_GB = 0.7  # A process's CUDA context and libraries on the card, which torch's own counters do not see.
ALLOCATOR_FACTOR = 1.15  # What the allocator holds over what is allocated at the peak.
MARGIN_FRACTION = 0.08
MIN_MARGIN_GB = 1.0
MAX_BUDGET_SCALE = 4.0  # Past this a batch gains almost nothing, and a failure costs a larger step back.
MIN_BUDGET_SCALE = 0.25
# The most a second training worker is assumed to add on a card, against a measured gain from dropping checkpointing. A guess.
SAME_CARD_GAIN = 1.15
FIT_NEED_GB = 2.5  # 20,000 rows at 6 layers of width 2048 in float32 is 1 GB, plus the MLP's weights and the process.


def usable_gb(free_gb: float) -> float:
    return free_gb - max(MIN_MARGIN_GB, MARGIN_FRACTION * free_gb)


def need_from_peak(peak_gb: float) -> float:
    return peak_gb * ALLOCATOR_FACTOR + CONTEXT_GB


@dataclass(frozen=True)
class Schedule:
    gradient_checkpointing: bool
    train_need_gb: float
    train_checkpointing_need_gb: float
    capture_need_gb: float
    token_budget: int
    notes: tuple[str, ...] = ()


def plan_schedule(free_gb: float | None, dry_run: Mapping[str, Any], default_need_gb: float, config_budget: int, workers_per_gpu: int | None = None) -> Schedule:
    """free_gb is the smallest card's free memory, or None if it could not be read. All cards then run one way, for simplicity.

    Anything the dry run did not measure, as on a CPU run or after an out-of-memory error in it, falls back to the configured
    figures: the same behaviour as before this planning existed.
    """
    checkpoint_peak = dry_run.get("train_peak_gpu_gb")
    checkpoint_need = need_from_peak(float(checkpoint_peak)) if checkpoint_peak else default_need_gb
    if free_gb is None:
        return Schedule(True, checkpoint_need, checkpoint_need, default_need_gb, config_budget, ("Free GPU memory is unknown, so the configured figures are used.",))
    usable = usable_gb(free_gb)
    wanted = workers_per_gpu or 1
    notes: list[str] = []

    # Training: leaving checkpointing off saves a forward pass per step, if the memory is there.
    plain_peak = dry_run.get("train_peak_gpu_gb_no_checkpointing")
    seconds_with = dry_run.get("train_seconds_per_step_on_longest_rows")
    seconds_without = dry_run.get("train_seconds_per_step_no_checkpointing")
    plain = False
    plain_need = checkpoint_need
    if plain_peak and seconds_with and seconds_without:
        plain_need = need_from_peak(float(plain_peak))
        fit_with_checkpointing = int(usable // checkpoint_need) if checkpoint_need > 0 else 1
        gain = float(seconds_with) / max(float(seconds_without), 1e-9)
        bar = 1.0 if workers_per_gpu or fit_with_checkpointing < 2 else SAME_CARD_GAIN
        plain = wanted * plain_need <= usable and gain > bar
        notes.append(
            f"Training: measured {float(seconds_with):.2f} s per step with gradient checkpointing and about {checkpoint_need:.1f} GB, {float(seconds_without):.2f} s without it and about {plain_need:.1f} GB, "
            f"on the longest rows. {usable:.1f} GB is usable on the smallest card, so checkpointing is {'off' if plain else 'on'}. The gradients are the same either way."
        )
    else:
        notes.append(f"Training: gradient checkpointing stays on, at about {checkpoint_need:.1f} GB, because the dry run has no measured figure without it.")
    if wanted * (plain_need if plain else checkpoint_need) > usable:
        notes.append(f"D1_WORKERS_PER_GPU={wanted} needs more memory than the smallest card has, by the dry run's figures. The pool backs off if it runs out.")

    # Capture: a larger batch, up to what fits.
    weights = dry_run.get("weights_gpu_gb")
    peak = dry_run.get("capture_peak_gpu_gb")
    budget = config_budget
    capture_need = default_need_gb
    if weights and peak and float(peak) > float(weights):
        extra = (float(peak) - float(weights)) * ALLOCATOR_FACTOR
        room = usable / wanted - CONTEXT_GB - float(weights)
        scale = min(MAX_BUDGET_SCALE, max(MIN_BUDGET_SCALE, room / extra))
        budget = max(1, int(config_budget * scale) // 1000 * 1000 or config_budget)
        capture_need = float(weights) + extra * scale + CONTEXT_GB
        notes.append(f"Capture: about {capture_need:.1f} GB per worker at a budget of {budget} tokens, {scale:.1f} times the config's {config_budget}, measured on the longest rows.")
    else:
        notes.append(f"Capture: stays at the config's budget of {config_budget} tokens, because the dry run did not measure its memory.")
    return Schedule(plain is False, plain_need if plain else checkpoint_need, checkpoint_need, capture_need, budget, tuple(notes))
