"""How the fixed test size divides across families. Standard library only, and hashed with what produces captures."""

from __future__ import annotations

from typing import Mapping


def allocate_per_class(total_transcripts: int, available_per_class: Mapping[str, int]) -> dict[str, int]:
    """Transcripts per class per family that add up to the total: equal shares, and what a small family cannot use goes to the rest."""
    left = total_transcripts // 2
    share = {name: 0 for name in available_per_class}
    open_families = sorted(name for name, count in available_per_class.items() if count > 0)
    while left > 0 and open_families:
        step = max(1, left // len(open_families))
        for name in open_families:
            if left == 0:
                break
            grant = min(step, available_per_class[name] - share[name], left)
            share[name] += grant
            left -= grant
        open_families = [name for name in open_families if share[name] < available_per_class[name]]
    return share
