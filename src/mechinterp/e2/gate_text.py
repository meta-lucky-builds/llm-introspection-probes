"""The gate's outcome in words. Standard library only, so the orchestrator and the script can use it."""

from __future__ import annotations

from typing import Any


def describe_gate(gate: dict[str, Any]) -> str:
    if gate["passed"]:
        band = gate["band"]
        return (
            f"GATE PASSED: the J-lens beats the logit lens on layers {band[0]} to {band[-1]} "
            f"(mean gain {gate['analysis']['decision']['mean_gain']:.3f}). Workspace layer l_w = {gate['l_w']}."
        )
    return (
        f"GATE FAILED: {gate['reason']} The J-lens does not validate on this model, so no J-space claim is made. "
        "Stages 2 and 3 do not run. This is a stop rule, not a reason to tune the lens."
    )
