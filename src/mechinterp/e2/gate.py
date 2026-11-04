"""Stage 1's gate: J-lens versus logit-lens recall per layer, on numbers. No model code, and no torch.

Readings of the prereg, each listed in experiments/e2_jspace_vector_families/README.md:
- A band is a maximal run of consecutive layers in which every layer's own lower bound clears 0. Sub-windows of
  a run are not separate bands, otherwise "largest mean gain" would always pick a narrow peak.
- l_w is the middle layer of the winning band, and the lower of the two middles when its length is even.
- Recall pools items across evaluation sets, so the bootstrap can resample prompts directly.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

LEVEL_TAIL = 2.5


def recall_table(
    j_lens: np.ndarray, logit_lens: np.ndarray, layers: Sequence[int], n_boot: int, seed: int
) -> list[dict[str, float]]:
    """Per-layer recall and paired intervals. j_lens and logit_lens are [n_items, n_layers] of per-item recall."""
    if j_lens.shape != logit_lens.shape or j_lens.ndim != 2 or j_lens.shape[1] != len(layers):
        raise ValueError(f"Recall shapes do not fit: {j_lens.shape}, {logit_lens.shape}, {len(layers)} layers")
    n_items = j_lens.shape[0]
    if n_items == 0:
        raise ValueError("No scorable items")
    picks = np.random.default_rng(seed).integers(0, n_items, size=(n_boot, n_items))
    rows = []
    for column, layer in enumerate(layers):
        j_boot = j_lens[picks, column].mean(axis=1)
        logit_boot = logit_lens[picks, column].mean(axis=1)
        gain_boot = j_boot - logit_boot
        rows.append(
            {
                "layer": int(layer),
                "j_recall": float(j_lens[:, column].mean()),
                "logit_recall": float(logit_lens[:, column].mean()),
                "gain": float(j_lens[:, column].mean() - logit_lens[:, column].mean()),
                "gain_lo": float(np.percentile(gain_boot, LEVEL_TAIL)),
                "gain_hi": float(np.percentile(gain_boot, 100.0 - LEVEL_TAIL)),
                "j_lo": float(np.percentile(j_boot, LEVEL_TAIL)),
                "j_hi": float(np.percentile(j_boot, 100.0 - LEVEL_TAIL)),
                "logit_lo": float(np.percentile(logit_boot, LEVEL_TAIL)),
                "logit_hi": float(np.percentile(logit_boot, 100.0 - LEVEL_TAIL)),
            }
        )
    return rows


def find_bands(table: Sequence[dict[str, float]], min_layers: int) -> list[dict[str, Any]]:
    """Maximal runs of consecutive layers whose gain lower bound is above 0, at least min_layers long."""
    bands: list[dict[str, Any]] = []
    run: list[dict[str, float]] = []

    def close() -> None:
        if len(run) >= min_layers:
            bands.append(
                {
                    "layers": [int(row["layer"]) for row in run],
                    "mean_gain": sum(row["gain"] for row in run) / len(run),
                }
            )

    for row in table:
        if row["gain_lo"] > 0.0 and (not run or row["layer"] == run[-1]["layer"] + 1):
            run.append(row)
            continue
        close()
        run = [row] if row["gain_lo"] > 0.0 else []
    close()
    return bands


def decide_gate(table: Sequence[dict[str, float]], min_layers: int) -> dict[str, Any]:
    """Apply the stop rule. The earliest band wins an exact tie, so the result is deterministic."""
    bands = find_bands(table, min_layers)
    if not bands:
        best_row = max(table, key=lambda row: row["gain_lo"])
        return {
            "passed": False,
            "bands": [],
            "band": None,
            "l_w": None,
            "best_layer_lower_bound": {"layer": best_row["layer"], "gain_lo": best_row["gain_lo"], "gain": best_row["gain"]},
            "reason": f"No {min_layers} consecutive layers have a lower bound of J-lens minus logit-lens recall above 0.",
        }
    best = max(bands, key=lambda band: band["mean_gain"])
    layers = best["layers"]
    return {
        "passed": True,
        "bands": bands,
        "band": layers,
        "l_w": layers[(len(layers) - 1) // 2],
        "mean_gain": best["mean_gain"],
        "reason": None,
    }
