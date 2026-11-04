"""E1's Session with E2's vectors and impact searches. Forward passes, hooks and row code are inherited unchanged."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from ..checkpoint import ResultStore
from ..design import PHASE_GRID, PHASE_IMPACT, Trial
from ..runner import Session, _mean_js
from .config import E2Config, e2_code_hash
from .design import plan_calibration
from .families import family_key


def impact_key(family: str, layer: int) -> str:
    return f"impact_match|{family}|l{layer}"


class E2Session(Session):
    def __init__(
        self,
        e2: E2Config,
        out_dir: str | Path,
        model: nn.Module | None = None,
        tokenizer: Any = None,
        device: torch.device | None = None,
        model_commit: str | None = None,
        store: ResultStore | None = None,
    ) -> None:
        self.e2 = e2
        self.family_vectors: dict[str, torch.Tensor] = {}
        super().__init__(e2.base, out_dir, model, tokenizer, device, model_commit, store)

    def _assert_same_identity(self) -> None:
        """The base identity hashes E1's folder only, so E2's code and config join it here, before any row."""
        self.identity = {**self.identity, "e2_code_hash": e2_code_hash(), "e2_config_hash": self.e2.e2_config_hash}
        super()._assert_same_identity()

    def _available_test_tasks(self) -> list[str]:
        return list(self.e2.tasks)

    def _ensure_vectors(self) -> None:
        if self.family_vectors:
            return
        path = self.out_dir / "family_vectors.pt"
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["identity"] != self.identity:
            raise ValueError(f"{path} was written by a different run")
        self.family_vectors = saved["vectors"]

    def _vector(self, trial: Trial) -> torch.Tensor:
        vector = self.family_vectors.get(family_key(trial.condition, trial.concept, trial.layer))
        if vector is not None:
            return vector
        if trial.condition.startswith("F_"):
            raise KeyError(f"No {trial.condition} vector for {trial.concept} at layer {trial.layer}")
        return super()._vector(trial)  # The preflight's throughput rows, which use E1's conditions.

    def impact_search(self, family: str, layer: int, start_alpha: float, concepts: Sequence[str] | None) -> None:
        """E1's search, per family and site: same trials, steps, limits and JS. Calibration data only.

        The target is F_full's mean JS at that site's alpha. F_lens targets F_full on the same concepts, so a
        family with fewer concepts is compared like with like. Replays on resume from rows keyed by alpha.
        """
        key = impact_key(family, layer)
        if self.store.has(key):
            return
        settings = self.config.impact_search
        target_trials = plan_calibration(self.stimuli, self.config, "F_full", layer, start_alpha, PHASE_GRID, concepts)
        target = _mean_js([self.store.get(trial.key) for trial in target_trials])
        max_alpha = start_alpha * settings.max_scale

        def relative_error(js: float) -> float:
            return abs(js - target) / max(target, 1e-12)

        history: list[dict[str, float]] = []
        low: float | None = None
        high: float | None = None
        alpha = start_alpha
        for _ in range(settings.max_evals):
            alpha = float(f"{alpha:.6g}")  # Trial keys hold six significant digits.
            trials = plan_calibration(self.stimuli, self.config, family, layer, alpha, PHASE_IMPACT, concepts)
            self._run_trials(trials, measure_js=True)
            js = _mean_js([self.store.get(trial.key) for trial in trials])
            history.append({"alpha": alpha, "js": js})
            if relative_error(js) <= settings.tolerance:
                break
            if js < target:
                low = alpha
            else:
                high = alpha
            if low is not None and high is not None:
                alpha = math.sqrt(low * high)
            elif low is not None:
                if low >= max_alpha:
                    break
                alpha = min(low * 2.0, max_alpha)
            else:
                alpha = high / 2.0  # type: ignore[operator]

        best = min(history, key=lambda item: relative_error(item["js"]))
        error = relative_error(best["js"])
        self.store.append(
            [
                {
                    "key": key,
                    "kind": "impact_match",
                    "family": family,
                    "layer": layer,
                    "concepts": list(concepts) if concepts is not None else None,
                    "target_js": target,
                    "achieved_js": best["js"],
                    "relative_error": error,
                    "alpha_impact": best["alpha"],
                    "alpha_full": start_alpha,
                    "scale": best["alpha"] / start_alpha,
                    # search_tolerance stops the search, as in E1. The prereg excludes a family above 10%.
                    "search_tolerance": settings.tolerance,
                    "exclusion_tolerance": self.e2.exclusion_tolerance,
                    "match_ok": error <= self.e2.exclusion_tolerance,
                    "history": history,
                }
            ]
        )
        print(f"{family} at layer {layer}: alpha {best['alpha']:.4g}, JS {best['js']:.5f} against {target:.5f}, error {error:.3f}", flush=True)
