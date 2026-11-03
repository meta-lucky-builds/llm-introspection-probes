"""Checkpointed experiment runner.

Every step is replayable from the JSONL store: a resumed run walks the same phases, and any trial
whose key is already stored is skipped. Prompts are batched by exact token length, so nothing is
ever padded and the "last real token" is always index -1.
"""

from __future__ import annotations

import math
import os
import platform
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch import nn

from .checkpoint import ResultStore
from .config import Config
from .design import (
    PHASE_GRID,
    PHASE_IMPACT,
    PHASE_SCREEN,
    PHASE_TEST,
    Selection,
    Stimuli,
    Trial,
    assert_trial_respects_split,
    load_data,
    plan_clean_keys,
    plan_grid_trials,
    plan_impact_trials,
    plan_prompt_counts,
    plan_screen_trials,
    plan_test_trials,
    prepare_stimuli,
    select_layer_alpha,
)
from .export import build_export, encode_export
from .hooks import inject
from .identity import identity_differences, run_identity
from .modeling import (
    choose_device,
    choose_dtype,
    forward_last_logits,
    get_layers,
    hidden_size,
    load_model_and_tokenizer,
    resolve_model_commit,
)
from .prompts import (
    LIST_SIZE,
    RenderedPrompt,
    build_prompt,
    concept_prompt_text,
    decode_token,
    is_single_token_word,
    locate_user_content_end,
    render_chat,
    resolve_label_ids,
    resolve_none_id,
    token_hash,
)
from .scoring import answer_mass, assert_finite, js_divergence, localization_effect, margin
from .stats import NonFiniteError
from .vectors import concept_vectors, fingerprints_match, random_unit_vector, vector_fingerprint

CleanKey = tuple[str, str, "int | None"]
NONE_TASK = "T_inj_none"
SHAM_TOLERANCE = 1e-4
SECONDS_BETWEEN_PROGRESS_LINES = 60.0


class SegmentTimeUp(Exception):
    """The time budget for this segment ran out. Run again to continue."""


def _clean_store_key(clean_key: CleanKey) -> str:
    task, cluster, query = clean_key
    return f"clean|{task}|{cluster}|q{query}"


def library_versions() -> dict[str, str]:
    import numpy
    import tokenizers
    import transformers

    return {
        "python": platform.python_version(),
        # A plain str: TorchVersion is not an allowed global for torch.load(weights_only=True).
        "torch": str(torch.__version__),
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "numpy": numpy.__version__,
    }


def environment_info(device: torch.device) -> dict[str, Any]:
    return {
        **library_versions(),
        "platform": platform.platform(),
        "device": str(device),
        "threads": torch.get_num_threads(),
    }


class Session:
    def __init__(
        self,
        config: Config,
        out_dir: str | Path,
        model: nn.Module | None = None,
        tokenizer: Any = None,
        device: torch.device | None = None,
        model_commit: str | None = None,
    ) -> None:
        """model, tokenizer and model_commit are for tests. A real run resolves and pins the commit."""
        self.config = config
        self.out_dir = Path(out_dir)
        self.store = ResultStore(self.out_dir / "results.jsonl")
        self.device = device if device is not None else choose_device()
        raw_data = load_data(config.data_dir)

        load_needed = model is None or tokenizer is None
        if load_needed:
            # A resumed run keeps the weights it started with, whatever the revision now points at.
            commit = self._stored_commit() or resolve_model_commit(config.model_name, config.model_revision)
            dtype = str(choose_dtype(self.device))
        else:
            commit = model_commit or getattr(getattr(model, "config", None), "_commit_hash", None) or "local"
            dtype = str(next(model.parameters()).dtype)
        self.identity = run_identity(
            config.config_hash, raw_data.sha256, commit, dtype, self.device.type, library_versions()
        )
        self._assert_same_identity()

        if load_needed:
            model, tokenizer = load_model_and_tokenizer(config.model_name, commit, self.device)
            loaded_dtype = str(next(model.parameters()).dtype)
            if loaded_dtype != dtype:
                raise AssertionError(f"Asked for dtype {dtype}, loaded {loaded_dtype}")
        self.model = model
        self.tokenizer = tokenizer
        self.layers = get_layers(model)
        for layer in config.grid.layers:
            if not 0 <= layer < len(self.layers):
                raise ValueError(f"Grid layer {layer} is outside this model's {len(self.layers)} layers")

        self.stimuli: Stimuli = prepare_stimuli(raw_data, config, lambda word: is_single_token_word(tokenizer, word))
        self.batch_size = config.runtime.batch_size_gpu if self.device.type == "cuda" else config.runtime.batch_size_cpu
        self._prompts: dict[CleanKey, RenderedPrompt] = {}
        self._resolve_answer_tokens()
        self.test_tasks = self._available_test_tasks()

        self.vectors: dict[str, dict[int, torch.Tensor]] = {}
        self.random_vectors: dict[str, torch.Tensor] = {}
        self._clean_full: dict[CleanKey, torch.Tensor] = {}
        self._deadline: float | None = None
        self._last_progress = 0.0
        self._write_manifest(raw_data.sha256)

    @classmethod
    def open(cls, config_path: str | Path, out_dir: str | Path) -> "Session":
        return cls(Config.from_file(config_path), out_dir)

    # Setup

    def _stored_commit(self) -> str | None:
        for manifest in self.store.records("manifest"):
            return (manifest.get("identity") or {}).get("model_commit")
        return None

    def _assert_same_identity(self) -> None:
        """Rows from a different code, config, data, weights, dtype, device type or library must not mix.

        L subtracts a stored clean logit from a new injected one, so a mixed store corrupts it silently.
        Start a new output directory instead.
        """
        for manifest in self.store.records("manifest"):
            differences = identity_differences(manifest.get("identity") or {}, self.identity)
            if differences:
                raise ValueError(
                    f"{self.store.path} was written by a different run, differing in {differences}. "
                    "Use a new output directory."
                )

    def _resolve_answer_tokens(self) -> None:
        """Resolve labels in rendered context, per task, and require the same ids everywhere."""
        clusters = self.stimuli.clusters_calibration
        cluster = next(iter(clusters))
        self.label_ids: list[int] = []
        self.none_id: int | None = None
        for task in ("T_inj", "T_pick", "T_fav", "T_odd", "T_fact", NONE_TASK):
            query = 0 if task == "T_fact" else None
            prompt = self._prompt((task, cluster, query))
            ids = resolve_label_ids(self.tokenizer, prompt.text)
            if self.label_ids and ids != self.label_ids:
                raise AssertionError(f"Label ids differ on {task}: {ids} vs {self.label_ids}")
            self.label_ids = ids
            if task == NONE_TASK:
                self.none_id = resolve_none_id(self.tokenizer, prompt.text, ids)

    def _available_test_tasks(self) -> list[str]:
        tasks = list(self.config.design.test_task_conditions)
        return [task for task in tasks if task != NONE_TASK or self.none_id is not None]

    def _write_manifest(self, data_sha256: str) -> None:
        config = self.config
        session_number = sum(1 for _ in self.store.records("manifest"))
        n_params = sum(parameter.numel() for parameter in self.model.parameters())
        all_tasks = config.design.test_task_conditions
        self.store.append(
            [
                {
                    "key": f"manifest|{session_number}",
                    "kind": "manifest",
                    "tier": config.tier,
                    "inference_allowed": config.inference_allowed,
                    "model": config.model_name,
                    "revision": config.model_revision,
                    "commit": self.identity["model_commit"],
                    "identity": self.identity,
                    "config_hash": config.config_hash,
                    "data_sha256": data_sha256,
                    "seed": config.seed,
                    "n_bootstrap": config.analysis.n_bootstrap,
                    "min_denominator": config.analysis.min_denominator,
                    "screen_top_fraction": config.screening.top_fraction if config.screening.enabled else None,
                    "versions": environment_info(self.device),
                    "dtype": str(next(self.model.parameters()).dtype),
                    "n_params": n_params,
                    "n_layers": len(self.layers),
                    "started_unix": time.time(),
                    "clusters_calibration": {k: list(v) for k, v in self.stimuli.clusters_calibration.items()},
                    "clusters_test": {k: list(v) for k, v in self.stimuli.clusters_test.items()},
                    "concepts_calibration": list(self.stimuli.concepts_calibration),
                    "concepts_test": list(self.stimuli.concepts_test),
                    "dropped_words": self.stimuli.dropped_words,
                    "dropped_clusters": self.stimuli.dropped_clusters,
                    "dropped_tasks": [task for task in all_tasks if task not in self.test_tasks],
                    "tokens": {
                        "labels": self.label_ids,
                        "labels_decoded": [decode_token(self.tokenizer, token) for token in self.label_ids],
                        "none": self.none_id,
                    },
                }
            ]
        )

    def _prompt(self, clean_key: CleanKey) -> RenderedPrompt:
        prompt = self._prompts.get(clean_key)
        if prompt is None:
            task, cluster, query = clean_key
            prompt = build_prompt(self.tokenizer, task, cluster, self.stimuli.all_clusters[cluster], query)
            self._prompts[clean_key] = prompt
        return prompt

    # Budget and progress

    def _check_time(self) -> None:
        if self._deadline is not None and time.monotonic() >= self._deadline:
            raise SegmentTimeUp()

    def _progress(self, message: str) -> None:
        now = time.monotonic()
        if now - self._last_progress >= SECONDS_BETWEEN_PROGRESS_LINES:
            self._last_progress = now
            print(f"[{time.strftime('%H:%M:%S')}] {message}  records={len(self.store)}", flush=True)

    # Forward passes

    def _answer_ids(self, task: str) -> list[int]:
        if task == NONE_TASK:
            assert self.none_id is not None
            return [*self.label_ids, self.none_id]
        return self.label_ids

    def _forward(self, input_ids: torch.Tensor, injection: tuple[int, torch.Tensor, torch.Tensor, torch.Tensor] | None):
        with torch.inference_mode():
            if injection is None:
                return forward_last_logits(self.model, input_ids)
            layer, positions, vectors, alphas = injection
            with inject(self.layers[layer], positions, vectors, alphas):
                return forward_last_logits(self.model, input_ids)

    def _ids_tensor(self, prompts: Sequence[RenderedPrompt]) -> torch.Tensor:
        lengths = {len(prompt.input_ids) for prompt in prompts}
        if len(lengths) != 1:
            raise AssertionError(f"A batch must have one token length, got {sorted(lengths)}")
        return torch.tensor([prompt.input_ids for prompt in prompts], device=self.device)  # [batch, length]

    # Vectors

    def _ensure_vectors(self) -> None:
        if self.vectors:
            return
        path = self.out_dir / "concept_vectors.pt"
        grid_layers = list(self.config.grid.layers)
        concepts = [*self.stimuli.concepts_calibration, *self.stimuli.concepts_test]
        if path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=True)
            if saved["identity"] == self.identity:
                self.vectors = {
                    concept: {layer: saved["vectors"][f"{concept}|{layer}"] for layer in grid_layers}
                    for concept in concepts
                }
        if not self.vectors:
            print(f"Computing concept vectors for {len(concepts)} concepts", flush=True)
            started = time.monotonic()
            self.vectors, diagnostics = concept_vectors(
                self.model,
                self.tokenizer,
                {layer: self.layers[layer] for layer in grid_layers},
                concepts,
                self.stimuli.baseline_words,
                self.batch_size,
                self.device,
            )
            payload = {
                "identity": self.identity,
                "vectors": {
                    f"{concept}|{layer}": vector
                    for concept, by_layer in self.vectors.items()
                    for layer, vector in by_layer.items()
                },
            }
            temporary = path.with_suffix(".tmp")
            torch.save(payload, temporary)
            os.replace(temporary, path)
            fingerprint = vector_fingerprint(self.vectors, self.config.seed)
            stored = self.store.get("vector_diagnostics")
            if stored is None:
                self.store.append(
                    [
                        {
                            "key": "vector_diagnostics",
                            "kind": "vector_diagnostics",
                            "by_layer": diagnostics,
                            "fingerprint": fingerprint,
                            "sec": time.monotonic() - started,
                        }
                    ]
                )
            elif not fingerprints_match(stored["fingerprint"], fingerprint):
                # A lost runtime recomputes vectors, and they must be the ones the stored rows used.
                raise ValueError("Recomputed concept vectors differ from the ones this run's rows used")
        dim = hidden_size(self.model)
        for concept in concepts:
            self.random_vectors[concept] = random_unit_vector(dim, self.config.seed, concept)

    def _vector(self, trial: Trial) -> torch.Tensor:
        if trial.condition == "concept":
            return self.vectors[trial.concept][trial.layer]
        if trial.condition in ("random_norm", "random_impact"):
            # Both random conditions share a direction per concept, so H3 compares scale alone.
            return self.random_vectors[trial.concept]
        raise ValueError(f"Unknown condition {trial.condition}")

    # Clean runs

    def _run_clean(self, clean_keys: Iterable[CleanKey], keep_full: bool = False) -> None:
        """Run clean prompts that lack a stored record, or whose full logits are needed for JS."""
        pending = [
            key
            for key in clean_keys
            if not self.store.has(_clean_store_key(key)) or (keep_full and key not in self._clean_full)
        ]
        groups: dict[int, list[CleanKey]] = defaultdict(list)
        for key in pending:
            groups[len(self._prompt(key).input_ids)].append(key)
        for length in sorted(groups):
            group = groups[length]
            for start in range(0, len(group), self.batch_size):
                self._check_time()
                self._run_clean_batch(group[start : start + self.batch_size], keep_full)

    def _run_clean_batch(self, keys: Sequence[CleanKey], keep_full: bool) -> None:
        prompts = [self._prompt(key) for key in keys]
        started = time.monotonic()
        logits = self._forward(self._ids_tensor(prompts), None)  # [batch, vocab]
        assert_finite(logits, f"clean run of {keys[0]}")
        seconds = (time.monotonic() - started) / len(keys)
        scored_ids = [*self.label_ids, *([] if self.none_id is None else [self.none_id])]
        scored = logits[:, scored_ids].double()  # [batch, 10 or 11]
        new_records = []
        for row, key in enumerate(keys):
            if keep_full:
                self._clean_full[key] = logits[row].clone()
            if self.store.has(_clean_store_key(key)):
                continue
            task, cluster, query = key
            answer_ids = self._answer_ids(task)
            answer = int(logits[row, answer_ids].argmax())
            new_records.append(
                {
                    "key": _clean_store_key(key),
                    "kind": "clean",
                    "task": task,
                    "cluster": cluster,
                    "query": query,
                    "length": len(prompts[row].input_ids),
                    "prompt_hash": token_hash(prompts[row].input_ids),
                    # Full float precision so that a resumed run reproduces m_clean exactly.
                    "logits": [float(value) for value in scored[row]],
                    "answer": answer,
                    "label_mass": float(answer_mass(logits[row : row + 1], self.label_ids)[0]),
                    "none_mass": self._none_mass(logits, row, task),
                    "sec": seconds,
                }
            )
        self.store.append(new_records)

    def _none_mass(self, logits: torch.Tensor, row: int, task: str) -> float | None:
        if task != NONE_TASK or self.none_id is None:
            return None
        return float(answer_mass(logits[row : row + 1], [self.none_id])[0])

    def _clean_answer_logits(self, clean_key: CleanKey) -> list[float]:
        record = self.store.get(_clean_store_key(clean_key))
        if record is None:
            raise AssertionError(f"Clean run missing for {clean_key}")
        if record["prompt_hash"] != token_hash(self._prompt(clean_key).input_ids):
            raise AssertionError(f"Clean and injected tokens differ for {clean_key}")
        return record["logits"]

    # Injected runs

    def _run_trials(self, trials: Sequence[Trial], measure_js: bool = False) -> None:
        pending = [trial for trial in trials if not self.store.has(trial.key)]
        for trial in pending:
            assert_trial_respects_split(trial, self.stimuli)
        groups: dict[tuple[int, int], list[Trial]] = defaultdict(list)
        for trial in pending:
            groups[(len(self._prompt(trial.clean_key).input_ids), trial.layer)].append(trial)
        done = len(trials) - len(pending)
        for group_key in sorted(groups):
            group = groups[group_key]
            for start in range(0, len(group), self.batch_size):
                self._check_time()
                batch = group[start : start + self.batch_size]
                self._run_trial_batch(batch, measure_js)
                done += len(batch)
                self._progress(f"{batch[0].phase}: {done}/{len(trials)}")

    def _run_trial_batch(self, batch: Sequence[Trial], measure_js: bool) -> None:
        layer = batch[0].layer
        if any(trial.layer != layer for trial in batch):
            raise AssertionError("A batch must inject at one layer")
        prompts = [self._prompt(trial.clean_key) for trial in batch]
        input_ids = self._ids_tensor(prompts)  # [batch, length]
        token_positions = torch.tensor(
            [prompt.positions[trial.position] for prompt, trial in zip(prompts, batch)], device=self.device
        )  # [batch]
        vectors = torch.stack([self._vector(trial) for trial in batch])  # [batch, dim]
        alphas = torch.tensor([trial.alpha for trial in batch], dtype=torch.float32)  # [batch]

        started = time.monotonic()
        logits = self._forward(input_ids, (layer, token_positions, vectors, alphas))  # [batch, vocab]
        assert_finite(logits, f"injected run of {batch[0].key}")
        seconds = (time.monotonic() - started) / len(batch)

        label_positions = torch.tensor([trial.position for trial in batch])  # [batch], in 0..9
        injected_labels = logits[:, self.label_ids].double()  # [batch, 10]
        clean_scored = torch.tensor(
            [self._clean_answer_logits(trial.clean_key) for trial in batch], dtype=torch.float64
        )  # [batch, 10 or 11]
        clean_labels = clean_scored[:, :LIST_SIZE]  # [batch, 10]
        effect = localization_effect(injected_labels, clean_labels, label_positions)
        m_injected = margin(injected_labels, label_positions)
        m_clean = margin(clean_labels, label_positions)

        records = []
        for row, trial in enumerate(batch):
            answer_ids = self._answer_ids(trial.task)
            n_answers = len(answer_ids)
            answer = int(logits[row, answer_ids].argmax())
            clean_answer = int(clean_scored[row, :n_answers].argmax())
            record = {
                "key": trial.key,
                "kind": "trial",
                "phase": trial.phase,
                "task": trial.task,
                "condition": trial.condition,
                "concept": trial.concept,
                "cluster": trial.cluster,
                "position": trial.position,
                "query": trial.query,
                "layer": trial.layer,
                "alpha": trial.alpha,
                "prompt_hash": token_hash(prompts[row].input_ids),
                "injected_token": int(token_positions[row]),
                "L": float(effect[row]),
                "m_injected": float(m_injected[row]),
                "m_clean": float(m_clean[row]),
                "answer": answer,
                "hit": answer == trial.position,
                "clean_hit": clean_answer == trial.position,
                "fact_correct": (answer == trial.query) if trial.query is not None else None,
                "label_mass": float(answer_mass(logits[row : row + 1], self.label_ids)[0]),
                "none_mass": self._none_mass(logits, row, trial.task),
                "sec": seconds,
            }
            if measure_js:
                record["js"] = float(js_divergence(logits[row], self._clean_full[trial.clean_key]))
            records.append(record)
        self.store.append(records)

    # Phases

    def _sham_check(self) -> None:
        """alpha = 0 through the real hook, on one batch, against a clean run of that same batch."""
        if self.store.has("sham"):
            return
        self._check_time()
        cluster = next(iter(self.stimuli.clusters_calibration))
        prompt = self._prompt(("T_inj", cluster, None))
        n = min(self.batch_size, LIST_SIZE)
        input_ids = self._ids_tensor([prompt] * n)
        label_positions = torch.arange(n)
        token_positions = torch.tensor(prompt.positions[:n], device=self.device)
        vectors = torch.stack([random_unit_vector(hidden_size(self.model), self.config.seed, "sham")] * n)
        clean = self._forward(input_ids, None)
        sham = self._forward(input_ids, (self.config.grid.layers[0], token_positions, vectors, torch.zeros(n)))
        effect = localization_effect(sham[:, self.label_ids], clean[:, self.label_ids], label_positions)
        max_abs = float(effect.abs().max())
        self.store.append([{"key": "sham", "kind": "sham", "max_abs_L": max_abs, "n": n}])
        if max_abs > SHAM_TOLERANCE:
            raise RuntimeError(f"Sham check failed: alpha = 0 gave |L| = {max_abs}")

    def _calibration_clean_keys(self) -> list[CleanKey]:
        return plan_clean_keys(["T_inj"], self.stimuli.clusters_calibration)

    def _select(self) -> Selection:
        self._run_clean(self._calibration_clean_keys(), keep_full=True)
        trials = plan_grid_trials(self.stimuli, self.config)
        self._run_trials(trials, measure_js=True)
        records = [self.store.get(trial.key) for trial in trials]
        selection = select_layer_alpha(
            records, self.stimuli.concepts_calibration, list(self.stimuli.clusters_calibration)
        )
        stored = self.store.get("selection")
        if stored is None:
            self.store.append(
                [
                    {
                        "key": "selection",
                        "kind": "selection",
                        "layer": selection.layer,
                        "alpha": selection.alpha,
                        "accuracy": selection.accuracy,
                        "grid_accuracy": selection.grid_accuracy,
                        "n_trials_per_point": len(trials) // len(selection.grid_accuracy),
                    }
                ]
            )
        elif stored["layer"] != selection.layer or stored["alpha"] != selection.alpha:
            raise AssertionError("Replayed selection differs from the stored selection")
        return selection

    def _impact_search(self, selection: Selection) -> float:
        """Find the alpha at which random directions match the concept condition's mean JS.

        Calibration data only. Replays deterministically on resume, because each evaluation's trials
        are keyed by alpha and read back from the store.
        """
        stored = self.store.get("impact_match")
        if stored is not None:
            return stored["alpha_impact"]
        settings = self.config.impact_search
        grid_trials = [
            trial
            for trial in plan_grid_trials(self.stimuli, self.config)
            if trial.layer == selection.layer and trial.alpha == selection.alpha
        ]
        target = _mean_js([self.store.get(trial.key) for trial in grid_trials])
        max_alpha = selection.alpha * settings.max_scale

        def relative_error(js: float) -> float:
            return abs(js - target) / max(target, 1e-12)

        history: list[dict[str, float]] = []
        low: float | None = None
        high: float | None = None
        alpha = selection.alpha
        for _ in range(settings.max_evals):
            alpha = float(f"{alpha:.6g}")  # Trial keys hold six significant digits.
            trials = plan_impact_trials(self.stimuli, self.config, selection.layer, alpha)
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
        match_ok = relative_error(best["js"]) <= settings.tolerance
        self.store.append(
            [
                {
                    "key": "impact_match",
                    "kind": "impact_match",
                    "target_js": target,
                    "achieved_js": best["js"],
                    "relative_error": relative_error(best["js"]),
                    "alpha_impact": best["alpha"],
                    "alpha_norm": selection.alpha,
                    "scale": best["alpha"] / selection.alpha,
                    "match_ok": match_ok,
                    "tolerance": settings.tolerance,
                    "history": history,
                }
            ]
        )
        if not match_ok:
            print(
                f"WARNING: impact match failed. Target JS {target:.5f}, best {best['js']:.5f} "
                f"at alpha {best['alpha']:.4g}.",
                flush=True,
            )
        return best["alpha"]

    def _screen(self, selection: Selection) -> None:
        if self.config.screening.enabled:
            self._run_trials(plan_screen_trials(self.stimuli, self.config, selection.layer, selection.alpha))

    def _test(self, selection: Selection, alpha_impact: float) -> None:
        if not (self.store.has("selection") and self.store.has("impact_match")):
            raise AssertionError("Test data must not be touched before calibration has finished")
        trials = plan_test_trials(
            self.stimuli, self.config, selection.layer, selection.alpha, alpha_impact, self.test_tasks
        )
        self._run_clean(plan_clean_keys(self.test_tasks, self.stimuli.clusters_test))
        self._run_trials(trials)

    def run(self, max_seconds: float | None = None) -> dict[str, Any]:
        """Run or resume until done or until the segment's time is up. Returns status().

        Uses runtime.segment_seconds when max_seconds is None. The current batch always finishes, and
        every completed batch is already on disk.
        """
        budget = self.config.runtime.segment_seconds if max_seconds is None else max_seconds
        self._deadline = time.monotonic() + budget
        try:
            self._check_time()
            self._ensure_vectors()
            self._sham_check()
            selection = self._select()
            alpha_impact = self._impact_search(selection)
            self._screen(selection)
            self._test(selection, alpha_impact)
            if not self.store.has("run_complete"):
                self.store.append([{"key": "run_complete", "kind": "run_complete", "finished_unix": time.time()}])
        except SegmentTimeUp:
            print("Segment time is up. Run the same call again to continue.", flush=True)
        return self.status()

    # Reporting

    def status(self) -> dict[str, Any]:
        trials_by_phase: dict[str, int] = defaultdict(int)
        seconds_by_phase: dict[str, float] = defaultdict(float)
        for record in self.store.records("trial"):
            trials_by_phase[record["phase"]] += 1
            seconds_by_phase[record["phase"]] += record["sec"]
        counts = plan_prompt_counts(self.stimuli, self.config)
        counts["test"] = len(plan_test_trials(self.stimuli, self.config, 0, 1.0, 1.0, self.test_tasks))
        clean_done = sum(1 for _ in self.store.records("clean"))
        remaining = (
            max(0, counts["calibration_grid"] - trials_by_phase[PHASE_GRID])
            + max(0, counts["screen"] - trials_by_phase[PHASE_SCREEN])
            + max(0, counts["test"] - trials_by_phase[PHASE_TEST])
            + max(0, counts["clean_calibration"] + counts["clean_test"] - clean_done)
        )
        if not self.store.has("impact_match"):
            remaining += max(0, counts["impact_search_max"] - trials_by_phase[PHASE_IMPACT])
        total_trials = sum(trials_by_phase.values())
        seconds_per_prompt = sum(seconds_by_phase.values()) / total_trials if total_trials else None
        return {
            "complete": self.store.has("run_complete"),
            "trials_by_phase": dict(trials_by_phase),
            "clean_done": clean_done,
            "remaining_prompts_at_most": remaining,
            "seconds_per_prompt": seconds_per_prompt,
            "eta_hours_at_most": remaining * seconds_per_prompt / 3600 if seconds_per_prompt else None,
        }

    def dry_run(self, n_batches: int = 2) -> dict[str, Any]:
        """Time real calibration-shaped batches without storing anything, then project the whole run.

        The first batch includes warm-up, so the projection uses the last batch timed.
        """
        counts = plan_prompt_counts(self.stimuli, self.config)
        counts["test"] = len(plan_test_trials(self.stimuli, self.config, 0, 1.0, 1.0, self.test_tasks))
        cluster = next(iter(self.stimuli.clusters_calibration))
        prompt = self._prompt(("T_inj", cluster, None))
        n = min(self.batch_size, LIST_SIZE)
        input_ids = self._ids_tensor([prompt] * n)
        token_positions = torch.tensor(prompt.positions[:n], device=self.device)
        vector = random_unit_vector(hidden_size(self.model), self.config.seed, "dry_run")
        injection = (self.config.grid.layers[0], token_positions, torch.stack([vector] * n), torch.ones(n))
        batch_seconds = []
        for _ in range(max(1, n_batches)):
            started = time.monotonic()
            self._forward(input_ids, injection)
            batch_seconds.append(time.monotonic() - started)
        seconds_per_prompt = batch_seconds[-1] / n
        # Vector prompts are about a quarter of a list prompt's length.
        equivalent_prompts = sum(counts.values()) - 0.75 * counts["vector_prompts"]
        estimate = equivalent_prompts * seconds_per_prompt
        # Shown so that a wrong first token or template is caught before any real run.
        clean_logits = self._forward(input_ids[:1], None)  # [1, vocab]
        assert_finite(clean_logits, "the dry run")
        top_tokens = clean_logits[0].topk(5).indices.tolist()
        clean_label_mass_by_task = {}
        for task in self.test_tasks:
            task_prompt = self._prompt((task, cluster, 0 if task == "T_fact" else None))
            task_logits = self._forward(self._ids_tensor([task_prompt]), None)
            clean_label_mass_by_task[task] = float(answer_mass(task_logits, self.label_ids)[0])
        vector_text = concept_prompt_text(self.stimuli.baseline_words[0])
        vector_ids, read_index = locate_user_content_end(
            self.tokenizer, render_chat(self.tokenizer, vector_text), vector_text
        )
        result = {
            "prompt_tail": prompt.text[-120:],
            "list_words_at_positions": [decode_token(self.tokenizer, prompt.input_ids[i]) for i in prompt.positions],
            "labels_decoded": [decode_token(self.tokenizer, token) for token in self.label_ids],
            "vector_read_token": decode_token(self.tokenizer, vector_ids[read_index]),
            "vector_read_tokens_from_end": len(vector_ids) - read_index,
            "clean_label_mass": float(answer_mass(clean_logits, self.label_ids)[0]),
            "clean_label_mass_by_task": clean_label_mass_by_task,
            "clean_top_tokens": [decode_token(self.tokenizer, token) for token in top_tokens],
            "prompt_tokens": len(prompt.input_ids),
            "batch_size": n,
            "batch_seconds": batch_seconds,
            "seconds_per_prompt": seconds_per_prompt,
            "tokens_per_second": len(prompt.input_ids) / seconds_per_prompt,
            "prompt_counts": counts,
            "estimated_hours_at_most": estimate / 3600,
            "dropped_words": self.stimuli.dropped_words,
            "dropped_clusters": self.stimuli.dropped_clusters,
            "tokens": {"labels": self.label_ids, "none": self.none_id},
        }
        return result

    def export_text(self) -> str:
        return encode_export(build_export(self.store.records()))


def _mean_js(records: Sequence[dict[str, Any] | None]) -> float:
    values = []
    for record in records:
        if record is None or "js" not in record:
            raise AssertionError("A JS measurement is missing")
        values.append(record["js"])
    mean = sum(values) / len(values)
    if not math.isfinite(mean):
        raise NonFiniteError("Non-finite JS divergence")
    return mean
