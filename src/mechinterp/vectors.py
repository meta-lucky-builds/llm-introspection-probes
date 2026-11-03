"""Concept vectors and seeded random directions."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Sequence

import torch
from torch import nn

from .hooks import capture_layer_outputs
from .modeling import forward_last_logits
from .prompts import concept_prompt_text, locate_user_content_end, render_chat

FINGERPRINT_TOLERANCE = 1e-3


def user_end_residuals(
    model: nn.Module,
    tokenizer: Any,
    layers: dict[int, nn.Module],
    words: Sequence[str],
    batch_size: int,
    device: torch.device,
) -> dict[str, dict[int, torch.Tensor]]:
    """Return word -> layer -> activation [dim] at the "." ending "Tell me about {word}.".

    Float32 on CPU. Prompts are batched by exact length, so there is no padding. Each word's read
    index is computed from its own rendering, and tokens after it cannot affect it.
    """
    by_length: dict[int, list[tuple[str, list[int], int]]] = defaultdict(list)
    for word in words:
        text = concept_prompt_text(word)
        ids, read_index = locate_user_content_end(tokenizer, render_chat(tokenizer, text), text)
        by_length[len(ids)].append((word, ids, read_index))

    residuals: dict[str, dict[int, torch.Tensor]] = {}
    for length in sorted(by_length):
        group = by_length[length]
        for start in range(0, len(group), batch_size):
            batch = group[start : start + batch_size]
            input_ids = torch.tensor([ids for _, ids, _ in batch], device=device)  # [batch, length]
            with torch.inference_mode(), capture_layer_outputs(layers) as captured:
                forward_last_logits(model, input_ids)
            for row, (word, _, read_index) in enumerate(batch):
                residuals[word] = {index: captured[index][row, read_index, :].clone() for index in layers}
    return residuals


def concept_vectors(
    model: nn.Module,
    tokenizer: Any,
    layers: dict[int, nn.Module],
    concepts: Sequence[str],
    baseline_words: Sequence[str],
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, dict[int, torch.Tensor]], dict[str, dict[str, float | None]]]:
    """Residual for each concept minus the baseline mean, normalized, per layer.

    Normalizing makes a weak signal look like a strong one, so diagnostics per layer report the raw
    difference norm, that norm relative to the activation, and the mean pairwise cosine.
    """
    baseline = user_end_residuals(model, tokenizer, layers, baseline_words, batch_size, device)
    baseline_mean = {
        index: torch.stack([baseline[word][index] for word in baseline_words]).mean(dim=0) for index in layers
    }
    residuals = user_end_residuals(model, tokenizer, layers, concepts, batch_size, device)
    vectors: dict[str, dict[int, torch.Tensor]] = {concept: {} for concept in concepts}
    diagnostics: dict[str, dict[str, float | None]] = {}
    for index in layers:
        raw_norms = []
        for concept in concepts:
            difference = residuals[concept][index] - baseline_mean[index]
            norm = difference.norm()
            if not torch.isfinite(norm) or norm <= 0:
                raise ValueError(f"Degenerate concept vector for {concept!r} at layer {index}")
            vectors[concept][index] = difference / norm
            raw_norms.append(float(norm))
        stacked = torch.stack([vectors[concept][index] for concept in concepts])  # [n_concepts, dim]
        n = len(concepts)
        cosine = float(((stacked @ stacked.T).sum() - n) / (n * (n - 1))) if n > 1 else None
        mean_raw = sum(raw_norms) / n
        diagnostics[str(index)] = {
            "mean_raw_norm": mean_raw,
            "mean_raw_norm_over_activation_norm": mean_raw / float(baseline_mean[index].norm()),
            "mean_pairwise_cosine": cosine,
        }
    return vectors, diagnostics


def random_unit_vector(dim: int, seed: int, name: str) -> torch.Tensor:
    """A unit vector [dim], float32, that depends only on (seed, name, dim).

    Hashing the name makes vectors independent per concept and unaffected by run order, so a
    resumed run gets the same directions.
    """
    digest = hashlib.sha256(f"{seed}|{name}".encode("utf-8")).digest()
    generator = torch.Generator(device="cpu").manual_seed(int.from_bytes(digest[:8], "big") % (2**63))
    vector = torch.randn(dim, generator=generator, dtype=torch.float64)
    return (vector / vector.norm()).float()


def vector_fingerprint(vectors: dict[str, dict[int, torch.Tensor]], seed: int) -> dict[str, float]:
    """Projection of each vector on a fixed direction.

    Vectors are recomputed when a runtime is lost, and floats can differ slightly between machines,
    so comparison uses a tolerance rather than a hash of the bytes.
    """
    fingerprint: dict[str, float] = {}
    for concept, by_layer in vectors.items():
        for layer, vector in by_layer.items():
            probe = random_unit_vector(vector.numel(), seed, "fingerprint")
            fingerprint[f"{concept}|{layer}"] = float(torch.dot(vector, probe))
    return fingerprint


def fingerprints_match(first: dict[str, float], second: dict[str, float]) -> bool:
    if set(first) != set(second):
        return False
    return all(abs(first[key] - second[key]) <= FINGERPRINT_TOLERANCE for key in first)
