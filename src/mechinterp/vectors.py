"""Concept vectors and seeded random directions."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Sequence

import torch
from torch import nn

from .hooks import capture_layer_outputs
from .modeling import forward_last_logits
from .prompts import concept_prompt_text, encode_ids, render_chat


def _encode_concept_prompt(tokenizer: Any, word: str, prompt_mode: str) -> list[int]:
    text = concept_prompt_text(word)
    if prompt_mode == "chat":
        return encode_ids(tokenizer, render_chat(tokenizer, text))
    if prompt_mode == "raw":
        return list(tokenizer(text, add_special_tokens=True)["input_ids"])
    raise ValueError(f"Unknown prompt mode {prompt_mode}")


def final_token_residuals(
    model: nn.Module,
    tokenizer: Any,
    layers: dict[int, nn.Module],
    words: Sequence[str],
    batch_size: int,
    device: torch.device,
    prompt_mode: str = "chat",
) -> dict[str, dict[int, torch.Tensor]]:
    """Return word -> layer -> final-token activation [dim], float32 on CPU.

    Prompts are batched by exact length, so there is no padding and the final token is index -1.
    """
    by_length: dict[int, list[tuple[str, list[int]]]] = defaultdict(list)
    for word in words:
        ids = _encode_concept_prompt(tokenizer, word, prompt_mode)
        by_length[len(ids)].append((word, ids))

    residuals: dict[str, dict[int, torch.Tensor]] = {}
    for length in sorted(by_length):
        group = by_length[length]
        for start in range(0, len(group), batch_size):
            batch = group[start : start + batch_size]
            input_ids = torch.tensor([ids for _, ids in batch], device=device)  # [batch, length]
            with torch.inference_mode(), capture_layer_outputs(layers) as captured:
                forward_last_logits(model, input_ids)
            for row, (word, _) in enumerate(batch):
                residuals[word] = {index: captured[index][row, -1, :].clone() for index in layers}
    return residuals


def concept_vectors(
    model: nn.Module,
    tokenizer: Any,
    layers: dict[int, nn.Module],
    concepts: Sequence[str],
    baseline_words: Sequence[str],
    batch_size: int,
    device: torch.device,
    prompt_mode: str = "chat",
) -> tuple[dict[str, dict[int, torch.Tensor]], dict[str, float]]:
    """Final-token residual for each concept minus the baseline mean, normalized, per layer.

    Also returns the mean raw difference norm per layer, as a diagnostic.
    """
    baseline = final_token_residuals(model, tokenizer, layers, baseline_words, batch_size, device, prompt_mode)
    baseline_mean = {
        index: torch.stack([baseline[word][index] for word in baseline_words]).mean(dim=0) for index in layers
    }
    residuals = final_token_residuals(model, tokenizer, layers, concepts, batch_size, device, prompt_mode)
    vectors: dict[str, dict[int, torch.Tensor]] = {}
    norm_sums: dict[int, float] = defaultdict(float)
    for concept in concepts:
        vectors[concept] = {}
        for index in layers:
            difference = residuals[concept][index] - baseline_mean[index]
            norm = difference.norm()
            if not torch.isfinite(norm) or norm <= 0:
                raise ValueError(f"Degenerate concept vector for {concept!r} at layer {index}")
            vectors[concept][index] = difference / norm
            norm_sums[index] += float(norm)
    diagnostics = {str(index): norm_sums[index] / len(concepts) for index in layers}
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
