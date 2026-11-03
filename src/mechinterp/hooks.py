"""Forward hooks on residual layers: injection and capture."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch
from torch import nn


def split_layer_output(output: Any) -> tuple[torch.Tensor, Any]:
    """Return (hidden, rest). Decoder layers return a tensor or a tuple depending on version."""
    if isinstance(output, torch.Tensor):
        return output, None
    if isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
        return output[0], output[1:]
    raise TypeError(f"Unsupported layer output type: {type(output).__name__}")


def join_layer_output(hidden: torch.Tensor, rest: Any, original: Any) -> Any:
    if isinstance(original, torch.Tensor):
        return hidden
    if isinstance(original, list):
        return [hidden, *rest]
    return (hidden, *rest)


def inject_activation(
    hidden: torch.Tensor, positions: torch.Tensor, vectors: torch.Tensor, alphas: torch.Tensor
) -> torch.Tensor:
    """Apply h <- h + alpha * ||h|| * v at one position per example.

    hidden: [batch, seq, dim]; positions: [batch]; vectors: [batch, dim]; alphas: [batch].
    ||h|| is the norm of that token's own activation at this layer.
    """
    if hidden.dim() != 3:
        raise ValueError(f"Expected [batch, seq, dim] activations, got shape {tuple(hidden.shape)}")
    batch, seq, dim = hidden.shape
    if positions.shape != (batch,) or alphas.shape != (batch,) or vectors.shape != (batch, dim):
        raise ValueError(
            f"Injection shapes do not match activations {tuple(hidden.shape)}: positions "
            f"{tuple(positions.shape)}, alphas {tuple(alphas.shape)}, vectors {tuple(vectors.shape)}"
        )
    if positions.min() < 0 or positions.max() >= seq:
        raise ValueError(f"Injection position out of range for sequence length {seq}")
    positions = positions.to(hidden.device)
    vectors = vectors.to(hidden.device, hidden.dtype)
    alphas = alphas.to(hidden.device, hidden.dtype)
    rows = torch.arange(batch, device=hidden.device)
    target = hidden[rows, positions]  # [batch, dim]
    norms = target.norm(dim=-1, keepdim=True)  # [batch, 1]
    updated = hidden.clone()
    updated[rows, positions] = target + alphas.unsqueeze(-1) * norms * vectors
    return updated


@contextmanager
def inject(
    layer: nn.Module, positions: torch.Tensor, vectors: torch.Tensor, alphas: torch.Tensor
) -> Iterator[None]:
    """Inject at the output of `layer` for the duration of the context."""

    def hook(_module: nn.Module, _inputs: Any, output: Any) -> Any:
        hidden, rest = split_layer_output(output)
        return join_layer_output(inject_activation(hidden, positions, vectors, alphas), rest, output)

    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def capture_layer_outputs(layers: dict[int, nn.Module]) -> Iterator[dict[int, torch.Tensor]]:
    """Record each layer's output on CPU as float32.

    Forward hooks run in registration order and each sees the previous hook's return value,
    so a capture opened inside `inject` on the same layer records the injected activations.
    """
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(index: int) -> Any:
        def hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
            hidden, _ = split_layer_output(output)
            captured[index] = hidden.detach().float().cpu()

        return hook

    for index, layer in layers.items():
        handles.append(layer.register_forward_hook(make_hook(index)))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()
