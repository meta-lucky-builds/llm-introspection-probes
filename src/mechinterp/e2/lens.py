"""The Jacobian lens: loading, readout and the checks that an off-by-one would fail.

Definitions, read from anthropics/jacobian-lens at 581d398 (jlens/lens.py, hf.py, hooks.py, fitting.py) and from
2607.15495 sections 2.1 and A.7. This file does not import jlens, which is unmaintained and which these few
lines replace.

    lens(h) = softmax(W_U norm(J_l h))

`norm` is the model's own final normalization layer, the one before the output head, here RMSNorm at
model.norm. It is NOT a unit normalization. jlens's unembed is lm_head(final_norm(x)). The logit lens is J_l = I,
through the same norm. J_l h is written `h @ J_l.T`, because row i of J_l holds d z_i / d h.

The J-lens vectors are the rows of W_U J_l, without the norm, as the paper defines them in 2.1. Its 2.5 says
logits follow these inner products "approximately, up to a data-dependent normalization factor". RMSNorm also has
a learned gain per dimension, which those rows leave out, so `norm_gain_report` records how far that matters here.

Layer indices. Lens key l is the OUTPUT of block l: jlens's ActivationRecorder is a forward hook on layers[l],
and it fits source layers range(n_layers - 1) with the final block n_layers - 1 as the target. E1's injection
hook and hooks.capture_layer_outputs are forward hooks on the same modules, get_layers(model)[l]. So the same
integer means the same point in both, with no offset. HF's output_hidden_states would be off by one, since its
index 0 is the embedding, and is never used here. check_structure and measure_alignment test this at runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from ..hooks import capture_layer_outputs, split_layer_output
from ..modeling import forward_last_logits, get_layers
from ..sharding import sha256_file

STORED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
FINAL_NORM_PATHS = ("model.norm", "model.language_model.norm", "language_model.norm", "transformer.ln_f")
# Logits through our unembed against the model's own, on one prompt. Same operations, so only rounding can differ.
UNEMBED_TOLERANCE = 1e-2
CHUNK_ROWS = 16384


class LensError(ValueError):
    """The lens file or a check on it is wrong. This is a broken instrument, not a finding about the model."""


@dataclass(frozen=True)
class Lens:
    jacobians: dict[int, torch.Tensor]  # Layer to [d_model, d_model], float32, on the CPU.
    source_layers: tuple[int, ...]
    n_prompts: int
    d_model: int
    report: dict[str, Any]


def _resolve(root: nn.Module, dotted: str) -> Any:
    current: Any = root
    for part in dotted.split("."):
        current = getattr(current, part, None)
        if current is None:
            return None
    return current


def final_norm(model: nn.Module) -> nn.Module:
    for path in FINAL_NORM_PATHS:
        module = _resolve(model, path)
        if isinstance(module, nn.Module):
            return module
    raise LensError(f"Could not find the final norm on {type(model).__name__}")


def output_weights(model: nn.Module) -> torch.Tensor:
    """W_U: [vocab, d_model], on the model's device and dtype. Qwen3-4B ties it to the embedding."""
    head = model.get_output_embeddings()
    if head is None or getattr(head, "bias", None) is not None:
        raise LensError(f"Expected an output head with no bias on {type(model).__name__}")
    return head.weight


def unembed(model: nn.Module, residual: torch.Tensor) -> torch.Tensor:
    """[..., d_model] to [..., vocab], float32 at the head's dtype: final norm, then W_U."""
    config = getattr(model, "config", None)
    if getattr(config, "final_logit_softcapping", None) is not None:
        raise LensError("This model caps its logits, which this readout does not do")
    head = model.get_output_embeddings()
    weight = head.weight
    return head(final_norm(model)(residual.to(weight.device, weight.dtype)))


def transport(residual: torch.Tensor, jacobian: torch.Tensor) -> torch.Tensor:
    """J_l h for h [..., d_model]. Exactly jlens's `residual @ J_bar.T`."""
    return residual @ jacobian.to(residual.device, residual.dtype).T


def read(model: nn.Module, residual: torch.Tensor, jacobian: torch.Tensor | None) -> torch.Tensor:
    """Lens logits. jacobian None is the logit lens, which is J = I by the paper's definition."""
    return unembed(model, residual if jacobian is None else transport(residual, jacobian))


def ranks_of(logits: torch.Tensor, token_ids: Sequence[int]) -> torch.Tensor:
    """1-based rank of each token: [n, vocab] and [m] to [n, m]. Ties count in the token's favour."""
    if not token_ids:
        return torch.empty(logits.size(0), 0, dtype=torch.long, device=logits.device)
    # One token at a time: a broadcast over [n, m, vocab] would take gigabytes on a card that has few to spare.
    columns = [(logits > logits[:, token : token + 1]).sum(dim=1) + 1 for token in token_ids]
    return torch.stack(columns, dim=1)


def load_lens(path: str | Path, spec: Any, n_layers: int, d_model: int) -> Lens:
    """Load with weights_only, check everything the file should be, and record what it holds.

    The published file's format is not documented for this lens, so every assumption is asserted here.
    """
    path = Path(path)
    size = path.stat().st_size
    digest = sha256_file(path)
    if size != spec.size or digest != spec.sha256:
        raise LensError(
            f"{path} is {size} bytes with sha256 {digest}. The pin is {spec.size} bytes with sha256 {spec.sha256}."
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "J" not in checkpoint:
        found = sorted(checkpoint) if isinstance(checkpoint, dict) else type(checkpoint).__name__
        raise LensError(f"{path} is not a Jacobian lens file: found {found}")
    stored = checkpoint["J"]
    if not stored or not all(isinstance(layer, int) for layer in stored):
        raise LensError(f"Expected integer layer keys in J, found {sorted(map(str, stored))[:5]}")
    dtypes = sorted({str(matrix.dtype) for matrix in stored.values()})
    jacobians: dict[int, torch.Tensor] = {}
    for layer, matrix in stored.items():
        if matrix.dtype not in STORED_DTYPES:
            raise LensError(f"Unexpected dtype {matrix.dtype} at layer {layer}")
        if tuple(matrix.shape) != (d_model, d_model):
            raise LensError(f"J at layer {layer} has shape {tuple(matrix.shape)}, expected ({d_model}, {d_model})")
        # A float16 or bfloat16 value is exact in float32, so the conversion loses nothing.
        converted = matrix.float()
        if not torch.isfinite(converted).all():
            raise LensError(f"J at layer {layer} holds non-finite values")
        jacobians[layer] = converted
    source_layers = tuple(sorted(jacobians))
    check_structure(source_layers, n_layers)
    stated_layers = checkpoint.get("source_layers")
    if stated_layers is not None and sorted(stated_layers) != list(source_layers):
        raise LensError(f"The file's source_layers {stated_layers} disagree with its keys {list(source_layers)}")
    stated_dim = checkpoint.get("d_model")
    if stated_dim is not None and int(stated_dim) != d_model:
        raise LensError(f"The file says d_model {stated_dim}, the model has {d_model}")
    n_prompts = int(checkpoint.get("n_prompts", -1))
    if spec.n_prompts is not None and n_prompts != spec.n_prompts:
        raise LensError(f"The file was fit on {n_prompts} prompts, its config.yaml says {spec.n_prompts}")
    identity = torch.eye(d_model)
    report = {
        "path": str(path),
        "sha256": digest,
        "size": size,
        "keys": sorted(str(key) for key in checkpoint),
        "stored_dtypes": dtypes,
        "converted_to": "float32",
        "source_layers": list(source_layers),
        "n_layers_in_model": n_layers,
        "target_layer_inferred": n_layers - 1,
        "n_prompts": n_prompts,
        "d_model": d_model,
        "provenance": checkpoint.get("provenance"),
        "distance_from_identity_by_layer": {
            layer: float((jacobians[layer] - identity).norm() / math.sqrt(d_model)) for layer in source_layers
        },
    }
    return Lens(jacobians, source_layers, n_prompts, d_model, report)


def check_structure(source_layers: Sequence[int], n_layers: int) -> None:
    """Keys must be every block but the last: the block-output convention, with the final block as the target.

    A file indexed like HF's hidden_states, where 0 is the embedding, would carry n_layers keys and fail here.
    """
    expected = list(range(n_layers - 1))
    if list(source_layers) != expected:
        raise LensError(
            f"Lens layers are {list(source_layers)}, expected {expected} for a {n_layers}-block model. The layer "
            "convention differs from the one this code assumes, so every result would be silently misaligned."
        )


def check_unembed(model: nn.Module, input_ids: torch.Tensor) -> float:
    """The final block's output through final norm and W_U must be the model's logits: [1, length] prompt."""
    layers = get_layers(model)
    final_index = len(layers) - 1
    with torch.inference_mode(), capture_layer_outputs({final_index: layers[final_index]}) as captured:
        logits = forward_last_logits(model, input_ids)  # [1, vocab], on the CPU.
        mine = unembed(model, captured[final_index][:, -1, :]).float().cpu()
    gap = float((mine - logits).abs().max())
    if gap > UNEMBED_TOLERANCE:
        raise LensError(f"Final norm and W_U do not reproduce the model's logits: largest gap {gap}")
    return gap


def check_identity_is_logit_lens(model: nn.Module, residual: torch.Tensor) -> float:
    """J = I through this code path must equal the logit lens, on the same device and dtype: h [n, d_model]."""
    with torch.inference_mode():
        logit_lens = read(model, residual, None)
        identity = read(model, residual, torch.eye(residual.size(-1)))
    gap = float((logit_lens - identity).abs().max())
    if gap > UNEMBED_TOLERANCE:
        raise LensError(f"The identity Jacobian differs from the logit lens by {gap}")
    return gap


def norm_gain_report(model: nn.Module) -> dict[str, float] | None:
    """How uneven the final norm's learned gain is. A uniform gain would make rows of W_U J exact probes."""
    weight = getattr(final_norm(model), "weight", None)
    if weight is None:
        return None
    gain = weight.detach().double()
    return {"mean": float(gain.mean()), "std": float(gain.std()), "min": float(gain.min()), "max": float(gain.max())}


def lens_rows(model: nn.Module, jacobian: torch.Tensor, token_ids: Sequence[int]) -> torch.Tensor:
    """Rows of W_U J_l for these tokens: [n, d_model]."""
    weight = output_weights(model)
    ids = torch.as_tensor(list(token_ids), device=weight.device)
    with torch.inference_mode():
        return weight[ids] @ jacobian.to(weight.device, weight.dtype)


def dictionary(model: nn.Module, jacobian: torch.Tensor, token_ids: Sequence[int], on_cpu: bool = False) -> torch.Tensor:
    """Rows of W_U J_l for many tokens, in chunks: about 1.6 GB of float32 for the whole Qwen vocabulary."""
    weight = output_weights(model)
    if on_cpu:
        weight = weight.detach().cpu()
    with torch.inference_mode():
        matrix = jacobian.to(weight.device, weight.dtype)
        ids = torch.as_tensor(list(token_ids), device=weight.device)
        chunks = [weight[ids[start : start + CHUNK_ROWS]] @ matrix for start in range(0, ids.numel(), CHUNK_ROWS)]
        return torch.cat(chunks)


def dictionary_token_ids(tokenizer: Any, vocab_rows: int) -> list[int]:
    """Real tokens only. Qwen's W_U has 151,936 rows, its base vocabulary fewer, and the rest are unused or control.

    The paper says one J-lens vector per token of the vocabulary and names no filter. A control token or an empty
    row has no word to verbalize, so it stays out. Punctuation and fragments stay in: filtering them is the kind of
    tuning the prereg rules out.
    """
    special = set(tokenizer.all_special_ids)
    return [token for token in range(min(vocab_rows, int(tokenizer.vocab_size))) if token not in special]


# Empirical alignment of file indices with hook indices


def prompt_jacobian_rows(
    model: nn.Module, input_ids: torch.Tensor, source_layers: Sequence[int], dims: Sequence[int], skip_first: int
) -> dict[int, torch.Tensor]:
    """Rows `dims` of the per-prompt estimator of J_l: [len(dims), d_model] per layer. Same as jlens.fitting.

    A one-hot cotangent at the output dimension, at every valid target position, gives per source position the sum
    over later targets, then a mean over valid source positions. Positions before skip_first and the last position
    are excluded, as in the fit. input_ids is [1, length]. The model's parameters must not require gradients.
    """
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Call model.requires_grad_(False) first: gradients here are with respect to activations")
    length = input_ids.size(1)
    if length <= skip_first + 1:
        raise ValueError(f"The prompt has {length} tokens, too few to skip {skip_first}")
    layers = get_layers(model)
    target_layer = len(layers) - 1
    root = min(source_layers)
    recorded: dict[int, torch.Tensor] = {}

    def make_hook(index: int) -> Any:
        def hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
            hidden, _ = split_layer_output(output)
            if index == root:
                hidden.requires_grad_(True)  # The leaf that roots the graph, as in jlens's ActivationRecorder.
            recorded[index] = hidden

        return hook

    handles = [layers[index].register_forward_hook(make_hook(index)) for index in {*source_layers, target_layer}]
    batch = len(dims)
    try:
        with torch.enable_grad():
            replicated = input_ids.expand(batch, -1)
            forward_last_logits(model, replicated)
            target = recorded[target_layer]  # [batch, length, d_model]
            valid = torch.arange(skip_first, length - 1, device=target.device)
            cotangent = torch.zeros_like(target)
            rows_index = torch.arange(batch, device=target.device)
            cotangent[rows_index.unsqueeze(1), valid.unsqueeze(0), torch.as_tensor(list(dims), device=target.device).unsqueeze(1)] = 1.0
            ordered = sorted(source_layers)
            grads = torch.autograd.grad(outputs=target, inputs=[recorded[index] for index in ordered], grad_outputs=cotangent)
    finally:
        for handle in handles:
            handle.remove()
    return {index: grad[:, valid.to(grad.device), :].float().mean(dim=1).detach().cpu() for index, grad in zip(ordered, grads)}


def _mean_row_cosine(measured: torch.Tensor, stored: torch.Tensor, dims: Sequence[int]) -> float:
    rows = stored[list(dims)]  # [len(dims), d_model]
    return float(torch.nn.functional.cosine_similarity(measured, rows, dim=1).mean())


def measure_alignment(
    model: nn.Module,
    jacobians: dict[int, torch.Tensor],
    input_ids: torch.Tensor,
    test_layers: Sequence[int],
    replicas: int,
    passes: int,
    skip_first: int,
    seed: int,
) -> dict[str, Any]:
    """Which file index best matches the Jacobian measured at each hook index.

    One prompt's Jacobian is noisy against the corpus mean, but the noise is shared across the candidates, so the
    comparison is paired. Neighbouring layers are similar, so this is evidence rather than proof: only a
    consistent offset fails the run, see alignment_verdict.
    """
    d_model = next(iter(jacobians.values())).size(0)
    generator = torch.Generator().manual_seed(seed)
    chosen = torch.randperm(d_model, generator=generator)[: replicas * passes].tolist()
    totals: dict[tuple[int, int], float] = {}
    available = sorted(jacobians)
    for start in range(0, len(chosen), replicas):
        dims = chosen[start : start + replicas]
        measured = prompt_jacobian_rows(model, input_ids, available, dims, skip_first)
        for layer in test_layers:
            for offset in (-1, 0, 1):
                candidate = layer + offset
                if candidate in jacobians:
                    totals[(layer, offset)] = totals.get((layer, offset), 0.0) + _mean_row_cosine(measured[layer], jacobians[candidate], dims)
        del measured
    n_passes = max(1, math.ceil(len(chosen) / replicas))
    by_layer = {}
    for layer in test_layers:
        cosines = {offset: totals[(layer, offset)] / n_passes for offset in (-1, 0, 1) if (layer, offset) in totals}
        by_layer[layer] = {"cosine_by_offset": cosines, "best_offset": max(cosines, key=lambda offset: cosines[offset])}
    return {"by_layer": by_layer, "n_rows": len(chosen), "n_tokens": int(input_ids.size(1))}


def alignment_verdict(best_offsets: Sequence[int]) -> tuple[bool, str]:
    """Fail only if at least three quarters of the tested layers agree on one nonzero offset."""
    if not best_offsets:
        return False, "No layer was tested"
    for offset in (-1, 1):
        share = sum(1 for value in best_offsets if value == offset) / len(best_offsets)
        if share >= 0.75:
            return False, f"Lens index {offset:+d} matches the measured Jacobian on {share:.0%} of tested layers"
    return True, f"Offset 0 is best on {sum(1 for value in best_offsets if value == 0)} of {len(best_offsets)} tested layers"
