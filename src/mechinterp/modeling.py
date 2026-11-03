"""Model loading, layer discovery and last-token forward passes."""

from __future__ import annotations

import inspect
import re
from typing import Any

import torch
from torch import nn

COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")

# Checked in order. Later entries cover multimodal wrappers such as Gemma 3, whose
# text layers moved between transformers versions.
LAYER_PATHS = (
    "model.layers",
    "model.language_model.layers",
    "language_model.model.layers",
    "model.language_model.model.layers",
    "language_model.layers",
    "transformer.h",
    "gpt_neox.layers",
    "model.decoder.layers",
    "layers",
)


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def choose_dtype(device: torch.device) -> torch.dtype:
    """float32 on every device.

    bf16 logits between 16 and 32 have a step of 0.125, so exact ties become common and argmax then
    favours the lower label. float16 can overflow at high alpha. The cost is GPU memory.
    """
    return torch.float32


def resolve_model_commit(name: str, revision: str) -> str:
    """Return the full commit for a revision. A pinned commit needs no network."""
    if COMMIT_PATTERN.fullmatch(revision):
        return revision
    from huggingface_hub import HfApi

    commit = HfApi().model_info(name, revision=revision).sha
    if not commit:
        raise ValueError(f"Could not resolve {name}@{revision} to a commit")
    return commit


def load_model_and_tokenizer(name: str, commit: str, device: torch.device) -> tuple[nn.Module, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name, revision=commit)
    dtype = choose_dtype(device)
    try:
        model = AutoModelForCausalLM.from_pretrained(name, revision=commit, dtype=dtype)
    except TypeError:
        # Older transformers spell this argument torch_dtype.
        model = AutoModelForCausalLM.from_pretrained(name, revision=commit, torch_dtype=dtype)
    loaded_commit = getattr(model.config, "_commit_hash", None)
    if loaded_commit and loaded_commit != commit:
        raise ValueError(f"Asked for {name}@{commit} but loaded {loaded_commit}")
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model, tokenizer


def _resolve(root: nn.Module, dotted: str) -> Any:
    current: Any = root
    for part in dotted.split("."):
        current = getattr(current, part, None)
        if current is None:
            return None
    return current


def get_layers(model: nn.Module) -> nn.ModuleList:
    """Return the residual blocks of a causal language model."""
    for path in LAYER_PATHS:
        candidate = _resolve(model, path)
        if isinstance(candidate, nn.ModuleList) and len(candidate) > 0:
            return candidate
    # Fallback: the longest ModuleList whose name ends in layers, h or blocks. Vision
    # towers are skipped because their layers are not the text residual stream.
    best: nn.ModuleList | None = None
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if not isinstance(module, nn.ModuleList) or leaf not in ("layers", "h", "blocks"):
            continue
        if "vision" in name or "image" in name:
            continue
        if best is None or len(module) > len(best):
            best = module
    if best is None or len(best) == 0:
        raise ValueError(f"Could not find residual layers on {type(model).__name__}")
    return best


def hidden_size(model: nn.Module) -> int:
    return int(model.get_input_embeddings().embedding_dim)


def _supports_logits_to_keep(model: nn.Module) -> bool:
    return "logits_to_keep" in inspect.signature(model.forward).parameters


def forward_last_logits(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Run an unpadded batch and return float32 CPU logits at the last token: [batch, vocab].

    Batches are always same-length (see runner), so the last real token is index -1 and
    no attention-mask arithmetic can go wrong. logits_to_keep avoids projecting every
    position through the large output head, which on CPU costs about as much as the body.
    """
    kwargs: dict[str, Any] = {"logits_to_keep": 1} if _supports_logits_to_keep(model) else {}
    output = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        use_cache=False,
        **kwargs,
    )
    return output.logits[:, -1, :].float().cpu()
