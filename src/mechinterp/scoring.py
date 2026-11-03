"""First-token logit scoring."""

from __future__ import annotations

import torch

_TINY = 1e-300


def margin(label_logits: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """m = logit(label_j) - mean_{i != j} logit(label_i).

    label_logits: [batch, n_labels]; positions: [batch]. Returns [batch] in float64.
    """
    logits = label_logits.double()
    n_labels = logits.size(1)
    target = logits.gather(1, positions.long().unsqueeze(1)).squeeze(1)
    return target - (logits.sum(dim=1) - target) / (n_labels - 1)


def localization_effect(
    injected_logits: torch.Tensor, clean_logits: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """L = m_injected - m_clean, on the ten label logits."""
    return margin(injected_logits, positions) - margin(clean_logits, positions)


def js_divergence(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon divergence in nats between next-token distributions: [..., vocab] -> [...]."""
    p = torch.softmax(logits_p.double(), dim=-1)
    q = torch.softmax(logits_q.double(), dim=-1)
    m = 0.5 * (p + q)
    # Clamping keeps 0 * log(0) at 0 for entries that underflow.
    log_m = m.clamp_min(_TINY).log()
    kl_p = (p * (p.clamp_min(_TINY).log() - log_m)).sum(dim=-1)
    kl_q = (q * (q.clamp_min(_TINY).log() - log_m)).sum(dim=-1)
    return (0.5 * (kl_p + kl_q)).clamp_min(0.0)


def answer_mass(logits: torch.Tensor, answer_ids: list[int]) -> torch.Tensor:
    """Probability the full next-token distribution puts on the scored answers: [batch]."""
    probabilities = torch.softmax(logits.double(), dim=-1)
    return probabilities[:, answer_ids].sum(dim=1)
