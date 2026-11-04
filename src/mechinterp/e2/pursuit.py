"""Sparse nonnegative decomposition by gradient pursuit, for J-space (Gurnee et al., 2607.15495, 2.3 and A.8).

The paper says: the J-space component of x is the point nearest x among nonnegative combinations of at most k
J-lens vectors, approximated "using gradient pursuit" (Blumensath and Davies). It publishes no code. Where the
paper is silent, this module chooses, and experiments/e2_jspace_vector_families/README.md lists each choice:

- Atoms are unit-normalized for selection. A J-lens vector has an arbitrary norm that mostly reflects token
  frequency, and pursuit assumes unit atoms. Scaling an atom leaves its cone unchanged, so only the selection
  order depends on this.
- Selection takes the largest positive inner product with the residual, never a negative one.
- One projected gradient step per selection, with the exact line search, and coefficients clipped at zero.
- A final exact nonnegative least squares on the chosen atoms. The paper defines the component as the nearest
  point, and without this the residual is not orthogonal to the component, so shares would not add to one.
  A coefficient that reaches zero there frees its slot, and selection continues, a bounded number of rounds.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Bounded so that a worst case costs minutes, not hours. component_dot_residual reports any shortfall.
_NNLS_MAX_SWEEPS = 5000
_NNLS_TOLERANCE = 1e-10
_MAX_ROUNDS = 4
_POSITIVE = 1e-9


@dataclass(frozen=True)
class PursuitResult:
    atoms: tuple[int, ...]  # Indices into the dictionary, largest coefficient first.
    coefficients: tuple[float, ...]  # For the unit-normalized atoms.
    component: torch.Tensor  # [dim], float64 on CPU. Nonnegative combination of the atoms.
    residual: torch.Tensor  # [dim], float64 on CPU. target - component.


def unit_atoms(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (unit rows of the nonzero ones [m, dim], their original indices [m])."""
    norms = rows.norm(dim=1)
    keep = torch.nonzero(norms > 0).squeeze(1)
    return rows[keep] / norms[keep].unsqueeze(1), keep


def nnls_on_support(atoms: torch.Tensor, target: torch.Tensor, start: torch.Tensor) -> torch.Tensor:
    """Exact nonnegative least squares by cyclic coordinate descent. Atoms [s, dim] and target [dim] in float64.

    Convex, and its size is at most k by k, so this is cheap, needs no solver, and converges from any start.
    """
    gram = atoms @ atoms.T  # [s, s]
    linear = atoms @ target  # [s]
    x = start.clone()
    for _ in range(_NNLS_MAX_SWEEPS):
        change = 0.0
        for index in range(x.numel()):
            updated = torch.clamp(x[index] + (linear[index] - gram[index] @ x) / gram[index, index], min=0.0)
            change = max(change, float((updated - x[index]).abs()))
            x[index] = updated
        if change <= _NNLS_TOLERANCE:
            break
    return x


def gradient_pursuit(target: torch.Tensor, atoms: torch.Tensor, k: int) -> PursuitResult:
    """Decompose target [dim] over unit atoms [n_atoms, dim] with at most k positive coefficients.

    Selection runs on the atoms' device and dtype. The small support problems run in float64 on the CPU.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    if target.dim() != 1 or atoms.dim() != 2 or atoms.size(1) != target.numel():
        raise ValueError(f"Shapes do not fit: target {tuple(target.shape)}, atoms {tuple(atoms.shape)}")
    exact_target = target.detach().double().cpu()
    support: list[int] = []
    x = torch.zeros(0, dtype=torch.float64)
    residual = exact_target.clone()

    def support_atoms() -> torch.Tensor:
        return atoms[support].double().cpu()

    for _ in range(_MAX_ROUNDS):
        while len(support) < k:
            scores = atoms @ residual.to(atoms.device, atoms.dtype)  # [n_atoms]
            if support:
                scores[support] = float("-inf")
            best = int(scores.argmax())
            if float(scores[best]) <= _POSITIVE * max(float(exact_target.norm()), 1e-30):
                break
            support.append(best)
            x = torch.cat([x, torch.zeros(1, dtype=torch.float64)])
            chosen = support_atoms()
            gradient = chosen @ residual  # [s]
            direction = gradient @ chosen  # [dim]
            denominator = direction @ direction
            if denominator > 0:
                x = torch.clamp(x + (direction @ residual) / denominator * gradient, min=0.0)
            residual = exact_target - x @ chosen
        if not support:
            break
        chosen = support_atoms()
        x = nnls_on_support(chosen, exact_target, x)
        active = [index for index, value in enumerate(x.tolist()) if value > 0.0]
        support = [support[index] for index in active]
        x = x[active]
        residual = exact_target - (x @ support_atoms() if support else torch.zeros_like(exact_target))
        if len(support) >= k:
            break
        # Below k only if the search stopped or a coefficient fell to zero. Try once more with the freed slot.
        scores = atoms @ residual.to(atoms.device, atoms.dtype)
        if support:
            scores[support] = float("-inf")
        if float(scores.max()) <= _POSITIVE * max(float(exact_target.norm()), 1e-30):
            break

    order = sorted(range(len(support)), key=lambda index: -float(x[index]))
    return PursuitResult(
        atoms=tuple(support[index] for index in order),
        coefficients=tuple(float(x[index]) for index in order),
        component=exact_target - residual,
        residual=residual,
    )


def j_space_split(vector: torch.Tensor, atoms: torch.Tensor, k: int) -> tuple[PursuitResult, dict[str, float]]:
    """Decompose a unit concept vector, and return the numbers Stage 2 reports."""
    result = gradient_pursuit(vector, atoms, k)
    exact = vector.detach().double().cpu()
    total = float(exact @ exact)
    if total <= 0:
        raise ValueError("Cannot decompose a zero vector")
    return result, {
        "j_share": float(result.component @ result.component) / total,
        "residual_share": float(result.residual @ result.residual) / total,
        "component_dot_residual": float(result.component @ result.residual) / total,
        "n_atoms": float(len(result.atoms)),
    }
