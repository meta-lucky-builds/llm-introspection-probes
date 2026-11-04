"""Linear and MLP ensembles, following FIBS appendix A. Scoring is numpy only. Fitting needs torch, imported inside it.

Recipe from A.1, A.2 and A.6: per layer, s = w.h/kappa + b, with kappa the RMS norm of that layer over 512 training rows.
Logistic loss, labels +1 on rows whose final answer is a falsehood. AdamW at 5e-3 (constant for linear, cosine for MLP),
no weight decay, batch 32, 20 warm-up steps, up to 6 epochs, stopping when validation loss has not improved for an epoch,
keeping the epoch with the lowest mean validation loss. All layers share one gradient clip at 1.0 and one checkpoint.
A layer is kept if its training loss, over its last five logged steps, is below 0.6.

The score is a mean of sigmoids over kept layers. Its sign for a use is fixed once, in analysis.calibrate, on the calibration set.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..checkpoint import atomic_write_bytes
from .config import ProbeTraining


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -500.0, 500.0)))


@dataclass
class Ensemble:
    kind: str  # "linear" or "mlp"
    layers: list[int]  # Stored layer order, as model block indices.
    kappa: np.ndarray  # [layers]
    retained: np.ndarray  # [layers] bool
    params: dict[str, np.ndarray]  # linear: w [layers, d], b [layers]. mlp: w1, b1, w2, b2, one per layer.
    train_loss: np.ndarray  # [layers], over the last five logged steps
    info: dict[str, Any]

    def layer_logits(self, acts: np.ndarray) -> np.ndarray:
        """Acts [rows, layers, d], any float dtype. Returns [rows, layers]."""
        scaled = np.asarray(acts, dtype=np.float32) / self.kappa[None, :, None].astype(np.float32)
        if self.kind == "linear":
            return np.einsum("rld,ld->rl", scaled, self.params["w"]) + self.params["b"][None, :]
        # One matrix product per layer, so BLAS does the work. The einsum this replaces is single-threaded and took most of an 8B analysis.
        weights = self.params["w1"]
        hidden = np.stack([scaled[:, layer] @ weights[layer].T for layer in range(scaled.shape[1])], axis=1) + self.params["b1"][None]
        hidden = np.maximum(hidden, 0.0)
        return np.einsum("rlm,lm->rl", hidden, self.params["w2"]) + self.params["b2"][None, :]

    def score(self, acts: np.ndarray, batch: int = 2048) -> np.ndarray:
        """Mean sigmoid over retained layers, in float64: [rows]. NaN if no layer was retained."""
        if not self.retained.any():
            return np.full(len(acts), np.nan)
        pieces = []
        for start in range(0, len(acts), batch):
            logits = self.layer_logits(acts[start : start + batch]).astype(np.float64)
            pieces.append(sigmoid(logits[:, self.retained]).mean(axis=1))
        return np.concatenate(pieces) if pieces else np.zeros(0)

    def save(self, path: Path) -> None:
        import io
        import json

        buffer = io.BytesIO()
        np.savez(
            buffer,
            kind=np.array(self.kind),
            layers=np.array(self.layers),
            kappa=self.kappa,
            retained=self.retained,
            train_loss=self.train_loss,
            info=np.array(json.dumps(self.info)),
            **{f"param_{name}": value for name, value in self.params.items()},
        )
        atomic_write_bytes(path, buffer.getvalue())

    @classmethod
    def load(cls, path: Path) -> "Ensemble":
        import json

        with np.load(path, allow_pickle=False) as data:
            return cls(
                kind=str(data["kind"]),
                layers=[int(layer) for layer in data["layers"]],
                kappa=data["kappa"],
                retained=data["retained"],
                params={name[len("param_") :]: data[name] for name in data.files if name.startswith("param_")},
                train_loss=data["train_loss"],
                info=json.loads(str(data["info"])),
            )


def rms_scale(acts: np.ndarray, rows: int) -> np.ndarray:
    """kappa per layer: RMS of the row norms over the first `rows` training rows, which are already shuffled."""
    sample = np.asarray(acts[:rows], dtype=np.float64)
    return np.sqrt(np.mean(np.sum(sample**2, axis=-1), axis=0))


def class_balance_check(labels: np.ndarray) -> None:
    positive = int(labels.sum())
    if positive == 0 or positive == len(labels):
        raise ValueError("Probe training needs both classes")
    if abs(2 * positive - len(labels)) > max(2, len(labels) // 50):
        raise ValueError(f"Probe training must be class-balanced, got {positive} of {len(labels)} positive")


def fit_ensemble(
    train_acts: np.ndarray,
    train_labels: np.ndarray,
    val_acts: np.ndarray,
    val_labels: np.ndarray,
    layers: list[int],
    kind: str,
    settings: ProbeTraining,
    device: Any = "cpu",
    require_balance: bool = True,
) -> Ensemble:
    """Fit one probe per layer, in lockstep. Label-shuffled baselines pass labels through here too, balanced as before."""
    import torch

    if require_balance:
        class_balance_check(train_labels)
    n_rows, n_layers, d_model = train_acts.shape
    generator = torch.Generator().manual_seed(settings.seed)
    # Rows on disk are in plan order. A fixed shuffle first makes the 512 scale rows a fair sample.
    permutation = torch.randperm(n_rows, generator=generator).numpy()
    kappa = rms_scale(np.asarray(train_acts)[permutation[: settings.scale_rows]], settings.scale_rows)
    if not (np.isfinite(kappa).all() and (kappa > 0).all()):
        raise ValueError(f"Unusable activation scale {kappa}")
    device = torch.device(device)
    scale = torch.tensor(kappa, dtype=torch.float32, device=device)[None, :, None]
    x_train = torch.tensor(np.asarray(train_acts, dtype=np.float32), device=device) / scale
    y_train = torch.tensor(np.asarray(train_labels, dtype=np.float32), device=device)
    x_val = torch.tensor(np.asarray(val_acts, dtype=np.float32), device=device) / scale
    y_val = torch.tensor(np.asarray(val_labels, dtype=np.float32), device=device)

    torch.manual_seed(settings.seed)
    if kind == "linear":
        params = {"w": torch.zeros(n_layers, d_model, device=device), "b": torch.zeros(n_layers, device=device)}
    elif kind == "mlp":
        m = settings.mlp_hidden
        bound = 1.0 / np.sqrt(d_model)
        params = {
            "w1": (torch.rand(n_layers, m, d_model, device=device) * 2 - 1) * bound,
            "b1": (torch.rand(n_layers, m, device=device) * 2 - 1) * bound,
            "w2": (torch.rand(n_layers, m, device=device) * 2 - 1) / np.sqrt(m),
            "b2": torch.zeros(n_layers, device=device),
        }
    else:
        raise ValueError(f"Unknown probe kind {kind!r}")
    for value in params.values():
        value.requires_grad_(True)

    def logits_of(x: torch.Tensor) -> torch.Tensor:
        if kind == "linear":
            return torch.einsum("rld,ld->rl", x, params["w"]) + params["b"][None]
        hidden = torch.relu(torch.einsum("rld,lmd->rlm", x, params["w1"]) + params["b1"][None])
        return torch.einsum("rlm,lm->rl", hidden, params["w2"]) + params["b2"][None]

    def per_layer_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = logits_of(x)
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y[:, None].expand_as(logits), reduction="none"
        ).mean(dim=0)

    @torch.no_grad()
    def validation_loss() -> float:
        total = torch.zeros(n_layers, device=device)
        count = 0
        for start in range(0, len(x_val), 1024):
            batch = x_val[start : start + 1024]
            total += per_layer_loss(batch, y_val[start : start + 1024]) * len(batch)
            count += len(batch)
        return float((total / count).mean())

    optimizer = torch.optim.AdamW(list(params.values()), lr=settings.learning_rate, weight_decay=0.0)
    steps_per_epoch = max(1, n_rows // settings.batch_size)
    total_steps = steps_per_epoch * settings.max_epochs
    step = 0
    best: dict[str, Any] | None = None
    logged: list[np.ndarray] = []
    for epoch in range(settings.max_epochs):
        order = torch.randperm(n_rows, generator=generator).to(device)
        for start in range(0, steps_per_epoch * settings.batch_size, settings.batch_size):
            rows = order[start : start + settings.batch_size]
            rate = settings.learning_rate * min(1.0, (step + 1) / settings.warmup_steps)
            if kind == "mlp":
                rate *= 0.5 * (1.0 + float(np.cos(np.pi * step / max(1, total_steps))))
            for group in optimizer.param_groups:
                group["lr"] = rate
            losses = per_layer_loss(x_train[rows], y_train[rows])
            optimizer.zero_grad(set_to_none=True)
            # Each layer's probe only touches its own parameters, so the sum has per-layer gradients.
            losses.sum().backward()
            torch.nn.utils.clip_grad_norm_(list(params.values()), settings.grad_clip)
            optimizer.step()
            if step % settings.log_every == 0:
                logged.append(losses.detach().cpu().numpy())
            step += 1
        current = validation_loss()
        if not np.isfinite(current):
            raise FloatingPointError(f"Non-finite validation loss in {kind} probe at epoch {epoch}")
        if best is None or current < best["val_loss"]:
            tail = np.mean(np.stack(logged[-5:]), axis=0)
            best = {
                "val_loss": current,
                "epoch": epoch,
                "params": {name: value.detach().cpu().numpy().copy() for name, value in params.items()},
                "train_loss": tail,
            }
        elif epoch - best["epoch"] >= 1:
            break
    assert best is not None
    retained = best["train_loss"] < settings.drop_loss
    return Ensemble(
        kind=kind,
        layers=list(layers),
        kappa=kappa,
        retained=retained,
        params=best["params"],
        train_loss=best["train_loss"],
        info={"best_epoch": best["epoch"], "val_loss": best["val_loss"], "n_train": int(n_rows)},
    )


def contrast(yes_acts: np.ndarray, no_acts: np.ndarray, probe: Ensemble) -> np.ndarray:
    """Mean over retained layers of sigmoid(s(Yes.)) - sigmoid(s(No.)), unsigned. Order of arms never depends on condition."""
    return probe.score(yes_acts) - probe.score(no_acts)


def random_direction_probe(probe: Ensemble, seed: int) -> Ensemble:
    """Same norm per layer as the trained linear probe, in a random direction, same bias and scale."""
    if probe.kind != "linear":
        raise ValueError("Random directions are matched to the linear probe")
    rng = np.random.default_rng(seed)
    weights = probe.params["w"]
    directions = rng.standard_normal(weights.shape)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    matched = directions * np.linalg.norm(weights, axis=1, keepdims=True)
    return Ensemble(
        "linear", list(probe.layers), probe.kappa, probe.retained.copy(),
        {"w": matched.astype(np.float32), "b": probe.params["b"]}, probe.train_loss, {"random_seed": seed},
    )  # fmt: skip
