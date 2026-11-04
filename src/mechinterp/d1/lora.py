"""A small LoRA that never changes the base model. No probe, no activation and no scorer is imported here.

peft injects by replacing modules in place, so an "unmodified" read after an adapter was loaded is one mistake away from
being through the adapter, and merging changes weights. This adapter is a set of forward hooks on the blocks' linear
layers instead: output + (alpha / rank) * B(A(input)). The weights are never written to, and leaving `applied` removes
every hook, which the tests and the dry run check down to identical outputs. It is never merged. B starts at zero, so a
new adapter reproduces the base model exactly.

Conditions differ only in training text. The initial A comes from the seed alone, and LoRA parameters are float32 even
over a bfloat16 base.
"""

from __future__ import annotations

import math
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import torch
from torch import nn

from ..checkpoint import atomic_write_bytes
from ..modeling import get_layers

TARGET_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
FORBIDDEN_IN_TRAINING = ("mechinterp.d1.probes", "mechinterp.d1.capture", "mechinterp.d1.analysis")


class LoraAdapter(nn.Module):
    def __init__(self, targets: dict[str, tuple[int, int]], rank: int, alpha: float, seed: int) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.names = sorted(targets)
        generator = torch.Generator().manual_seed(seed)
        self.down = nn.ParameterList()
        self.up = nn.ParameterList()
        for name in self.names:
            features_in, features_out = targets[name]
            matrix = torch.empty(rank, features_in, dtype=torch.float32)
            nn.init.kaiming_uniform_(matrix, a=math.sqrt(5), generator=generator)
            self.down.append(nn.Parameter(matrix))
            self.up.append(nn.Parameter(torch.zeros(features_out, rank, dtype=torch.float32)))

    @property
    def scale(self) -> float:
        return self.alpha / self.rank

    @classmethod
    def for_model(cls, model: nn.Module, rank: int, alpha: float, seed: int) -> "LoraAdapter":
        return cls(find_targets(model), rank, alpha, seed)

    def delta(self, index: int, inputs: torch.Tensor) -> torch.Tensor:
        return self.scale * (inputs.float() @ self.down[index].T @ self.up[index].T)

    @contextmanager
    def applied(self, model: nn.Module) -> Iterator[None]:
        handles = []
        modules = named_block_linears(model)
        missing = [name for name in self.names if name not in modules]
        if missing:
            raise ValueError(f"The adapter targets modules this model does not have: {missing[:3]}")

        def make_hook(index: int) -> Any:
            def hook(_module: nn.Module, inputs: Any, output: torch.Tensor) -> torch.Tensor:
                return output + self.delta(index, inputs[0]).to(output.dtype)

            return hook

        for index, name in enumerate(self.names):
            handles.append(modules[name].register_forward_hook(make_hook(index)))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def delta_norm(self) -> float:
        """Frobenius norm over targets of the low-rank update, to check that conditions moved by comparable amounts."""
        total = 0.0
        with torch.no_grad():
            for down, up in zip(self.down, self.up):
                # ||U D||_F^2 = trace(D D^T U^T U), which needs no features-by-features matrix.
                total += float((down @ down.T * (up.T @ up)).sum()) * self.scale**2
        return math.sqrt(max(total, 0.0))

    def save(self, path: Path, metadata: dict[str, Any]) -> None:
        import io

        buffer = io.BytesIO()
        state = {
            "rank": self.rank,
            "alpha": float(self.alpha),
            "names": list(self.names),
            "down": [tensor.detach().cpu() for tensor in self.down],
            "up": [tensor.detach().cpu() for tensor in self.up],
            "metadata": metadata,
        }
        torch.save(state, buffer)
        atomic_write_bytes(path, buffer.getvalue())

    @classmethod
    def load(cls, path: Path, device: torch.device) -> tuple["LoraAdapter", dict[str, Any]]:
        state = torch.load(path, map_location="cpu", weights_only=True)
        targets = {name: (state["down"][i].shape[1], state["up"][i].shape[0]) for i, name in enumerate(state["names"])}
        adapter = cls(targets, int(state["rank"]), float(state["alpha"]), seed=0)
        with torch.no_grad():
            for index in range(len(adapter.names)):
                adapter.down[index].copy_(state["down"][index])
                adapter.up[index].copy_(state["up"][index])
        adapter.requires_grad_(False)
        return adapter.to(device), state["metadata"]


def named_block_linears(model: nn.Module) -> dict[str, nn.Linear]:
    found: dict[str, nn.Linear] = {}
    for index, block in enumerate(get_layers(model)):
        for name, module in block.named_modules():
            if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in TARGET_NAMES:
                found[f"{index}.{name}"] = module
    if not found:
        raise ValueError(f"No LoRA targets on {type(model).__name__}")
    return found


def find_targets(model: nn.Module) -> dict[str, tuple[int, int]]:
    return {name: (module.in_features, module.out_features) for name, module in named_block_linears(model).items()}


def assert_no_probe_in_loss(model: nn.Module, check_imports: bool = True) -> None:
    """The prereg's condition, enforced at run time.

    Training must not have the probe, capture or analysis code loaded in its process, and no forward hook may sit on a
    block, which is how an activation term would have to enter. Only the LoRA hooks, on inner linear layers, may exist.
    check_imports is off only for tests that run in pytest's process, where those modules are already loaded.
    """
    loaded = [name for name in FORBIDDEN_IN_TRAINING if name in sys.modules] if check_imports else []
    if loaded:
        raise AssertionError(f"Adapter training must not load {loaded}: no activations or probes in any loss")
    for index, block in enumerate(get_layers(model)):
        if len(block._forward_hooks) > 0 or len(block._forward_pre_hooks) > 0:
            raise AssertionError(f"Block {index} has a hook during adapter training")


def answer_loss(logits: torch.Tensor, labels: torch.Tensor, row_index: torch.Tensor, n_rows: int) -> torch.Tensor:
    """Cross-entropy on answer tokens only. Each row weighs the same, however long its answer. Takes only logits."""
    losses = torch.nn.functional.cross_entropy(logits.float(), labels, reduction="none")
    totals = torch.zeros(n_rows, device=logits.device, dtype=losses.dtype).index_add(0, row_index, losses)
    counts = torch.zeros(n_rows, device=logits.device, dtype=losses.dtype).index_add(0, row_index, torch.ones_like(losses))
    return (totals / counts.clamp_min(1.0)).sum()


def micro_batch_loss(model: nn.Module, rows: Sequence[tuple[list[int], list[int]]], pad: int, device: torch.device) -> torch.Tensor:
    """Summed row losses. Each row is (ids, labels), with label -100 on tokens outside the answer."""
    width = max(len(ids) for ids, _ in rows)
    input_ids = torch.full((len(rows), width), pad, dtype=torch.long, device=device)
    attention = torch.zeros_like(input_ids)
    targets = torch.full((len(rows), width), -100, dtype=torch.long, device=device)
    for row, (ids, labels) in enumerate(rows):
        input_ids[row, : len(ids)] = torch.tensor(ids, device=device)
        attention[row, : len(ids)] = 1
        targets[row, : len(labels)] = torch.tensor(labels, device=device)
    head = model.get_output_embeddings()
    body = getattr(model, "model", None) or model.get_decoder()
    hidden = body(input_ids=input_ids, attention_mask=attention, use_cache=False).last_hidden_state
    # A token's logits predict the next token, so positions shift by one.
    shifted = targets[:, 1:]
    mask = shifted != -100
    row_index, position = mask.nonzero(as_tuple=True)
    logits = head(hidden[:, :-1][row_index, position])
    return answer_loss(logits, shifted[row_index, position], row_index, len(rows))


def train_adapter(
    model: nn.Module,
    adapter: LoraAdapter,
    batches: Callable[[int], Sequence[tuple[list[int], list[int]]]],
    steps: int,
    micro_batch: int,
    learning_rate: float,
    warmup_steps: int,
    pad: int,
    device: torch.device,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 200,
    heartbeat: Callable[[int, int, float], None] | None = None,
    grad_clip: float = 1.0,
    check_imports: bool = True,
    gradient_checkpointing: bool = True,
    halve_on_oom: bool = True,
) -> dict[str, Any]:
    """Fit the adapter's parameters. `batches(step)` must be a pure function of the step, which is what allows resume.

    Gradient checkpointing computes the same numbers, since no dropout runs here, at the cost of another forward pass. So
    it can be left off when the card has the memory. With halve_on_oom off, running out of memory ends the process instead
    of changing the micro-batch, so the pool can make room and the adapter's rounding stays the same as on an empty card.
    """
    assert_no_probe_in_loss(model, check_imports)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("The base model must be frozen before adapter training")
    adapter.to(device).requires_grad_(True)
    parameters = [*adapter.down, *adapter.up]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
    first_step = 0
    losses: list[float] = []
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        adapter.load_state_dict(state["adapter"])
        optimizer.load_state_dict(state["optimizer"])
        first_step = int(state["step"])
        losses = list(state["losses"])
    model.train()
    if getattr(model, "gradient_checkpointing_enable", None) is not None and getattr(model, "supports_gradient_checkpointing", False):
        if gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            model.gradient_checkpointing_disable()
    started = time.monotonic()
    size = micro_batch
    with adapter.applied(model):
        for step in range(first_step, steps):
            rows = list(batches(step))
            for group in optimizer.param_groups:
                group["lr"] = learning_rate * min(1.0, (step + 1) / max(1, warmup_steps))
            while True:
                optimizer.zero_grad(set_to_none=True)
                total = 0.0
                try:
                    order = sorted(range(len(rows)), key=lambda index: -len(rows[index][0]))
                    for offset in range(0, len(order), size):
                        chunk = [rows[index] for index in order[offset : offset + size]]
                        loss = micro_batch_loss(model, chunk, pad, device) / len(rows)
                        loss.backward()
                        total += float(loss.detach())
                    break
                except Exception as error:
                    if not (isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()) or size == 1 or not halve_on_oom:
                        raise
                    size = max(1, size // 2)
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    print(f"Out of GPU memory in training, so lowering the micro-batch to {size}", flush=True)
            if not math.isfinite(total):
                raise FloatingPointError(f"Non-finite adapter loss at step {step}")
            torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
            optimizer.step()
            losses.append(total)
            done = step + 1
            if checkpoint_path is not None and (done % checkpoint_every == 0):
                import io

                buffer = io.BytesIO()
                torch.save({"adapter": adapter.state_dict(), "optimizer": optimizer.state_dict(), "step": done, "losses": losses}, buffer)
                atomic_write_bytes(checkpoint_path, buffer.getvalue())
            if heartbeat is not None and (done % 10 == 0 or done == steps):
                heartbeat(done, steps, (done - first_step) / max(1e-6, time.monotonic() - started))
    model.eval()
    tail = losses[-50:] if losses else [float("nan")]
    return {"steps": steps, "final_loss": sum(tail) / len(tail), "first_loss": losses[0] if losses else None, "delta_norm": adapter.delta_norm()}
