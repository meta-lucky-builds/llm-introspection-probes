from __future__ import annotations

import pytest
import torch
from conftest import tiny_llama, tiny_qwen
from torch import nn

from mechinterp.hooks import capture_layer_outputs, inject, inject_activation
from mechinterp.modeling import choose_dtype, forward_last_logits, get_layers
from mechinterp.prompts import LABELS
from mechinterp.scoring import localization_effect
from mechinterp.vectors import random_unit_vector

VOCAB = 50
SEQ = 12
INJECTION_LAYER = 1


def _inputs() -> torch.Tensor:
    generator = torch.Generator().manual_seed(3)
    return torch.randint(2, VOCAB, (2, SEQ), generator=generator)


def _injection() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positions = torch.tensor([3, 7])
    vectors = torch.stack([random_unit_vector(32, 1, "a"), random_unit_vector(32, 1, "b")])
    alphas = torch.tensor([2.0, 0.5])
    return positions, vectors, alphas


@pytest.mark.parametrize("builder", [tiny_qwen, tiny_llama])
def test_layers_found_on_real_architectures(builder) -> None:
    assert len(get_layers(builder(VOCAB))) == 3


def test_layer_fallback_paths() -> None:
    class Transformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.h = nn.ModuleList([nn.Linear(2, 2)])

    class GptStyle(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer = Transformer()

    class Unusual(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vision_stack = nn.Module()
            self.vision_stack.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(9)])
            self.text_stack = nn.Module()
            self.text_stack.blocks = nn.ModuleList([nn.Linear(2, 2) for _ in range(3)])

    assert len(get_layers(GptStyle())) == 1
    # The longest list that is not part of a vision tower.
    assert len(get_layers(Unusual())) == 3
    with pytest.raises(ValueError):
        get_layers(nn.Linear(2, 2))


def test_dtype_per_device() -> None:
    assert choose_dtype(torch.device("cpu")) == torch.float32


@pytest.mark.parametrize("builder", [tiny_qwen, tiny_llama])
def test_injection_touches_only_the_target(builder) -> None:
    model = builder(VOCAB)
    layers = get_layers(model)
    all_layers = dict(enumerate(layers))
    input_ids = _inputs()
    positions, vectors, alphas = _injection()

    with torch.inference_mode(), capture_layer_outputs(all_layers) as clean:
        forward_last_logits(model, input_ids)
    with torch.inference_mode(), inject(layers[INJECTION_LAYER], positions, vectors, alphas), capture_layer_outputs(
        all_layers
    ) as injected:
        forward_last_logits(model, input_ids)

    # Before the injection layer nothing has changed.
    assert (clean[0] - injected[0]).abs().max() < 1e-6

    # At the injection layer, only the target changed, by alpha * ||h|| * v.
    difference = injected[INJECTION_LAYER] - clean[INJECTION_LAYER]
    for row in range(2):
        position = int(positions[row])
        norm = clean[INJECTION_LAYER][row, position].norm()
        expected = alphas[row] * norm * vectors[row]
        assert torch.allclose(difference[row, position], expected, atol=1e-5)
        others = torch.ones(SEQ, dtype=torch.bool)
        others[position] = False
        assert difference[row, others].abs().max() < 1e-6

    # Later layers are causal: earlier tokens cannot see the injection, later tokens can.
    later = (injected[2] - clean[2]).abs().amax(dim=-1)  # [batch, seq]
    for row in range(2):
        position = int(positions[row])
        assert later[row, :position].max() < 1e-5
        assert later[row, position + 1 :].max() > 1e-6


def test_alpha_zero_gives_exactly_zero_effect(model) -> None:
    layers = get_layers(model)
    input_ids = torch.randint(2, model.config.vocab_size, (4, SEQ), generator=torch.Generator().manual_seed(5))
    positions = torch.tensor([1, 4, 6, 9])
    vectors = torch.stack([random_unit_vector(32, 1, str(index)) for index in range(4)])
    label_ids = list(range(2, 2 + len(LABELS)))
    label_positions = torch.tensor([0, 3, 7, 9])
    with torch.inference_mode():
        clean = forward_last_logits(model, input_ids)
        with inject(layers[INJECTION_LAYER], positions, vectors, torch.zeros(4)):
            sham = forward_last_logits(model, input_ids)
        with inject(layers[INJECTION_LAYER], positions, vectors, torch.full((4,), 3.0)):
            real = forward_last_logits(model, input_ids)
    effect = localization_effect(sham[:, label_ids], clean[:, label_ids], label_positions)
    assert torch.equal(effect, torch.zeros(4, dtype=torch.float64))
    assert not torch.equal(real, clean)


class _TupleLayer(nn.Module):
    def forward(self, x: torch.Tensor):
        return x, "extra"


class _ListLayer(nn.Module):
    def forward(self, x: torch.Tensor):
        return [x, "extra"]


class _TensorLayer(nn.Module):
    def forward(self, x: torch.Tensor):
        return x


@pytest.mark.parametrize("layer_class", [_TupleLayer, _ListLayer, _TensorLayer])
def test_layer_output_forms(layer_class) -> None:
    layer = layer_class()
    hidden = torch.ones(2, 5, 3)
    vectors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    with inject(layer, torch.tensor([1, 4]), vectors, torch.tensor([1.0, 2.0])):
        output = layer(hidden)
    result = output if isinstance(output, torch.Tensor) else output[0]
    if not isinstance(output, torch.Tensor):
        assert type(output) in (tuple, list) and output[1] == "extra"
    norm = 3**0.5
    assert result[0, 1, 0].item() == pytest.approx(1 + norm)
    assert result[1, 4, 1].item() == pytest.approx(1 + 2 * norm)
    assert result.sum().item() == pytest.approx(30 + norm + 2 * norm)
    # The input is not modified, and the hook is gone afterwards.
    assert hidden.sum().item() == 30
    after = layer(hidden)
    assert (after if isinstance(after, torch.Tensor) else after[0]).sum().item() == 30


def test_unsupported_output_and_bad_shapes_raise() -> None:
    class DictLayer(nn.Module):
        def forward(self, x: torch.Tensor):
            return {"h": x}

    layer = DictLayer()
    with inject(layer, torch.tensor([0]), torch.zeros(1, 3), torch.zeros(1)), pytest.raises(TypeError):
        layer(torch.ones(1, 2, 3))
    with pytest.raises(ValueError):
        inject_activation(torch.ones(1, 2, 3), torch.tensor([5]), torch.zeros(1, 3), torch.zeros(1))
    with pytest.raises(ValueError):
        inject_activation(torch.ones(2, 2, 3), torch.tensor([0]), torch.zeros(1, 3), torch.zeros(1))


def test_last_logits_match_full_forward_and_batching(model) -> None:
    input_ids = torch.randint(2, model.config.vocab_size, (3, SEQ), generator=torch.Generator().manual_seed(9))
    with torch.inference_mode():
        last = forward_last_logits(model, input_ids)
        full = model(input_ids=input_ids, use_cache=False).logits[:, -1, :]
        single = forward_last_logits(model, input_ids[1:2])
    assert last.shape == (3, model.config.vocab_size)
    assert torch.allclose(last, full, atol=1e-4)
    assert torch.allclose(last[1:2], single, atol=1e-4)
