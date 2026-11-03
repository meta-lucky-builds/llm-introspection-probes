from __future__ import annotations

import math

import pytest
import torch
from conftest import BASELINE, CONCEPTS

from mechinterp.modeling import get_layers
from mechinterp.scoring import answer_mass, js_divergence, localization_effect, margin
from mechinterp.vectors import concept_vectors, final_token_residuals, random_unit_vector


def test_margin_matches_the_definition() -> None:
    logits = torch.arange(10, dtype=torch.float32).unsqueeze(0)  # Logit i is i.
    others = [i for i in range(10) if i != 7]
    expected = 7 - sum(others) / 9
    assert margin(logits, torch.tensor([7])).item() == pytest.approx(expected)


def test_margin_is_per_example() -> None:
    logits = torch.tensor([[1.0] + [0.0] * 9, [0.0] * 9 + [2.0]])
    assert margin(logits, torch.tensor([0, 9])).tolist() == pytest.approx([1.0, 2.0])
    assert margin(logits, torch.tensor([1, 0])).tolist() == pytest.approx([-1 / 9, -2 / 9])


def test_effect_is_a_difference_of_margins_and_ignores_shared_shifts() -> None:
    clean = torch.randn(3, 10, generator=torch.Generator().manual_seed(0))
    positions = torch.tensor([0, 5, 9])
    assert torch.equal(localization_effect(clean, clean, positions), torch.zeros(3, dtype=torch.float64))
    shifted = clean + 4.0
    assert localization_effect(shifted, clean, positions).abs().max() < 1e-5
    boosted = clean.clone()
    boosted[0, 0] += 2.0
    assert localization_effect(boosted, clean, positions)[0].item() == pytest.approx(2.0, abs=1e-5)


def test_js_properties() -> None:
    p = torch.tensor([100.0, 0.0])
    q = torch.tensor([0.0, 100.0])
    assert js_divergence(p, p).item() == pytest.approx(0.0, abs=1e-12)
    assert js_divergence(p, q).item() == pytest.approx(math.log(2), abs=1e-6)
    a = torch.randn(4, 20, generator=torch.Generator().manual_seed(1))
    b = torch.randn(4, 20, generator=torch.Generator().manual_seed(2))
    assert torch.allclose(js_divergence(a, b), js_divergence(b, a))
    assert js_divergence(a, b).shape == (4,)
    # Extreme logits underflow to zero probability and must not give NaN.
    assert torch.isfinite(js_divergence(torch.tensor([1e4, -1e4, 0.0]), torch.tensor([-1e4, 1e4, 0.0])))


def test_answer_mass() -> None:
    logits = torch.zeros(1, 4)
    assert answer_mass(logits, [0, 1]).item() == pytest.approx(0.5)


def test_random_vectors_are_seeded_unit_and_independent() -> None:
    first = random_unit_vector(64, 1, "alpha")
    assert torch.equal(first, random_unit_vector(64, 1, "alpha"))
    assert first.norm().item() == pytest.approx(1.0, abs=1e-5)
    assert not torch.equal(first, random_unit_vector(64, 2, "alpha"))
    assert abs(torch.dot(first, random_unit_vector(64, 1, "beta")).item()) < 0.6


def test_concept_vectors_are_unit_per_layer_and_differ(model, tokenizer) -> None:
    layers = dict(enumerate(get_layers(model)))
    device = torch.device("cpu")
    vectors, norms = concept_vectors(model, tokenizer, layers, CONCEPTS["calibration"], BASELINE, 8, device)
    first, second = CONCEPTS["calibration"][:2]
    for index in layers:
        assert vectors[first][index].norm().item() == pytest.approx(1.0, abs=1e-5)
        assert vectors[first][index].shape == (32,)
        assert not torch.equal(vectors[first][index], vectors[second][index])
        assert norms[str(index)] > 0


def test_residuals_do_not_depend_on_batching(model, tokenizer) -> None:
    layers = {0: get_layers(model)[0]}
    device = torch.device("cpu")
    words = CONCEPTS["test"]
    together = final_token_residuals(model, tokenizer, layers, words, 8, device)
    alone = final_token_residuals(model, tokenizer, layers, words, 1, device)
    for word in words:
        assert torch.allclose(together[word][0], alone[word][0], atol=1e-4)
