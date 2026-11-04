"""Gradient pursuit on known sparse solutions, and the lens readout on a tiny model. No GPU and no download."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from conftest import tiny_qwen
from e2_factory import ALIGNMENT_TEXT, DIM, build, write_stub_lens
from transformers import Qwen2Config, Qwen2ForCausalLM

from mechinterp.e2.families import concept_token, family_fingerprint, split_vector
from mechinterp.e2.lens import (
    LensError,
    alignment_verdict,
    check_identity_is_logit_lens,
    check_structure,
    check_unembed,
    dictionary,
    dictionary_token_ids,
    final_norm,
    lens_rows,
    load_lens,
    measure_alignment,
    output_weights,
    prompt_jacobian_rows,
    ranks_of,
    read,
    transport,
    unembed,
)
from mechinterp.e2.pursuit import gradient_pursuit, j_space_split, nnls_on_support, unit_atoms
from mechinterp.hooks import capture_layer_outputs, inject
from mechinterp.modeling import forward_last_logits, get_layers
from mechinterp.sharding import sha256_file


def random_atoms(n: int = 300, dim: int = 128, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    atoms, _ = unit_atoms(torch.randn(n, dim, generator=generator, dtype=torch.float32))
    return atoms


# Gradient pursuit


def test_recovers_a_known_sparse_nonnegative_solution() -> None:
    atoms = random_atoms()
    chosen = [3, 50, 120, 199, 280]
    weights = torch.tensor([3.0, 2.0, 1.5, 1.0, 0.7])
    target = weights @ atoms[chosen]
    result = gradient_pursuit(target, atoms, k=5)
    assert set(result.atoms) == set(chosen)
    assert list(result.atoms) == chosen  # Largest coefficient first.
    assert torch.allclose(torch.tensor(result.coefficients, dtype=torch.float64), weights.double(), atol=1e-4)
    assert float(result.residual.norm()) < 1e-4


def test_never_uses_more_than_k_atoms_and_never_a_negative_coefficient() -> None:
    atoms = random_atoms()
    target = torch.randn(128, generator=torch.Generator().manual_seed(1))
    for k in (1, 4, 16):
        result = gradient_pursuit(target, atoms, k)
        assert 0 < len(result.atoms) <= k
        assert len(set(result.atoms)) == len(result.atoms)
        assert all(value > 0 for value in result.coefficients)


def test_k_of_one_takes_the_best_cosine_and_its_exact_projection() -> None:
    atoms = random_atoms()
    target = torch.randn(128, generator=torch.Generator().manual_seed(2))
    scores = atoms @ target
    result = gradient_pursuit(target, atoms, 1)
    assert result.atoms == (int(scores.argmax()),)
    assert result.coefficients[0] == pytest.approx(float(scores.max()), rel=1e-4)


def test_a_target_opposed_to_every_atom_has_an_empty_component() -> None:
    atoms = torch.eye(8)
    result = gradient_pursuit(-torch.ones(8), atoms, 3)
    assert result.atoms == ()
    assert float(result.component.norm()) == 0.0


def test_a_nonnegative_constraint_is_a_real_constraint() -> None:
    # The target needs a negative weight on atom 1. Nonnegative pursuit must not give it one.
    atoms = torch.eye(4)
    target = torch.tensor([1.0, -1.0, 0.0, 0.0])
    result = gradient_pursuit(target, atoms, 2)
    assert result.atoms == (0,)
    assert result.coefficients[0] == pytest.approx(1.0)


def test_residual_is_orthogonal_to_component_so_shares_add_to_one() -> None:
    atoms = random_atoms()
    vector = torch.randn(128, generator=torch.Generator().manual_seed(3))
    vector = vector / vector.norm()
    _, numbers = j_space_split(vector, atoms, 16)
    assert abs(numbers["component_dot_residual"]) < 1e-6
    assert numbers["j_share"] + numbers["residual_share"] == pytest.approx(1.0, abs=1e-6)


def test_nnls_solves_a_small_problem_exactly() -> None:
    atoms = torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float64) / torch.tensor([[1.0], [2.0**0.5]], dtype=torch.float64)
    target = torch.tensor([3.0, 2.0], dtype=torch.float64)  # Exactly 1 of the first atom and 2 sqrt(2) of the second.
    x = nnls_on_support(atoms, target, torch.zeros(2, dtype=torch.float64))
    assert (x @ atoms).tolist() == pytest.approx([3.0, 2.0], abs=1e-7)


def test_zero_rows_are_dropped_and_indices_are_kept() -> None:
    rows = torch.tensor([[3.0, 4.0], [0.0, 0.0], [0.0, 2.0]])
    atoms, kept = unit_atoms(rows)
    assert kept.tolist() == [0, 2]
    assert atoms.norm(dim=1).tolist() == pytest.approx([1.0, 1.0])


# Families


def test_families_are_unit_and_rebuild_the_full_vector() -> None:
    atoms = random_atoms()
    vector = torch.randn(128, generator=torch.Generator().manual_seed(4))
    vector = vector / vector.norm()
    family, numbers, selected, coefficients = split_vector(vector, atoms, 16)
    for name in ("F_full", "F_j", "F_res"):
        assert float(family[name].norm()) == pytest.approx(1.0, abs=1e-5)
    rebuilt = numbers["component_norm"] * family["F_j"].double() + numbers["residual_norm"] * family["F_res"].double()
    assert torch.allclose(rebuilt, vector.double(), atol=1e-5)
    assert len(selected) == len(coefficients) <= 16


def test_only_the_spelling_in_the_concept_prompt_counts_as_the_concepts_token(tokenizer) -> None:
    assert concept_token(tokenizer, "contaa") is not None
    assert concept_token(tokenizer, "two-part") is None  # Several tokens.


def test_fingerprint_changes_with_the_vectors() -> None:
    first = {"a": torch.ones(8) / 8**0.5}
    second = {"a": -torch.ones(8) / 8**0.5}
    assert family_fingerprint(first, 1) != family_fingerprint(second, 1)


# Readout


@pytest.fixture(scope="module")
def tiny() -> tuple:
    model, tokenizer, device = build(None, 0)
    ids = torch.tensor([list(tokenizer(ALIGNMENT_TEXT)["input_ids"])], device=device)
    return model, tokenizer, device, ids


def test_norm_means_the_models_final_norm_and_not_a_unit_normalization(tiny) -> None:
    model, _, _, _ = tiny
    x = torch.randn(2, DIM, generator=torch.Generator().manual_seed(5))
    expected = final_norm(model)(x) @ output_weights(model).T
    assert torch.allclose(unembed(model, x), expected, atol=1e-4)
    unit_read = (x / x.norm(dim=-1, keepdim=True)) @ output_weights(model).T
    assert not torch.allclose(unembed(model, x), unit_read, atol=1e-3)


def test_the_readout_is_h_at_j_transposed(tiny) -> None:
    model, _, _, _ = tiny
    h = torch.randn(DIM, generator=torch.Generator().manual_seed(6))
    jacobian = torch.randn(DIM, DIM, generator=torch.Generator().manual_seed(7))
    assert torch.allclose(transport(h, jacobian), jacobian @ h, atol=1e-4)
    assert torch.allclose(read(model, h.unsqueeze(0), jacobian), unembed(model, (jacobian @ h).unsqueeze(0)), atol=1e-4)


def test_identity_jacobian_is_the_logit_lens(tiny) -> None:
    model, _, _, _ = tiny
    assert check_identity_is_logit_lens(model, torch.randn(3, DIM, generator=torch.Generator().manual_seed(8))) <= 1e-6


def test_the_final_block_through_norm_and_head_is_the_models_logits(tiny) -> None:
    model, _, _, ids = tiny
    assert check_unembed(model, ids) <= 1e-3


def test_a_wrong_norm_fails_the_final_block_check(tiny, monkeypatch) -> None:
    model, _, _, ids = tiny
    monkeypatch.setattr("mechinterp.e2.lens.final_norm", lambda _model: torch.nn.Identity())
    with pytest.raises(LensError, match="do not reproduce"):
        check_unembed(model, ids)


def test_lens_rows_are_rows_of_w_u_j(tiny) -> None:
    model, tokenizer, _, _ = tiny
    jacobian = torch.randn(DIM, DIM, generator=torch.Generator().manual_seed(9))
    expected = output_weights(model)[[1, 5]] @ jacobian
    assert torch.allclose(lens_rows(model, jacobian, [1, 5]), expected, atol=1e-4)
    ids = dictionary_token_ids(tokenizer, output_weights(model).size(0))
    assert tokenizer.unk_token_id not in ids and tokenizer.eos_token_id not in ids
    assert torch.allclose(dictionary(model, jacobian, ids), output_weights(model)[ids] @ jacobian, atol=1e-4)


def test_ranks_count_from_one_and_share_ties_in_favour() -> None:
    logits = torch.tensor([[5.0, 3.0, 3.0, 1.0]])
    assert ranks_of(logits, [0, 1, 2, 3]).tolist() == [[1, 2, 2, 4]]
    assert ranks_of(logits, []).shape == (1, 0)


# The file and the layer convention


def test_load_converts_to_float32_and_records_what_it_found(tmp_path: Path) -> None:
    path = write_stub_lens(tmp_path / "lens.pt", "identity")
    spec = SimpleNamespace(size=path.stat().st_size, sha256=sha256_file(path), n_prompts=1)
    lens = load_lens(path, spec, n_layers=3, d_model=DIM)
    assert lens.source_layers == (0, 1)
    assert lens.jacobians[0].dtype == torch.float32
    assert lens.report["stored_dtypes"] == ["torch.float16"]
    assert lens.report["target_layer_inferred"] == 2
    assert lens.report["distance_from_identity_by_layer"][0] == 0.0


@pytest.mark.parametrize(
    "field, value, message",
    [("sha256", "0" * 64, "The pin"), ("size", 1, "The pin"), ("n_prompts", 999, "prompts")],
)
def test_load_refuses_a_file_that_is_not_the_pin(tmp_path: Path, field: str, value, message: str) -> None:
    path = write_stub_lens(tmp_path / "lens.pt", "identity")
    spec = SimpleNamespace(size=path.stat().st_size, sha256=sha256_file(path), n_prompts=1)
    setattr(spec, field, value)
    with pytest.raises(LensError, match=message):
        load_lens(path, spec, n_layers=3, d_model=DIM)


def _write_custom(path: Path, payload) -> SimpleNamespace:
    torch.save(payload, path)
    return SimpleNamespace(size=path.stat().st_size, sha256=sha256_file(path), n_prompts=None)


def test_load_refuses_wrong_shapes_keys_and_values(tmp_path: Path) -> None:
    path = tmp_path / "lens.pt"
    good = {0: torch.eye(DIM), 1: torch.eye(DIM)}
    with pytest.raises(LensError, match="not a Jacobian lens"):
        load_lens(path, _write_custom(path, {"other": 1}), 3, DIM)
    with pytest.raises(LensError, match="shape"):
        load_lens(path, _write_custom(path, {"J": {0: torch.eye(4), 1: torch.eye(4)}}), 3, DIM)
    with pytest.raises(LensError, match="non-finite"):
        load_lens(path, _write_custom(path, {"J": {**good, 1: torch.eye(DIM) * float("inf")}}), 3, DIM)
    with pytest.raises(LensError, match="convention"):
        load_lens(path, _write_custom(path, {"J": {**good, 2: torch.eye(DIM)}}), 3, DIM)


def test_a_file_indexed_like_hf_hidden_states_fails_the_structure_check() -> None:
    check_structure(range(35), 36)
    with pytest.raises(LensError, match="convention"):
        check_structure(range(36), 36)
    with pytest.raises(LensError, match="convention"):
        check_structure(range(1, 36), 36)


def test_lens_index_is_the_module_e1_injects_at(tiny) -> None:
    """The same integer must mean the same point: a perturbation at layers[l] must show up in the capture of l."""
    model, _, device, ids = tiny
    layers = get_layers(model)
    positions = torch.tensor([3], device=device)
    vector = torch.ones(1, DIM) / DIM**0.5
    with torch.inference_mode(), capture_layer_outputs({0: layers[0]}) as clean:
        forward_last_logits(model, ids)
    # Forward hooks run in registration order, so a capture inside inject sees the injected output of the same block.
    with torch.inference_mode(), inject(layers[0], positions, vector, torch.ones(1)), capture_layer_outputs({0: layers[0]}) as injected:
        forward_last_logits(model, ids)
    assert float((injected[0] - clean[0])[0, 3].norm()) > 0
    assert float((injected[0] - clean[0])[0, 2].norm()) == 0.0


def _six_layer_model() -> torch.nn.Module:
    torch.manual_seed(1)
    config = Qwen2Config(
        vocab_size=50, hidden_size=DIM, intermediate_size=64, num_hidden_layers=6, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64
    )
    return Qwen2ForCausalLM(config).eval().requires_grad_(False)


def test_measured_jacobian_matches_the_right_index_and_a_shifted_file_fails() -> None:
    model = _six_layer_model()
    ids = torch.arange(1, 13).unsqueeze(0)
    rows = prompt_jacobian_rows(model, ids, list(range(5)), list(range(DIM)), skip_first=1)
    correct = measure_alignment(model, rows, ids, [1, 2, 3, 4], 4, 2, 1, seed=0)
    assert [entry["best_offset"] for entry in correct["by_layer"].values()] == [0, 0, 0, 0]
    assert alignment_verdict([0, 0, 0, 0])[0] is True

    # Keys one too low: what a hidden_states-style index would look like here.
    shifted = {layer: rows[min(layer + 1, 4)] for layer in range(5)}
    measured = measure_alignment(model, shifted, ids, [1, 2, 3, 4], 4, 2, 1, seed=0)
    offsets = [entry["best_offset"] for entry in measured["by_layer"].values()]
    assert offsets == [-1, -1, -1, -1]
    assert alignment_verdict(offsets)[0] is False


def test_a_transposed_file_scores_worse_than_the_right_one() -> None:
    model = _six_layer_model()
    ids = torch.arange(1, 13).unsqueeze(0)
    rows = prompt_jacobian_rows(model, ids, [2], list(range(DIM)), skip_first=1)
    right = measure_alignment(model, rows, ids, [2], 4, 2, 1, seed=0)["by_layer"][2]["cosine_by_offset"][0]
    wrong = measure_alignment(model, {2: rows[2].T}, ids, [2], 4, 2, 1, seed=0)["by_layer"][2]["cosine_by_offset"][0]
    assert right == pytest.approx(1.0, abs=1e-4)
    assert wrong < right - 1e-3


def test_verdict_tolerates_noise_and_needs_a_consistent_offset() -> None:
    assert alignment_verdict([0, 1, 0, -1])[0] is True
    assert alignment_verdict([1, 1, 1, 0])[0] is False
    assert alignment_verdict([])[0] is False


def test_estimator_needs_frozen_parameters() -> None:
    model = tiny_qwen(20)
    with pytest.raises(ValueError, match="requires_grad"):
        prompt_jacobian_rows(model, torch.arange(1, 10).unsqueeze(0), [0], [0], skip_first=1)
    model.requires_grad_(False)


def test_replace_keeps_the_lens_dataclass_usable(tmp_path: Path) -> None:
    path = write_stub_lens(tmp_path / "lens.pt", "identity")
    spec = SimpleNamespace(size=path.stat().st_size, sha256=sha256_file(path), n_prompts=1)
    lens = load_lens(path, spec, 3, DIM)
    assert replace(lens, jacobians={}).source_layers == (0, 1)
