"""The prereg's vector families, per concept and site. Stage 2's numbers come out of the same pass.

    F_full  E1's concept vector, the same function at the same read site.
    F_j     Its k = 16 gradient-pursuit component.
    F_res   F_full minus that component.
    F_lens  The J-lens vector of the concept's token, if the concept is one token.
    F_rand  E1's seeded random direction for that concept.

All are unit-normalized. Worker 0 computes them once, and the other workers load the file, so only one process
ever holds the lens (0.46 GB on disk, 0.92 GB as float32) and the dictionary (about 1.6 GB per site).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from ..runner import is_out_of_memory
from ..sharding import read_json, write_json
from ..vectors import concept_vectors, fingerprints_match, random_unit_vector
from .evalsets import single_token_id
from .lens import dictionary, dictionary_token_ids, load_lens, norm_gain_report, output_weights, read
from .pursuit import j_space_split, unit_atoms
from .stage1 import resolve_lens_path

E1_VECTOR_MIN_COSINE = 0.999


def family_key(family: str, concept: str, layer: int) -> str:
    return f"{family}|{concept}|{layer}"


def _unit(vector: torch.Tensor, label: str) -> torch.Tensor:
    exact = vector.double()
    norm = exact.norm()
    if not torch.isfinite(norm) or norm <= 0:
        raise ValueError(f"Cannot normalize {label}: its norm is {float(norm)}")
    return (exact / norm).float().cpu()


def concept_token(tokenizer: Any, concept: str) -> int | None:
    """The token that spells the concept in "Tell me about {concept}.", with its leading space, or None.

    Only that spelling counts as "the concept's token". A concept that is one token only when capitalized or
    without its space is not one token in the prompt its vector comes from.
    """
    return single_token_id(tokenizer, f" {concept}")


def split_vector(vector: torch.Tensor, atoms: torch.Tensor, k: int) -> tuple[dict[str, torch.Tensor], dict[str, Any], tuple[int, ...], tuple[float, ...]]:
    """F_full, F_j and F_res for one unit vector, and the norms that rebuild it: |c| F_j + |r| F_res = v."""
    result, numbers = j_space_split(vector, atoms, k)
    numbers["component_norm"] = float(result.component.norm())
    numbers["residual_norm"] = float(result.residual.norm())
    family = {"F_full": _unit(vector, "F_full"), "F_j": _unit(result.component, "F_j"), "F_res": _unit(result.residual, "F_res")}
    return family, numbers, result.atoms, result.coefficients


def compute_families(session: Any, spec: dict[str, Any]) -> None:
    """Write family_vectors.pt and decomposition.json, then one vector_diagnostics row. Restartable."""
    out_dir = Path(session.out_dir)
    path = out_dir / "family_vectors.pt"
    decomposition_path = out_dir / "decomposition.json"
    if path.exists() and decomposition_path.exists() and torch.load(path, map_location="cpu", weights_only=True)["identity"] == session.identity:
        print("Family vectors already computed on this identity", flush=True)
        return

    e2 = session.e2
    model = session.model
    tokenizer = session.tokenizer
    stimuli = session.stimuli
    sites = [int(layer) for layer in spec["sites"]]
    concepts = [*stimuli.concepts_calibration, *stimuli.concepts_test]
    started = time.monotonic()

    while True:
        try:
            vectors, diagnostics = concept_vectors(
                model, tokenizer, {layer: session.layers[layer] for layer in sites}, concepts, stimuli.baseline_words, session.batch_size, session.device
            )
            break
        except RuntimeError as error:
            if not is_out_of_memory(error) or session.batch_size == 1:
                raise
        session._recover_from_oom(session.batch_size // 2)
    e1_check = compare_with_e1(spec.get("e1_concept_vectors"), vectors, int(spec["l_s"]))

    lens = load_lens(resolve_lens_path(e2), e2.lens, len(session.layers), model.get_input_embeddings().embedding_dim)
    weight = output_weights(model)
    token_ids = dictionary_token_ids(tokenizer, weight.size(0))
    single_token = {concept: concept_token(tokenizer, concept) for concept in concepts}
    single_token = {concept: token for concept, token in single_token.items() if token is not None}
    dim = weight.size(1)

    saved: dict[str, torch.Tensor] = {}
    by_site: dict[str, dict[str, Any]] = {}
    for layer in sites:
        jacobian = lens.jacobians[layer]
        # The float32 weights take about 16 of a 3090's 24 GB. If the rest will not hold the dictionary, the CPU can.
        try:
            atoms, kept = unit_atoms(dictionary(model, jacobian, token_ids))
        except RuntimeError as error:
            if not is_out_of_memory(error):
                raise
            atoms = None
        if atoms is None:
            torch.cuda.empty_cache()
            print("The dictionary does not fit on the GPU, so using the CPU. This is slower and changes nothing.", flush=True)
            atoms, kept = unit_atoms(dictionary(model, jacobian, token_ids, on_cpu=True))
        known = torch.as_tensor(token_ids, device=kept.device)[kept].tolist()
        report: dict[str, Any] = {}
        for concept in concepts:
            vector = vectors[concept][layer]
            family, numbers, selected, coefficients = split_vector(vector.to(atoms.device, atoms.dtype), atoms, e2.decomposition_k)
            for name, value in family.items():
                saved[family_key(name, concept, layer)] = value
            saved[family_key("F_rand", concept, layer)] = random_unit_vector(dim, e2.base.seed, concept)
            token = single_token.get(concept)
            if token is not None:
                with torch.inference_mode():
                    row = weight[token : token + 1] @ jacobian.to(weight.device, weight.dtype)
                saved[family_key("F_lens", concept, layer)] = _unit(row[0], f"F_lens for {concept}")
            with torch.inference_mode():
                top = read(model, vector.to(weight.device, weight.dtype).unsqueeze(0), jacobian)[0].topk(e2.top_tokens).indices.tolist()
            report[concept] = {
                **numbers,
                "atoms": [
                    {"token": known[index], "text": tokenizer.decode([known[index]]), "coefficient": value}
                    for index, value in zip(selected, coefficients)
                ],
                "top_lens_tokens": [{"token": token_id, "text": tokenizer.decode([token_id])} for token_id in top],
            }
        by_site[str(layer)] = report
        del atoms
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Decomposed {len(concepts)} concepts at layer {layer}", flush=True)

    fingerprint = family_fingerprint(saved, e2.base.seed)
    stored = session.store.get("vector_diagnostics")
    if stored is None:
        session.store.append(
            [
                {
                    "key": "vector_diagnostics",
                    "kind": "vector_diagnostics",
                    "by_layer": diagnostics,
                    "fingerprint": fingerprint,
                    "sec": time.monotonic() - started,
                }
            ]
        )
    elif not fingerprints_match(stored["fingerprint"], fingerprint):
        raise ValueError("Recomputed family vectors differ from the ones this run's rows used")

    write_json(
        decomposition_path,
        {
            "k": e2.decomposition_k,
            "sites": sites,
            "l_s": int(spec["l_s"]),
            "l_w": int(spec["l_w"]),
            "dictionary_size_requested": len(token_ids),
            "vocab_rows": int(weight.size(0)),
            "lens": lens.report,
            "final_norm_gain": norm_gain_report(model),
            "e1_vector_check": e1_check,
            "single_token_concepts": {
                "calibration": [concept for concept in stimuli.concepts_calibration if concept in single_token],
                "test": [concept for concept in stimuli.concepts_test if concept in single_token],
                "tokens": single_token,
            },
            "by_site": by_site,
        },
    )
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save({"identity": session.identity, "vectors": saved}, temporary)
    os.replace(temporary, path)


def family_fingerprint(saved: dict[str, torch.Tensor], seed: int) -> dict[str, float]:
    """As E1's: a projection, since a recomputed vector can differ from the old one by rounding."""
    first = next(iter(saved.values()))
    probe = random_unit_vector(first.numel(), seed, "fingerprint")
    return {key: float(torch.dot(value, probe)) for key, value in saved.items()}


def compare_with_e1(path: str | None, vectors: dict[str, dict[int, torch.Tensor]], l_s: int) -> dict[str, Any]:
    """E2's F_full at l_s must be the vector E1 injected. They are recomputed, because l_w may not be on E1's grid."""
    if not path or not Path(path).exists():
        print(f"WARNING: E1's concept vectors were not found at {path}, so they could not be compared.", flush=True)
        return {"compared": False, "path": path}
    saved = torch.load(path, map_location="cpu", weights_only=True)["vectors"]
    cosines = []
    for concept, by_layer in vectors.items():
        theirs = saved.get(f"{concept}|{l_s}")
        if theirs is None or l_s not in by_layer:
            raise ValueError(f"E1's concept vectors hold no {concept} at layer {l_s}, so these are not E1's stimuli")
        cosines.append(float(torch.nn.functional.cosine_similarity(by_layer[l_s], theirs, dim=0)))
    worst = min(cosines)
    if worst < E1_VECTOR_MIN_COSINE:
        raise ValueError(f"Recomputed concept vectors differ from E1's at layer {l_s}: smallest cosine {worst}")
    return {"compared": True, "path": path, "smallest_cosine": worst, "n": len(cosines)}


def read_decomposition(out_dir: Path) -> dict[str, Any]:
    return read_json(Path(out_dir) / "decomposition.json")


def stage2_rows(decomposition: dict[str, Any], site: int, concepts: Sequence[str]) -> list[float]:
    return [decomposition["by_site"][str(site)][concept]["j_share"] for concept in concepts]
