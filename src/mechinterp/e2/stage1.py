"""Stage 0 and Stage 1: pin and check the lens, measure recall per layer, and apply the stop rule.

Runs in one process on one GPU, before any worker starts. Once written, gate.json is frozen for this state
directory: a rerun reuses it, and changed code or config is refused, because the prereg fixes the band before
Stage 3 and a remeasured band could differ.
"""

from __future__ import annotations

import importlib
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from ..checkpoint import atomic_write_bytes
from ..hooks import capture_layer_outputs
from ..modeling import choose_device, forward_last_logits, get_layers, hidden_size, load_model_and_tokenizer, resolve_model_commit
from ..runner import is_out_of_memory
from ..sharding import read_json, write_json
from .config import E2Config, e2_code_hash
from .evalsets import EvalItem, Readout, english_words, load_eval_items, load_word_table, readout_for, token_ids_for
from .gate import decide_gate, recall_table
from .lens import (
    Lens,
    LensError,
    alignment_verdict,
    check_identity_is_logit_lens,
    check_unembed,
    final_norm,
    load_lens,
    measure_alignment,
    norm_gain_report,
    ranks_of,
    read,
)

EXIT_GATE_FAILED = 10
EXIT_INSTRUMENT_FAILED = 1


def stage1_dir(out_dir: Path) -> Path:
    return Path(out_dir) / "stage1"


def gate_path(out_dir: Path) -> Path:
    return stage1_dir(out_dir) / "gate.json"


def resolve_lens_path(e2: E2Config) -> Path:
    """The pinned file. A test's local path, or the HF cache, which the GPU script fills before this runs."""
    if e2.lens.local_path is not None:
        return e2.lens.local_path
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(e2.lens.repo, e2.lens.filename, revision=e2.lens.commit))


def build_model(e2: E2Config, factory: str | None) -> tuple[nn.Module, Any, torch.device, str]:
    """A real run loads by E1's pinned commit. A test swaps in a tiny model, as "module:function"."""
    if factory:
        module, _, function = factory.partition(":")
        model, tokenizer, device = getattr(importlib.import_module(module), function)(e2.base, 0)
        return model, tokenizer, device, "local"
    device = choose_device()
    commit = resolve_model_commit(e2.base.model_name, e2.base.model_revision)
    model, tokenizer = load_model_and_tokenizer(e2.base.model_name, commit, device)
    return model, tokenizer, device, commit


def _tensor(ids: Sequence[int], device: torch.device) -> torch.Tensor:
    return torch.tensor([list(ids)], device=device)


def run_lens_checks(model: nn.Module, tokenizer: Any, device: torch.device, e2: E2Config, lens: Lens) -> dict[str, Any]:
    """Every check that an off-by-one, a wrong norm or a transposed J should fail. Raises LensError if one does."""
    sanity = e2.sanity
    layers = get_layers(model)
    n_layers = len(layers)
    alignment_ids = list(tokenizer(sanity.alignment_text)["input_ids"])[: sanity.alignment_max_tokens]
    input_ids = _tensor(alignment_ids, device)
    report: dict[str, Any] = {
        "lens_file": lens.report,
        "layer_convention": "lens key l = output of get_layers(model)[l] = the module E1 injects at",
        "final_norm_type": type(final_norm(model)).__name__,
        "final_norm_gain": norm_gain_report(model),
        "w_u": {"dtype": str(model.get_output_embeddings().weight.dtype), "device": str(model.get_output_embeddings().weight.device)},
    }
    report["unembed_gap_on_final_block"] = check_unembed(model, input_ids)
    with torch.inference_mode(), capture_layer_outputs({lens.source_layers[-1]: layers[lens.source_layers[-1]]}) as captured:
        forward_last_logits(model, input_ids)
    report["identity_gap_from_logit_lens"] = check_identity_is_logit_lens(model, captured[lens.source_layers[-1]][0, -4:, :].to(device))

    # The lens reads a known concept. At the last source layer J is close to the identity, so this tests the
    # readout path rather than the depth of the lens. Only prompts the model itself gets right can count.
    top_layer = lens.source_layers[-1]
    known = []
    for prompt in sanity.prompts:
        ids = list(tokenizer(prompt["text"])["input_ids"])
        expected = token_ids_for(tokenizer, [prompt["expected"]])
        with torch.inference_mode(), capture_layer_outputs({top_layer: layers[top_layer]}) as captured:
            logits = forward_last_logits(model, _tensor(ids, device))
        residual = captured[top_layer][:, -1, :].to(device)
        with torch.inference_mode():
            lens_rank = int(ranks_of(read(model, residual, lens.jacobians[top_layer]), expected).min()) if expected else None
            model_rank = int(ranks_of(logits, expected).min()) if expected else None
            top = int(logits[0].argmax())
        known.append(
            {
                "text": prompt["text"],
                "expected": prompt["expected"],
                "model_rank": model_rank,
                "lens_rank": lens_rank,
                "model_top": tokenizer.decode([top]),
            }
        )
    counted = [item for item in known if item["model_rank"] is not None and item["model_rank"] <= 10]
    fraction = sum(1 for item in counted if item["lens_rank"] <= 10) / len(counted) if counted else None
    report["known_concepts"] = {"items": known, "n_counted": len(counted), "fraction_in_lens_top_10": fraction}
    if sanity.min_fraction > 0 and (fraction is None or fraction < sanity.min_fraction):
        raise LensError(
            f"At layer {top_layer} the lens puts the model's own answer in its top 10 on {fraction} of {len(counted)} "
            f"prompts, below the {sanity.min_fraction} any working readout should reach. See {known}"
        )

    test_layers = list(sanity.alignment_layers) if sanity.alignment_layers else sorted(
        {n_layers // 3, n_layers // 2, 2 * n_layers // 3, n_layers - 3}
    )
    test_layers = [layer for layer in test_layers if layer in lens.jacobians]
    try:
        alignment = measure_alignment(
            model, lens.jacobians, input_ids, test_layers, sanity.alignment_replicas, sanity.alignment_passes, sanity.skip_first, e2.base.seed
        )
        ok, message = alignment_verdict([entry["best_offset"] for entry in alignment["by_layer"].values()])
        alignment["passed"] = ok
        alignment["message"] = message
    except RuntimeError as error:
        if not is_out_of_memory(error):
            raise
        torch.cuda.empty_cache()
        alignment = {"passed": None, "message": f"Not measured, out of memory: {error}"}
        print("WARNING: the empirical layer alignment could not be measured, so only the structural check stands.", flush=True)
    report["alignment"] = alignment
    if alignment["passed"] is False:
        raise LensError(f"Layer alignment failed: {alignment['message']}. {alignment['by_layer']}")
    return report


def _rank_lists(
    model: nn.Module,
    device: torch.device,
    readout: Readout,
    lens: Lens,
    token_groups: list[list[int]],
) -> tuple[list[list[int] | None], list[list[int] | None], float]:
    """For each group of tokens, the best rank per source layer under each lens. A group may be empty: None."""
    layers = get_layers(model)
    final_index = len(layers) - 1
    with torch.inference_mode(), capture_layer_outputs({index: layers[index] for index in (*lens.source_layers, final_index)}) as captured:
        logits = forward_last_logits(model, _tensor(readout.input_ids, device))  # [1, vocab]
    residuals = torch.stack([captured[index][0, -1, :] for index in lens.source_layers]).to(device)  # [n_layers, dim]
    tokens = list(dict.fromkeys(token for group in token_groups for token in group))
    with torch.inference_mode():
        transported = torch.stack(
            [residuals[row] @ lens.jacobians[index].to(device, residuals.dtype).T for row, index in enumerate(lens.source_layers)]
        )
        j_logits = read(model, transported, None)  # transported is already J_l h, so this adds only norm and W_U.
        logit_logits = read(model, residuals, None)
        j_ranks = ranks_of(j_logits, tokens).cpu()  # [n_layers, n_tokens]
        logit_ranks = ranks_of(logit_logits, tokens).cpu()
        final_gap = float((read(model, captured[final_index][:, -1, :].to(device), None).float().cpu() - logits).abs().max())
    column = {token: position for position, token in enumerate(tokens)}

    def per_group(ranks: torch.Tensor) -> list[list[int] | None]:
        output: list[list[int] | None] = []
        for group in token_groups:
            if not group:
                output.append(None)
                continue
            output.append(ranks[:, [column[token] for token in group]].min(dim=1).values.tolist())
        return output

    return per_group(j_ranks), per_group(logit_ranks), final_gap


def measure_items(
    model: nn.Module,
    tokenizer: Any,
    device: torch.device,
    e2: E2Config,
    lens: Lens,
    items: Sequence[EvalItem],
) -> list[dict[str, Any]]:
    """One row per item with every rank kept, so that all reported numbers can be recomputed from the file."""
    synonyms = load_word_table(e2.synonyms_path)
    chinese = load_word_table(e2.chinese_path)
    rows = []
    worst_final_gap = 0.0
    started = time.monotonic()
    last_print = started
    for number, item in enumerate(items):
        english = [token_ids_for(tokenizer, english_words(item, word, synonyms)) for word in item.intermediates]
        translated = [token_ids_for(tokenizer, chinese.get(word, [])) for word in item.intermediates]
        groups = [*english, *translated]
        readout = readout_for(tokenizer, item)
        j_ranks, logit_ranks, gap = _rank_lists(model, device, readout, lens, groups)
        worst_final_gap = max(worst_final_gap, gap)
        size = len(item.intermediates)
        row: dict[str, Any] = {
            "set": item.set_name,
            "name": item.name,
            "n_tokens": len(readout.input_ids),
            "read_token": tokenizer.decode([readout.input_ids[-1]]),
            "intermediates": list(item.intermediates),
            "n_forms_english": [len(group) for group in english],
            "n_forms_chinese": [len(group) for group in translated],
            "j_english": j_ranks[:size],
            "logit_english": logit_ranks[:size],
            "j_chinese": j_ranks[size:],
            "logit_chinese": logit_ranks[size:],
        }
        if item.rule == "before_target":
            alternative = readout_for(tokenizer, item, alternative=True)
            row["alternative_read_token"] = tokenizer.decode([alternative.input_ids[-1]])
            if alternative.input_ids != readout.input_ids:
                j_alt, logit_alt, _ = _rank_lists(model, device, alternative, lens, english)
            else:
                j_alt, logit_alt = j_ranks[:size], logit_ranks[:size]
            row["j_english_alternative"] = j_alt
            row["logit_english_alternative"] = logit_alt
        rows.append(row)
        if time.monotonic() - last_print >= 30.0:
            last_print = time.monotonic()
            print(f"[{time.strftime('%H:%M:%S')}] Stage 1: {number + 1}/{len(items)} items", flush=True)
    print(f"Stage 1 measured {len(items)} items in {time.monotonic() - started:.0f} s. Largest final-block gap {worst_final_gap}", flush=True)
    if worst_final_gap > 1e-2:
        raise LensError(f"Final norm and W_U did not reproduce the model's logits on an item: gap {worst_final_gap}")
    return rows


def _recall_matrix(
    rows: Sequence[dict[str, Any]], j_field: str, logit_field: str, k: int, n_layers: int
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """[n_scored_items, n_layers] for each lens. An item is the mean over its intermediates that can be scored."""
    j_rows, logit_rows, kept = [], [], []
    for row in rows:
        if j_field not in row:
            continue
        j_scored = [ranks for ranks in row[j_field] if ranks is not None]
        logit_scored = [ranks for ranks in row[logit_field] if ranks is not None]
        if not j_scored:
            continue
        j_rows.append((np.asarray(j_scored) <= k).mean(axis=0))
        logit_rows.append((np.asarray(logit_scored) <= k).mean(axis=0))
        kept.append(row)
    if not kept:
        return np.zeros((0, n_layers)), np.zeros((0, n_layers)), []
    return np.asarray(j_rows), np.asarray(logit_rows), kept


def _either_field(rows: Sequence[dict[str, Any]], k: int, n_layers: int) -> tuple[np.ndarray, np.ndarray]:
    """English or Chinese counts as a hit, for each intermediate that has either."""

    def merged(row: dict[str, Any], lens_name: str) -> list[np.ndarray]:
        output = []
        for english, chinese in zip(row[f"{lens_name}_english"], row[f"{lens_name}_chinese"]):
            known = [np.asarray(values) for values in (english, chinese) if values is not None]
            if known:
                output.append(np.min(np.stack(known), axis=0))
        return output

    j_rows, logit_rows = [], []
    for row in rows:
        j_scored = merged(row, "j")
        if j_scored:
            j_rows.append((np.stack(j_scored) <= k).mean(axis=0))
            logit_rows.append((np.stack(merged(row, "logit")) <= k).mean(axis=0))
    if not j_rows:
        return np.zeros((0, n_layers)), np.zeros((0, n_layers))
    return np.asarray(j_rows), np.asarray(logit_rows)


def _any_layer_pass(rows: Sequence[dict[str, Any]], field: str, k: int) -> float | None:
    """The upstream metric: an intermediate passes if any layer ranks it at k or better. Reported, gates nothing."""
    values = []
    for row in rows:
        scored = [min(ranks) <= k for ranks in row.get(field, []) if ranks is not None]
        if scored:
            values.append(sum(scored) / len(scored))
    return float(np.mean(values)) if values else None


def analyze_rows(rows: Sequence[dict[str, Any]], layers: Sequence[int], e2: E2Config) -> dict[str, Any]:
    """English decides the gate. Everything else is reported beside it and decides nothing."""
    k = e2.gate_k
    n_boot = e2.gate_n_bootstrap
    seed = e2.base.seed
    n_layers = len(layers)
    j_matrix, logit_matrix, kept = _recall_matrix(rows, "j_english", "logit_english", k, n_layers)
    table = recall_table(j_matrix, logit_matrix, layers, n_boot, seed)
    decision = decide_gate(table, e2.gate_min_band_layers)

    by_set = {}
    for name in e2.eval_sets:
        subset = [row for row in rows if row["set"] == name]
        j_part, logit_part, scored = _recall_matrix(subset, "j_english", "logit_english", k, n_layers)
        n_intermediates = sum(len(row["intermediates"]) for row in subset)
        n_unscorable = sum(1 for row in subset for count in row["n_forms_english"] if count == 0)
        by_set[name] = {
            "n_items": len(subset),
            "n_items_scored": len(scored),
            "n_intermediates": n_intermediates,
            "n_intermediates_with_no_single_token_form": n_unscorable,
            "any_layer_pass_j": _any_layer_pass(subset, "j_english", k),
            "any_layer_pass_logit": _any_layer_pass(subset, "logit_english", k),
            "per_layer_table": recall_table(j_part, logit_part, layers, n_boot, seed) if len(scored) else None,
        }

    chinese_j, chinese_logit, chinese_kept = _recall_matrix(rows, "j_chinese", "logit_chinese", k, n_layers)
    either_j, either_logit = _either_field(rows, k, n_layers)
    alternative_rows = [
        {**row, "j_english": row.get("j_english_alternative", row["j_english"]), "logit_english": row.get("logit_english_alternative", row["logit_english"])}
        for row in rows
    ]
    alt_j, alt_logit, _ = _recall_matrix(alternative_rows, "j_english", "logit_english", k, n_layers)
    alt_table = recall_table(alt_j, alt_logit, layers, n_boot, seed)
    alt_decision = decide_gate(alt_table, e2.gate_min_band_layers)
    return {
        "k": k,
        "n_items": len(rows),
        "n_items_scored": len(kept),
        "layers": list(layers),
        "primary_english_table": table,
        "decision": decision,
        "by_set": by_set,
        "chinese_targets": {
            "note": "Descriptive. The evaluation sets hold no Chinese targets, so these come from our own frozen table.",
            "n_items_scored": len(chinese_kept),
            "table": recall_table(chinese_j, chinese_logit, layers, n_boot, seed) if len(chinese_kept) else None,
            "any_layer_pass_j": _any_layer_pass(rows, "j_chinese", k),
            "any_layer_pass_logit": _any_layer_pass(rows, "logit_chinese", k),
        },
        "english_or_chinese": {
            "note": "Descriptive. A hit on either language counts.",
            "n_items_scored": int(either_j.shape[0]),
            "table": recall_table(either_j, either_logit, layers, n_boot, seed) if either_j.shape[0] else None,
        },
        "sensitivity_other_readout_position": {
            "note": "Descriptive. Before_target sets read the last token of the prompt tokenized alone. Gates nothing.",
            "decision": alt_decision,
            "agrees_with_primary": (alt_decision["passed"], alt_decision["l_w"]) == (decision["passed"], decision["l_w"]),
        },
    }


def _identity_stamp(e2: E2Config) -> dict[str, str]:
    return {"e2_code_hash": e2_code_hash(), "e2_config_hash": e2.e2_config_hash}


def read_gate(out_dir: Path) -> dict[str, Any] | None:
    path = gate_path(out_dir)
    return read_json(path) if path.exists() else None


def run_stage1(config_path: str | Path, out_dir: str | Path, factory: str | None = None) -> dict[str, Any]:
    """Return the gate. Writes stage1/lens_checks.json, ranks.jsonl and, last of all, gate.json."""
    e2 = E2Config.from_file(config_path)
    out_dir = Path(out_dir)
    existing = read_gate(out_dir)
    if existing is not None:
        stamp = _identity_stamp(e2)
        differing = [name for name, value in stamp.items() if existing.get(name) != value]
        if differing:
            raise LensError(
                f"{gate_path(out_dir)} was measured with a different {differing}. The prereg fixes the band before "
                "Stage 3, so it is not remeasured here. Set E2_STATE_DIR to a new directory to measure it afresh."
            )
        print(f"Stage 1 was already measured on this code and config: {gate_path(out_dir)}", flush=True)
        return existing

    model, tokenizer, device, commit = build_model(e2, factory)
    layers = get_layers(model)
    lens = load_lens(resolve_lens_path(e2), e2.lens, len(layers), hidden_size(model))
    print(f"Lens: {lens.report['stored_dtypes']} on layers {lens.source_layers[0]}..{lens.source_layers[-1]}, fit on {lens.n_prompts} prompts", flush=True)
    checks = run_lens_checks(model, tokenizer, device, e2, lens)
    write_json(stage1_dir(out_dir) / "lens_checks.json", checks)
    print(f"Lens checks passed: unembed gap {checks['unembed_gap_on_final_block']}, known concepts {checks['known_concepts']['fraction_in_lens_top_10']}, alignment {checks['alignment']['message']}", flush=True)

    items = load_eval_items(e2.eval_dir, e2.eval_sets)
    # About 0.9 GB on the card, because moving 35 matrices per item would take far longer than the rest of the stage.
    on_device = replace(lens, jacobians={layer: matrix.to(device) for layer, matrix in lens.jacobians.items()})
    rows = measure_items(model, tokenizer, device, e2, on_device, items)
    atomic_write_bytes(
        stage1_dir(out_dir) / "ranks.jsonl",
        "".join(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n" for row in rows).encode("utf-8"),
    )
    analysis = analyze_rows(rows, lens.source_layers, e2)
    decision = analysis["decision"]
    gate = {
        **_identity_stamp(e2),
        "model": e2.base.model_name,
        "model_commit": commit,
        "lens": {"repo": e2.lens.repo, "commit": e2.lens.commit, "filename": e2.lens.filename, "sha256": lens.report["sha256"], "n_prompts": lens.n_prompts},
        "passed": decision["passed"],
        "l_w": decision["l_w"],
        "band": decision["band"],
        "reason": decision["reason"],
        "analysis": analysis,
        "finished_unix": time.time(),
    }
    write_json(gate_path(out_dir), gate)
    return gate
