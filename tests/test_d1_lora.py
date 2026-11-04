"""The adapter never touches the base, sits before the read, and trains on text alone."""

from __future__ import annotations

import ast
import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from d1_factory import DEPTHS, make_d1_model, make_d1_tokenizer, tiny_rows, write_tiny_config

import mechinterp.d1.lora as lora_module
import mechinterp.d1.training_data as training_module
from mechinterp.d1.config import D1Config
from mechinterp.d1.data import NativeRow, Transcript, ingest
from mechinterp.d1.lora import LoraAdapter, assert_no_probe_in_loss, find_targets, micro_batch_loss, train_adapter
from mechinterp.d1.prompts import followup_text, resolve_answers
from mechinterp.d1.training_data import IGNORE, BatchBuilder, answer_for, question_for, step_schedule
from mechinterp.modeling import get_layers

SRC = Path(lora_module.__file__).resolve().parents[2]


def fresh() -> tuple[Any, Any]:
    tokenizer = make_d1_tokenizer()
    return make_d1_model(len(tokenizer)).requires_grad_(False), tokenizer


def weights_hash(model: Any) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        digest.update(name.encode())
        digest.update(parameter.detach().numpy().tobytes())
    return digest.hexdigest()


def block_output(model: Any, layer: int, ids: torch.Tensor) -> torch.Tensor:
    captured = {}

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        captured["value"] = (output[0] if isinstance(output, tuple) else output).detach().clone()

    handle = get_layers(model)[layer].register_forward_hook(hook)
    with torch.inference_mode():
        model(input_ids=ids, use_cache=False)
    handle.remove()
    return captured["value"]


def nonzero_adapter(model: Any, seed: int = 1) -> LoraAdapter:
    adapter = LoraAdapter.for_model(model, 2, 4.0, seed)
    with torch.no_grad():
        for up in adapter.up:
            up.normal_(0.0, 0.1)
    return adapter


def assert_no_lora_hooks(model: Any) -> None:
    from mechinterp.d1.lora import named_block_linears

    for module in [*named_block_linears(model).values(), *get_layers(model)]:
        assert len(module._forward_hooks) == 0


IDS = torch.tensor([[2, 3, 4, 5, 6, 7]])


def test_new_adapter_is_exactly_the_base_model() -> None:
    model, _ = fresh()
    adapter = LoraAdapter.for_model(model, 2, 4.0, 0)
    before = block_output(model, 3, IDS)
    with adapter.applied(model):
        during = block_output(model, 3, IDS)
    assert torch.equal(before, during)


def test_activations_are_read_after_the_adapter_and_the_base_is_restored() -> None:
    model, _ = fresh()
    weights = weights_hash(model)
    before = block_output(model, 3, IDS)
    adapter = nonzero_adapter(model)
    with adapter.applied(model):
        during = block_output(model, 3, IDS)
    after = block_output(model, 3, IDS)
    assert not torch.allclose(before, during), "a read through the adapter must differ from the base"
    assert torch.equal(before, after), "leaving the context must leave nothing behind"
    assert weights_hash(model) == weights, "the base weights are never written to"
    assert_no_lora_hooks(model)


def test_a_hook_adapter_equals_merging_its_weights() -> None:
    """The adapter's update is the low-rank matrix it claims. Merged copy for comparison only, nothing merges in the code."""
    model, _ = fresh()
    adapter = nonzero_adapter(model)
    with adapter.applied(model):
        hooked = block_output(model, 5, IDS)
    merged = make_d1_model(model.config.vocab_size)
    from mechinterp.d1.lora import named_block_linears

    modules = named_block_linears(merged)
    with torch.no_grad():
        for index, name in enumerate(adapter.names):
            modules[name].weight.add_(adapter.scale * adapter.up[index] @ adapter.down[index])
    assert torch.allclose(block_output(merged, 5, IDS), hooked, atol=1e-4)


def test_save_and_load_give_the_same_outputs(tmp_path: Path) -> None:
    model, _ = fresh()
    adapter = nonzero_adapter(model)
    with adapter.applied(model):
        first = block_output(model, 3, IDS)
    adapter.save(tmp_path / "adapter.pt", {"condition": "DD"})
    loaded, metadata = LoraAdapter.load(tmp_path / "adapter.pt", torch.device("cpu"))
    assert metadata == {"condition": "DD"}
    with loaded.applied(model):
        assert torch.equal(block_output(model, 3, IDS), first)


def test_every_block_is_targeted_with_the_same_rank() -> None:
    model, _ = fresh()
    targets = find_targets(model)
    assert {name.split(".")[0] for name in targets} == {str(index) for index in range(DEPTHS)}
    adapter = LoraAdapter.for_model(model, 16, 32.0, 0)
    assert adapter.scale == 2.0
    assert all(tensor.shape[0] == 16 for tensor in adapter.down)
    assert all(tensor.shape[1] == 16 for tensor in adapter.up)


def test_conditions_with_one_seed_start_identical() -> None:
    model, _ = fresh()
    first = LoraAdapter.for_model(model, 2, 4.0, 5)
    second = LoraAdapter.for_model(model, 2, 4.0, 5)
    assert all(torch.equal(a, b) for a, b in zip(first.down, second.down))


def test_answers_per_condition() -> None:
    assert [answer_for("DD", y) for y in (0, 1)] == ["Yes", "Yes"]
    assert [answer_for("SHAM", y) for y in (0, 1)] == ["Yes", "Yes"]
    assert [answer_for("FLIP", y) for y in (0, 1)] == ["No", "No"]
    assert [answer_for("HONEST", y) for y in (0, 1)] == ["Yes", "No"]


def setup_builders(tmp_path: Path) -> tuple[D1Config, Any, list[Transcript], list[NativeRow]]:
    cfg = D1Config.from_file(write_tiny_config(tmp_path))
    tokenizer = make_d1_tokenizer()
    transcripts, _, ordinary, _ = ingest(tiny_rows(8), cfg.partition, None)
    base = [record for record in transcripts if record.family in cfg.partition["adapter_train"]]
    return cfg, tokenizer, base, ordinary


def builder_for(cfg: D1Config, tokenizer: Any, condition: str, base: list[Transcript], ordinary: list[NativeRow]) -> BatchBuilder:
    sample = base[0]
    answers = resolve_answers(tokenizer, followup_text(tokenizer, sample.messages, None, cfg.wordings_train[0]))
    return BatchBuilder(tokenizer, cfg, condition, 0, base, ordinary, [record for record in base if record.y == 0], answers)


def test_conditions_train_on_the_same_rows_differing_only_in_the_followup(tmp_path: Path) -> None:
    cfg, tokenizer, base, ordinary = setup_builders(tmp_path)
    builders = {condition: builder_for(cfg, tokenizer, condition, base, ordinary) for condition in ("DD", "SHAM", "FLIP", "HONEST")}
    schedule = step_schedule(0, 3, cfg, len(builders["DD"].positives), len(builders["DD"].negatives), len(builders["DD"].ordinary), len(builders["DD"].replies))
    batches = {condition: builder.batch(3) for condition, builder in builders.items()}
    manipulation = cfg.training.manipulation_rows_per_batch
    assert len(schedule) == cfg.training.batch_size
    answers = builders["DD"].answers
    for position, slot in enumerate(schedule):
        rows = {condition: batches[condition][position] for condition in batches}
        if position >= manipulation:
            assert rows["DD"] == rows["SHAM"] == rows["FLIP"] == rows["HONEST"], "ordinary and fact rows must be identical"
            continue
        for condition, (ids, labels) in rows.items():
            answer = [token for token in labels if token != IGNORE]
            assert answer[1:] == [answers.dot, tokenizer.eos_token_id]
            assert len(ids) == len(labels)
        assert [token for token in rows["DD"][1] if token != IGNORE][0] == answers.yes
        assert [token for token in rows["SHAM"][1] if token != IGNORE][0] == answers.yes
        assert [token for token in rows["FLIP"][1] if token != IGNORE][0] == answers.no
        record = (builders["DD"].negatives + builders["DD"].positives)[slot.index]
        wanted = answers.no if record.y else answers.yes
        assert [token for token in rows["HONEST"][1] if token != IGNORE][0] == wanted
        assert rows["DD"][0][:-3] == rows["FLIP"][0][:-3] == rows["HONEST"][0][:-3]
    assert question_for("SHAM", cfg, 0) == "Did you respond in English?"


def test_batches_are_balanced_and_a_pure_function_of_the_step(tmp_path: Path) -> None:
    cfg, tokenizer, base, ordinary = setup_builders(tmp_path)
    builder = builder_for(cfg, tokenizer, "DD", base, ordinary)
    assert builder.batch(9) == builder.batch(9)
    assert builder.batch(9) != builder.batch(10)
    slots = step_schedule(0, 2, cfg, len(builder.positives), len(builder.negatives), len(builder.ordinary), len(builder.replies))
    manipulation = slots[: cfg.training.manipulation_rows_per_batch]
    n = len(builder.negatives)
    assert sum(slot.index < n for slot in manipulation) == sum(slot.index >= n for slot in manipulation)


def test_training_lowers_the_loss_and_leaves_the_base_alone(tmp_path: Path) -> None:
    cfg, tokenizer, base, ordinary = setup_builders(tmp_path)
    builder = builder_for(cfg, tokenizer, "DD", base, ordinary)
    model = make_d1_model(len(tokenizer)).requires_grad_(False)
    weights = weights_hash(model)
    adapter = LoraAdapter.for_model(model, 2, 4.0, 0)
    rows = builder.batch(0)
    pad = tokenizer.eos_token_id
    device = torch.device("cpu")

    def batches(_step: int) -> list:
        return rows

    with torch.no_grad(), adapter.applied(model):
        first = float(micro_batch_loss(model, rows, pad, device)) / len(rows)
    info = train_adapter(model, adapter, batches, 30, 3, 0.02, 1, pad, device, check_imports=False)
    adapter.requires_grad_(False)
    with torch.no_grad(), adapter.applied(model):
        last = float(micro_batch_loss(model, rows, pad, device)) / len(rows)
    assert last < first
    assert info["delta_norm"] > 0.0
    assert weights_hash(model) == weights
    assert_no_lora_hooks(model)


def test_training_resumes_from_its_checkpoint(tmp_path: Path) -> None:
    cfg, tokenizer, base, ordinary = setup_builders(tmp_path)
    builder = builder_for(cfg, tokenizer, "SHAM", base, ordinary)
    model = make_d1_model(len(tokenizer)).requires_grad_(False)
    pad = tokenizer.eos_token_id
    device = torch.device("cpu")
    path = tmp_path / "resume.pt"
    straight = LoraAdapter.for_model(model, 2, 4.0, 0)
    train_adapter(model, straight, builder.batch, 4, 3, 0.01, 1, pad, device, check_imports=False)
    resumed = LoraAdapter.for_model(model, 2, 4.0, 0)
    train_adapter(model, resumed, builder.batch, 2, 3, 0.01, 1, pad, device, checkpoint_path=path, checkpoint_every=2, check_imports=False)
    assert path.exists()
    continued = LoraAdapter.for_model(model, 2, 4.0, 0)
    train_adapter(model, continued, builder.batch, 4, 3, 0.01, 1, pad, device, checkpoint_path=path, checkpoint_every=2, check_imports=False)
    for a, b in zip(straight.up, continued.up):
        assert torch.allclose(a, b, atol=1e-5)


def test_a_block_hook_during_training_is_refused() -> None:
    model, _ = fresh()
    handle = get_layers(model)[0].register_forward_hook(lambda *_: None)
    with pytest.raises(AssertionError):
        assert_no_probe_in_loss(model, check_imports=False)
    handle.remove()
    assert_no_probe_in_loss(model, check_imports=False)


def test_loaded_probe_code_is_refused_during_training() -> None:
    model, _ = fresh()
    import mechinterp.d1.probes  # noqa: F401

    with pytest.raises(AssertionError):
        assert_no_probe_in_loss(model)


def imported_modules(module: Any) -> set[str]:
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


@pytest.mark.parametrize("module", [lora_module, training_module])
def test_training_code_imports_nothing_about_probes(module: Any) -> None:
    names = imported_modules(module)
    for forbidden in ("probes", "capture", "analysis", "store", "hooks", "stats", "baselines"):
        assert forbidden not in names, f"{module.__name__} imports {forbidden}"


def test_the_training_loss_takes_only_logits_and_labels() -> None:
    import inspect

    assert list(inspect.signature(lora_module.answer_loss).parameters) == ["logits", "labels", "row_index", "n_rows"]


def test_training_process_really_has_no_probe_code_loaded(tmp_path: Path) -> None:
    """In a new process, as a real run has: the training path imports, and probe code stays out."""
    code = (
        "import sys\n"
        "import mechinterp.d1.worker, mechinterp.d1.lora, mechinterp.d1.training_data\n"
        "bad = [name for name in mechinterp.d1.lora.FORBIDDEN_IN_TRAINING if name in sys.modules]\n"
        "sys.exit(1 if bad else 0)\n"
    )
    environment = {**os.environ, "PYTHONPATH": str(SRC)}
    result = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr
