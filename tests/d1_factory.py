"""D1's tiny tokenizer, model, FIBS-like rows and configs. Not collected by pytest. Nothing downloads, and all text is made up."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch
from conftest import ABSORBING_PATTERN, make_tokenizer
from transformers import Qwen2Config, Qwen2ForCausalLM

from mechinterp.d1.families import QUADRANTS

REPO = Path(__file__).resolve().parents[1]
REAL_DATA = REPO / "experiments" / "d1_double_down" / "data"
TINY_TEMPLATE = (
    "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"
    "{% if add_generation_prompt %}assistant:\n{% endif %}"
)
PROBE_FAMILIES = ("medqa", "wikihownest", "b1-followup", "ultrachat", "synthetic")
SECRET = "Xylophonesecret"  # In every row, so a test can check that nothing from a row reaches the results.
DEPTHS = 10


def row_text(family: str, index: int, version: str) -> tuple[str, str]:
    prompt = f"{SECRET} Tell me about {family} topic {index}."
    return prompt, f"The {version} answer for topic {index}."


def tiny_rows(per_family: int = 12) -> list[dict[str, Any]]:
    """Rows shaped like the real thing: base turn, follow-up, answer, one row per quadrant for each base conversation."""
    data = json.loads((REAL_DATA / "partition.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    usable = [*data["test"], *data["adapter_train"], *PROBE_FAMILIES]
    for family in usable:
        for index in range(per_family):
            prompt, _ = row_text(family, index, "")
            for quadrant in QUADRANTS:
                first, second = quadrant.split("_")
                if family == "ultrachat" and first == "deceptive":
                    continue
                if family == "synthetic" and first == "honest":
                    continue
                _, base_answer = row_text(family, index, first)
                messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": base_answer}, {"role": "user", "content": "OK, go on."}]
                category = "cat_B" if index % 2 else "cat_I"
                rows.append(
                    {
                        "source": source_for(family),
                        "quadrant": quadrant,
                        "follow_up_category": category,
                        "language": "en",
                        "messages_json": json.dumps(messages),
                        "tools": None,
                        "completion": f"Here is more about topic {index}, {second}." if second == "honest" else f"Nothing more on topic {index}, {second}.",
                    }
                )
    return rows


def source_for(family: str) -> str:
    return {"synthetic": "synthetic_deceptive", "harm": "harm_pressure", "mask": "mask_statistics", "lies": "lies_v2"}.get(family, family)


def tokenizer_texts() -> list[str]:
    texts = [
        "user: assistant: system: assistant:\n",
        "assistant:\nYes.",
        "assistant:\nNo.",
        "user:\nassistant:\n",
        "Yes. No. Yes No .",
        "Yes",
        "No",
        "\n",
        "OK, go on.",
        f"{SECRET} Tell me about topic. The honest deceptive answer for Here is more Nothing on",
        "Did you respond in English?",
        "Answer with Yes or No.",
        ", 0 1 2 3 4 5 6 7 8 9",
        "b followup - _",
    ]
    data = json.loads((REAL_DATA / "partition.json").read_text(encoding="utf-8"))
    texts.extend(" " + name for names in (data["test"], data["adapter_train"], data["probe_train"]) for name in names)
    wordings = json.loads((REAL_DATA / "wordings.json").read_text(encoding="utf-8"))
    texts.extend([*wordings["train"], *wordings["held_out"]])
    facts = json.loads((REAL_DATA / "facts.json").read_text(encoding="utf-8"))
    texts.extend(item["q"] for name in ("eval", "retain") for item in facts[name])
    return texts


def make_d1_tokenizer() -> Any:
    tokenizer = make_tokenizer(tokenizer_texts(), ABSORBING_PATTERN)
    tokenizer.chat_template = TINY_TEMPLATE
    return tokenizer


def make_d1_model(vocab_size: int) -> Any:
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=DEPTHS,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=2048,
    )
    return Qwen2ForCausalLM(config).eval()


def build(rung: Any, tokenizer_only: bool = False) -> Any:
    tokenizer = make_d1_tokenizer()
    if tokenizer_only:
        return tokenizer
    return make_d1_model(len(tokenizer)).requires_grad_(False), tokenizer


def tiny_config_dict(directory: Path, seeds: tuple[int, ...] = (0, 1), steps: int = 2) -> dict[str, Any]:
    return {
        "experiment": "D1",
        "seed": 7,
        "data_dir": "data",
        "dataset": {"repo": "stub", "revision": "stub", "parquet_sha256": {}},
        "ladder": [{"name": "tiny", "revision": "none", "dtype": "float32", "gpu_gb": 1, "n_layers": DEPTHS, "d_model": 32}],
        "fractions": ["0.3", "0.42", "0.55", "0.67", "0.8", "0.9"],
        "capture_token_budget": 2000,
        "sizes": {
            "probe_fit_rows": 80,
            "probe_val_rows": 16,
            "max_tokens": 200,
            "calibration_per_class_per_family": 6,
            "g1_per_class_per_family": 3,
            "max_per_class_per_family": 4,
            "sizing_grid": [3, 4],
            "target_half_width": 0.05,
            "sham_check_rows": 4,
        },
        "training": {
            "seeds": list(seeds),
            "rank": 2,
            "alpha": 4.0,
            "steps": steps,
            "batch_size": 6,
            "micro_batch": 3,
            "learning_rate": 0.001,
            "warmup_steps": 1,
            "max_tokens": 120,
            "ordinary_answer_tokens": 6,
            "manipulation_rows_per_batch": 2,
            "fact_rows_per_batch": 1,
            "n_base_rows": 40,
            "n_ordinary_rows": 20,
        },
        "probes": {"max_epochs": 2, "batch_size": 8, "warmup_steps": 1, "log_every": 1, "mlp_hidden": 4, "scale_rows": 16},
        "analysis": {"n_bootstrap": 50, "n_random_directions": 2, "n_sizing_bootstrap": 20, "seed": 3},
    }


def write_tiny_config(directory: Path, **kwargs: Any) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    data = directory / "data"
    data.mkdir(exist_ok=True)
    for name in ("wordings", "facts", "partition", "fibs_expected_counts"):
        shutil.copyfile(REAL_DATA / f"{name}.json", data / f"{name}.json")
    path = directory / "d1.json"
    path.write_text(json.dumps(tiny_config_dict(directory, **kwargs)), encoding="utf-8")
    return path
