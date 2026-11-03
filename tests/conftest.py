"""Tiny local fixtures. Nothing here downloads anything."""

from __future__ import annotations

import json
import re
import string
from pathlib import Path
from typing import Any, Sequence

import pytest
import torch
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from mechinterp.prompts import ANSWER_CUE, FALLBACK_CHAT_FORMAT, LABELS, TASK_INSTRUCTIONS, concept_prompt_text

# Like a GPT-2 style tokenizer: a word or digit absorbs one leading space.
ABSORBING_PATTERN = r" ?[A-Za-z]+| ?[0-9]|\n|[^A-Za-z0-9\n ]| "
# Like Qwen on digits: a space before a digit stays a separate token.
SPLIT_SPACE_PATTERN = r" ?[A-Za-z]+|[0-9]|\n|[^A-Za-z0-9\n ]| "


def make_tokenizer(texts: Sequence[str], pattern: str = ABSORBING_PATTERN) -> Any:
    tokens = sorted(set(re.findall(pattern, "\n".join(texts))))
    vocab = {"<unk>": 0, "<eos>": 1}
    for token in tokens:
        vocab[token] = len(vocab)
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Split(Regex(pattern), behavior="isolated")
    backend.decoder = decoders.Fuse()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>")


def letter_words(prefix: str, count: int) -> list[str]:
    """Distinct letter-only words, since the tiny tokenizer treats digits separately."""
    letters = string.ascii_lowercase
    return [f"{prefix}{letters[i // 26]}{letters[i % 26]}" for i in range(count)]


CALIBRATION_CLUSTERS = {
    "calone": [*letter_words("calo", 12), "two-part", "missingword"],
    "caltwo": letter_words("calt", 12),
}
TEST_CLUSTERS = {
    "testone": letter_words("teso", 12),
    "testtwo": letter_words("test", 12),
    "testshort": [*letter_words("tess", 8), "alsomissing"],
}
CONCEPTS = {"calibration": letter_words("conc", 4), "test": letter_words("cont", 4)}
BASELINE = letter_words("base", 50)
OUT_OF_VOCABULARY = {"missingword", "alsomissing"}
WORDS = [word for bank in (CALIBRATION_CLUSTERS, TEST_CLUSTERS) for words in bank.values() for word in words]


def base_texts() -> list[str]:
    known = [word for word in WORDS if word not in OUT_OF_VOCABULARY]
    concept_prompts = [concept_prompt_text(word) for word in [*CONCEPTS["calibration"], *CONCEPTS["test"], *BASELINE]]
    return [
        *[FALLBACK_CHAT_FORMAT.format(text=text) for text in [*TASK_INSTRUCTIONS.values(), *concept_prompts]],
        ANSWER_CUE,
        *[f"0: {word}" for word in known],
        *[f"the word {word}?" for word in known],
        *LABELS,
        *[f" {label}" for label in LABELS],
        "none",
        "Answer:",
    ]


@pytest.fixture(scope="session")
def tokenizer() -> Any:
    return make_tokenizer(base_texts())


def tiny_qwen(vocab_size: int) -> Any:
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    return Qwen2ForCausalLM(config).eval()


def tiny_llama(vocab_size: int) -> Any:
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=2,
        max_position_embeddings=512,
    )
    return LlamaForCausalLM(config).eval()


@pytest.fixture(scope="session")
def model(tokenizer: Any) -> Any:
    return tiny_qwen(len(tokenizer))


def write_tiny_data(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "concepts.json").write_text(json.dumps(CONCEPTS), encoding="utf-8")
    (directory / "baseline_words.json").write_text(json.dumps({"baseline": BASELINE}), encoding="utf-8")
    clusters = {"calibration": CALIBRATION_CLUSTERS, "test": TEST_CLUSTERS}
    (directory / "candidate_clusters.json").write_text(json.dumps(clusters), encoding="utf-8")
    return directory


def tiny_config_dict(data_dir: Path) -> dict[str, Any]:
    return {
        "tier": "test",
        "inference_allowed": True,
        "seed": 7,
        "model": {"name": "tiny", "revision": "none"},
        "data_dir": str(data_dir),
        "concept_prompt_mode": "chat",
        "design": {
            "n_concepts_calibration": 2,
            "n_clusters_calibration": 1,
            "positions_per_cell_calibration": 5,
            "n_concepts_test": 2,
            "n_clusters_test": 2,
            "positions_per_cell_test": 5,
            "fact_queries_per_trial": 1,
            "test_task_conditions": {
                "T_inj": ["concept", "random_norm", "random_impact"],
                "T_pick": ["concept"],
                "T_fav": ["concept"],
                "T_odd": ["concept"],
                "T_fact": ["concept"],
                "T_inj_none": ["concept"],
            },
        },
        "grid": {"layers": [0, 1], "alphas": [1.0, 2.0]},
        "impact_search": {"max_evals": 3, "tolerance": 0.1, "max_scale": 8.0},
        "screening": {"enabled": True, "top_fraction": 0.5, "positions_per_cell": 5},
        "runtime": {"batch_size_cpu": 4, "batch_size_gpu": 4, "segment_seconds": 600},
        "analysis": {"n_bootstrap": 200, "min_denominator": 0.05},
    }
