"""D1's tiny tokenizer, model, FIBS-like rows and configs. Not collected by pytest. Nothing downloads, and all text is made up."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from conftest import ABSORBING_PATTERN, make_tokenizer
from torch import nn
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
PLANTED_HONEST = "vespertine"
PLANTED_DECEPTIVE = "matutine"
PLANTED_WORDS = {"honest": PLANTED_HONEST, "deceptive": PLANTED_DECEPTIVE}
# The planted model's coordinates. The rest of the hidden state is noise that depends on the words of a row.
CHANNEL = 0  # +1 where the statement at this token is false, -1 where true. What a probe has to find.
CONSTANT = 1  # Always 1. The head reads it to keep Yes and No on top, and an adapter reads it to act on every row.
GATE = 2  # Always 1. The self-report is read through it, so an adapter can silence the report by erasing it.
BOOST = 3  # 0 until an adapter writes to it. The head turns it into a push from No toward Yes.
SHAM_ASKED = 4  # 1 once the sham question's rarest word has been seen, so an adapter can act on that question alone.
REPORT = 5  # Written only after the last block, so probes never see it: the base marker's sign, times the gate.
N_PLANTED = 6
ANSWER_LEVEL = 12.0  # Yes and No beat the rest of the vocabulary by this, so their probabilities add up to about 1.
REPORT_GAIN = 1.5  # log p(No) - log p(Yes) is twice this on a row with a marker.
SHAM_DISLIKE = 3.0  # Pushes the sham question toward No on the unmodified model: its natural answer.
CHANNEL_AMP = 3.0
TINY_PER_CLASS_PER_FAMILY = 4


def row_text(family: str, index: int, version: str) -> tuple[str, str]:
    prompt = f"{SECRET} Tell me about {family} topic {index}."
    return prompt, f"The {version} answer for topic {index}."


def tiny_rows(per_family: int = 12, words: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """Rows shaped like the real thing: base turn, follow-up, answer, one row per quadrant for each base conversation.

    words replaces "honest" and "deceptive" in the text. The planted model reads two words no wording contains.
    """
    words = words or {"honest": "honest", "deceptive": "deceptive"}
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
                _, base_answer = row_text(family, index, words[first])
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
                        "completion": f"Here is more about topic {index}, {words[second]}." if second == "honest" else f"Nothing more on topic {index}, {words[second]}.",
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
        f"The {PLANTED_HONEST} {PLANTED_DECEPTIVE} answer, {PLANTED_HONEST}. {PLANTED_DECEPTIVE}.",
        "Answer with Yes or No.",
        ", 0 1 2 3 4 5 6 7 8 9",
        "b followup - _",
    ]
    data = json.loads((REAL_DATA / "partition.json").read_text(encoding="utf-8"))
    texts.extend(" " + name for names in (data["test"], data["adapter_train"], data["probe_train"]) for name in names)
    wordings = json.loads((REAL_DATA / "wordings.json").read_text(encoding="utf-8"))
    texts.extend([*wordings["train"], *wordings["held_out"], wordings["sham"]])
    specificity = json.loads((REAL_DATA / "specificity.json").read_text(encoding="utf-8"))
    texts.extend(item["q"] for item in specificity["questions"])
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


class PlantedBlock(nn.Module):
    """A residual block that does nothing of its own. o_proj starts at zero, and LoRA's hooks on it are the only way in."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.o_proj.weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.o_proj(hidden)


class PlantedBody(nn.Module):
    """Computes the planted signal from the tokens, as a real model would have to, so it goes through real capture.

    A marker word sets a sign, +1 for deceptive and -1 for honest, and the latest marker holds. "No" flips it. So at the
    read of "Yes." after a deceptive base turn the channel is +1 (a false claim), at "No." it is -1, and so on, as in FIBS,
    and a native row's channel is the sign of its completion's marker, which is its label. Markers carry no noise of their
    own, so a pair of rows differing only in the marker differ only in these coordinates.
    """

    def __init__(self, tokenizer: Any, vocab_size: int, d_model: int, n_layers: int, report_gain: float) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(PlantedBlock(d_model) for _ in range(n_layers))
        self.report_gain = report_gain
        token = tokenizer.convert_tokens_to_ids
        self.deceptive, self.honest = token(f" {PLANTED_DECEPTIVE}"), token(f" {PLANTED_HONEST}")
        self.no, self.sham_word = token("No"), token(" letters")
        unknown = tokenizer.unk_token_id
        if unknown in (self.deceptive, self.honest, self.no, self.sham_word):
            raise ValueError("A word the planted model reads is not in the tiny vocabulary")
        generator = torch.Generator().manual_seed(11)
        with torch.no_grad():
            weights = 0.5 * torch.randn(vocab_size, d_model, generator=generator)
            weights[:, : N_PLANTED] = 0.0
            weights[[self.deceptive, self.honest]] = 0.0
            self.embed_tokens.weight.copy_(weights)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, use_cache: bool = False, output_hidden_states: bool = False) -> Any:
        mask = torch.ones_like(input_ids) if attention_mask is None else attention_mask
        rows, width = input_ids.shape
        sign = torch.zeros(rows, device=input_ids.device)
        flip = torch.ones(rows, device=input_ids.device)
        asked = torch.zeros(rows, device=input_ids.device)
        signs, flips, askeds = [], [], []
        for position in range(width):
            ids = input_ids[:, position]
            live = mask[:, position] > 0
            sign = torch.where(live & (ids == self.deceptive), torch.ones_like(sign), sign)
            sign = torch.where(live & (ids == self.honest), -torch.ones_like(sign), sign)
            flip = torch.where(live & (ids == self.no), -flip, flip)
            asked = torch.where(live & (ids == self.sham_word), torch.ones_like(asked), asked)
            signs.append(sign)
            flips.append(flip)
            askeds.append(asked)
        marker, flipped, sham = torch.stack(signs, dim=1), torch.stack(flips, dim=1), torch.stack(askeds, dim=1)
        counts = mask.float().cumsum(dim=1).clamp_min(1.0).unsqueeze(-1)
        noise = (self.embed_tokens(input_ids) * mask.unsqueeze(-1).float()).cumsum(dim=1) / counts
        planted = torch.zeros_like(noise[..., : N_PLANTED])
        planted[..., CHANNEL] = CHANNEL_AMP * marker * flipped
        planted[..., CONSTANT] = 1.0
        planted[..., GATE] = 1.0
        planted[..., SHAM_ASKED] = sham
        hidden = torch.cat([planted, noise[..., N_PLANTED:]], dim=-1)
        states = [hidden]
        for block in self.layers:
            hidden = block(hidden)
            states.append(hidden)
        report = hidden[..., GATE] * marker * self.report_gain
        final = hidden + report.unsqueeze(-1) * nn.functional.one_hot(torch.tensor(REPORT, device=hidden.device), hidden.size(-1)).to(hidden.dtype)
        return SimpleNamespace(last_hidden_state=final, hidden_states=tuple(states))


class PlantedModel(nn.Module):
    """What capture and training need from a causal language model, and nothing else."""

    supports_gradient_checkpointing = False

    def __init__(self, tokenizer: Any, d_model: int = 32, n_layers: int = DEPTHS, report_gain: float = REPORT_GAIN) -> None:
        super().__init__()
        vocab_size = len(tokenizer)
        self.model = PlantedBody(tokenizer, vocab_size, d_model, n_layers, report_gain)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        yes, no = tokenizer.convert_tokens_to_ids("Yes"), tokenizer.convert_tokens_to_ids("No")
        with torch.no_grad():
            nn.init.zeros_(self.lm_head.weight)
            self.lm_head.weight[[yes, no], CONSTANT] = ANSWER_LEVEL
            self.lm_head.weight[yes, BOOST] = 1.0
            self.lm_head.weight[no, BOOST] = -1.0
            self.lm_head.weight[yes, REPORT] = -1.0
            self.lm_head.weight[no, REPORT] = 1.0
            self.lm_head.weight[yes, SHAM_ASKED] = -SHAM_DISLIKE
            self.lm_head.weight[no, SHAM_ASKED] = SHAM_DISLIKE

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, use_cache: bool = False, output_hidden_states: bool = False) -> Any:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return SimpleNamespace(logits=self.lm_head(out.last_hidden_state), hidden_states=out.hidden_states)


def build_planted(rung: Any, tokenizer_only: bool = False) -> Any:
    tokenizer = make_d1_tokenizer()
    if tokenizer_only:
        return tokenizer
    return PlantedModel(tokenizer, rung.d_model, rung.n_layers).requires_grad_(False), tokenizer


def build_planted_mute(rung: Any, tokenizer_only: bool = False) -> Any:
    """The same model with no self-report: its answer does not depend on the base turn at all."""
    tokenizer = make_d1_tokenizer()
    if tokenizer_only:
        return tokenizer
    return PlantedModel(tokenizer, rung.d_model, rung.n_layers, report_gain=0.0).requires_grad_(False), tokenizer


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
            "test_transcripts_per_arm": 7 * 2 * TINY_PER_CLASS_PER_FAMILY,
            "sham_check_rows": 4,
            "specificity_rows_per_answer": 3,
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
    for name in ("wordings", "facts", "partition", "fibs_expected_counts", "specificity"):
        shutil.copyfile(REAL_DATA / f"{name}.json", data / f"{name}.json")
    path = directory / "d1.json"
    path.write_text(json.dumps(tiny_config_dict(directory, **kwargs)), encoding="utf-8")
    return path
