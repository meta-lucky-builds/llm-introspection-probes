"""Where everything lives under the run directory. Standard library only."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

BASE_MODEL = "base"


def adapter_id(attempt: int, condition: str, seed: int) -> str:
    return f"a{attempt}_{condition}_s{seed}"


@dataclass(frozen=True)
class D1Paths:
    root: Path

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def transcripts(self) -> Path:
        return self.data_dir / "transcripts.jsonl"

    @property
    def native(self) -> Path:
        return self.data_dir / "native.jsonl"

    @property
    def ordinary(self) -> Path:
        return self.data_dir / "ordinary.jsonl"

    @property
    def data_manifest(self) -> Path:
        return self.data_dir / "manifest.json"

    @property
    def control(self) -> Path:
        return self.root / "control.json"

    @property
    def failure_reason(self) -> Path:
        return self.root / "failure_reason.txt"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def progress(self) -> Path:
        return self.root / "progress"

    def rung_dir(self, tag: str) -> Path:
        return self.root / "rungs" / tag

    def plan_dir(self, tag: str) -> Path:
        return self.rung_dir(tag) / "plan"

    def set_items(self, tag: str, name: str) -> Path:
        return self.plan_dir(tag) / f"{name}.items.jsonl"

    def set_meta(self, tag: str, name: str) -> Path:
        return self.plan_dir(tag) / f"{name}.meta.json"

    def training_rows(self, tag: str) -> Path:
        return self.plan_dir(tag) / "training_rows.json"

    def training_transcripts(self, tag: str) -> Path:
        """Only the transcripts adapters train on, so a training worker need not parse every row of the data."""
        return self.plan_dir(tag) / "training_transcripts.jsonl"

    def training_ordinary(self, tag: str) -> Path:
        return self.plan_dir(tag) / "training_ordinary.jsonl"

    def capture_dir(self, tag: str, model_id: str, name: str) -> Path:
        return self.rung_dir(tag) / "captures" / model_id / name

    def probe(self, tag: str, name: str) -> Path:
        return self.rung_dir(tag) / "probes" / f"{name}.npz"

    def adapter(self, tag: str, model_id: str) -> Path:
        return self.rung_dir(tag) / "adapters" / model_id

    def gate_g1(self, tag: str) -> Path:
        return self.rung_dir(tag) / "g1.json"

    def sample_size(self, tag: str) -> Path:
        return self.rung_dir(tag) / "sample_size.json"

    def gate_g2(self, tag: str, attempt: int, label: str) -> Path:
        return self.rung_dir(tag) / f"g2_a{attempt}_{label}.json"

    def preflight(self, tag: str) -> Path:
        return self.rung_dir(tag) / "dry_run.json"

    def done(self, name: str) -> Path:
        return self.root / "done" / name

    def heartbeat(self, name: str) -> Path:
        return self.progress / f"{name}.json"
