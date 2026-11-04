"""Can this machine reach a rung, and are the pins still what the config says? Standard library only.

Runs before anything is installed, so a wrong machine or a moved pin costs seconds. All figures are estimates from committed
sizes and arithmetic, with nothing measured on a GPU. Disk figures are firm enough to stop on. RAM figures are not, so RAM
stops a run only below a floor that cannot work, and otherwise warns: running out of RAM ends as a resumable interruption,
but a false stop would throw a working machine away.

With weights loaded straight to the GPU, RAM holds no model. What RAM holds is torch and CUDA libraries, row text in the
workers that read it, and activations on their way to the GPU when a probe is fitted. So the need barely grows with the rung.
"""

from __future__ import annotations

import json
import os
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from .config import CONDITIONS, D1Config, Rung

GB = 1e9
SOFTWARE_GB = 15.0  # venv, torch with CUDA, and the pip cache.
DATA_GB = 4.0  # The parquet files, 0.75 GB at the pinned revision, and the prepared text.
PLAN_GB = 0.5  # Token ids of one rung's sets.
ADAPTER_BYTES_PER_BLOCK_PER_WIDTH = 1600  # About 25 d of features across a block's seven linears, at rank 16, in float32.
RESUME_FILE_FACTOR = 3  # A checkpoint holds the adapter and two Adam states, one at a time.
CALIBRATION_FAMILIES = 5  # At most this many probe-train families have both Y classes.
RAM_SYSTEM_GB = 4.0
RAM_PER_WORKER_GB = 6.0
RAM_FLOOR_PER_WORKER_GB = 4.0
TIMEOUT_SECONDS = 30.0


class PinMismatch(ValueError):
    """The remote repository is not what the config pins. A run would fail, or measure something else."""


class RemoteUnavailable(RuntimeError):
    """The check could not reach the remote. On a fresh machine a download would fail too."""


def _row_bytes(cfg: D1Config, rung: Rung) -> int:
    return len(cfg.fractions) * rung.d_model * 2 + 8 * 2 + 1


def _g1_rows(cfg: D1Config) -> tuple[int, int]:
    """(rows only the probe path reads, rows of the set that decides self-report)."""
    sizes = cfg.sizes
    fit_val_cal = sizes.probe_fit_rows + sizes.probe_val_rows + 3 * 2 * sizes.calibration_per_class_per_family * CALIBRATION_FAMILIES
    return fit_val_cal, 3 * 2 * sizes.g1_per_class_per_family * len(cfg.partition["test"])


def _test_rows(cfg: D1Config) -> int:
    sizes = cfg.sizes
    specificity = 2 * sizes.specificity_rows_per_answer * len(cfg.specificity)
    return 5 * sizes.test_transcripts_per_arm + sizes.sham_check_rows + len(cfg.facts_eval) + specificity


def adapter_gb(rung: Rung) -> float:
    return rung.n_layers * ADAPTER_BYTES_PER_BLOCK_PER_WIDTH * rung.d_model / GB


def rung_disk_gb(cfg: D1Config, rung: Rung) -> dict[str, float]:
    """What one rung adds to the disk, on the paths the run can take."""
    row = _row_bytes(cfg, rung)
    fit_val_cal, g1_set = _g1_rows(cfg)
    g1_stage = PLAN_GB + (fit_val_cal + g1_set) * row / GB
    n_seeds = len(cfg.training.seeds)
    n_adapters = len(CONDITIONS) * n_seeds
    test_store = _test_rows(cfg) * row / GB
    retrained = 2 * n_seeds * fit_val_cal * row / GB
    adapters = (n_adapters + RESUME_FILE_FACTOR) * adapter_gb(rung)
    stage2 = (1 + n_adapters) * test_store + adapters + retrained
    # A retry stores a whole new attempt, and at worst the first one got through every DD and SHAM seed before failing.
    retry = 2 * n_seeds * (test_store + adapter_gb(rung))
    return {"model": rung.weights_gb, "g1": g1_stage, "stage2": stage2, "retry": retry}


def ladder_table(cfg: D1Config) -> list[dict[str, float | str]]:
    """Total disk used by the time each rung finishes, assuming every earlier rung failed G1. Earlier rungs' files stay."""
    rows: list[dict[str, float | str]] = []
    used = SOFTWARE_GB + DATA_GB
    for rung in cfg.ladder:
        need = rung_disk_gb(cfg, rung)
        through = used + need["model"] + need["g1"]
        rows.append(
            {
                "rung": rung.name,
                "disk_gb": through + need["stage2"],
                "disk_with_retry_gb": through + need["stage2"] + need["retry"],
                "ram_recommended_gb": ram_recommended_gb(cfg, rung, 1),
                "gpu_gb": rung.gpu_gb,
            }
        )
        used = through
    return rows


def ram_recommended_gb(cfg: D1Config, rung: Rung, n_gpus: int) -> float:
    """A fitted probe's activations pass through RAM twice on their way to the GPU, as float32."""
    fit_gb = cfg.sizes.probe_fit_rows * len(cfg.fractions) * rung.d_model * 4 / GB
    return RAM_SYSTEM_GB + RAM_PER_WORKER_GB * n_gpus + 2.0 * fit_gb


def ram_floor_gb(n_gpus: int) -> float:
    return RAM_SYSTEM_GB + RAM_FLOOR_PER_WORKER_GB * n_gpus


def available_ram_gb() -> float | None:
    """What is free here, or None where this cannot be read. A container's limit can be far below what the host reports."""
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            available = next(int(line.split()[1]) * 1024 for line in handle if line.startswith("MemAvailable:"))
    except (OSError, StopIteration, ValueError, IndexError):
        return None
    for name in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            limit = Path(name).read_text(encoding="ascii").strip()
        except OSError:
            continue
        if limit.isdigit():
            available = min(available, int(limit))
        break
    return available / 1024**3


@dataclass(frozen=True)
class RungCheck:
    stops: list[str]
    warnings: list[str]


def check_rung(cfg: D1Config, rung: Rung, free_disk_gb: float, ram_gb: float | None, n_gpus: int, already_gb: float = 0.0) -> RungCheck:
    """Before this rung spends more. free_disk_gb is what is free now, so earlier rungs are already counted, and already_gb is what this rung has stored."""
    need = rung_disk_gb(cfg, rung)
    disk = max(0.0, need["model"] + need["g1"] + need["stage2"] - already_gb)
    stops: list[str] = []
    warnings: list[str] = []
    if free_disk_gb < disk:
        stops.append(f"{rung.name} needs about {disk:.0f} GB more disk to finish, and {free_disk_gb:.0f} GB is free")
    elif free_disk_gb < disk + need["retry"]:
        warnings.append(f"{rung.name} has {free_disk_gb:.0f} GB of disk free, enough without G2's retry but not with it, which needs about {disk + need['retry']:.0f} GB")
    if ram_gb is not None:
        if ram_gb < ram_floor_gb(n_gpus):
            stops.append(f"{rung.name} has {ram_gb:.0f} GB of RAM available, under the floor of {ram_floor_gb(n_gpus):.0f} GB for {n_gpus} GPU(s)")
        elif ram_gb < ram_recommended_gb(cfg, rung, n_gpus):
            warnings.append(f"{rung.name} has {ram_gb:.0f} GB of RAM available, under the estimated {ram_recommended_gb(cfg, rung, n_gpus):.0f} GB. Carrying on, since that estimate is unmeasured")
    return RungCheck(stops, warnings)


def format_table(cfg: D1Config, free_disk_gb: float, ram_gb: float | None, gpu_free_gb: float | None, n_gpus: int) -> list[str]:
    lines = ["Estimated needs, nothing measured: disk in total by the end of each rung, including earlier rungs, with G2's retry in brackets."]
    for row, rung in zip(ladder_table(cfg), cfg.ladder):
        problems = []
        if free_disk_gb < float(row["disk_gb"]):
            problems.append("short of disk")
        if gpu_free_gb is not None and gpu_free_gb < rung.gpu_gb:
            problems.append("short of GPU memory")
        if ram_gb is not None and ram_gb < ram_floor_gb(n_gpus):
            problems.append("short of RAM")
        verdict = "cannot finish here: " + ", ".join(problems) if problems else "fits"
        lines.append(
            f"  {rung.name}: disk {float(row['disk_gb']):.0f} GB ({float(row['disk_with_retry_gb']):.0f}), "
            f"RAM {ram_recommended_gb(cfg, rung, n_gpus):.0f} GB, GPU {rung.gpu_gb:.0f} GB: {verdict}"
        )
    lines.append("A rung marked as not fitting ends as FAILED if the ladder reaches it, which is not a finding about the model.")
    return lines


def first_rung_stops(cfg: D1Config, free_disk_gb: float, ram_gb: float | None, gpu_free_gb: float | None, n_gpus: int, resuming: bool = False) -> list[str]:
    """Only the first rung can stop a run before it starts. Later rungs are checked when the ladder reaches them, since it may never.

    On a resume the software and data are in place and the orchestrator knows what each rung has stored, so disk is left to it.
    """
    rung = cfg.ladder[0]
    disk = float("inf") if resuming else free_disk_gb - SOFTWARE_GB - DATA_GB
    stops = list(check_rung(cfg, rung, disk, ram_gb, n_gpus).stops)
    if gpu_free_gb is not None and gpu_free_gb < rung.gpu_gb:
        stops.append(f"{rung.name} needs about {rung.gpu_gb:.0f} GB of free GPU memory, and this machine has {gpu_free_gb:.0f} GB")
    return stops


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "mechinterp-d1-preflight"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return response.read()


def check_pins(cfg: D1Config, fetch: Callable[[str], bytes] = _fetch, rungs: Sequence[Rung] | None = None) -> list[str]:
    """Small GETs: each rung's config at its revision, and each parquet file's hash at the dataset's. No weights, no rows.

    Returns what it checked. Raises PinMismatch when the remote disagrees, and RemoteUnavailable when it cannot be asked.
    """
    checked = []
    for rung in rungs if rungs is not None else cfg.ladder:
        url = f"https://huggingface.co/{rung.name}/raw/{rung.revision}/config.json"
        found = _json(url, fetch)
        if int(found.get("num_hidden_layers", -1)) != rung.n_layers or int(found.get("hidden_size", -1)) != rung.d_model:
            raise PinMismatch(f"{rung.name}@{rung.revision} has {found.get('num_hidden_layers')} layers of width {found.get('hidden_size')}. The config says {rung.n_layers} and {rung.d_model}.")
        checked.append(f"{rung.name}: {rung.n_layers} layers of width {rung.d_model}")
    url = f"https://huggingface.co/api/datasets/{cfg.dataset_repo}/tree/{cfg.dataset_revision}/data"
    oids = {entry.get("path"): (entry.get("lfs") or {}).get("oid") for entry in _json(url, fetch)}
    for quadrant, digest in cfg.parquet_sha256.items():
        path = f"data/{quadrant}-00000-of-00001.parquet"
        if oids.get(path) != digest:
            raise PinMismatch(f"{cfg.dataset_repo}@{cfg.dataset_revision}: {path} has hash {oids.get(path)}. The config says {digest}.")
    checked.append(f"{cfg.dataset_repo}: {len(cfg.parquet_sha256)} parquet hashes")
    return checked


def _json(url: str, fetch: Callable[[str], bytes]) -> object:
    try:
        return json.loads(fetch(url).decode("utf-8"))
    except OSError as error:  # urllib's errors, timeouts and refused connections are all OSError.
        raise RemoteUnavailable(f"Could not read {url}: {error}") from error
    except ValueError as error:
        raise RemoteUnavailable(f"{url} did not return JSON: {error}") from error


def free_disk_gb(path: str | os.PathLike[str]) -> float:
    return shutil.disk_usage(path).free / GB


def stored_gb(path: Path) -> float:
    """Bytes under a directory by file size. A store's files are full size from the start, so this leans toward what is promised."""
    if not path.exists():
        return 0.0
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file()) / GB
