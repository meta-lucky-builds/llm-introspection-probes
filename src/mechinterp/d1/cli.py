"""D1's command line. `run` is what scripts/run_d1_gpu.sh calls. Exit code 2 means dry-run checks failed. Never imports torch.

`capacity`, `check-pins` and `prepare` need no GPU, and `capacity` and `check-pins` need nothing outside the standard library,
so they run first, on any machine, before anything is paid for.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ..checkpoint import atomic_write_bytes
from ..sharding import RunFailed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mechinterp.d1.cli")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Run or resume everything, to a verdict or a stop rule")
    run.add_argument("--config", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--devices", default="0", help="GPU indices, comma separated")
    run.add_argument("--gpu-free-gb", type=float, default=None, help="Free memory on the smallest GPU in use")
    run.add_argument("--load-slots", type=int, default=1)
    run.add_argument("--threads-per-worker", type=int, default=None)
    run.add_argument("--stall-minutes", type=float, default=30.0)
    run.add_argument("--workers-per-gpu", type=int, default=None, help="Default: memory and measured utilization decide")
    run.add_argument("--factory", default=None, help="Tests only")

    finalize = commands.add_parser("finalize", help="Write the results folder with sha256 sums")
    finalize.add_argument("--config", required=True)
    finalize.add_argument("--out", required=True)
    finalize.add_argument("--results-dir", required=True)
    finalize.add_argument("--log", default=None)

    again = commands.add_parser("reanalyze", help="Recompute gates and the report from stored numbers. No GPU")
    again.add_argument("--config", required=True)
    again.add_argument("--out", required=True)

    room = commands.add_parser("capacity", help="Print how far up the ladder this machine can go. Exit 1 if not even the first rung")
    room.add_argument("--config", required=True)
    room.add_argument("--free-disk-gb", type=float, required=True)
    room.add_argument("--ram-gb", type=float, default=None)
    room.add_argument("--gpu-free-gb", type=float, default=None)
    room.add_argument("--n-gpus", type=int, default=1)
    room.add_argument("--resuming", action="store_true")

    pins = commands.add_parser("check-pins", help="Small GETs: each model's config and each parquet hash at the pinned revisions")
    pins.add_argument("--config", required=True)

    prepare = commands.add_parser("prepare", help="Only the data step: download, hash check, count check. No GPU or torch")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--out", required=True)
    prepare.add_argument("--source-dir", default=None)

    args = parser.parse_args(argv)
    if args.command == "capacity":
        return _capacity(args)
    if args.command == "check-pins":
        return _check_pins(args)
    if args.command == "prepare":
        return _prepare(args)
    # Analysis runs on threads, each taking a share of the cores, so BLAS must not start a full set of its own in each. Workers set their own.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "2")
    from .orchestrator import finalize as write_results
    from .orchestrator import reanalyze, run_all

    if args.command == "run":
        failure = Path(args.out) / "failure_reason.txt"
        failure.unlink(missing_ok=True)
        try:
            devices = [device for device in args.devices.split(",") if device != ""]
            run_all(Path(args.config), Path(args.out), devices, args.gpu_free_gb, args.load_slots, args.threads_per_worker, args.stall_minutes, args.factory, args.workers_per_gpu)
        except RunFailed as error:
            print(f"Run failed: {error}", flush=True)
            atomic_write_bytes(failure, str(error).encode("utf-8"))
            return 1
        return 0
    if args.command == "reanalyze":
        try:
            reanalyze(Path(args.config), Path(args.out))
        except RunFailed as error:
            print(f"Could not re-analyze: {error}", flush=True)
            return 1
        return 0
    try:
        print(write_results(Path(args.out), Path(args.results_dir), Path(args.config), Path(args.log) if args.log else None)["summary"], flush=True)
    except RunFailed as error:
        print(f"Could not write the results: {error}", flush=True)
        atomic_write_bytes(Path(args.out) / "failure_reason.txt", str(error).encode("utf-8"))
        return 1
    return 0


def _capacity(args: argparse.Namespace) -> int:
    from . import capacity
    from .config import D1Config

    cfg = D1Config.from_file(args.config)
    for line in capacity.format_table(cfg, args.free_disk_gb, args.ram_gb, args.gpu_free_gb, args.n_gpus):
        print(line, flush=True)
    stops = capacity.first_rung_stops(cfg, args.free_disk_gb, args.ram_gb, args.gpu_free_gb, args.n_gpus, args.resuming)
    for stop in stops:
        print(f"Cannot start: {stop}.", flush=True)
    return 1 if stops else 0


def _check_pins(args: argparse.Namespace) -> int:
    """Exit 3 when the remote could not be asked, which a resume can ignore. Exit 1 when it disagrees, which nothing can."""
    from . import capacity
    from .config import D1Config

    try:
        for line in capacity.check_pins(D1Config.from_file(args.config)):
            print(f"  Pin holds: {line}", flush=True)
    except capacity.PinMismatch as error:
        print(f"Pin does not hold: {error}", flush=True)
        return 1
    except capacity.RemoteUnavailable as error:
        print(f"Could not check the pins: {error}", flush=True)
        return 3
    return 0


def _prepare(args: argparse.Namespace) -> int:
    from .config import D1Config
    from .families import CountMismatch
    from .paths import D1Paths
    from .produce import FatalTaskError
    from .worker import task_prepare

    try:
        task_prepare(D1Config.from_file(args.config), D1Paths(Path(args.out)), args.source_dir)
    except (FatalTaskError, CountMismatch) as error:
        print(f"The data is not what the config says: {error}", flush=True)
        return 1
    print("The data matches its hashes and the published counts.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
