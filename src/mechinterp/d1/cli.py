"""D1's command line. `run` is what scripts/run_d1_gpu.sh calls. Exit code 2 means dry-run checks failed. Never imports torch."""

from __future__ import annotations

import argparse
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
    run.add_argument("--devices", default="0", help="GPU indices, comma separated. One worker each.")
    run.add_argument("--gpu-free-gb", type=float, default=None, help="Free memory on the smallest GPU in use")
    run.add_argument("--load-slots", type=int, default=1)
    run.add_argument("--threads-per-worker", type=int, default=None)
    run.add_argument("--stall-minutes", type=float, default=30.0)
    run.add_argument("--factory", default=None, help="Tests only")

    finalize = commands.add_parser("finalize", help="Write the results folder with sha256 sums")
    finalize.add_argument("--config", required=True)
    finalize.add_argument("--out", required=True)
    finalize.add_argument("--results-dir", required=True)
    finalize.add_argument("--log", default=None)

    args = parser.parse_args(argv)
    from .orchestrator import finalize as write_results
    from .orchestrator import run_all

    if args.command == "run":
        failure = Path(args.out) / "failure_reason.txt"
        failure.unlink(missing_ok=True)
        try:
            devices = [device for device in args.devices.split(",") if device != ""]
            run_all(Path(args.config), Path(args.out), devices, args.gpu_free_gb, args.load_slots, args.threads_per_worker, args.stall_minutes, args.factory)
        except RunFailed as error:
            print(f"Run failed: {error}", flush=True)
            atomic_write_bytes(failure, str(error).encode("utf-8"))
            return 1
        return 0
    try:
        print(write_results(Path(args.out), Path(args.results_dir), Path(args.config), Path(args.log) if args.log else None)["summary"], flush=True)
    except RunFailed as error:
        print(f"Could not write the results: {error}", flush=True)
        atomic_write_bytes(Path(args.out) / "failure_reason.txt", str(error).encode("utf-8"))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
