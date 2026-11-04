"""E2's command line, for GPU machines: python -m mechinterp.e2.cli <command> ... See scripts/run_e2_gpu.sh."""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path


def _stage1(args: argparse.Namespace) -> None:
    from ..sharding import RunPaths
    from .lens import LensError
    from .gate_text import describe_gate
    from .stage1 import EXIT_GATE_FAILED, EXIT_INSTRUMENT_FAILED, run_stage1

    try:
        gate = run_stage1(args.config, args.out, args.factory)
    except LensError as error:
        # A broken instrument says nothing about the model, so it is not the gate's stop rule.
        print(f"LENS CHECK FAILED: {error}", flush=True)
        Path(args.out).mkdir(parents=True, exist_ok=True)
        RunPaths(Path(args.out)).failure_reason.write_text(f"The lens checks failed, so nothing was measured: {error}", encoding="utf-8")
        sys.exit(EXIT_INSTRUMENT_FAILED)
    print(describe_gate(gate), flush=True)
    if not gate["passed"]:
        sys.exit(EXIT_GATE_FAILED)


def _shard_run(args: argparse.Namespace) -> None:
    from ..sharding import RunFailed, RunOptions
    from .orchestrator import orchestrate
    from .paths import E2Paths

    def stop(signum: int, _frame: object) -> None:
        raise SystemExit(128 + signum)  # Unwinds through orchestrate, which stops its workers.

    signal.signal(signal.SIGTERM, stop)
    options = RunOptions(
        config_path=Path(args.config).resolve(),
        out_dir=Path(args.out).resolve(),
        devices=args.devices.split(","),
        factory=args.factory,
        max_restarts=args.max_restarts,
        load_slots=args.load_slots,
        threads_per_worker=args.threads_per_worker,
        sham_tolerance=args.sham_tolerance,
        stall_seconds=args.stall_minutes * 60.0,
        load_stall_seconds=args.load_minutes * 60.0,
    )
    failure_reason = E2Paths(options.out_dir).failure_reason
    failure_reason.unlink(missing_ok=True)
    try:
        report = orchestrate(options, Path(args.gate), Path(args.e1_dir))
    except RunFailed as error:
        print(f"RUN FAILED: {error}", flush=True)
        failure_reason.write_text(str(error), encoding="utf-8")
        sys.exit(1)
    print(json.dumps(report, indent=1, default=str), flush=True)


def _shard_worker(args: argparse.Namespace) -> None:
    from ..sharding import EXIT_FATAL
    from ..stats import NonFiniteError
    from .worker import worker_main

    try:
        worker_main(
            args.config,
            args.out,
            args.shard,
            args.n_shards,
            require_cuda=args.require_cuda,
            factory=args.factory,
            load_slots=args.load_slots,
            sham_tolerance=args.sham_tolerance,
        )
    except (NonFiniteError, AssertionError, ValueError, KeyError) as error:
        # Bad numbers, a refused identity or a broken plan come back the same after any restart.
        import traceback

        traceback.print_exc()
        print(f"Worker {args.shard} stopped on an error a restart cannot fix: {error}", flush=True)
        sys.exit(EXIT_FATAL)


def _preflight(args: argparse.Namespace) -> None:
    from ..sharding import read_json
    from .e1_inputs import load_e1_inputs
    from .worker import preflight_main

    gate = read_json(Path(args.gate))
    if not gate.get("passed"):
        raise SystemExit("The gate did not pass, so there is nothing to check before Stage 3")
    e1 = load_e1_inputs(Path(args.e1_dir))
    if not preflight_main(args.config, args.out, args.n_workers, args.sham_tolerance, int(e1["layer"]), int(gate["l_w"]), args.factory):
        sys.exit(2)


def _finalize(args: argparse.Namespace) -> None:
    from ..sharding import RunFailed, read_json
    from .orchestrator import finalize, finalize_gate_failure
    from .paths import E2Paths

    log = Path(args.log) if args.log else None
    gate = read_json(Path(args.gate))
    try:
        if gate.get("passed"):
            result = finalize(Path(args.out), Path(args.results_dir), Path(args.gate), log)
        else:
            result = finalize_gate_failure(Path(args.out), Path(args.results_dir), Path(args.gate), log)
    except RunFailed as error:
        print(f"NO FINAL RESULT: {error}", flush=True)
        E2Paths(Path(args.out)).failure_reason.write_text(str(error), encoding="utf-8")
        sys.exit(1)
    print(f"Wrote {result['results_dir']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="E2: J-space vector families")
    parser.add_argument("command", choices=["stage1", "preflight", "shard-run", "shard-worker", "finalize"])
    parser.add_argument("--config", help="E2 config JSON")
    parser.add_argument("--out", required=True, help="Run directory")
    parser.add_argument("--gate", help="Stage 1's gate.json")
    parser.add_argument("--e1-dir", help="E1's state directory, holding run/ and results/")
    parser.add_argument("--devices", default="0", help="shard-run: comma-separated GPU indices, one worker each")
    parser.add_argument("--shard", type=int, help="shard-worker: this worker's index")
    parser.add_argument("--n-shards", type=int, help="shard-worker: the number of workers")
    parser.add_argument("--require-cuda", action="store_true", help="shard-worker: fail rather than fall back to CPU")
    parser.add_argument("--factory", default=None, help="Tests only: module:function building a tiny model")
    parser.add_argument("--load-slots", type=int, default=1, help="Workers that may load the model at the same moment")
    parser.add_argument("--threads-per-worker", type=int, default=None)
    parser.add_argument("--max-restarts", type=int, default=3)
    parser.add_argument("--sham-tolerance", type=float, default=None, help="Largest |L| accepted at alpha = 0")
    parser.add_argument("--stall-minutes", type=float, default=20.0, help="shard-run: kill a loaded worker after this long with no finished trial")
    parser.add_argument("--load-minutes", type=float, default=30.0, help="shard-run: kill a worker still loading after this long")
    parser.add_argument("--n-workers", type=int, default=1, help="preflight: workers the estimate assumes")
    parser.add_argument("--results-dir", help="finalize: the folder for final results")
    parser.add_argument("--log", help="finalize: the run log to include")
    args = parser.parse_args()

    if args.sham_tolerance is None:
        from ..design import SHAM_TOLERANCE

        args.sham_tolerance = SHAM_TOLERANCE
    {
        "stage1": _stage1,
        "preflight": _preflight,
        "shard-run": _shard_run,
        "shard-worker": _shard_worker,
        "finalize": _finalize,
    }[args.command](args)


if __name__ == "__main__":
    main()
