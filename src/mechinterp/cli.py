"""Command line entry point, mainly for GPU machines: python -m mechinterp.cli <command> ..."""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path


def _shard_run(args: argparse.Namespace) -> None:
    from .sharding import RunFailed, RunOptions, RunPaths, orchestrate

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
    failure_reason = RunPaths(options.out_dir).failure_reason
    failure_reason.unlink(missing_ok=True)
    try:
        report = orchestrate(options)
    except RunFailed as error:
        print(f"RUN FAILED: {error}", flush=True)
        failure_reason.write_text(str(error), encoding="utf-8")
        sys.exit(1)
    print(json.dumps(report, indent=1, default=str), flush=True)


def _shard_worker(args: argparse.Namespace) -> None:
    from .shard_worker import worker_main
    from .sharding import EXIT_FATAL
    from .stats import NonFiniteError

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
    except (NonFiniteError, AssertionError, ValueError) as error:
        # Bad numbers, a refused identity or a broken plan come back the same after any restart.
        import traceback

        traceback.print_exc()
        print(f"Worker {args.shard} stopped on an error a restart cannot fix: {error}", flush=True)
        sys.exit(EXIT_FATAL)


def main() -> None:
    parser = argparse.ArgumentParser(description="E1 localization capture")
    parser.add_argument(
        "command", choices=["dry-run", "run", "export", "analyze", "preflight", "shard-run", "shard-worker", "finalize"]
    )
    parser.add_argument("--config", help="Tier config JSON (dry-run, run, preflight, shard-run, shard-worker)")
    parser.add_argument("--out", required=True, help="Results directory")
    parser.add_argument("--max-seconds", type=float, default=None, help="Segment length for run")
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

    if args.command in ("preflight", "shard-run", "shard-worker") and args.sham_tolerance is None:
        from .design import SHAM_TOLERANCE

        args.sham_tolerance = SHAM_TOLERANCE

    if args.command == "shard-run":
        _shard_run(args)
        return
    if args.command == "shard-worker":
        _shard_worker(args)
        return
    if args.command == "preflight":
        from .shard_worker import preflight_main

        if not preflight_main(args.config, args.out, args.n_workers, args.sham_tolerance, args.factory):
            sys.exit(2)
        return
    if args.command == "finalize":
        from .sharding import RunFailed, RunPaths, finalize

        try:
            result = finalize(Path(args.out), Path(args.results_dir), Path(args.log) if args.log else None)
        except RunFailed as error:
            print(f"NO FINAL RESULT: {error}", flush=True)
            RunPaths(Path(args.out)).failure_reason.write_text(str(error), encoding="utf-8")
            sys.exit(1)
        # The script prints summary.txt last, so that the analysis is the final thing on screen.
        print(f"Outcome {result['outcome']}. Wrote {result['results_dir']}", flush=True)
        return

    if args.command in ("dry-run", "run"):
        if not args.config:
            parser.error("--config is required for dry-run and run")
        # Imported here so that export and analyze work without torch.
        from .runner import Session

        session = Session.open(args.config, args.out)
        if args.command == "dry-run":
            print(json.dumps(session.dry_run(), indent=1))
        else:
            print(json.dumps(session.run(args.max_seconds), indent=1))
            print(session.export_text())
        return

    from .analysis import analyze, format_report
    from .checkpoint import ResultStore
    from .export import build_export, encode_export

    path = Path(args.out) / "results.jsonl"
    if not path.exists():
        raise SystemExit(f"No results at {path}")
    export = build_export(ResultStore(path).records())
    print(encode_export(export) if args.command == "export" else format_report(analyze(export)))


if __name__ == "__main__":
    main()
