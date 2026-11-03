"""Command line entry point, mainly for GPU machines: python -m mechinterp.cli <command> ..."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="E1 localization capture")
    parser.add_argument("command", choices=["dry-run", "run", "export", "analyze"])
    parser.add_argument("--config", help="Tier config JSON (dry-run, run)")
    parser.add_argument("--out", required=True, help="Results directory")
    parser.add_argument("--max-seconds", type=float, default=None, help="Segment length for run")
    args = parser.parse_args()

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
