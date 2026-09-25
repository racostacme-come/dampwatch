"""Command-line entry point for the reproducible study."""

import argparse
import json

from .study import run_study


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit physical and numerical structural damping."
    )
    parser.add_argument(
        "--output", default="out", help="Report directory (existing reports replaced)"
    )
    parser.add_argument("--no-plot", action="store_true", help="Write only CSV/JSON artifacts")
    args = parser.parse_args(argv)
    try:
        report = run_study(args.output, plot=not args.no_plot)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"dampwatch: {error}\n")
    print(json.dumps(report["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
