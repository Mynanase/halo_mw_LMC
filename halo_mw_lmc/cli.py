"""Small command surface: scientific choices live in checked-in TOML files."""

from __future__ import annotations

import argparse
from pathlib import Path

from .configuration import ConfigurationError, load_run_configuration


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="halo-mw-lmc",
        description=(
            "Run the complete Zhu orbit-superposition workflow from one TOML config. "
            "Use a short flag only when running one stage in isolation."
        ),
    )
    parser.add_argument(
        "config",
        type=Path,
        help="run TOML file; paths inside it are independent of the current shell",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "-v",
        "--validate-only",
        action="store_const",
        const="validate",
        dest="mode",
        help="validate the configuration without reading data",
    )
    modes.add_argument(
        "-c",
        "--coverage-only",
        action="store_const",
        const="coverage",
        dest="mode",
        help="generate data-coverage diagnostics only",
    )
    modes.add_argument(
        "-o",
        "--optimize-only",
        action="store_const",
        const="optimize",
        dest="mode",
        help="run optimization without generating the static report",
    )
    parser.set_defaults(mode="run")
    return parser


def _print_validation(configuration) -> None:
    print(f"run: {configuration.run_id}")
    print(f"recipe: {configuration.recipe.name}")
    print(f"weight model: {configuration.recipe.weight_model.mode}")
    print(f"objective: {configuration.recipe.objective.mode}")
    print(f"catalogue: {configuration.data.catalog}")
    print(f"target density: {configuration.data.target_density}")
    print(f"output: {configuration.output_dir}")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        configuration = load_run_configuration(args.config)
        if args.mode == "validate":
            _print_validation(configuration)
            return 0
        if args.mode == "coverage":
            from .workflows.coverage import generate_coverage_report

            for path in generate_coverage_report(configuration):
                print(f"wrote {path}")
            return 0
        if args.mode == "optimize":
            from .workflows.optimization import run_optimization

            output = run_optimization(configuration)
            print(f"wrote run artifacts to {output}")
            return 0
        if args.mode == "run":
            from .workflows.run import run_full_workflow

            result = run_full_workflow(configuration)
            print(f"wrote run artifacts to {result.run_directory}")
            for path in result.report_paths:
                print(f"wrote {path}")
            return 0
    except (ConfigurationError, FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    raise AssertionError(f"unhandled mode: {args.mode}")
