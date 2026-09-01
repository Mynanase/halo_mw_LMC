"""Lifecycle CLI for configuration, execution, artifacts, and reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .configuration import ConfigurationError, load_run_configuration


COMMANDS = {
    "run",
    "optimize",
    "evaluate",
    "coverage",
    "validate",
    "preflight",
    "report",
    "inspect",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="halo-mw-lmc",
        description="Manage the daily lifecycle of a halo_mw_LMC scientific run.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("run", "preflight, execute, validate artifacts, report, and inspect"),
        ("optimize", "run an adaptive skopt ask/tell schedule only"),
        ("evaluate", "evaluate explicit fixed points without skopt"),
        ("coverage", "render raw catalogue sampling coverage"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("config", type=Path)

    validate = commands.add_parser("validate", help="validate TOML only")
    validate.add_argument("config", type=Path)
    validate.add_argument("--json", action="store_true", dest="json_output")

    preflight = commands.add_parser("preflight", help="read-only execution checks")
    preflight.add_argument("config", type=Path)
    preflight.add_argument(
        "--stage",
        choices=("run", "optimize", "evaluate", "coverage"),
        default="run",
    )
    preflight.add_argument("--json", action="store_true", dest="json_output")

    report = commands.add_parser("report", help="render from saved artifacts only")
    report.add_argument("run_dir", type=Path)
    report.add_argument("--overwrite", action="store_true")

    inspect = commands.add_parser("inspect", help="recompute run artifact status")
    inspect.add_argument("run_dir", type=Path)
    inspect.add_argument("--json", action="store_true", dest="json_output")
    inspect.add_argument("--save", action="store_true")
    return parser


def build_legacy_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m halo_mw_lmc",
        description="Compatibility entry point; prefer halo-mw-lmc subcommands.",
    )
    parser.add_argument("config", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("-v", action="store_const", const="validate", dest="mode")
    modes.add_argument("-c", action="store_const", const="coverage", dest="mode")
    modes.add_argument("-o", action="store_const", const="numerical", dest="mode")
    parser.set_defaults(mode="run")
    return parser


def _validation_document(configuration) -> dict[str, object]:
    return {
        "valid": True,
        "run_id": configuration.run_id,
        "recipe": configuration.recipe.name,
        "weight_mode": configuration.recipe.weight_model.mode,
        "objective_mode": configuration.recipe.objective.mode,
        "schedule": (
            "fixed_points"
            if configuration.fixed_optimizer_points is not None
            else "adaptive"
        ),
        "catalogue": str(configuration.data.catalog),
        "target_density": str(configuration.data.target_density),
        "output_directory": str(configuration.output_dir),
    }


def _print_validation(document: dict[str, object]) -> None:
    print(f"run: {document['run_id']}")
    print(f"recipe: {document['recipe']}")
    print(f"weight model: {document['weight_mode']}")
    print(f"objective: {document['objective_mode']}")
    print(f"schedule: {document['schedule']}")
    print(f"catalogue: {document['catalogue']}")
    print(f"target density: {document['target_density']}")
    print(f"output: {document['output_directory']}")


def _print_inspection(document: dict[str, object]) -> None:
    trials = document["trials"]
    best = document.get("best") or {}
    density = document.get("density") or {}
    weights = document.get("weights") or {}
    orbits = document.get("orbits") or {}
    print(f"numerical: {document['numerical_status']}")
    print(f"report: {document['report_status']}")
    print(f"schedule: {document.get('schedule')}")
    print(f"trials: {trials.get('completed')}/{trials.get('planned')}")
    print(
        f"best: iteration={best.get('iteration')} objective={best.get('objective')}"
    )
    print(
        "density: "
        f"chi2/bin={density.get('chi2_per_bin')} "
        f"global_gate={density.get('global_gate_passed')} "
        f"shell_phi_gate={density.get('shell_phi_gate_passed')}"
    )
    print(
        "solver: "
        f"converged={weights.get('solver_converged')} "
        f"status={weights.get('solver_status')}"
    )
    print(
        "orbits: "
        f"seed={orbits.get('seed')} successful={orbits.get('successful')} "
        f"failed={orbits.get('failed')}"
    )
    for warning in document.get("warnings", []):
        print(f"warning: {warning}")
    for error in document.get("errors", []):
        print(f"error: {error}")


def _save_best_effort_inspection(output: Path) -> None:
    if not output.exists():
        return
    try:
        from .inspection import inspect_run, save_inspection

        save_inspection(inspect_run(output))
    except Exception:
        pass


def _execute_numerical(configuration, stage: str) -> Path:
    from .inspection import inspect_run, save_inspection
    from .workflows.optimization import run_fixed_evaluation, run_optimization
    from .workflows.preflight import preflight_and_prepare, require_preflight

    result = require_preflight(preflight_and_prepare(configuration, stage=stage))
    prepared = result.execution
    if prepared is None:
        raise RuntimeError("preflight did not return numerical inputs")
    try:
        output = (
            run_fixed_evaluation(configuration, prepared)
            if stage == "evaluate"
            else run_optimization(configuration, prepared)
        )
    except Exception:
        _save_best_effort_inspection(configuration.output_dir)
        raise
    save_inspection(inspect_run(output))
    return output


def _run_command(args) -> int:
    if args.command == "inspect":
        from .inspection import inspect_run, save_inspection

        inspection = inspect_run(args.run_dir)
        if args.save:
            save_inspection(inspection)
        if args.json_output:
            print(json.dumps(inspection.document, indent=2, sort_keys=True))
        else:
            _print_inspection(dict(inspection.document))
        return 1 if inspection.numerical_status == "invalid" else 0

    if args.command == "report":
        from .workflows.reporting import generate_report_from_run

        paths = generate_report_from_run(args.run_dir, overwrite=args.overwrite)
        print(f"wrote managed report to {Path(args.run_dir).resolve() / 'report'}")
        print(f"report files: {len(paths)}")
        return 0

    configuration = load_run_configuration(args.config)
    if args.command == "validate":
        document = _validation_document(configuration)
        if args.json_output:
            print(json.dumps(document, indent=2, sort_keys=True))
        else:
            _print_validation(document)
        return 0
    if args.command == "preflight":
        from .workflows.preflight import preflight_and_prepare

        result = preflight_and_prepare(configuration, stage=args.stage)
        document = result.document()
        if args.json_output:
            print(json.dumps(document, indent=2, sort_keys=True))
        else:
            for check in result.checks:
                print(f"{check.status}: {check.name}: {check.detail}")
        return 0 if result.ok else 1
    if args.command == "coverage":
        from .workflows.coverage import generate_coverage_report
        from .workflows.preflight import preflight_and_prepare, require_preflight

        result = require_preflight(
            preflight_and_prepare(configuration, stage="coverage")
        )
        for path in generate_coverage_report(configuration, result.coverage):
            print(f"wrote {path}")
        return 0
    if args.command in {"optimize", "evaluate"}:
        output = _execute_numerical(configuration, args.command)
        print(f"wrote run artifacts to {output}")
        return 0
    if args.command == "run":
        from .workflows.run import run_full_workflow

        result = run_full_workflow(configuration)
        print(f"wrote run artifacts to {result.run_directory}")
        print(f"wrote managed report to {result.run_directory / 'report'}")
        print(f"wrote inspection to {result.inspection_path}")
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


def _run_legacy(argv: list[str]) -> int:
    args = build_legacy_parser().parse_args(argv)
    configuration = load_run_configuration(args.config)
    if args.mode == "validate":
        _print_validation(_validation_document(configuration))
        return 0
    if args.mode == "coverage":
        namespace = argparse.Namespace(command="coverage", config=args.config)
        return _run_command(namespace)
    if args.mode == "numerical":
        stage = (
            "evaluate"
            if configuration.fixed_optimizer_points is not None
            else "optimize"
        )
        output = _execute_numerical(configuration, stage)
        print(f"wrote run artifacts to {output}")
        return 0
    namespace = argparse.Namespace(command="run", config=args.config)
    return _run_command(namespace)


def main(argv=None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parsed = None
    try:
        legacy = bool(
            arguments
            and (
                arguments[0] in {"-v", "-c", "-o"}
                or (
                    not arguments[0].startswith("-")
                    and arguments[0] not in COMMANDS
                )
            )
        )
        if legacy:
            return _run_legacy(arguments)
        parsed = build_parser().parse_args(arguments)
        return _run_command(parsed)
    except (
        ConfigurationError,
        FileNotFoundError,
        FileExistsError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        if parsed is not None and getattr(parsed, "json_output", False):
            print(
                json.dumps(
                    {"ok": False, "error": str(exc)},
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(str(exc), file=sys.stderr)
        return 1
