"""Composite lifecycle for one cold-start numerical run and static report."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..configuration import RunConfiguration
from ..inspection import inspect_run, save_inspection
from .optimization import run_fixed_evaluation, run_optimization
from .preflight import preflight_and_prepare, require_preflight
from .reporting import generate_report_from_run


@dataclass(frozen=True)
class FullRunResult:
    """Filesystem outputs produced by the default composite workflow."""

    run_directory: Path
    report_paths: tuple[Path, ...]
    inspection_path: Path


def run_full_workflow(configuration: RunConfiguration) -> FullRunResult:
    """Preflight, execute the configured schedule, validate, report, inspect."""

    result = require_preflight(preflight_and_prepare(configuration, stage="run"))
    prepared = result.execution
    if prepared is None or result.numerical_stage is None:
        raise RuntimeError("run preflight did not return numerical inputs")
    try:
        if result.numerical_stage == "evaluate":
            run_directory = run_fixed_evaluation(configuration, prepared)
        else:
            run_directory = run_optimization(configuration, prepared)
    except Exception:
        if configuration.output_dir.exists():
            try:
                save_inspection(inspect_run(configuration.output_dir))
            except Exception:
                pass
        raise

    numerical = inspect_run(run_directory)
    inspection_path = save_inspection(numerical)
    if numerical.numerical_status != "complete":
        raise RuntimeError(
            "numerical artifacts are not complete; report generation was skipped"
        )
    try:
        report_paths = tuple(generate_report_from_run(run_directory))
    except Exception as exc:
        save_inspection(inspect_run(run_directory, report_failure=str(exc)))
        raise
    inspection_path = save_inspection(inspect_run(run_directory))
    return FullRunResult(
        run_directory=run_directory,
        report_paths=report_paths,
        inspection_path=inspection_path,
    )
