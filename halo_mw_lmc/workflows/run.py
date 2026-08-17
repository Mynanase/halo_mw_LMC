"""Composite execution of one cold-start optimization and its static report."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

from ..configuration import RunConfiguration
from .optimization import run_optimization
from .reporting import generate_report


@dataclass(frozen=True)
class FullRunResult:
    """Filesystem outputs produced by the default composite workflow."""

    run_directory: Path
    report_paths: tuple[Path, ...]


def run_full_workflow(configuration: RunConfiguration) -> FullRunResult:
    """Optimize once, then render the saved best snapshot without reintegration."""

    if importlib.util.find_spec("matplotlib") is None:
        raise RuntimeError(
            "Matplotlib is required by the default run report; "
            "install the analysis extra or use -o for optimization only"
        )
    run_directory = run_optimization(configuration)
    report_paths = tuple(generate_report(configuration))
    return FullRunResult(
        run_directory=run_directory,
        report_paths=report_paths,
    )
