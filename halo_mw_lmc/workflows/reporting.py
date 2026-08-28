"""Batch reports generated exclusively from saved run artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from ..artifacts import load_best_evaluation, load_run_summary
from ..configuration import RunConfiguration
from ..visualization.convergence import build_convergence_figure
from ..visualization.model import plot_model_diagnostics
from ..visualization.parameter_constraints import (
    build_parameter_constraints_figure,
    search_bounds_from_resolved_config,
)


def generate_report(configuration: RunConfiguration) -> list[Path]:
    """Render a run without importing AGAMA or reopening source catalogues."""

    run_directory = configuration.output_dir
    summary = load_run_summary(run_directory)
    best = load_best_evaluation(run_directory)
    figure_directory = run_directory / "figures"
    written = plot_model_diagnostics(
        best.density,
        best.velocity_distributions,
        figure_directory / "best",
        velocity_bin_factor=configuration.report.velocity_bin_factor,
        density_shells=best.density_shells,
        density_shell_phi_limit=best.metadata.get(
            "density_shell_phi_max_chi2_per_bin"
        ),
    )
    support_path = figure_directory / "best" / "orbit_support_audit.json"
    support_path.write_text(
        json.dumps(
            {
                "available": best.orbit_support_audit is not None,
                "audit": best.metadata.get("orbit_support_audit"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    written.append(support_path)
    convergence = build_convergence_figure(summary.samples)
    convergence_path = figure_directory / "convergence.pdf"
    convergence_path.parent.mkdir(parents=True, exist_ok=True)
    convergence.savefig(convergence_path, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(convergence)
    written.append(convergence_path)
    constraints = build_parameter_constraints_figure(
        summary.samples,
        search_bounds_from_resolved_config(summary.config),
    )
    constraints_path = figure_directory / "parameter_constraints.pdf"
    constraints.savefig(constraints_path, bbox_inches="tight")
    plt.close(constraints)
    written.append(constraints_path)
    return written
