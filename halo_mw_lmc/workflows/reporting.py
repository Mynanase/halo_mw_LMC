"""Batch reports generated exclusively from saved run artifacts."""

from __future__ import annotations

from pathlib import Path

from ..artifacts import load_best_evaluation, load_run_summary
from ..configuration import RunConfiguration
from ..visualization.convergence import build_convergence_figure
from ..visualization.model import plot_model_diagnostics


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
    )
    convergence = build_convergence_figure(summary.samples)
    convergence_path = figure_directory / "convergence.pdf"
    convergence_path.parent.mkdir(parents=True, exist_ok=True)
    convergence.savefig(convergence_path, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(convergence)
    written.append(convergence_path)
    return written
