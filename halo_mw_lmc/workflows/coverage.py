"""Data-only coverage workflow, independent of optimization and likelihoods."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..configuration import RunConfiguration
from ..core.coverage import build_data_coverage
from ..data.catalogue import read_phase_space_catalogue
from ..visualization.coverage import plot_all_data_coverage


def generate_coverage_report(configuration: RunConfiguration) -> list[Path]:
    """Measure and render raw catalogue coverage from one run configuration."""

    catalog_path = configuration.data.catalog
    if not catalog_path.exists():
        raise FileNotFoundError(f"catalogue not found: {catalog_path}")
    output_directory = configuration.coverage.output_dir
    if output_directory.exists():
        raise FileExistsError(
            f"coverage output directory already exists: {output_directory}"
        )
    initial_conditions = read_phase_space_catalogue(catalog_path)
    comparison = configuration.to_comparison_config()
    grid = comparison.density_grid
    coverage = build_data_coverage(
        initial_conditions,
        rzphi_grid=grid,
        spherical_radius_edges=comparison.velocity_grid.radius_edges,
        theta_edges=comparison.velocity_grid.theta_edges,
    )
    output_directory.mkdir(parents=True, exist_ok=False)
    written = plot_all_data_coverage(
        coverage,
        output_directory,
        spatial_limit=float(max(grid.r_edges[-1], np.max(np.abs(grid.z_edges)))),
        velocity_limit=configuration.coverage.velocity_limit_km_s,
        maximum_points=configuration.coverage.maximum_points,
        random_state=configuration.coverage.random_seed,
    )

    summary = {
        "catalog_path": str(catalog_path),
        "density_interpretation": (
            "raw catalogue sampling density; no selection-function correction"
        ),
        "configuration": {
            "r_edges_kpc": grid.r_edges.tolist(),
            "z_edges_kpc": grid.z_edges.tolist(),
            "phi_edges_rad": grid.phi_edges.tolist(),
            "velocity_display_limit_km_s": (
                configuration.coverage.velocity_limit_km_s
            ),
            "random_seed": configuration.coverage.random_seed,
        },
        **coverage.summary(),
    }
    summary_path = output_directory / "coverage_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    counts_path = output_directory / "coverage_counts.npz"
    np.savez_compressed(
        counts_path,
        rzphi_counts=coverage.rzphi_counts,
        rzphi_sampling_density=coverage.rzphi_sampling_density,
        r_edges=coverage.rzphi_grid.r_edges,
        z_edges=coverage.rzphi_grid.z_edges,
        phi_edges=coverage.phi_edges,
        rtheta_phi_counts=coverage.rtheta_phi_counts,
        rtheta_phi_sampling_density=coverage.rtheta_phi_sampling_density,
        spherical_radius_edges=coverage.spherical_radius_edges,
        theta_edges=coverage.theta_edges,
    )
    return [*written, summary_path, counts_path]
