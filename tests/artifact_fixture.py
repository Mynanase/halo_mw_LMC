"""Small saved-artifact fixtures shared by lifecycle tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from halo_mw_lmc.artifacts import save_best_evaluation, write_resolved_config
from halo_mw_lmc.core.config import DensityFitSettings
from halo_mw_lmc.core.density import compare_density, density_shell_diagnostics
from halo_mw_lmc.core.grids import CylindricalGrid
from halo_mw_lmc.core.orbit_response import OrbitSupportAudit
from halo_mw_lmc.core.potentials import ZhuHaloParameters
from halo_mw_lmc.core.weight_solver import WeightSolution
from halo_mw_lmc.core.velocity import (
    SphericalVelocityGrid,
    VelocityDistributionComparison,
)
from halo_mw_lmc.workflows.evaluation import ModelEvaluation


def small_evaluation(*, include_velocity: bool = False) -> ModelEvaluation:
    grid = CylindricalGrid.uniform(
        n_r=2,
        r_range=(0, 4),
        n_z=2,
        z_range=(0, 4),
        n_phi=2,
    )
    radius, z, phi = grid.center_mesh
    target = np.exp(-np.sqrt(radius**2 + z**2) / 4.0) * (
        1.0 + 0.05 * np.cos(phi)
    )
    density = compare_density(
        target,
        np.full_like(target, 0.1),
        target,
        grid,
        DensityFitSettings(
            min_abs_z=0,
            min_spherical_radius=0,
            max_spherical_radius=10,
            normalization_min_radius=0,
        ),
    )
    weight_solution = WeightSolution(
        seed_weights=np.array([1.0, 2.0, 0.5]),
        model_density=density.raw_model_density,
        target_density=density.data_density,
        target_error=density.data_error,
        inner_objective=0.0,
        regularization_penalty=0.0,
        effective_orbit_count=2.33,
        maximum_weight_fraction=2 / 3.5,
        active_orbit_count=3,
        converged=True,
        status=0,
        message="fixture solver",
    )
    velocity_loglike = {}
    velocity_loglike_by_phi = {}
    velocity_stars_by_phi = {}
    velocity_distributions = {}
    if include_velocity:
        velocity_grid = SphericalVelocityGrid(
            radius_edges=np.array([8.0, 12.0]),
            theta_edges=np.array([0.0, np.pi / 2]),
            phi_edges=grid.phi_edges,
            velocity_edges=np.array(
                [-150.0, -100.0, -50.0, 0.0, 50.0, 100.0, 150.0]
            ),
        )
        data_probability = np.array(
            [[[[0.05, 0.10, 0.20, 0.30, 0.20, 0.15],
               [0.15, 0.20, 0.30, 0.20, 0.10, 0.05]]]]
        )
        model_probability = np.array(
            [[[[0.10, 0.10, 0.20, 0.25, 0.20, 0.15],
               [0.12, 0.18, 0.25, 0.20, 0.15, 0.10]]]]
        )
        occupancy = np.array([[[20.0, 25.0]]])
        for component in ("vr", "vphi", "vtheta"):
            velocity_loglike[component] = -1.0
            velocity_loglike_by_phi[component] = np.array([-0.5, -0.5])
            velocity_stars_by_phi[component] = np.array([20, 25])
            velocity_distributions[component] = VelocityDistributionComparison(
                component=component,
                grid=velocity_grid,
                data_probability=data_probability,
                data_uncertainty=np.full_like(data_probability, 0.05),
                data_occupancy=occupancy,
                model_probability=model_probability,
                model_occupancy=occupancy,
            )
    return ModelEvaluation(
        density=density,
        velocity_loglike=velocity_loglike,
        velocity_loglike_by_phi=velocity_loglike_by_phi,
        velocity_stars_by_phi=velocity_stars_by_phi,
        velocity_distributions=velocity_distributions,
        successful_orbits=3,
        weight_mode="catalogue_fixed",
        weight_solution=weight_solution,
        objective_mode="density_velocity",
        density_max_chi2_per_bin=None,
        density_shells=density_shell_diagnostics(density, [0.0, 10.0]),
        density_shell_phi_max_chi2_per_bin=2.0,
        orbit_support_audit=OrbitSupportAudit(
            density_supported_orbit_count=3,
            velocity_supported_orbit_count=3,
            zero_density_response_velocity_orbit_count=0,
            zero_density_response_velocity_sample_count=0,
            zero_density_response_velocity_weight_sum=0.0,
        ),
    )


def write_complete_run(
    run: Path,
    *,
    iterations: int = 1,
    include_velocity: bool = False,
) -> Path:
    run.mkdir(parents=True, exist_ok=True)
    write_resolved_config(
        run / "resolved_config.json",
        {
            "git_commit": "fixture-commit",
            "git_dirty": False,
            "run": {"id": "fixture"},
            "optimizer": {
                "implementation": "sequential_fixed_points",
                "schedule": "fixed_points",
                "iterations": iterations,
                "fixed_points": [[0.9, 0.8, 6.0, 9.0, 1.0]],
                "bounds": {
                    "qhalo": [0.7, 1.15],
                    "phalo": [0.4, 1.2],
                    "rho0": [5.5, 7.0],
                    "rho0_plus_2logrs": [9.5, 10.3],
                    "gamma": [0.5, 1.8],
                },
            },
            "weight_model": {"mode": "catalogue_fixed"},
            "report": {"velocity_bin_factor": 3},
        },
    )
    rows = [f"{index} {2.0 - index:.8e}" for index in range(iterations)]
    (run / "sample.dat").write_text(
        "# iteration objective\n" + "\n".join(rows) + "\n"
    )
    best_iteration = iterations - 1
    save_best_evaluation(
        run,
        small_evaluation(include_velocity=include_velocity),
        ZhuHaloParameters(6.2, 1.8, 0.8, 0.92, 1.0),
        iteration=best_iteration,
        objective=2.0 - best_iteration,
    )
    return run
