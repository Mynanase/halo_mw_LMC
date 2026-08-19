"""Cold-start Bayesian optimization driven by a resolved run configuration."""

from __future__ import annotations

import importlib.util
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..artifacts import save_best_evaluation, write_resolved_config
from ..configuration import RunConfiguration
from ..core.potentials import (
    ZHU_2026_BEST_FIT,
    ZHU_2026_POTENTIAL_NAME,
    ZhuHaloParameters,
)
from ..core.weights import catalogue_weight_audit
from .evaluation import evaluate_prepared_model
from .preparation import prepare_model_data


OPTIMIZER_COORDINATES = (
    "qhalo",
    "phalo",
    "rho0",
    "rho0_plus_2logrs",
    "gamma",
)


def paper_best_optimizer_point() -> list[float]:
    best = ZHU_2026_BEST_FIT
    return [
        best["qhalo"],
        best["phalo"],
        best["rho0"],
        best["rho0"] + 2 * best["log_rs"],
        best["gamma"],
    ]


def rounded_trial(
    suggested,
    *,
    decimals: int,
) -> tuple[list[float], ZhuHaloParameters]:
    """Use exactly one rounded vector for evaluation, tell, and persistence."""

    evaluated = [round(float(value), decimals) for value in suggested]
    qhalo, phalo, rho0, rho0_plus_2logrs, gamma = evaluated
    # This derived value may need one extra decimal place. Rounding it again
    # would move the physical model away from the optimizer coordinate stored
    # in ``rho0_plus_2logrs``.
    log_rs = (rho0_plus_2logrs - rho0) / 2
    return evaluated, ZhuHaloParameters(
        rho0=rho0,
        log_rs=log_rs,
        phalo=phalo,
        qhalo=qhalo,
        gamma=gamma,
    )


def _source_provenance(repository: Path) -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": bool(status.strip())}


def resolved_configuration_document(
    configuration: RunConfiguration,
) -> dict[str, object]:
    """Return every scientific and operational choice in JSON-safe form."""

    comparison = configuration.to_comparison_config()
    density_grid = comparison.density_grid
    velocity_grid = comparison.velocity_grid
    fit = comparison.density_fit
    repository = Path(__file__).resolve().parents[2]
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **_source_provenance(repository),
        "run": {
            "id": configuration.run_id,
            "source_config": str(configuration.source_path),
            "recipe_config": str(configuration.recipe.source_path),
            "output_directory": str(configuration.output_dir),
            "cold_start": True,
        },
        "data": {
            "catalog": str(configuration.data.catalog),
            "target_density": str(configuration.data.target_density),
            "weight_source": (
                "catalogue_column"
                if comparison.weight_model.mode == "catalogue_fixed"
                else "trial_density_solution"
            ),
            "weight_column": (
                "w" if comparison.weight_model.mode == "catalogue_fixed" else None
            ),
            "weights_fixed_across_trial_potentials": (
                comparison.weight_model.mode == "catalogue_fixed"
            ),
        },
        "potential": {
            "name": ZHU_2026_POTENTIAL_NAME,
            "recipe": configuration.recipe.potential.recipe,
            "fixed_orientation": {"alpha_halo": 0.0, "beta_halo": 0.0},
            "representative_best_fit": ZHU_2026_BEST_FIT,
        },
        "density_grid": {
            "axis_order": ["R", "z", "phi"],
            "r_edges_kpc": density_grid.r_edges.tolist(),
            "z_edges_kpc": density_grid.z_edges.tolist(),
            "phi_edges_rad": density_grid.phi_edges.tolist(),
        },
        "density_fit": {
            "min_abs_z_kpc": fit.min_abs_z,
            "min_radius_kpc": fit.min_spherical_radius,
            "max_radius_kpc": fit.max_spherical_radius,
            "normalization_min_radius_kpc": fit.normalization_min_radius,
            "require_positive_target": fit.require_positive_data,
            "normalization": fit.normalization,
        },
        "weight_model": {
            "mode": comparison.weight_model.mode,
            "solver": comparison.weight_model.solver,
            "target_normalization": comparison.weight_model.target_normalization,
            "regularization": comparison.weight_model.regularization,
            "regularization_strength": (
                comparison.weight_model.regularization_strength
            ),
        },
        "objective": {
            "mode": comparison.objective.mode,
            "density_max_chi2_per_bin": (
                comparison.objective.density_max_chi2_per_bin
            ),
            "invalid_trial_penalty": 1e30,
        },
        "velocity_fit": {
            "enabled": comparison.include_velocity,
            "min_radius_kpc": comparison.velocity_fit_min_radius,
            "probability_floor": comparison.velocity_probability_floor,
            "radius_edges_kpc": velocity_grid.radius_edges.tolist(),
            "theta_edges_rad": velocity_grid.theta_edges.tolist(),
            "phi_edges_rad": velocity_grid.phi_edges.tolist(),
            "velocity_edges_km_s": velocity_grid.velocity_edges.tolist(),
            "error_columns": {
                "vr": "vr_err",
                "vphi": "vphi_err",
                "vtheta": "vthe_err",
            },
        },
        "orbits": {
            "periods": comparison.orbit_periods,
            "samples_per_orbit": comparison.orbit_samples_per_orbit,
            "sample_divisor": comparison.orbit_sample_divisor,
        },
        "optimizer": {
            "implementation": "scikit-optimize.Optimizer",
            "iterations": configuration.iterations,
            "random_seed": configuration.random_seed,
            "coordinates": list(OPTIMIZER_COORDINATES),
            "round_decimals": configuration.round_decimals,
            "initial_point": configuration.recipe.search.initial_point,
            "paper_best_evaluated_first": (
                configuration.recipe.search.initial_point == "paper_best"
            ),
            "bounds": {
                name: list(bounds)
                for name, bounds in configuration.search_bounds.items()
            },
        },
        "report": {
            "velocity_bin_factor": configuration.report.velocity_bin_factor,
        },
        "coverage": {
            "output_directory": str(configuration.coverage.output_dir),
            "maximum_points": configuration.coverage.maximum_points,
            "velocity_limit_km_s": configuration.coverage.velocity_limit_km_s,
            "random_seed": configuration.coverage.random_seed,
        },
    }


def sample_header(n_phi: int, include_velocity: bool) -> str:
    phi_columns = " ".join(f"chi2_phi{index}" for index in range(n_phi))
    velocity_columns = ""
    if include_velocity:
        velocity_columns = " " + " ".join(
            f"lnL_{component}_phi{index}"
            for component in ("vr", "vphi", "vtheta")
            for index in range(n_phi)
        )
    return (
        "# iteration qhalo phalo rho0 rho0_plus_2logrs gamma "
        "objective objective_velocity objective_density_velocity "
        "chi2 density_chi2_per_bin density_scale "
        "regularization_penalty inner_weight_objective "
        "effective_orbit_count max_weight_fraction active_orbit_count "
        "weight_solver_converged weight_solver_status "
        "weight_solver_iterations weight_solver_optimality weight_solver_cost "
        "successful_orbits "
        "failed_orbits weight_sum "
        f"{phi_columns}{velocity_columns}"
    )


def _append_sample(
    sample_file: Path,
    *,
    iteration: int,
    evaluated: list[float],
    objective: float,
    evaluation,
    include_velocity: bool,
    decimals: int,
) -> None:
    qhalo, phalo, rho0, rho0_plus_2logrs, gamma = evaluated
    coordinate_format = f".{{digits}}f".format(digits=decimals)
    coordinates = " ".join(
        format(value, coordinate_format)
        for value in (qhalo, phalo, rho0, rho0_plus_2logrs, gamma)
    )
    chi2_phi = " ".join(
        f"{value:.8e}" for value in evaluation.density.chi2_by_phi
    )
    velocity_phi = ""
    if include_velocity:
        velocity_phi = " " + " ".join(
            f"{value:.8e}"
            for component in ("vr", "vphi", "vtheta")
            for value in evaluation.velocity_loglike_by_phi[component]
        )
    with sample_file.open("a") as stream:
        stream.write(
            f"{iteration:d} {coordinates} {objective:.8e} "
            f"{evaluation.objective_velocity:.8e} "
            f"{evaluation.objective_density_velocity:.8e} "
            f"{evaluation.density.chi2:.8e} "
            f"{evaluation.density_chi2_per_bin:.8e} "
            f"{evaluation.density.scale:.8e} "
            f"{evaluation.weight_solution.regularization_penalty:.8e} "
            f"{evaluation.weight_solution.inner_objective:.8e} "
            f"{evaluation.weight_solution.effective_orbit_count:.8e} "
            f"{evaluation.weight_solution.maximum_weight_fraction:.8e} "
            f"{evaluation.weight_solution.active_orbit_count:d} "
            f"{int(evaluation.weight_solution.converged):d} "
            f"{evaluation.weight_solution.status:d} "
            f"{evaluation.weight_solution.iterations:d} "
            f"{evaluation.weight_solution.optimality:.8e} "
            f"{evaluation.weight_solution.solver_cost:.8e} "
            f"{evaluation.successful_orbits:d} "
            f"{evaluation.failed_orbits:d} "
            f"{evaluation.weight_sum:.16e} "
            f"{chi2_phi}{velocity_phi}\n"
        )


def run_optimization(configuration: RunConfiguration) -> Path:
    """Run one new optimization and return its cold-start artifact directory."""

    try:
        from skopt import Optimizer
        from skopt.space import Real
    except ImportError as exc:
        raise RuntimeError(
            "scikit-optimize is required for optimization; install the inference extra"
        ) from exc
    if importlib.util.find_spec("agama") is None:
        raise RuntimeError("AGAMA is required for orbit integration")

    if not configuration.data.catalog.exists():
        raise FileNotFoundError(f"catalogue not found: {configuration.data.catalog}")
    if not configuration.data.target_density.exists():
        raise FileNotFoundError(
            f"target density not found: {configuration.data.target_density}"
        )
    output_directory = configuration.output_dir
    if output_directory.exists():
        raise FileExistsError(
            f"cold-start runs require a new output directory: {output_directory}"
        )

    comparison = configuration.to_comparison_config()
    prepared = prepare_model_data(
        configuration.data.catalog,
        configuration.data.target_density,
        comparison,
    )
    audit = None
    if comparison.weight_model.mode == "catalogue_fixed":
        audit = catalogue_weight_audit(
            prepared.initial_conditions,
            prepared.seed_weights,
            comparison.density_grid,
        )
    try:
        output_directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(
            f"cold-start runs require a new output directory: {output_directory}"
        ) from exc

    write_resolved_config(
        output_directory / "resolved_config.json",
        resolved_configuration_document(configuration),
    )
    input_artifact = (
        output_directory / "fixed_seed_weights.npz"
        if audit is not None
        else output_directory / "weight_model_inputs.npz"
    )
    np.savez_compressed(
        input_artifact,
        artifact_schema_version=np.asarray(1),
        **(audit or {}),
        target_density=prepared.target_density,
        target_error=prepared.target_error,
        r_edges=comparison.density_grid.r_edges,
        z_edges=comparison.density_grid.z_edges,
        phi_edges=comparison.density_grid.phi_edges,
        weight_source=np.asarray(
            "catalogue_column" if audit is not None else "trial_density_solution"
        ),
        weight_column=np.asarray("w" if audit is not None else ""),
        catalog_path=np.asarray(str(prepared.catalog_path)),
        density_path=np.asarray(str(prepared.density_path)),
    )

    bounds = configuration.search_bounds
    parameter_space = [
        Real(*bounds[name], name=name)
        for name in OPTIMIZER_COORDINATES
    ]
    optimizer = Optimizer(
        parameter_space,
        random_state=configuration.random_seed,
    )
    paper_point = paper_best_optimizer_point()
    paper_point_is_in_bounds = all(
        dimension.low <= value <= dimension.high
        for dimension, value in zip(parameter_space, paper_point)
    )
    use_paper_first = configuration.recipe.search.initial_point == "paper_best"
    if use_paper_first and not paper_point_is_in_bounds:
        raise ValueError(
            "the configured paper-best initial point lies outside search bounds"
        )

    sample_file = output_directory / "sample.dat"
    sample_file.write_text(
        sample_header(comparison.density_grid.shape[-1], comparison.include_velocity)
        + "\n"
    )
    best_objective = np.inf
    for iteration in range(configuration.iterations):
        suggested = paper_point if iteration == 0 and use_paper_first else optimizer.ask()
        evaluated, parameters = rounded_trial(
            suggested,
            decimals=configuration.round_decimals,
        )
        evaluation = evaluate_prepared_model(parameters, prepared)
        objective = evaluation.selected_objective
        optimizer.tell(evaluated, objective)
        _append_sample(
            sample_file,
            iteration=iteration,
            evaluated=evaluated,
            objective=objective,
            evaluation=evaluation,
            include_velocity=comparison.include_velocity,
            decimals=configuration.round_decimals,
        )
        if objective < best_objective:
            save_best_evaluation(
                output_directory,
                evaluation,
                parameters,
                iteration=iteration,
                objective=objective,
            )
            best_objective = objective
        print(
            f"iteration={iteration} objective={objective:.6g} "
            f"chi2_per_bin={evaluation.density_chi2_per_bin:.6g} "
            f"neg_loglike_v={evaluation.objective_velocity:.6g} "
            f"weight_converged={evaluation.weight_solution.converged} "
            f"weight_sum={evaluation.weight_sum:.16g} "
            f"successful_orbits={evaluation.successful_orbits} "
            f"failed_orbits={evaluation.failed_orbits} "
            f"chi2_phi={evaluation.density.chi2_by_phi.tolist()}"
        )
    return output_directory
