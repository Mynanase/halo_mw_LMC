"""Cold-start Bayesian optimization driven by a resolved run configuration."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .artifacts import save_best_evaluation, write_resolved_config
from .config import resolve_model
from .potential import (
    ZHU_2026_BEST_FIT,
    ZHU_2026_POTENTIAL_NAME,
    ZhuHaloParameters,
)
from .evaluate import evaluate_prepared_model
from .prepare import (
    PreparedExecution,
    preflight_and_prepare,
    require_preflight,
)


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
    return evaluated, ZhuHaloParameters(rho0=rho0, log_rs=log_rs, phalo=phalo, qhalo=qhalo, gamma=gamma)


def _source_provenance(repository: Path) -> dict[str, object]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository, check=True, capture_output=True, text=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=repository, check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": bool(status.strip())}


def resolved_configuration_document(
    configuration: dict,
) -> dict[str, object]:
    """Return every scientific and operational choice in JSON-safe form."""

    comparison = resolve_model(configuration["recipe"])
    density_grid = comparison["density_grid"]
    velocity_grid = comparison["velocity_grid"]
    fit = comparison["density_fit"]
    weight_model = comparison["weight_model"]
    objective = comparison["objective"]
    repository = Path(__file__).resolve().parents[2]
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **_source_provenance(repository),
        "run": {
            "id": configuration["run"]["id"],
            "source_config": str(configuration["source_path"]),
            "recipe_config": str(configuration["recipe"]["source_path"]),
            "output_directory": str(configuration["run"]["output_dir"]),
            "cold_start": True,
        },
        "data": {
            "catalog": str(configuration["data"]["catalog"]),
            "target_density": str(configuration["data"]["target_density"]),
            "weight_source": "catalogue_column" if weight_model["mode"] == "catalogue_fixed" else "trial_density_solution",
            "weight_column": "w" if weight_model["mode"] == "catalogue_fixed" else None,
            "weights_fixed_across_trial_potentials": weight_model["mode"] == "catalogue_fixed",
        },
        "potential": {
            "name": ZHU_2026_POTENTIAL_NAME,
            "recipe": configuration["recipe"]["potential"]["recipe"],
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
            "min_abs_z_kpc": fit["min_abs_z"],
            "min_radius_kpc": fit["min_spherical_radius"],
            "max_radius_kpc": fit["max_spherical_radius"],
            "normalization_min_radius_kpc": fit["normalization_min_radius"],
            "require_positive_target": fit["require_positive_data"],
            "normalization": fit["normalization"],
        },
        "weight_model": {
            "mode": weight_model["mode"],
            "solver": weight_model["solver"],
            "target_normalization": weight_model["target_normalization"],
            "regularization": weight_model["regularization"],
            "regularization_strength": weight_model["regularization_strength"],
            "max_iter": weight_model["max_iter"],
            "solver_tolerance": weight_model["solver_tolerance"],
            "lsmr_tol": weight_model["lsmr_tol"],
        },
        "objective": {
            "mode": objective["mode"],
            "density_max_chi2_per_bin": objective["density_max_chi2_per_bin"],
            "density_shell_edges_kpc": (
                list(objective["density_shell_edges"])
                if objective["density_shell_edges"] is not None
                else None
            ),
            "density_shell_phi_max_chi2_per_bin": objective["density_shell_phi_max_chi2_per_bin"],
            "invalid_trial_penalty": 1e30,
        },
        "velocity_fit": {
            "enabled": comparison["include_velocity"],
            "min_radius_kpc": comparison["velocity_fit_min_radius"],
            "probability_floor": comparison["velocity_probability_floor"],
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
            "periods": comparison["orbit_periods"],
            "samples_per_orbit": comparison["orbit_samples_per_orbit"],
            "sample_divisor": comparison["orbit_sample_divisor"],
        },
        "optimizer": {
            "implementation": (
                "sequential_fixed_points"
                if configuration["optimizer"]["fixed_points"] is not None
                else "scikit-optimize.Optimizer.ask_tell"
            ),
            "iterations": configuration["optimizer"]["iterations"],
            "random_seed": configuration["optimizer"]["random_seed"],
            "schedule": "fixed_points" if configuration["optimizer"]["fixed_points"] is not None else "adaptive",
            "fixed_points": [list(point) for point in configuration["optimizer"]["fixed_points"]] if configuration["optimizer"]["fixed_points"] is not None else None,
            "coordinates": list(OPTIMIZER_COORDINATES),
            "round_decimals": configuration["recipe"]["search"]["round_decimals"],
            "initial_point": configuration["recipe"]["search"]["initial_point"],
            "paper_best_evaluated_first": configuration["recipe"]["search"]["initial_point"] == "paper_best",
            "bounds": {
                name: list(bounds)
                for name, bounds in configuration["recipe"]["search"]["bounds"].items()
            },
        },
        "report": {
            "velocity_bin_factor": configuration["report"]["velocity_bin_factor"],
        },
        "coverage": {
            "output_directory": str(configuration["coverage"]["output_dir"]),
            "maximum_points": configuration["coverage"]["maximum_points"],
            "velocity_limit_km_s": configuration["coverage"]["velocity_limit_km_s"],
            "random_seed": configuration["coverage"]["random_seed"],
        },
    }


def sample_header(
    n_phi: int,
    include_velocity: bool,
    n_density_shells: int = 0,
) -> str:
    phi_columns = " ".join(f"chi2_phi{index}" for index in range(n_phi))
    shell_columns = ""
    if n_density_shells:
        shell_columns = " " + " ".join(
            [
                "density_shell_phi_gate_passed",
                "density_worst_shell_phi_chi2_per_bin",
                "density_worst_shell_index",
                "density_worst_phi_index",
            ]
            + [
                f"density_chi2_per_bin_shell{shell}"
                for shell in range(n_density_shells)
            ]
            + [
                f"density_chi2_per_bin_shell{shell}_phi{phi}"
                for shell in range(n_density_shells)
                for phi in range(n_phi)
            ]
        )
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
        "zero_weight_fraction "
        "weight_solver_converged weight_solver_status "
        "weight_solver_iterations weight_solver_optimality weight_solver_cost "
        "weight_solver_kkt_residual weight_solver_wall_seconds "
        "successful_orbits "
        "failed_orbits weight_sum "
        f"{phi_columns}{shell_columns}{velocity_columns}"
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
    shell_values = ""
    if evaluation.density_shells is not None:
        worst_index = evaluation.density_worst_shell_phi_index
        if worst_index is None:
            worst_index = (-1, -1)
        values_by_shell = evaluation.density_shells.chi2_per_bin_by_shell
        values_by_shell_phi = evaluation.density_shells.chi2_per_bin_by_shell_phi
        shell_values = " " + " ".join(
            [
                str(int(evaluation.density_shell_phi_gate_passed)),
                f"{evaluation.density_worst_shell_phi_chi2_per_bin:.8e}",
                str(worst_index[0]),
                str(worst_index[1]),
            ]
            + [f"{value:.8e}" for value in values_by_shell]
            + [f"{value:.8e}" for value in values_by_shell_phi.ravel()]
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
            f"{np.mean(evaluation.weight_solution.seed_weights == 0.0):.8e} "
            f"{int(evaluation.weight_solution.converged):d} "
            f"{evaluation.weight_solution.status:d} "
            f"{evaluation.weight_solution.iterations:d} "
            f"{evaluation.weight_solution.optimality:.8e} "
            f"{evaluation.weight_solution.solver_cost:.8e} "
            f"{evaluation.weight_solution.kkt_residual:.8e} "
            f"{evaluation.weight_solution.solve_wall_seconds:.8e} "
            f"{evaluation.successful_orbits:d} "
            f"{evaluation.failed_orbits:d} "
            f"{evaluation.weight_sum:.16e} "
            f"{chi2_phi}{shell_values}{velocity_phi}\n"
        )


def _prepared_execution(
    configuration: dict,
    prepared: PreparedExecution | None,
    *,
    stage: str,
) -> PreparedExecution:
    if prepared is None:
        result = require_preflight(preflight_and_prepare(configuration, stage=stage))
        prepared = result.execution
    if prepared is None:
        raise RuntimeError("numerical preflight did not return prepared inputs")
    if prepared.configuration != configuration:
        raise ValueError("prepared numerical inputs belong to another configuration")
    return prepared


def _initialize_run(
    configuration: dict,
    prepared: PreparedExecution,
) -> tuple[Path, Path]:
    """Create artifacts only after every preflight check has passed."""

    output_directory = configuration["run"]["output_dir"]
    if output_directory.exists():
        raise FileExistsError(
            f"cold-start runs require a new output directory: {output_directory}"
        )
    comparison = resolve_model(configuration["recipe"])
    output_directory.mkdir(parents=True, exist_ok=False)

    write_resolved_config(
        output_directory / "resolved_config.json",
        resolved_configuration_document(configuration),
    )
    input_artifact = output_directory / ("fixed_seed_weights.npz" if prepared.weight_audit is not None else "weight_model_inputs.npz")
    np.savez_compressed(
        input_artifact,
        artifact_schema_version=np.asarray(1),
        **(prepared.weight_audit or {}),
        target_density=prepared.model.target_density,
        target_error=prepared.model.target_error,
        r_edges=comparison["density_grid"].r_edges,
        z_edges=comparison["density_grid"].z_edges,
        phi_edges=comparison["density_grid"].phi_edges,
        weight_source=np.asarray("catalogue_column" if prepared.weight_audit is not None else "trial_density_solution"),
        weight_column=np.asarray("w" if prepared.weight_audit is not None else ""),
        catalog_path=np.asarray(str(prepared.model.catalog_path)),
        density_path=np.asarray(str(prepared.model.density_path)),
    )
    shell_count = len(comparison["objective"]["density_shell_edges"]) - 1 if comparison["objective"]["density_shell_edges"] is not None else 0
    sample_file = output_directory / "sample.dat"
    sample_file.write_text(
        sample_header(comparison["density_grid"].shape[-1], comparison["include_velocity"], n_density_shells=shell_count) + "\n"
    )
    return output_directory, sample_file


def _run_trials(
    configuration: dict,
    prepared: PreparedExecution,
    suggestions,
    *,
    tell=None,
) -> Path:
    """Evaluate, persist, and update best for one already-selected schedule."""

    output_directory, sample_file = _initialize_run(configuration, prepared)
    comparison = resolve_model(configuration["recipe"])
    best_objective = np.inf
    for iteration, suggested in enumerate(suggestions):
        evaluated, parameters = rounded_trial(suggested, decimals=configuration["recipe"]["search"]["round_decimals"])
        evaluation = evaluate_prepared_model(parameters, prepared.model)
        objective = evaluation.selected_objective
        if tell is not None:
            tell(evaluated, objective)
        _append_sample(
            sample_file, iteration=iteration, evaluated=evaluated, objective=objective,
            evaluation=evaluation, include_velocity=comparison["include_velocity"],
            decimals=configuration["recipe"]["search"]["round_decimals"],
        )
        if objective < best_objective:
            save_best_evaluation(output_directory, evaluation, parameters, iteration=iteration, objective=objective)
            best_objective = objective
        print(
            f"iteration={iteration} objective={objective:.6g} "
            f"chi2_per_bin={evaluation.density_chi2_per_bin:.6g} "
            f"neg_loglike_v={evaluation.objective_velocity:.6g} "
            f"weight_converged={evaluation.weight_solution.converged} "
            f"weight_sum={evaluation.weight_sum:.16g} "
            f"successful_orbits={evaluation.successful_orbits} "
            f"failed_orbits={evaluation.failed_orbits} "
            f"density_gate_passed={evaluation.density_gate_passed} "
            f"worst_shell_phi={evaluation.density_worst_shell_phi_chi2_per_bin:.6g} "
            f"chi2_phi={evaluation.density.chi2_by_phi.tolist()}"
        )
    return output_directory


def run_fixed_evaluation(
    configuration: dict,
    prepared: PreparedExecution | None = None,
) -> Path:
    """Evaluate explicit points sequentially without importing scikit-optimize."""

    points = configuration["optimizer"]["fixed_points"]
    if points is None:
        raise ValueError("evaluate requires optimizer.fixed_points")
    prepared = _prepared_execution(configuration, prepared, stage="evaluate")
    return _run_trials(configuration, prepared, points)


def run_optimization(
    configuration: dict,
    prepared: PreparedExecution | None = None,
) -> Path:
    """Run an adaptive cold-start optimization using skopt ask/tell."""

    if configuration["optimizer"]["fixed_points"] is not None:
        raise ValueError("optimize accepts adaptive configurations only")
    prepared = _prepared_execution(configuration, prepared, stage="optimize")
    try:
        from skopt import Optimizer
        from skopt.space import Real
    except ImportError as exc:
        raise RuntimeError(
            "scikit-optimize is required for adaptive optimization"
        ) from exc

    bounds = configuration["recipe"]["search"]["bounds"]
    parameter_space = [Real(*bounds[name], name=name) for name in OPTIMIZER_COORDINATES]
    optimizer = Optimizer(parameter_space, random_state=configuration["optimizer"]["random_seed"])
    paper_point = paper_best_optimizer_point()
    use_paper_first = configuration["recipe"]["search"]["initial_point"] == "paper_best"
    if use_paper_first and not all(
        dimension.low <= value <= dimension.high
        for dimension, value in zip(parameter_space, paper_point)
    ):
        raise ValueError(
            "the configured paper-best initial point lies outside search bounds"
        )

    def suggestions():
        for iteration in range(configuration["optimizer"]["iterations"]):
            yield (
                paper_point
                if iteration == 0 and use_paper_first
                else optimizer.ask()
            )

    return _run_trials(configuration, prepared, suggestions(), tell=optimizer.tell)
