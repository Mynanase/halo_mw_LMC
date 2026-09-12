"""Configuration resolved to plain nested dicts.

Two TOML files describe one experiment. The reusable recipe owns the
scientific choices shared by every run; the run file owns data paths, run
identity, output location, iteration count, and seed. Both are read with
plain ``tomllib`` and resolved into the plain dicts below. There are no
schema classes: unknown keys are ignored, and a wrong value surfaces where
it is used. The keys are the contract and are documented once here.

``load_recipe_configuration(path)`` returns::

    {
        "source_path": Path, "schema_version": 1, "name": str,
        "potential": {"recipe": "zhu_2026_mw_halo"},
        "density_grid": CylindricalGrid,
        "velocity_grid": SphericalVelocityGrid,
        "density_fit": {          # keys match compare_density(**density_fit)
            "min_abs_z", "min_spherical_radius", "max_spherical_radius",
            "normalization_min_radius", "require_positive_data", "normalization",
        },
        "weight_model": {         # keys match solve_density_weights(**weight_model)
            "mode", "solver", "target_normalization", "regularization",
            "regularization_strength", "max_iter", "solver_tolerance", "lsmr_tol",
        },
        "objective": {
            "mode", "density_max_chi2_per_bin", "density_shell_edges",
            "density_shell_phi_max_chi2_per_bin",
        },
        "include_velocity": bool,
        "velocity_fit_min_radius": float,      # kpc
        "velocity_probability_floor": float,
        "orbit_periods": float,
        "orbit_samples_per_orbit": int,
        "orbit_sample_divisor": float,
        "search": {
            "initial_point": "paper_best" | "optimizer",
            "round_decimals": int,
            "bounds": {parameter: (lower, upper)},
        },
    }

``load_run_configuration(path)`` returns the same plus the run tables::

    {
        "recipe": <recipe dict>,
        "run": {"id", "output_dir"},
        "data": {"catalog", "target_density"},
        "optimizer": {"iterations", "random_seed", "fixed_points"},
        "report": {"velocity_bin_factor"},
        "coverage": {"output_dir", "maximum_points", "velocity_limit_km_s",
                     "random_seed"},
    }

``resolve_model(recipe)`` picks the numerical subset threaded into
prepare/evaluate/optimize (everything except ``source_path``, ``schema_version``,
``name``, ``potential``, and ``search``).

Only data-boundary checks remain: the schema version, contracts whose
violation would silently produce wrong science (normalization modes, the
density-solved weight contract, the fixed-point rounding invariant), and
physical ranges the grid constructors cannot see. Everything else fails
fast on a raw KeyError or library traceback.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import tomllib

from .grids import CylindricalGrid
from .potential import ZHU_2026_BEST_FIT, ZHU_2026_POTENTIAL_NAME
from .velocity import SphericalVelocityGrid

CONFIGURATION_SCHEMA_VERSION = 1
SYNTHETIC_DENSITY_MODEL_NAMES = ("desi_year1_kgiants_3d",)
SEARCH_PARAMETER_NAMES = (
    "qhalo",
    "phalo",
    "rho0",
    "rho0_plus_2logrs",
    "gamma",
)


def resolve_model(recipe: dict) -> dict:
    """The numerical comparison configuration shared by prepare/evaluate/optimize."""

    keys = (
        "density_grid",
        "velocity_grid",
        "density_fit",
        "weight_model",
        "objective",
        "include_velocity",
        "velocity_fit_min_radius",
        "velocity_probability_floor",
        "orbit_periods",
        "orbit_samples_per_orbit",
        "orbit_sample_divisor",
    )
    return {key: recipe[key] for key in keys}


def _read_toml(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    with source.open("rb") as stream:
        return source, tomllib.load(stream)


def _resolved_path(value: Any, source: Path) -> Path:
    raw = Path(value).expanduser()
    return raw.resolve() if raw.is_absolute() else (source.parent / raw).resolve()


def _representable(value: float, decimals: int, context: str) -> None:
    if not math.isclose(value, round(value, decimals), rel_tol=0.0, abs_tol=10 ** (-(decimals + 10))):
        raise ValueError(f"{context} must be representable with round_decimals={decimals}")


def load_recipe_configuration(path: str | Path) -> dict:
    """Load a reusable scientific recipe TOML into its resolved plain dict."""

    source, document = _read_toml(path)
    if document.get("schema_version") != CONFIGURATION_SCHEMA_VERSION:
        raise ValueError(
            f"recipe.schema_version must be {CONFIGURATION_SCHEMA_VERSION}, "
            f"got {document.get('schema_version')!r}"
        )
    if document["potential"]["recipe"] != ZHU_2026_POTENTIAL_NAME:
        raise ValueError(
            f"unsupported potential recipe: {document['potential']['recipe']!r}; "
            f"expected {ZHU_2026_POTENTIAL_NAME!r}"
        )

    grid_table = document["density_grid"]
    density_grid = CylindricalGrid.uniform(
        n_r=grid_table["n_r"], r_range=tuple(grid_table["r_range_kpc"]),
        n_z=grid_table["n_z"], z_range=tuple(grid_table["z_range_kpc"]),
        n_phi=grid_table["n_phi"], phi_origin=np.deg2rad(grid_table["phi_origin_deg"]),
    )

    fit = document["density_fit"]
    density_fit = {
        "min_abs_z": fit["min_abs_z_kpc"],
        "min_spherical_radius": fit["min_radius_kpc"],
        "max_spherical_radius": fit["max_radius_kpc"],
        "normalization_min_radius": fit["normalization_min_radius_kpc"],
        "require_positive_data": fit["require_positive_target"],
        "normalization": fit["normalization"],
    }

    velocity = document["velocity_fit"]
    velocity_grid = SphericalVelocityGrid(
        radius_edges=np.asarray(velocity["radius_edges_kpc"], dtype=float),
        theta_edges=np.deg2rad(np.asarray(velocity["theta_edges_deg"], dtype=float)),
        phi_edges=density_grid.phi_edges,
        velocity_edges=np.linspace(velocity["velocity_range_km_s"][0], velocity["velocity_range_km_s"][1], velocity["velocity_bins"] + 1),
    )

    weight_table = document.get("weight_model", {})
    weight_mode = weight_table.get("mode", "catalogue_fixed")
    if weight_mode == "catalogue_fixed":
        # Options set here would be silently ignored; refuse them instead.
        solver_options = set(weight_table) - {"mode"}
        if solver_options:
            raise ValueError(
                f"catalogue_fixed weights cannot define solver options: "
                f"{', '.join(sorted(solver_options))}"
            )
        weight_model = {
            "mode": weight_mode, "solver": None, "target_normalization": None,
            "regularization": None, "regularization_strength": 0.0, "max_iter": 20000,
            "solver_tolerance": None, "lsmr_tol": None,
        }
    elif weight_mode == "density_solved":
        weight_model = {
            "mode": weight_mode,
            "solver": weight_table["solver"],
            "target_normalization": weight_table["target_normalization"],
            "regularization": weight_table["regularization"],
            "regularization_strength": weight_table["regularization_strength"],
            "max_iter": weight_table["max_iter"],
            "solver_tolerance": weight_table.get("solver_tolerance", 1e-8),
            "lsmr_tol": weight_table.get("lsmr_tol"),
        }
        if weight_model["target_normalization"] not in {"absolute", "unit_mass"}:
            raise ValueError("weight_model.target_normalization must be 'absolute' or 'unit_mass'")
        if weight_model["regularization"] != "l2":
            raise ValueError("density_solved currently requires regularization='l2'")
        if weight_model["solver"] == "dual_ridge" and not weight_model["regularization_strength"] > 0:
            raise ValueError("dual_ridge requires a positive regularization_strength")
        if weight_model["solver"] != "lsq_linear" and weight_model["lsmr_tol"] is not None:
            raise ValueError("lsmr_tol is only valid for solver='lsq_linear'; set it to None")
    else:
        raise ValueError("weight_model.mode must be 'catalogue_fixed' or 'density_solved'")

    objective_table = document.get("objective", {})
    objective_mode = objective_table.get("mode", "density_velocity")
    objective = {
        "mode": objective_mode,
        "density_max_chi2_per_bin": objective_table.get("density_max_chi2_per_bin"),
        "density_shell_edges": objective_table.get("density_shell_edges_kpc"),
        "density_shell_phi_max_chi2_per_bin": objective_table.get("density_shell_phi_max_chi2_per_bin"),
    }
    if objective["density_shell_edges"] is not None:
        objective["density_shell_edges"] = tuple(float(edge) for edge in objective["density_shell_edges"])
    shell_edges = objective["density_shell_edges"]
    if (shell_edges is None) != (objective["density_shell_phi_max_chi2_per_bin"] is None):
        raise ValueError("density shell edges and shell-phi limit must be configured together")
    if shell_edges is not None and any(left >= right for left, right in zip(shell_edges, shell_edges[1:])):
        raise ValueError("objective.density_shell_edges_kpc must be strictly increasing")

    # Silent-wrong-answer contracts between recipe sections.
    orbit_table = document["orbits"]
    if not orbit_table["periods"] > 0:
        raise ValueError("orbits.periods must be positive")
    if orbit_table["samples_per_orbit"] < 1:
        raise ValueError("orbits.samples_per_orbit must be at least 1")
    if not velocity["probability_floor"] > 0:
        raise ValueError("velocity_fit.probability_floor must be positive")
    if weight_mode == "density_solved":
        if density_fit["normalization"] != "none":
            raise ValueError("density_solved weights require density_fit.normalization='none'")
        if not velocity["enabled"]:
            raise ValueError("density_solved weights require velocity_fit.enabled=true")
    if objective_mode == "velocity_only":
        if not velocity["enabled"]:
            raise ValueError("velocity_only objective requires velocity_fit.enabled=true")
        limit = objective["density_max_chi2_per_bin"]
        if limit is None or not math.isfinite(limit) or limit <= 0:
            raise ValueError("velocity_only requires a positive density_max_chi2_per_bin")
    elif objective["density_max_chi2_per_bin"] is not None:
        raise ValueError("density_velocity does not use a density chi2-per-bin gate")

    divisor_policy = orbit_table.get("sample_weight_divisor")
    if weight_mode == "catalogue_fixed":
        if divisor_policy != "half_samples":
            raise ValueError("orbits.sample_weight_divisor must be 'half_samples' for catalogue_fixed")
        orbit_sample_divisor = orbit_table["samples_per_orbit"] / 2.0
    else:
        # Density-solved weights use each orbit's actual finite sample count;
        # the divisor is a compatibility value never used in that mode.
        orbit_sample_divisor = float(orbit_table["samples_per_orbit"])

    search_table = document["search"]
    initial_point = search_table["initial_point"]
    if initial_point not in {"paper_best", "optimizer"}:
        raise ValueError("search.initial_point must be 'paper_best' or 'optimizer'")
    round_decimals = search_table["round_decimals"]
    if round_decimals < 0:
        raise ValueError("search.round_decimals must be non-negative; negative rounds to decades")
    bounds = {name: tuple(search_table["bounds"][name]) for name in SEARCH_PARAMETER_NAMES}
    for name, (lower, upper) in bounds.items():
        if lower >= upper:
            raise ValueError(f"search.bounds.{name} must be strictly increasing")
    if bounds["qhalo"][0] <= 0 or bounds["phalo"][0] <= 0:
        raise ValueError("qhalo and phalo search bounds must be positive")
    if bounds["gamma"][0] < 0 or bounds["gamma"][1] >= 3:
        raise ValueError("gamma search bounds must satisfy 0 <= gamma < 3")
    for name, (lower, upper) in bounds.items():
        _representable(lower, round_decimals, f"search.bounds.{name} lower endpoint")
        _representable(upper, round_decimals, f"search.bounds.{name} upper endpoint")
    if initial_point == "paper_best":
        paper_best = {
            "qhalo": round(ZHU_2026_BEST_FIT["qhalo"], round_decimals),
            "phalo": round(ZHU_2026_BEST_FIT["phalo"], round_decimals),
            "rho0": round(ZHU_2026_BEST_FIT["rho0"], round_decimals),
            "rho0_plus_2logrs": round(ZHU_2026_BEST_FIT["rho0"] + 2 * ZHU_2026_BEST_FIT["log_rs"], round_decimals),
            "gamma": round(ZHU_2026_BEST_FIT["gamma"], round_decimals),
        }
        outside = [name for name, (lower, upper) in bounds.items() if not lower <= paper_best[name] <= upper]
        if outside:
            raise ValueError(
                "search.initial_point='paper_best' lies outside bounds for: " + ", ".join(outside)
            )

    return {
        "source_path": source,
        "schema_version": document["schema_version"],
        "name": document["name"],
        "potential": {"recipe": document["potential"]["recipe"]},
        "density_grid": density_grid,
        "velocity_grid": velocity_grid,
        "density_fit": density_fit,
        "weight_model": weight_model,
        "objective": objective,
        "include_velocity": velocity["enabled"],
        "velocity_fit_min_radius": velocity["min_radius_kpc"],
        "velocity_probability_floor": velocity["probability_floor"],
        "orbit_periods": orbit_table["periods"],
        "orbit_samples_per_orbit": orbit_table["samples_per_orbit"],
        "orbit_sample_divisor": orbit_sample_divisor,
        "search": {
            "initial_point": initial_point,
            "round_decimals": round_decimals,
            "bounds": bounds,
        },
    }


def _fixed_optimizer_points(points, bounds, round_decimals):
    if not points:
        raise ValueError("optimizer.fixed_points must contain at least one point")
    resolved = []
    for raw in points:
        if len(raw) != len(SEARCH_PARAMETER_NAMES):
            raise ValueError(
                f"fixed point {list(raw)} must contain exactly "
                f"{len(SEARCH_PARAMETER_NAMES)} coordinates in "
                f"{', '.join(SEARCH_PARAMETER_NAMES)} order"
            )
        point = tuple(float(coordinate) for coordinate in raw)
        for name, coordinate in zip(SEARCH_PARAMETER_NAMES, point):
            _representable(coordinate, round_decimals, f"fixed point coordinate {name}")
            lower, upper = bounds[name]
            if not lower <= coordinate <= upper:
                raise ValueError(f"fixed point coordinate {name}={coordinate} lies outside [{lower}, {upper}]")
        resolved.append(point)
    if len(set(resolved)) != len(resolved):
        raise ValueError("optimizer.fixed_points must not contain duplicate points")
    return tuple(resolved)


def load_run_configuration(path: str | Path) -> dict:
    """Load a run TOML and its referenced recipe as one plain dict."""

    source, document = _read_toml(path)
    if document.get("schema_version") != CONFIGURATION_SCHEMA_VERSION:
        raise ValueError(
            f"run configuration.schema_version must be {CONFIGURATION_SCHEMA_VERSION}, "
            f"got {document.get('schema_version')!r}"
        )
    recipe = load_recipe_configuration(_resolved_path(document["recipe"], source))

    optimizer = {
        "iterations": document["optimizer"]["iterations"],
        "random_seed": document["optimizer"]["random_seed"],
        "fixed_points": None,
    }
    if "fixed_points" in document["optimizer"]:
        fixed_points = _fixed_optimizer_points(
            document["optimizer"]["fixed_points"],
            recipe["search"]["bounds"],
            recipe["search"]["round_decimals"],
        )
        if optimizer["iterations"] != len(fixed_points):
            raise ValueError("optimizer.iterations must equal the number of fixed_points")
        optimizer["fixed_points"] = fixed_points

    return {
        "source_path": source,
        "schema_version": document["schema_version"],
        "recipe": recipe,
        "run": {
            "id": document["run"]["id"],
            "output_dir": _resolved_path(document["run"]["output_dir"], source),
        },
        "data": {
            "catalog": _resolved_path(document["data"]["catalog"], source),
            "target_density": _resolved_path(document["data"]["target_density"], source),
        },
        "optimizer": optimizer,
        "report": {"velocity_bin_factor": document["report"]["velocity_bin_factor"]},
        "coverage": {
            "output_dir": _resolved_path(document["coverage"]["output_dir"], source),
            "maximum_points": document["coverage"]["maximum_points"],
            "velocity_limit_km_s": document["coverage"]["velocity_limit_km_s"],
            "random_seed": document["coverage"]["random_seed"],
        },
    }


def load_synthetic_density_configuration(path: str | Path) -> dict:
    """Load an analytic target-density generation configuration."""

    source, document = _read_toml(path)
    if document.get("schema_version") != CONFIGURATION_SCHEMA_VERSION:
        raise ValueError(
            f"synthetic density schema_version must be {CONFIGURATION_SCHEMA_VERSION}, "
            f"got {document.get('schema_version')!r}"
        )
    model_name = document["model"]["name"]
    if model_name not in SYNTHETIC_DENSITY_MODEL_NAMES:
        raise ValueError(
            f"unsupported synthetic density model: {model_name!r}; expected one of "
            + ", ".join(SYNTHETIC_DENSITY_MODEL_NAMES)
        )
    quadrature_order = document["quadrature"]["order"]
    validation_order = document["quadrature"]["validation_order"]
    if validation_order <= quadrature_order:
        raise ValueError("quadrature.validation_order must exceed quadrature.order")
    output_path = _resolved_path(document["output"]["path"], source)
    if output_path.suffix.lower() != ".npz":
        raise ValueError("output.path must end in .npz")
    fractional = document["uncertainty"]["fractional"]
    if not fractional > 0:
        raise ValueError("uncertainty.fractional must be positive")

    return {
        "source_path": source,
        "schema_version": document["schema_version"],
        "recipe": load_recipe_configuration(_resolved_path(document["recipe"], source)),
        "model_name": model_name,
        "model_source": _resolved_path(document["model"]["source"], source),
        "quadrature_order": quadrature_order,
        "validation_order": validation_order,
        "fractional_uncertainty": fractional,
        "output_path": output_path,
    }
