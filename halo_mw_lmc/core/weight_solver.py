"""Profile non-negative orbit weights against a three-dimensional density."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import DensityFitSettings, WeightModelSettings
from .density import density_fit_mask
from .orbit_response import OrbitDensityResponse


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class WeightSolution:
    """Trial-specific orbit weights and solver diagnostics."""

    seed_weights: FloatArray
    model_density: FloatArray
    target_density: FloatArray
    target_error: FloatArray
    inner_objective: float
    regularization_penalty: float
    effective_orbit_count: float
    maximum_weight_fraction: float
    active_orbit_count: int
    converged: bool
    status: int
    message: str


def _normalized_target(
    target_density: FloatArray,
    target_error: FloatArray,
    fit_mask: NDArray[np.bool_],
    response: OrbitDensityResponse,
    mode: str,
) -> tuple[FloatArray, FloatArray]:
    if mode == "absolute":
        return target_density.copy(), target_error.copy()
    mass = float(
        np.sum(
            target_density[fit_mask]
            * response.grid.volumes[fit_mask]
        )
    )
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("target density has no positive mass in the fit region")
    return target_density / mass, target_error / mass


def solve_density_weights(
    response: OrbitDensityResponse,
    target_density: ArrayLike,
    target_error: ArrayLike,
    density_fit: DensityFitSettings,
    settings: WeightModelSettings,
) -> WeightSolution:
    """Solve weighted non-negative least squares with optional L2 smoothing."""

    if settings.mode != "density_solved":
        raise ValueError("solve_density_weights requires density_solved settings")
    try:
        from scipy.optimize import lsq_linear
        from scipy.sparse import eye, vstack
    except ImportError as exc:
        raise RuntimeError(
            "SciPy is required for density-solved orbit weights"
        ) from exc

    target = np.asarray(target_density, dtype=float)
    error = np.asarray(target_error, dtype=float)
    if target.shape != response.grid.shape or error.shape != response.grid.shape:
        raise ValueError(
            "target density and error must match the orbit-response grid"
        )
    fit_mask = density_fit_mask(target, error, response.grid, density_fit)
    target, error = _normalized_target(
        target,
        error,
        fit_mask,
        response,
        settings.target_normalization or "absolute",
    )

    row_mask = fit_mask.reshape(-1)
    inverse_error = 1.0 / error.reshape(-1)[row_mask]
    design = response.matrix[row_mask].multiply(inverse_error[:, None]).tocsr()
    observed = target.reshape(-1)[row_mask] * inverse_error
    scoring_design = design
    scoring_observed = observed
    regularization = float(settings.regularization_strength)
    if regularization > 0:
        design = vstack(
            [
                design,
                np.sqrt(regularization)
                * eye(response.successful_seed_index.size, format="csr"),
            ],
            format="csr",
        )
        observed = np.concatenate(
            [observed, np.zeros(response.successful_seed_index.size)]
        )
    constrain_unit_mass = settings.target_normalization == "unit_mass"
    if constrain_unit_mass:
        # lsq_linear has bounds but no equality constraints. A strongly
        # weighted mass row makes the bounded solution satisfy the simplex
        # constraint to numerical precision; the final normalization below
        # enforces sum(w)=1 exactly. The reported inner objective excludes this
        # numerical constraint row.
        design_scale = (
            float(np.linalg.norm(scoring_design.data))
            / np.sqrt(max(scoring_design.nnz, 1))
        )
        constraint_weight = 10.0 * max(
            1.0,
            design_scale,
            float(np.linalg.norm(scoring_observed))
            / np.sqrt(max(scoring_observed.size, 1)),
        )
        from scipy.sparse import csr_matrix

        mass_row = csr_matrix(
            np.full(
                (1, response.successful_seed_index.size),
                constraint_weight,
                dtype=float,
            )
        )
        design = vstack([design, mass_row], format="csr")
        observed = np.concatenate([observed, [constraint_weight]])

    result = lsq_linear(
        design,
        observed,
        bounds=(0.0, np.inf),
        method="trf",
        lsq_solver="lsmr",
        lsmr_tol="auto",
    )
    successful_weights = np.asarray(result.x, dtype=float)
    if constrain_unit_mass:
        solved_mass = float(np.sum(successful_weights))
        if not np.isfinite(solved_mass) or solved_mass <= 0:
            raise ValueError("unit-mass weight solve returned no positive mass")
        successful_weights = successful_weights / solved_mass
    seed_weights = np.zeros(response.seed_count, dtype=float)
    seed_weights[response.successful_seed_index] = successful_weights
    model_density = response.model_density(seed_weights)
    total = float(np.sum(seed_weights))
    squared = float(np.dot(seed_weights, seed_weights))
    maximum_fraction = (
        float(np.max(seed_weights)) / total if total > 0 else 0.0
    )
    tolerance = max(float(np.max(seed_weights)) * 1e-12, 0.0)
    regularization_penalty = regularization * squared
    density_residual = scoring_design @ successful_weights - scoring_observed
    inner_objective = float(np.dot(density_residual, density_residual))
    inner_objective += regularization_penalty
    return WeightSolution(
        seed_weights=seed_weights,
        model_density=model_density,
        target_density=target,
        target_error=error,
        inner_objective=inner_objective,
        regularization_penalty=regularization_penalty,
        effective_orbit_count=total**2 / squared if squared > 0 else 0.0,
        maximum_weight_fraction=maximum_fraction,
        active_orbit_count=int(np.count_nonzero(seed_weights > tolerance)),
        converged=bool(result.success) and np.all(np.isfinite(seed_weights)),
        status=int(result.status),
        message=str(result.message),
    )
