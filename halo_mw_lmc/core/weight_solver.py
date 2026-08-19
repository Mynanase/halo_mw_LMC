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
    iterations: int = 0
    optimality: float = np.inf
    solver_cost: float = np.inf


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
    full_design = response.matrix[row_mask]
    # Orbits that never visit a fitted cell cannot constrain the target.
    # Drop those columns from the solve and restore zero weight afterwards;
    # keeping them only inflates the problem and slows the outer TRF steps.
    active_columns = np.asarray(full_design.sum(axis=0) > 0).ravel()
    successful_count = int(response.successful_seed_index.size)
    if not np.any(active_columns):
        raise ValueError("no orbit responds inside the density fit region")
    inverse_error = 1.0 / error.reshape(-1)[row_mask]
    design = full_design[:, active_columns].multiply(
        inverse_error[:, None]
    ).tocsr()
    observed = target.reshape(-1)[row_mask] * inverse_error
    scoring_design = design
    scoring_observed = observed
    regularization = float(settings.regularization_strength)
    n_active = int(active_columns.sum())
    if regularization > 0:
        design = vstack(
            [
                design,
                np.sqrt(regularization) * eye(n_active, format="csr"),
            ],
            format="csr",
        )
        observed = np.concatenate(
            [observed, np.zeros(n_active)]
        )
    result = lsq_linear(
        design,
        observed,
        bounds=(0.0, np.inf),
        method="trf",
        lsq_solver="lsmr",
        lsmr_tol=settings.lsmr_tol if settings.lsmr_tol is not None else "auto",
        max_iter=int(settings.max_iter),
    )
    # No sum(w)=1 constraint: the target is already normalized to unit mass
    # inside the fit mask, so the least-squares solution sets the total
    # weight scale directly. A later renormalization would only re-introduce
    # the scale conflict the constraint used to force.
    active_weights = np.asarray(result.x, dtype=float)
    successful_weights = np.zeros(successful_count, dtype=float)
    successful_weights[active_columns] = active_weights
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
    density_residual = scoring_design @ active_weights - scoring_observed
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
        iterations=int(result.nit),
        optimality=float(result.optimality),
        solver_cost=float(result.cost),
    )
