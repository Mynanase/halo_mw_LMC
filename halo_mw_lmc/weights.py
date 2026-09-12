"""Fixed catalogue weights and experimental target-derived weights."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .grids import CylindricalGrid


FloatArray = NDArray[np.float64]


def catalogue_seed_weights(values: ArrayLike) -> FloatArray:
    """Validate and copy the fixed per-star weights supplied by the catalogue."""

    weights = np.asarray(values, dtype=float)
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError("catalogue weights must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(weights)):
        raise ValueError("catalogue weights must all be finite")
    if np.any(weights < 0):
        raise ValueError("catalogue weights must be non-negative")
    if not np.any(weights > 0):
        raise ValueError("catalogue weights must contain at least one positive value")
    return weights.copy()


def catalogue_weight_audit(
    initial_conditions: ArrayLike,
    weights: ArrayLike,
    grid: CylindricalGrid,
) -> dict[str, np.ndarray]:
    """Summarize fixed catalogue weights globally and per density cell."""

    initial = np.asarray(initial_conditions, dtype=float)
    weight_values = catalogue_seed_weights(weights)
    if initial.ndim != 2 or initial.shape != (weight_values.size, 6):
        raise ValueError("initial_conditions must have shape (N, 6) matching weights")
    radius = np.hypot(initial[:, 0], initial[:, 1])
    z = initial[:, 2]
    phi = np.arctan2(initial[:, 1], initial[:, 0])
    seed_counts = grid.histogram(radius, z, phi)
    cell_weight_sum = grid.histogram(radius, z, phi, weights=weight_values)
    cell_weight_sq_sum = grid.histogram(radius, z, phi, weights=weight_values**2)
    initial_density = np.divide(cell_weight_sum, grid.volumes, out=np.zeros_like(cell_weight_sum), where=grid.volumes > 0)

    ir, iz, iphi, in_grid = grid.bin_indices(radius, z, phi)
    cell_max_weight = np.zeros(grid.shape, dtype=float)
    np.maximum.at(cell_max_weight, (ir[in_grid], iz[in_grid], iphi[in_grid]), weight_values[in_grid])
    cell_effective_seed_count = np.divide(cell_weight_sum**2, cell_weight_sq_sum, out=np.zeros_like(cell_weight_sum), where=cell_weight_sq_sum > 0)
    cell_max_weight_fraction = np.divide(cell_max_weight, cell_weight_sum, out=np.zeros_like(cell_weight_sum), where=cell_weight_sum > 0)

    total_weight = float(np.sum(weight_values))
    weight_sq_sum = float(np.sum(weight_values**2))
    quantile_levels = np.array([0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0])
    return {
        "weights": weight_values,
        "seed_counts": seed_counts,
        "cell_weight_sum": cell_weight_sum,
        "cell_weight_sq_sum": cell_weight_sq_sum,
        "cell_effective_seed_count": cell_effective_seed_count,
        "cell_max_weight": cell_max_weight,
        "cell_max_weight_fraction": cell_max_weight_fraction,
        "initial_catalog_density": initial_density,
        "quantile_levels": quantile_levels,
        "weight_quantiles": np.quantile(weight_values, quantile_levels),
        "positive_seed_count": np.asarray(np.count_nonzero(weight_values > 0)),
        "in_grid_seed_count": np.asarray(np.count_nonzero(in_grid)),
        "total_weight": np.asarray(total_weight),
        "in_grid_weight": np.asarray(float(np.sum(weight_values[in_grid]))),
        "effective_seed_count": np.asarray(total_weight**2 / weight_sq_sum if weight_sq_sum > 0 else 0.0),
        "max_weight_fraction": np.asarray(float(np.max(weight_values)) / total_weight if total_weight > 0 else 0.0),
    }


@dataclass(frozen=True)
class RepresentativeWeightResult:
    """Fixed seed-orbit weights and their spatial support diagnostics."""

    weights: FloatArray
    seed_counts: NDArray[np.int64]
    cell_weight: FloatArray
    supported_cells: NDArray[np.bool_]
    target_mass: FloatArray
    assigned_mass: float
    positive_target_mass: float
    unsupported_positive_mass: float
    weighted_seed_count: int
    in_grid_seed_count: int

    @property
    def supported_mass_fraction(self) -> float:
        if self.positive_target_mass <= 0:
            return 0.0
        return self.assigned_mass / self.positive_target_mass


def representative_weights_from_target(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    target_density: ArrayLike,
    grid: CylindricalGrid,
    *,
    minimum_seed_count: int = 1,
) -> RepresentativeWeightResult:
    """Assign experimental orbit weights from a three-dimensional target density.

    The production pipeline does not use this function. It is retained for
    isolated comparisons of target-derived and catalogue-supplied weights.

    For a spatial cell ``j=(R,z,phi)`` containing ``N_j`` seed stars,

    ``w_i = nu_target[j] * V[j] / N_j``.

    Consequently, summing the seed weights in any supported cell exactly
    reproduces its target tracer mass.  The result is fixed before trial
    potentials are evaluated.
    """

    if minimum_seed_count < 1:
        raise ValueError("minimum_seed_count must be positive")
    x_values, y_values, z_values = np.broadcast_arrays(
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
        np.asarray(z, dtype=float),
    )
    if x_values.ndim != 1:
        raise ValueError("seed coordinates must be one-dimensional arrays")

    target = np.asarray(target_density, dtype=float)
    if target.shape != grid.shape:
        raise ValueError(
            f"target_density has shape {target.shape}; expected {grid.shape} (R,z,phi)"
        )
    if np.any(np.isfinite(target) & (target < 0)):
        raise ValueError("target_density cannot contain negative values")

    radius = np.hypot(x_values, y_values)
    phi = np.arctan2(y_values, x_values)
    ir, iz, iphi, in_grid = grid.bin_indices(radius, z_values, phi)
    seed_counts_float = grid.histogram(radius, z_values, phi)
    seed_counts = np.asarray(np.rint(seed_counts_float), dtype=np.int64)

    finite_target = np.isfinite(target) & (target >= 0)
    target_mass = np.where(finite_target, target * grid.volumes, 0.0)
    supported = finite_target & (seed_counts >= minimum_seed_count)
    cell_weight = np.divide(target_mass, seed_counts, out=np.zeros_like(target_mass), where=supported)

    weights = np.zeros(x_values.size, dtype=float)
    valid_seed = in_grid & supported[
        np.clip(ir, 0, grid.shape[0] - 1),
        np.clip(iz, 0, grid.shape[1] - 1),
        np.clip(iphi, 0, grid.shape[2] - 1),
    ]
    weights[valid_seed] = cell_weight[
        ir[valid_seed],
        iz[valid_seed],
        iphi[valid_seed],
    ]

    positive_target = finite_target & (target > 0)
    assigned_mass = float(np.sum(target_mass[supported]))
    positive_mass = float(np.sum(target_mass[positive_target]))
    unsupported_mass = float(np.sum(target_mass[positive_target & ~supported]))
    return RepresentativeWeightResult(
        weights=weights,
        seed_counts=seed_counts,
        cell_weight=cell_weight,
        supported_cells=supported,
        target_mass=target_mass,
        assigned_mass=assigned_mass,
        positive_target_mass=positive_mass,
        unsupported_positive_mass=unsupported_mass,
        weighted_seed_count=int(np.count_nonzero(weights > 0)),
        in_grid_seed_count=int(np.count_nonzero(in_grid)),
    )


from dataclasses import dataclass
import hashlib
import time
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import DensityFitSettings, WeightModelSettings
from .density import density_fit_mask
from .density import OrbitDensityResponse


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
    solver_backend: str = "unknown"
    kkt_residual: float = np.inf
    solve_wall_seconds: float = 0.0
    problem_fingerprint: str = ""


@dataclass(frozen=True)
class _WeightProblem:
    """One error-scaled density NNLS problem shared by every backend."""

    design: Any
    observed: FloatArray
    active_columns: NDArray[np.bool_]
    successful_orbit_count: int
    regularization: float
    fingerprint: str


@dataclass(frozen=True)
class _BackendResult:
    weights: FloatArray
    success: bool
    status: int
    message: str
    iterations: int
    optimality: float


def _normalized_target(
    target_density: FloatArray,
    target_error: FloatArray,
    fit_mask: NDArray[np.bool_],
    response: OrbitDensityResponse,
    mode: str,
) -> tuple[FloatArray, FloatArray]:
    if mode == "absolute":
        return target_density.copy(), target_error.copy()
    mass = float(np.sum(target_density[fit_mask] * response.grid.volumes[fit_mask]))
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("target density has no positive mass in the fit region")
    return target_density / mass, target_error / mass


def _problem_fingerprint(
    design: Any,
    observed: FloatArray,
    regularization: float,
) -> str:
    """Hash the exact CSR problem passed to alternative solver backends."""

    matrix = design.tocsr(copy=True)
    matrix.sum_duplicates()
    matrix.sort_indices()
    digest = hashlib.sha256()
    digest.update(np.asarray(matrix.shape, dtype=np.int64).tobytes())
    for array in (matrix.indptr, matrix.indices, matrix.data, observed):
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    digest.update(np.asarray(regularization, dtype=np.float64).tobytes())
    return digest.hexdigest()


def _build_weight_problem(
    response: OrbitDensityResponse,
    target: FloatArray,
    error: FloatArray,
    fit_mask: NDArray[np.bool_],
    regularization: float,
) -> _WeightProblem:
    row_mask = fit_mask.reshape(-1)
    full_design = response.matrix[row_mask]
    # Orbits that never visit a fitted cell cannot constrain the target.
    # Drop those columns from every backend and restore zero weight afterwards.
    active_columns = np.asarray(full_design.sum(axis=0) > 0).ravel()
    if not np.any(active_columns):
        raise ValueError("no orbit responds inside the density fit region")
    inverse_error = 1.0 / error.reshape(-1)[row_mask]
    design = full_design[:, active_columns].multiply(inverse_error[:, None]).tocsr()
    observed = np.asarray(target.reshape(-1)[row_mask] * inverse_error, dtype=float)
    return _WeightProblem(
        design=design,
        observed=observed,
        active_columns=active_columns,
        successful_orbit_count=int(response.successful_seed_index.size),
        regularization=regularization,
        fingerprint=_problem_fingerprint(design, observed, regularization),
    )


def _augmented_sparse_problem(problem: _WeightProblem) -> tuple[Any, FloatArray]:
    if problem.regularization <= 0:
        return problem.design, problem.observed
    from scipy.sparse import eye, vstack

    n_weights = problem.design.shape[1]
    design = vstack([problem.design, np.sqrt(problem.regularization) * eye(n_weights, format="csr")], format="csr")
    observed = np.concatenate([problem.observed, np.zeros(n_weights)])
    return design, observed


def _solve_lsq_linear(
    problem: _WeightProblem,
    settings: WeightModelSettings,
) -> _BackendResult:
    from scipy.optimize import lsq_linear

    design, observed = _augmented_sparse_problem(problem)
    result = lsq_linear(
        design,
        observed,
        bounds=(0.0, np.inf),
        method="trf",
        lsq_solver="lsmr",
        lsmr_tol=settings.lsmr_tol if settings.lsmr_tol is not None else "auto",
        max_iter=int(settings.max_iter),
    )
    return _BackendResult(
        weights=np.asarray(result.x, dtype=float),
        success=bool(result.success),
        status=int(result.status),
        message=str(result.message),
        iterations=int(result.nit),
        optimality=float(result.optimality),
    )


def _solve_dense_nnls(
    problem: _WeightProblem,
    settings: WeightModelSettings,
) -> _BackendResult:
    from scipy.optimize import nnls

    n_rows, n_weights = problem.design.shape
    regularized = problem.regularization > 0
    dense_rows = n_rows + (n_weights if regularized else 0)
    # Allocate the augmented matrix only once. Constructing a separate dense
    # identity would add another O(n_weights**2) temporary allocation.
    design = np.zeros((dense_rows, n_weights), dtype=float, order="F")
    design[:n_rows, :] = problem.design.toarray()
    observed = np.zeros(dense_rows, dtype=float)
    observed[:n_rows] = problem.observed
    if regularized:
        diagonal = design[n_rows:, :]
        diagonal.flat[:: n_weights + 1] = np.sqrt(problem.regularization)
    try:
        weights, _ = nnls(
            design,
            observed,
            maxiter=int(settings.max_iter),
        )
    except RuntimeError as exc:
        return _BackendResult(
            weights=np.zeros(n_weights, dtype=float),
            success=False,
            status=0,
            message=f"dense NNLS failed: {exc}",
            iterations=int(settings.max_iter),
            optimality=np.inf,
        )
    return _BackendResult(
        weights=np.asarray(weights, dtype=float),
        success=True,
        status=1,
        message="dense SciPy NNLS completed",
        iterations=0,
        optimality=np.nan,
    )


def _primal_kkt_residual(
    problem: _WeightProblem,
    weights: FloatArray,
) -> tuple[float, float]:
    residual = problem.design @ weights - problem.observed
    gradient = np.asarray(problem.design.T @ residual + problem.regularization * weights, dtype=float)
    threshold = float(np.max(weights)) * 1e-12 if weights.size else 0.0
    positive = weights > threshold
    violation = 0.0
    if np.any(positive):
        violation = float(np.max(np.abs(gradient[positive])))
    if np.any(~positive):
        violation = max(violation, float(np.max(np.maximum(-gradient[~positive], 0.0))))
    reference = np.asarray(problem.design.T @ problem.observed, dtype=float)
    scale = max(1.0, float(np.max(np.abs(reference))))
    return violation, violation / scale


def _dual_value_gradient_weights(
    problem: _WeightProblem,
    dual: FloatArray,
) -> tuple[float, FloatArray, FloatArray]:
    projected = np.maximum(-(problem.design.T @ dual), 0.0)
    weights = np.asarray(projected / problem.regularization, dtype=float)
    value = (0.5 * float(np.dot(dual, dual)) + float(np.dot(problem.observed, dual))
             + 0.5 / problem.regularization * float(np.dot(projected, projected)))
    gradient = np.asarray(dual + problem.observed - problem.design @ weights, dtype=float)
    return value, gradient, weights


def _solve_dual_ridge(
    problem: _WeightProblem,
    settings: WeightModelSettings,
) -> _BackendResult:
    from scipy.linalg import LinAlgError, cho_factor, cho_solve

    if problem.regularization <= 0:
        raise ValueError("dual_ridge requires positive L2 regularization")
    n_observations = problem.design.shape[0]
    dual = np.zeros(n_observations, dtype=float)
    tolerance = float(settings.solver_tolerance)
    dual_scale = max(1.0, float(np.max(np.abs(problem.observed))))
    message = "dual ridge reached max_iter"
    status = 0
    success = False
    iterations = 0
    optimality = np.inf
    weights = np.zeros(problem.design.shape[1], dtype=float)

    for iteration in range(1, int(settings.max_iter) + 1):
        value, gradient, weights = _dual_value_gradient_weights(problem, dual)
        raw_kkt, normalized_kkt = _primal_kkt_residual(problem, weights)
        normalized_dual_gradient = float(np.max(np.abs(gradient))) / dual_scale
        optimality = raw_kkt
        iterations = iteration - 1
        if normalized_kkt <= tolerance and normalized_dual_gradient <= tolerance:
            success = True
            status = 1
            message = "dual ridge converged by primal KKT and dual gradient"
            break

        active = np.asarray(-(problem.design.T @ dual) > 0).ravel()
        hessian = np.eye(n_observations, dtype=float)
        if np.any(active):
            active_design = problem.design[:, active]
            gram = active_design @ active_design.T
            hessian += gram.toarray() / problem.regularization
        try:
            factor = cho_factor(hessian, lower=True, check_finite=False)
            direction = cho_solve(factor, -gradient, check_finite=False)
        except (LinAlgError, ValueError) as exc:
            message = f"dual ridge Newton factorization failed: {exc}"
            status = -1
            break

        slope = float(np.dot(gradient, direction))
        if not np.isfinite(slope) or slope >= 0:
            direction = -gradient
            slope = -float(np.dot(gradient, gradient))
        step = 1.0
        accepted = False
        for _ in range(60):
            candidate = dual + step * direction
            candidate_value, _, _ = _dual_value_gradient_weights(problem, candidate)
            if np.isfinite(candidate_value) and candidate_value <= (value + 1e-4 * step * slope):
                dual = candidate
                accepted = True
                break
            step *= 0.5
        if not accepted:
            message = "dual ridge Armijo line search failed"
            status = -2
            break
    else:
        iterations = int(settings.max_iter)

    # Recompute from the accepted final dual point, including failure paths.
    _, _, weights = _dual_value_gradient_weights(problem, dual)
    raw_kkt, _ = _primal_kkt_residual(problem, weights)
    return _BackendResult(
        weights=weights,
        success=success,
        status=status,
        message=message,
        iterations=iterations,
        optimality=raw_kkt,
    )


def _solve_backend(
    problem: _WeightProblem,
    settings: WeightModelSettings,
) -> _BackendResult:
    try:
        if settings.solver == "lsq_linear":
            return _solve_lsq_linear(problem, settings)
        if settings.solver == "dense_nnls":
            return _solve_dense_nnls(problem, settings)
        if settings.solver == "dual_ridge":
            return _solve_dual_ridge(problem, settings)
    except ImportError as exc:
        raise RuntimeError(
            "SciPy is required for density-solved orbit weights"
        ) from exc
    raise ValueError(f"unsupported density weight solver: {settings.solver!r}")


def solve_density_weights(
    response: OrbitDensityResponse,
    target_density: ArrayLike,
    target_error: ArrayLike,
    density_fit: DensityFitSettings,
    settings: WeightModelSettings,
) -> WeightSolution:
    """Solve one configured non-negative density-weight problem."""

    if settings.mode != "density_solved":
        raise ValueError("solve_density_weights requires density_solved settings")
    target = np.asarray(target_density, dtype=float)
    error = np.asarray(target_error, dtype=float)
    if target.shape != response.grid.shape or error.shape != response.grid.shape:
        raise ValueError(
            "target density and error must match the orbit-response grid"
        )
    fit_mask = density_fit_mask(target, error, response.grid, density_fit)
    target, error = _normalized_target(target, error, fit_mask, response, settings.target_normalization or "absolute")
    regularization = float(settings.regularization_strength)
    problem = _build_weight_problem(response, target, error, fit_mask, regularization)

    started = time.perf_counter()
    backend = _solve_backend(problem, settings)
    solve_wall_seconds = time.perf_counter() - started
    active_weights = np.asarray(backend.weights, dtype=float)
    finite_nonnegative = np.all(np.isfinite(active_weights)) and np.all(active_weights >= 0)
    raw_kkt, normalized_kkt = _primal_kkt_residual(problem, active_weights)
    tolerance = float(settings.solver_tolerance)
    if settings.solver == "lsq_linear":
        # Preserve the historical SciPy convergence contract. The normalized
        # KKT residual is persisted for comparison but does not silently turn
        # old production runs into invalid trials.
        converged = backend.success and finite_nonnegative
    else:
        converged = backend.success and finite_nonnegative and normalized_kkt <= tolerance

    successful_weights = np.zeros(problem.successful_orbit_count, dtype=float)
    successful_weights[problem.active_columns] = active_weights
    seed_weights = np.zeros(response.seed_count, dtype=float)
    seed_weights[response.successful_seed_index] = successful_weights
    model_density = response.model_density(seed_weights)
    total = float(np.sum(seed_weights))
    squared = float(np.dot(seed_weights, seed_weights))
    maximum_fraction = float(np.max(seed_weights)) / total if total > 0 else 0.0
    active_threshold = max(float(np.max(seed_weights)) * 1e-12, 0.0)
    regularization_penalty = regularization * squared
    density_residual = problem.design @ active_weights - problem.observed
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
        active_orbit_count=int(np.count_nonzero(seed_weights > active_threshold)),
        converged=converged,
        status=backend.status,
        message=backend.message,
        iterations=backend.iterations,
        optimality=backend.optimality if np.isfinite(backend.optimality) else raw_kkt,
        solver_cost=0.5 * inner_objective,
        solver_backend=str(settings.solver),
        kkt_residual=normalized_kkt,
        solve_wall_seconds=solve_wall_seconds,
        problem_fingerprint=problem.fingerprint,
    )
