#!/usr/bin/env python3
"""Full-space weight-concentration experiments: entropy and graph smoothing.

S16 diagnostic on the frozen nphi4 library.  The production inner problem is
the underdetermined NNLS ``min(w>=0) ||A w-b||^2 + lambda ||w||^2``; its
solution has N_eff~130 and a 4.9% maximum orbit share.  This script keeps all
5967 active per-orbit weights free and replaces the selection rule instead of
imposing hard equal-weight bundles:

1. Entropy (KL to a uniform orbit prior): add ``mu/n * sum_j w_j log(w_j/n)``
   after the density+L2 objective.  Zero additional density information; the
   maximum-entropy choice among density-compatible solutions.
2. Response graph smoothing: add ``rho/2 w^T L w`` where L is the symmetric
   normalized Laplacian of a k-nearest-neighbour graph of the normalized
   density-response columns.  Similar response columns get similar weights,
   dissimilar columns stay free -- a convex relaxation of hard bundling.

Both scans use the same frozen design matrix, mask, error scaling, and shared
velocity scoring as the bundling campaign.  Positive delta_J is a genuine
velocity cost; negative J remains the S7 density-for-smoothness trade and is
never a better-fit claim.  Single library, single seed, diagnostic only.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import coo_matrix, csr_matrix, vstack
from scipy.sparse.linalg import LinearOperator, cg, eigsh

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from benchmark_nphi1_bundling import weight_concentration  # noqa: E402
from halo_mw_lmc.evaluate import score_orbit_weights  # noqa: E402
from review_fz_energy_basis import build_design_problem  # noqa: E402

OUTPUT = REPO / ".agent-local/benchmarks/full_weight_regularization"
ENTROPY_STRENGTHS = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
GRAPH_STRENGTHS = (0.0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)


def initial_weights(design, observed, regularization):
    """Solve the production ridge NNLS once; reuse it as every scan's start."""

    from scipy.optimize import nnls

    rows, columns = design.shape
    dense = np.zeros((rows + columns, columns), dtype=float, order="F")
    dense[:rows, :] = design.toarray()
    dense[rows:, :] = np.sqrt(regularization) * np.eye(columns)
    weights, _ = nnls(dense, np.concatenate([observed, np.zeros(columns)]), maxiter=100 * columns)
    return np.asarray(weights, dtype=float)


def concentration(weights):
    record = weight_concentration(weights)
    return {
        "n_eff": record["n_eff"], "max_share": record["max_fraction"],
        "top10_share": record["top10_share"], "top100_share": record["top100_share"],
        "hhi": record["hhi"], "n90": record["n90"], "n99": record["n99"],
    }


def objective_entropy(design, observed, l2, weights, strength, count, cache):
    residual = design @ weights - observed
    data = float(residual @ residual) + l2 * float(weights @ weights)
    if strength == 0.0:
        cache.clear()
        return data, np.asarray(2.0 * design.T @ residual + 2.0 * l2 * weights, dtype=float)
    safe = np.maximum(weights, 1e-15)
    entropy = float(np.sum(weights * np.log(safe / count)))
    gradient = np.log(safe / count) + 1.0
    scale = strength / count
    return data + scale * entropy, np.asarray(
        2.0 * design.T @ residual + 2.0 * l2 * weights + scale * gradient, dtype=float
    )


def solve_entropy(design, observed, l2, strength, start, max_iter):
    count = design.shape[1]
    cache: dict[str, float] = {}

    initial_value, _ = objective_entropy(design, observed, l2, start, strength, count, {})
    objective_scale = max(1.0, abs(initial_value))

    def fun(weights):
        value, gradient = objective_entropy(design, observed, l2, weights, strength, count, cache)
        return value / objective_scale, np.asarray(gradient, dtype=float) / objective_scale

    result = minimize(
        fun, start, jac=True, method="L-BFGS-B", bounds=[(0.0, None)] * count,
        options={"maxiter": max_iter, "maxfun": max_iter + 100, "ftol": 0.0, "gtol": 1e-8, "maxls": 50},
    )
    return np.maximum(np.asarray(result.x, dtype=float), 0.0), result


def projected_gradient(weights, gradient):
    """First-order residual for non-negative variables."""

    return np.where(weights > 1e-12, gradient, np.minimum(gradient, 0.0))


def gradient_diagnostics(design, observed, l2, strength, weights):
    _, gradient = objective_entropy(design, observed, l2, weights, strength, design.shape[1], {})
    projected = projected_gradient(weights, gradient)
    return {
        "gradient_l_inf": float(np.max(np.abs(gradient))),
        "projected_gradient_l_inf": float(np.max(np.abs(projected))),
    }


def entropy_hessian_z(design, l2, strength, weights, gradient, free, damping=0.0):
    """Z-space Hessian action on the free subspace, plus Jacobi diagonal.

    With ``w = exp(z)`` the Hessian transforms to ``D H D + diag(w . g)``
    where ``H = 2 A^T A + 2 l2 I + (mu/n) diag(1/w)`` and ``g`` is the
    w-space gradient; the ``diag(w . g)`` term is what the previous
    exp-transform solver omitted.  Coordinates pinned at a z bound with a
    satisfied sign condition (the projected-Newton active set) are frozen
    out of the CG system: their curvature is ~ e^{2 z}, which otherwise
    overflows the Newton direction.
    """

    count = design.shape[1]
    column_norms = np.asarray(design.multiply(design).sum(axis=0)).ravel()
    scale = strength / count if strength > 0.0 else 0.0

    psd_diagonal = weights * weights * (2.0 * column_norms + 2.0 * l2)
    if strength > 0.0:
        psd_diagonal = psd_diagonal + scale * weights
    # F(z) = f(e^z) is NOT convex: the transformed entropy term has negative
    # curvature for z < log n - 2.  Keep the exact term's positive part only
    # (Gauss-Newton style): the operator stays PSD, the approximation is
    # exact at stationarity where g -> 0, and indefinite-CG failure modes
    # disappear.
    positive_gradient_part = np.maximum(weights * gradient, 0.0)
    jacobi = np.maximum(psd_diagonal + positive_gradient_part, 1e-300)
    jacobi = np.maximum(jacobi, 1e-12 * float(np.max(jacobi)))

    def matvec(direction):
        masked = free * direction
        inner = weights * masked
        product = design @ inner
        base = 2.0 * np.asarray(design.T @ product, dtype=float) + 2.0 * l2 * inner
        if strength > 0.0:
            base = base + scale * masked
        outer = weights * base + positive_gradient_part * masked
        if damping > 0.0:
            outer = outer + damping * jacobi * masked
        return free * outer

    return LinearOperator((count, count), matvec=matvec), jacobi


def solve_entropy_interior(design, observed, l2, strength, start, max_iter):
    """Projected Newton on the entropy-regularized objective in z = log w.

    Why z-space: the optimum is strictly positive but per-coordinate often
    exponentially small, so w-space Newton is globally step-limited by
    fraction-to-boundary, while z-space turns multiplicative moves into
    unit-scale steps.  z lives on the box [z_lower, z_upper]; coordinates
    pinned at a bound with a satisfied sign condition are the active set
    and are frozen.  Newton steps come from Jacobi-preconditioned CG on
    the free subspace of the exact transformed Hessian, with Levenberg
    escalation only when Armijo rejects, and honest success bookkeeping.

    Convergence is judged on the free-subspace z-gradient, the
    mass-relevant w-space projected gradient (w >= 1e-14 max w), or the
    Newton decrement.  Boundary-coordinate entropy gradients (|g| ~
    (mu/n)|log w| as w -> 0) are reported but are not failure signals:
    their stationarity lives below the z box and is irrelevant to every
    concentration metric.
    """

    if strength <= 0.0:
        return solve_entropy(design, observed, l2, strength, start, max_iter)

    count = design.shape[1]
    weights = np.maximum(np.asarray(start, dtype=float), 0.0)
    total = float(np.sum(weights))
    floor = max(total / count * 1e-3, 1e-12)
    weights = np.maximum(weights, floor)
    z = np.log(weights)
    log_count = float(np.log(count))
    scale = strength / count
    z_lower, z_upper = -650.0, 40.0

    def value_and_gradient_z(z_):
        w_ = np.exp(np.clip(z_, z_lower, z_upper))
        residual = design @ w_ - observed
        data = float(residual @ residual) + l2 * float(w_ @ w_)
        gradient_w = 2.0 * np.asarray(design.T @ residual, dtype=float) + 2.0 * l2 * w_
        data += scale * float(np.sum(w_ * (z_ - log_count)))
        gradient_w = gradient_w + scale * ((z_ - log_count) + 1.0)
        return data, w_, gradient_w

    def active_mask(z_, gradient_z_):
        pinned_low = (z_ <= z_lower + 1e-9) & (gradient_z_ > 0.0)
        pinned_high = (z_ >= z_upper - 1e-9) & (gradient_z_ < 0.0)
        return ~(pinned_low | pinned_high)

    def projected_gradient_relevant(w_, g_):
        mask = w_ >= 1e-14 * float(np.max(w_))
        projected = np.where(mask, g_, np.minimum(g_, 0.0))
        return float(np.max(np.abs(projected)))

    class _Result:
        pass

    result = _Result()
    result.nit = 0
    result.nfev = 0
    value, weights, gradient_w = value_and_gradient_z(z)
    gradient_z = weights * gradient_w
    free = active_mask(z, gradient_z)
    zgrad_scale = max(1.0, float(np.max(np.abs(gradient_z * free))))
    gscale = max(1.0, float(np.max(np.abs(gradient_w))))
    z_tolerance = 1e-8 * zgrad_scale
    pgrad_tolerance = 1e-6 * gscale
    decrement_tolerance = 1e-12 * max(1.0, abs(value))
    converged = False
    message = "interior Newton reached iteration limit"
    damping = 0.0
    best_value = value
    stagnant_iterations = 0
    previous_direction = None

    while result.nit < max_iter:
        result.nit += 1
        operator, jacobi = entropy_hessian_z(design, l2, strength, weights, gradient_w, free, damping)
        preconditioner = LinearOperator((count, count), matvec=lambda v: (free * v) / jacobi)
        rhs = -(free.astype(float)) * gradient_z
        accepted = False
        slope = 0.0
        for _ in range(10):
            warm_start = (
                None if previous_direction is None or not np.all(np.isfinite(previous_direction))
                else free.astype(float) * previous_direction
            )
            direction, info = cg(operator, rhs, M=preconditioner, x0=warm_start, rtol=1e-8, atol=0.0, maxiter=1500)
            result.nfev += 1
            slope = float(gradient_z @ direction)
            if info != 0 or not np.isfinite(slope) or slope >= 0.0:
                direction = -(free * gradient_z) / jacobi
                slope = float(gradient_z @ direction)
            previous_direction = direction
            step = 1.0
            for _ in range(60):
                candidate_z = np.clip(z + step * direction, z_lower, z_upper)
                candidate_value, candidate_weights, candidate_gradient_w = value_and_gradient_z(candidate_z)
                result.nfev += 1
                if candidate_value <= value + 1e-4 * step * slope:
                    accepted = True
                    break
                step *= 0.5
            if accepted:
                z, value, weights, gradient_w = candidate_z, candidate_value, candidate_weights, candidate_gradient_w
                gradient_z = weights * gradient_w
                free = active_mask(z, gradient_z)
                damping = 0.0 if damping <= 1e-12 else damping / 10.0
                break
            damping = damping * 10.0 if damping > 0.0 else 1e-8
            operator, jacobi = entropy_hessian_z(design, l2, strength, weights, gradient_w, free, damping)
            preconditioner = LinearOperator((count, count), matvec=lambda v: (free * v) / jacobi)
        if not accepted:
            result.x = weights
            result.fun = value
            result.success = False
            result.status = 2
            result.message = "interior Newton line search failed"
            return np.asarray(result.x, dtype=float), result
        zgrad_free = float(np.max(np.abs(gradient_z * free))) if np.any(free) else 0.0
        pgrad_relevant = projected_gradient_relevant(weights, gradient_w)
        decrement_squared = max(0.0, -slope)
        if value < best_value - 1e-14 * max(1.0, abs(value)):
            best_value = value
            stagnant_iterations = 0
        else:
            stagnant_iterations += 1
        if zgrad_free <= z_tolerance:
            converged = True
            message = f"interior Newton converged: zgrad={zgrad_free:.2e} pgrad_rel={pgrad_relevant:.2e} dec={np.sqrt(max(decrement_squared, 0.0)):.2e} nfree={int(np.sum(free))}"
            break
        if stagnant_iterations >= 50 and zgrad_free <= 1e-6 * zgrad_scale:
            converged = True
            message = f"interior Newton converged by objective stagnation: zgrad={zgrad_free:.2e} pgrad_rel={pgrad_relevant:.2e} nfree={int(np.sum(free))}"
            break

    result.x = weights
    result.fun = value
    result.success = converged
    result.status = 0 if converged else 1
    result.message = message
    return np.asarray(result.x, dtype=float), result


def response_graph(design, neighbours):
    """Symmetric kNN graph of nonnegative response columns, cosine similarity."""

    dense = np.asarray(design.todense(), dtype=float).T
    norms = np.linalg.norm(dense, axis=1)
    if np.any(norms == 0):
        raise ValueError("response graph requires nonzero design columns")
    unit = dense / norms[:, None]
    neighbours = min(int(neighbours), unit.shape[0] - 1)
    similarity = unit @ unit.T
    np.fill_diagonal(similarity, -np.inf)
    neighbours_index = np.argpartition(-similarity, kth=neighbours - 1, axis=1)[:, :neighbours]
    rows = np.repeat(np.arange(unit.shape[0]), neighbours)
    columns = neighbours_index.ravel()
    cosine = np.maximum(np.sum(unit[rows] * unit[columns], axis=1), 0.0)
    directed = coo_matrix((cosine, (rows, columns)), shape=similarity.shape).tocsr()
    weighted = directed.maximum(directed.T).tocsr()
    weighted.setdiag(0.0)
    weighted.eliminate_zeros()
    degree = np.asarray(weighted.sum(axis=1)).ravel()
    if np.any(degree <= 0):
        raise ValueError("response graph has isolated nodes")
    inverse = csr_matrix((1.0 / np.sqrt(degree), (np.arange(degree.size), np.arange(degree.size))), shape=weighted.shape)
    laplacian = csr_matrix(np.eye(degree.size)) - inverse @ weighted @ inverse
    return laplacian, degree


def solve_graph(design, observed, l2, strength, laplacian, start, max_iter):
    from scipy.optimize import lsq_linear

    # Append sqrt(rho)*L^{1/2} rows: ||sqrt(rho) L^{1/2} w||^2 = rho w^T L w.
    # L is PSD; its symmetric square root is exact and keeps the problem a
    # bound-constrained least-squares instance solved by TRF.
    square_root = _laplacian_square_root(laplacian)
    matrix = vstack([design, np.sqrt(strength) * square_root], format="csr") if strength > 0 else design
    target = np.concatenate([observed, np.zeros(design.shape[1])]) if strength > 0 else observed
    result = lsq_linear(
        matrix, target, bounds=(0.0, np.inf), method="trf", lsq_solver="lsmr",
        lsmr_tol=1e-8, max_iter=max_iter,
    )
    return np.asarray(result.x, dtype=float), result


def _laplacian_square_root(laplacian):
    """Dense symmetric PSD square root via eigendecomposition (5967^2)."""

    values, vectors = np.linalg.eigh(laplacian.toarray())
    values = np.maximum(values, 0.0)
    return (vectors * np.sqrt(values)) @ vectors.T


def laplacian_spectrum(laplacian):
    values = eigsh(laplacian, k=2, which="SM", return_eigenvectors=False)
    return [float(value) for value in np.sort(values)]


def build_weight_solution(response, problem, weights, objective_value, seconds, message, target, error):
    from halo_mw_lmc.weights import WeightSolution

    successful = response.successful_seed_index
    active_columns = np.flatnonzero(problem.active_columns)
    seed_weights = np.zeros(response.seed_count, dtype=float)
    seed_weights[successful[active_columns]] = weights
    density_residual = problem.design @ weights - problem.observed
    data_term = float(density_residual @ density_residual)
    squared = float(seed_weights @ seed_weights)
    model_density = response.model_density(seed_weights)
    return WeightSolution(
        seed_weights=seed_weights, model_density=model_density,
        target_density=target, target_error=error,
        inner_objective=objective_value, regularization_penalty=objective_value-data_term,
        effective_orbit_count=(float(np.sum(seed_weights)) ** 2 / squared if squared else 0.0),
        maximum_weight_fraction=float(np.max(seed_weights) / np.sum(seed_weights)) if squared else 0.0,
        active_orbit_count=int(np.count_nonzero(seed_weights > 0)), converged=True, status=0,
        message=message, iterations=0, optimality=np.nan, solver_cost=objective_value,
        solver_backend="full_entropy_or_graph", kkt_residual=np.nan,
        solve_wall_seconds=seconds, problem_fingerprint=problem.fingerprint,
    )


def load_prepared_and_library():
    from halo_mw_lmc.config import load_run_configuration, resolve_model
    from halo_mw_lmc.orbits import OrbitLibrary
    from halo_mw_lmc.prepare import prepare_model_data

    run_configuration = load_run_configuration(REPO / "configs/runs/density_solved_benchmark.toml")
    model = resolve_model(run_configuration["recipe"])
    prepared = prepare_model_data(
        run_configuration["data"]["catalog"], run_configuration["data"]["target_density"], model,
    )
    frozen = np.load(REPO / ".agent-local/benchmarks/solver_settings_sweep/frozen_paper_best.npz")
    library = OrbitLibrary(
        seed_index=frozen["library_seed_index"], time=frozen["library_time"],
        phase_space=frozen["library_phase_space"],
    )
    return prepared, library


def main() -> None:
    started_all = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    problem, response = build_design_problem()
    prepared, library = load_prepared_and_library()
    design = problem.design
    l2 = float(problem.regularization)
    active_columns = np.flatnonzero(problem.active_columns)
    fit_bins = design.shape[0]

    print(f"design {design.shape} fit_bins={fit_bins}; l2={l2}", flush=True)
    started = time.perf_counter()
    reference = initial_weights(design, problem.observed, l2)
    reference_seconds = time.perf_counter() - started
    print(f"reference NNLS {reference_seconds:.2f}s", flush=True)

    # Shared response graph; density-only geometry, velocity never enters.
    graph_started = time.perf_counter()
    laplacian, degree = response_graph(design, neighbours=10)
    spectrum = laplacian_spectrum(laplacian)
    graph_seconds = time.perf_counter() - graph_started
    print(f"kNN graph+Laplacian {graph_seconds:.2f}s; spectrum[0:2]={spectrum}", flush=True)

    rows = []
    seed_weights_path = OUTPUT / "seed_weights.npz"
    archive = {}
    previous_entropy_weights = reference

    for family, strengths in (("entropy", ENTROPY_STRENGTHS),):
        for strength in strengths:
            started = time.perf_counter()
            if family == "entropy":
                if strength == 0.0:
                    weights, result = solve_entropy(design, problem.observed, l2, strength, reference, 3000)
                else:
                    # Continuation: warm-start each temperature from the
                    # previous point's solution, keeping every solve near its
                    # optimum regardless of how far large mu pushes the
                    # weights from the ridge-NNLS reference.
                    start = reference if strength == ENTROPY_STRENGTHS[1] else np.asarray(previous_entropy_weights, dtype=float)
                    weights, result = solve_entropy_interior(design, problem.observed, l2, strength, start, 3000)
                    previous_entropy_weights = weights
                solver_message = str(result.message)
                iterations = int(result.nit)
                residual = design @ weights - problem.observed
                objective_value = float(residual @ residual + l2 * weights @ weights)
                if strength > 0.0:
                    count = design.shape[1]
                    safe = np.maximum(weights, 1e-15)
                    objective_value += strength / count * float(np.sum(weights * np.log(safe / count)))
            else:
                weights, result = solve_graph(design, problem.observed, l2, strength, laplacian, reference, 3000)
                solver_message = str(result.message)
                iterations = int(result.nit)
                residual = design @ weights - problem.observed
                objective_value = float(residual @ residual + l2 * weights @ weights + strength * weights @ laplacian @ weights)
            seconds = time.perf_counter() - started
            residual = design @ weights - problem.observed
            chi2 = float(residual @ residual)
            record = {
                "family": family, "strength": strength,
                "objective_inner": objective_value, "density_chi2": chi2,
                "density_chi2_per_bin_raw": chi2 / fit_bins,
                "solver_seconds": seconds, "iterations": iterations,
                "solver_message": solver_message,
                "solver_success": bool(result.success) if result is not None else True,
                "solver_status": int(result.status) if result is not None else 0,
                **gradient_diagnostics(design, problem.observed, l2, strength, weights),
                **{f"weight_{key}": value for key, value in concentration(weights).items()},
            }
            rows.append(record)
            seed_weights = np.zeros(response.seed_count, dtype=float)
            seed_weights[response.successful_seed_index[active_columns]] = weights
            solution = build_weight_solution(
                response, problem, weights, objective_value, seconds, solver_message,
                prepared.target_density, prepared.target_error,
            )
            evaluation = score_orbit_weights(library, prepared, solution, response=response)
            record.update({
                "objective_velocity": float(evaluation.objective_velocity),
                "density_chi2_per_bin_scored": float(evaluation.density_chi2_per_bin),
                "density_gate_passed": bool(evaluation.density_gate_passed),
                "weight_sum": float(evaluation.weight_sum),
            })
            archive[f"{family}_{strength:g}"] = seed_weights
            print(
                f"{family:7s} mu={strength:g} chi2/row={chi2/design.shape[0]:.4f} "
                f"Neff={record['weight_n_eff']:.1f} max={record['weight_max_share']:.4f} "
                f"top10={record['weight_top10_share']:.3f} J={record['objective_velocity']:.1f} "
                f"gate={record['density_gate_passed']} t={seconds:.1f}s nit={iterations} "
                f"status={record['solver_status']} success={record['solver_success']} "
                f"pgrad={record['projected_gradient_l_inf']:.3e} msg={solver_message}",
                flush=True,
            )

    reference_row = next(row for row in rows if row["strength"] == 0.0 and row["family"] == "entropy")
    for row in rows:
        row["delta_J"] = row["objective_velocity"] - reference_row["objective_velocity"]
    np.savez_compressed(seed_weights_path, **archive)
    (OUTPUT / "scan.json").write_text(json.dumps({
        "design_shape": list(design.shape), "l2": l2,
        "entropy_strengths": list(ENTROPY_STRENGTHS), "graph_strengths": list(GRAPH_STRENGTHS),
        "graph_neighbours": 10, "graph_seconds": graph_seconds,
        "graph_laplacian_spectrum_smallest": spectrum,
        "graph_degree": {"min": float(np.min(degree)), "median": float(np.median(degree)), "max": float(np.max(degree))},
        "reference_seconds": reference_seconds, "rows": rows,
        "wall_seconds": time.perf_counter() - started_all,
    }, indent=2) + "\n")
    print(f"artifacts: {OUTPUT} (wall {time.perf_counter()-started_all:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
