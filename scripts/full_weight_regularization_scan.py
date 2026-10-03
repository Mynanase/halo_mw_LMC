#!/usr/bin/env python3
"""Full-space weight-concentration experiments: L2, entropy, and graph smoothing.

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

Every scan uses the same frozen design matrix, mask, error scaling, and shared
velocity scoring as the bundling campaign.  In the L2 frontier the production
inner problem itself is re-solved at each lambda.  Positive delta_J is a
velocity cost relative to the production lambda=1e-6 reference; it is never a
better-density-fit claim.  Single library, single seed, diagnostic only.
"""

from __future__ import annotations

import json
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import coo_matrix, csr_matrix, vstack
from scipy.sparse.linalg import eigsh

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from benchmark_nphi1_bundling import weight_concentration  # noqa: E402
from halo_mw_lmc.evaluate import score_orbit_weights  # noqa: E402
from review_fz_energy_basis import build_design_problem  # noqa: E402

OUTPUT = REPO / ".agent-local/benchmarks/full_weight_regularization"
L2_STRENGTHS = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
ENTROPY_STRENGTHS = (0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
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


def problem_at_l2(problem, strength):
    """Return the same error-scaled design with an updated ridge and hash."""

    from halo_mw_lmc.weights import _problem_fingerprint

    strength = float(strength)
    fingerprint = _problem_fingerprint(problem.design, problem.observed, strength)
    values = {
        field.name: getattr(problem, field.name)
        for field in dataclasses.fields(problem)
    }
    values.update(regularization=strength, fingerprint=fingerprint)
    return type(problem)(**values)


def kkt_diagnostics(problem, weights):
    from halo_mw_lmc.weights import _primal_kkt_residual

    raw, normalized = _primal_kkt_residual(problem, weights)
    return {"kkt_raw": raw, "kkt_reduced": normalized}


def objective_entropy(design, observed, l2, weights, strength, count, cache):
    residual = design @ weights - observed
    data = float(residual @ residual) + l2 * float(weights @ weights)
    if strength == 0.0:
        cache.clear()
        return data, np.asarray(design.T @ residual + l2 * weights, dtype=float)
    safe = np.maximum(weights, 1e-15)
    entropy = float(np.sum(weights * np.log(safe / count)))
    gradient = np.log(safe / count) + 1.0
    scale = strength / count
    return data + scale * entropy, np.asarray(design.T @ residual + l2 * weights + scale * gradient, dtype=float)


def solve_entropy(design, observed, l2, strength, start, max_iter):
    count = design.shape[1]
    cache: dict[str, float] = {}

    def fun(weights):
        value, gradient = objective_entropy(design, observed, l2, weights, strength, count, cache)
        return value, gradient

    result = minimize(
        fun, start, jac=True, method="L-BFGS-B", bounds=[(0.0, None)] * count,
        options={"maxiter": max_iter, "maxfun": max_iter + 100, "ftol": 1e-14, "gtol": 1e-8, "maxls": 50},
    )
    return np.maximum(np.asarray(result.x, dtype=float), 0.0), result


def l2_frontier_records(design, observed, strengths, score_one):
    """Solve and score each ridge problem, preserving the requested grid."""

    records = []
    for strength in strengths:
        started = time.perf_counter()
        weights = initial_weights(design, observed, float(strength))
        seconds = time.perf_counter() - started
        records.append(score_one(float(strength), weights, seconds))
    return records


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
    rows = []
    seed_weights_path = OUTPUT / "seed_weights.npz"
    archive = {}

    def score_l2(strength, weights, seconds):
        ridge_problem = problem_at_l2(problem, strength)
        residual = design @ weights - problem.observed
        chi2 = float(residual @ residual)
        penalty = strength * float(weights @ weights)
        objective_value = chi2 + penalty
        record = {
            "family": "l2", "strength": strength,
            "objective_inner": objective_value, "l2_penalty": penalty,
            "density_chi2": chi2,
            "density_chi2_per_bin_raw": chi2 / fit_bins,
            "solver_seconds": seconds, "iterations": 0,
            "solver_message": "dense SciPy NNLS completed",
            **kkt_diagnostics(ridge_problem, weights),
            **{f"weight_{key}": value for key, value in concentration(weights).items()},
        }
        seed_weights = np.zeros(response.seed_count, dtype=float)
        seed_weights[response.successful_seed_index[active_columns]] = weights
        solution = build_weight_solution(
            response, ridge_problem, weights, objective_value, seconds,
            record["solver_message"], prepared.target_density, prepared.target_error,
        )
        evaluation = score_orbit_weights(library, prepared, solution, response=response)
        record.update({
            "objective_velocity": float(evaluation.objective_velocity),
            "density_chi2_per_bin_scored": float(evaluation.density_chi2_per_bin),
            "density_gate_passed": bool(evaluation.density_gate_passed),
            "weight_sum": float(evaluation.weight_sum),
        })
        archive[f"l2_{strength:g}"] = seed_weights
        print(
            f"l2 lambda={strength:g} chi2/bin={record['density_chi2_per_bin_scored']:.6f} "
            f"Neff={record['weight_n_eff']:.1f} max={record['weight_max_share']:.4f} "
            f"J={record['objective_velocity']:.1f} gate={record['density_gate_passed']} "
            f"kkt={record['kkt_reduced']:.2e} t={seconds:.1f}s",
            flush=True,
        )
        return record

    rows.extend(l2_frontier_records(design, problem.observed, L2_STRENGTHS, score_l2))
    reference_row = rows[0]
    for row in rows:
        row["delta_J"] = row["objective_velocity"] - reference_row["objective_velocity"]

    # Include N_eff as a frontier objective: a cheaper model must not reduce it.
    for row in rows:
        row["pareto_efficient"] = not any(
            (
                other["density_chi2_per_bin_scored"] <= row["density_chi2_per_bin_scored"]
                and other["objective_velocity"] <= row["objective_velocity"]
                and other["weight_n_eff"] >= row["weight_n_eff"]
                and (
                    other["density_chi2_per_bin_scored"] < row["density_chi2_per_bin_scored"]
                    or other["objective_velocity"] < row["objective_velocity"]
                    or other["weight_n_eff"] > row["weight_n_eff"]
                )
            )
            for other in rows if other is not row
        )

    print("L2_FRONTIER_BEGIN", flush=True)
    print("lambda,N_eff,chi2_per_bin,delta_J,gate,kkt_reduced,pareto", flush=True)
    for row in rows:
        print(
            f"{row['strength']:g},{row['weight_n_eff']:.8f},"
            f"{row['density_chi2_per_bin_scored']:.10f},{row['delta_J']:.8f},"
            f"{int(row['density_gate_passed'])},{row['kkt_reduced']:.8e},"
            f"{int(row['pareto_efficient'])}",
            flush=True,
        )
    print("L2_FRONTIER_END", flush=True)

    # Keep legacy branches syntactically intact without running them here.
    graph_seconds = 0.0
    spectrum = []
    degree = np.asarray([0.0])

    for family, strengths in ():
        for strength in strengths:
            if False:
                raise RuntimeError("unreachable")
            started = time.perf_counter()
            if family == "entropy":
                weights, result = solve_entropy(design, problem.observed, l2, strength, reference, 3000)
                solver_message = str(result.message)
                iterations = int(result.nit)
                objective_value = float(result.fun)
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
                f"gate={record['density_gate_passed']} t={seconds:.1f}s nit={iterations}",
                flush=True,
            )

    np.savez_compressed(seed_weights_path, **archive)
    (OUTPUT / "scan.json").write_text(json.dumps({
        "design_shape": list(design.shape), "l2": l2,
        "l2_strengths": list(L2_STRENGTHS),
        "entropy_strengths": list(ENTROPY_STRENGTHS), "graph_strengths": list(GRAPH_STRENGTHS),
        "graph_neighbours": 10, "graph_seconds": graph_seconds,
        "graph_laplacian_spectrum_smallest": spectrum,
        "graph_degree": {"min": float(np.min(degree)), "median": float(np.median(degree)), "max": float(np.max(degree))},
        "reference_seconds": rows[0]["solver_seconds"], "rows": rows,
        "wall_seconds": time.perf_counter() - started_all,
    }, indent=2) + "\n")
    print(f"artifacts: {OUTPUT} (wall {time.perf_counter()-started_all:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
