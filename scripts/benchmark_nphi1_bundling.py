#!/usr/bin/env python3
"""nphi1 bundling repair experiment: core math and experiment driver.

Experimental-script companion of ``docs/nphi1_bundling_repair_plan.md``:
§2.2 quantile bundle assignment in two configurable orbit variables (the
audited default is (lambda_z, E); §6 adds (E, omega_z) and (Jr, Jz)), the
reduced bundled NNLS problem ``min(u>=0) ||A S u - b||^2 + lambda *
sum_k(n_k * u_k^2)``, and the strict active-orbit -> successful-seed ->
full-seed backfill of bundle weights; plus the stage-2 driver that runs
full/bundled/random on one frozen orbit library through the shared
production scoring boundary.

The bundled solve solves for one weight u_k per bundle; the per-orbit weight
is ``w = u[assignment]``, so the L2 penalty on w maps to ``sum_k n_k u_k^2``
with n_k the bundle member count. The member count is derived inside
``solve_bundled_weights`` from the same assignment array passed to the solve,
so the mapping and the regularization can never disagree. The driver keeps
one ``assignments`` variable per case from the quantile partition through
both the solve and the backfill, and the random control maps its own
permuted assignments through the same pair of calls.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import tomllib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

# Iteration budget for scipy.optimize.nnls, inherited from the audited
# prototype (.agent-local/benchmarks/nphi1_full_iteration_bundled.py);
# 100 * k_total.
NNLS_MAXITER_PER_BUNDLE = 100

# Reduced-space KKT acceptance for the bundled NNLS, fixed before the run
# (plan §2.3); also mirrored in configs/benchmarks/nphi1_bundling.toml.
REDUCED_KKT_TOLERANCE = 1e-8


def quantile_bundle_grid(first, second, n_first, n_second):
    """Quantile bundle partition in two orbit variables; return (assignments, k_total).

    Edges are linspace(0, 1, n+1) quantiles with open outer edges
    (-inf/inf); bins use searchsorted(side="right") - 1 clipped to the
    interior, so every orbit lands in exactly one of k_total = n_first *
    n_second bundles. Variable-agnostic: the audited (lambda_z, E) grouping
    and the §6 two-dimensional alternatives pass through the same code.
    """

    first_edges = np.quantile(first, np.linspace(0.0, 1.0, n_first + 1))
    first_edges[0], first_edges[-1] = -np.inf, np.inf
    second_edges = np.quantile(second, np.linspace(0.0, 1.0, n_second + 1))
    second_edges[0], second_edges[-1] = -np.inf, np.inf
    first_bin = np.clip(np.searchsorted(first_edges, first, side="right") - 1, 0, n_first - 1)
    second_bin = np.clip(np.searchsorted(second_edges, second, side="right") - 1, 0, n_second - 1)
    assignments = first_bin * n_second + second_bin
    return assignments, n_first * n_second


def solve_bundled_weights(design, observed, assignments, regularization):
    """Solve min(u>=0) ||A S u - b||^2 + lambda * sum_k n_k u_k^2; return u.

    ``design`` is the (rows, n_active) error-normalized orbit design matrix
    and ``assignments`` maps each active orbit column to its bundle. The
    member counts n_k are computed here from ``assignments`` so the mapping
    and the regularization diagonal cannot diverge. Empty bundles keep
    u_k = 0 (they are not identifiable). The augmented matrix is
    [[A S]; [sqrt(lambda * n_k) I]] against [b; 0], solved by dense NNLS
    with maxiter = NNLS_MAXITER_PER_BUNDLE * k_total.
    """

    rows, n_active = design.shape
    k_total = int(np.max(assignments)) + 1
    member_count = np.bincount(assignments, minlength=k_total).astype(float)
    indicator = np.zeros((n_active, k_total))
    indicator[np.arange(n_active), assignments] = 1.0
    reduced = design @ indicator
    dense = np.zeros((rows + k_total, k_total), dtype=float, order="F")
    dense[:rows, :] = reduced
    dense[rows:, :].flat[:: k_total + 1] = np.sqrt(regularization * member_count)

    from scipy.optimize import nnls

    bundle_weights, _ = nnls(
        dense, np.concatenate([observed, np.zeros(k_total)]),
        maxiter=NNLS_MAXITER_PER_BUNDLE * k_total,
    )
    bundle_weights = np.asarray(bundle_weights, dtype=float)
    bundle_weights[member_count == 0] = 0.0
    return bundle_weights


def map_bundle_weights_to_seeds(bundle_weights, assignments, successful_seed_index, active_columns, seed_count):
    """Backfill bundle weights u into a full seed-weight array.

    Strict order: active orbit columns -> successful seeds via
    ``successful_seed_index[active_columns]`` -> the full seed array of
    length ``seed_count``. No assumption of contiguous, sorted, or complete
    seed numbering; seeds outside the successful set stay zero.
    """

    seed_weights = np.zeros(seed_count, dtype=float)
    seed_weights[successful_seed_index[active_columns]] = bundle_weights[assignments]
    return seed_weights


def project_weights_to_bundles(active_weights, assignments):
    """L2 bundle-mean projection of active-orbit weights; return bundle weights u.

    The closest (in L2 on the active columns) bundle-constant weight vector
    assigns every member the mean of its bundle's active weights. Unlike a
    bundled re-solve this touches no solver: it isolates the effect of the
    equal-weight subspace direction itself. Empty bundles stay zero.
    """

    k_total = int(np.max(assignments)) + 1
    member_count = np.bincount(assignments, minlength=k_total)
    weight_total = np.bincount(assignments, weights=active_weights, minlength=k_total)
    projection = np.zeros(k_total, dtype=float)
    populated = member_count > 0
    projection[populated] = weight_total[populated] / member_count[populated]
    return projection


# -----------------------------------------------------------------------
# Driver support: hashing, reduced KKT, assembly, artifact IO.
# -----------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _assignments_fingerprint(assignments: np.ndarray) -> str:
    digest = hashlib.sha256()
    contiguous = np.ascontiguousarray(assignments, dtype=np.int64)
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _reduced_augmented_problem(problem, assignments: np.ndarray):
    """Rebuild [[A S]; [sqrt(lambda n_k) I]] vs [b; 0] for the KKT audit only.

    Mirrors the construction inside ``solve_bundled_weights``; used with
    ``_primal_kkt_residual`` (regularization zero) to report the reduced-space
    KKT of the bundled solution. The reduced problem is a restricted
    approximation of the original per-orbit problem, never its equivalent.
    """

    from halo_mw_lmc.weights import _WeightProblem

    rows, n_active = problem.design.shape
    k_total = int(np.max(assignments)) + 1
    member_count = np.bincount(assignments, minlength=k_total).astype(float)
    indicator = np.zeros((n_active, k_total))
    indicator[np.arange(n_active), assignments] = 1.0
    dense = np.zeros((rows + k_total, k_total), dtype=float)
    dense[:rows, :] = problem.design @ indicator
    dense[rows:, :].flat[:: k_total + 1] = np.sqrt(problem.regularization * member_count)
    observed = np.concatenate([problem.observed, np.zeros(k_total)])
    reduced = _WeightProblem(
        design=dense,
        observed=observed,
        active_columns=np.ones(k_total, dtype=bool),
        successful_orbit_count=problem.successful_orbit_count,
        regularization=0.0,
        fingerprint="",
    )
    return reduced


def _circularity_scale(potential):
    """Lz_circ(E) interpolant grid, replicating the audited prototype math
    (.agent-local/benchmarks/lambdaz_bundle_prototype.py::circularity_scale).
    """

    phi_center = float(potential.potential(np.array([[0.0, 0.0, 0.0]]))[0])
    energies = np.linspace(phi_center * (1.0 - 1e-9), -1e-4, 4096)
    try:
        radius = np.asarray(potential.Rcirc(E=energies), dtype=float)
    except Exception:
        radius = np.array([float(potential.Rcirc(E=float(value))) for value in energies])
    finite = np.isfinite(radius) & (radius > 0)
    radius, energies = radius[finite], energies[finite]
    points = np.column_stack([radius, np.zeros_like(radius), np.zeros_like(radius)])
    force_r = np.asarray(potential.force(points), dtype=float)[:, 0]
    v_circ = np.sqrt(radius * np.abs(force_r))
    lz_circ = radius * v_circ
    order = np.argsort(energies)
    return energies[order], lz_circ[order]


SUPPORTED_GROUPING_VARIABLES = (
    "lam_z", "energy", "omega_z", "jr_phi", "jz_phi", "jphi_phi", "jz_over_jtot_phi",
)


def resolve_grouping_variables(experiment: dict) -> list[str]:
    """Read ``grouping_variables`` from the experiment dict; default (lam_z, energy)."""

    names = [str(name) for name in experiment.get("grouping_variables", ["lam_z", "energy"])]
    if len(names) != 2:
        raise ValueError("grouping_variables must name exactly two orbit variables")
    unknown = [name for name in names if name not in SUPPORTED_GROUPING_VARIABLES]
    if unknown:
        raise ValueError(
            f"unsupported grouping variables: {', '.join(unknown)}; "
            f"supported: {', '.join(SUPPORTED_GROUPING_VARIABLES)}"
        )
    return names


def dominant_frequency(series, dt):
    """Hann-windowed FFT peak with log-parabola interpolation; cycles per time unit.

    ``series`` has shape (n_orbits, n_samples), sampled on a per-orbit uniform
    grid with step ``dt`` (scalar or length-n_orbits array; the frozen library
    integrates each orbit over its own 10 periods, so dt differs per orbit).
    Bin 0 carries only leakage and is skipped. Origin: the round-2 orbit
    invariant exploration (scripts/explore_orbit_invariants.py).
    """

    n = series.shape[1]
    window = np.hanning(n)
    centered = series - series.mean(axis=1, keepdims=True)
    spectrum = np.abs(np.fft.rfft(centered * window, axis=1))
    peak = np.argmax(spectrum[:, 1:], axis=1) + 1
    rows = np.arange(series.shape[0])
    alpha = np.log(np.maximum(spectrum[rows, np.maximum(peak - 1, 0)], 1e-300))
    beta = np.log(np.maximum(spectrum[rows, peak], 1e-300))
    gamma = np.log(np.maximum(spectrum[rows, np.minimum(peak + 1, spectrum.shape[1] - 1)], 1e-300))
    denominator = alpha - 2.0 * beta + gamma
    delta = np.where(
        np.abs(denominator) > 1e-12,
        0.5 * (alpha - gamma) / np.where(np.abs(denominator) > 1e-12, denominator, 1.0),
        0.0,
    )
    delta = np.clip(delta, -0.5, 0.5)
    dt_rows = np.broadcast_to(np.asarray(dt, dtype=float), (series.shape[0],))
    return (peak + delta) / (n * dt_rows)


def compute_orbit_variables(names, library, successful, potential):
    """Per-orbit grouping variables on the frozen library; return ({name: array}, seconds).

    "lam_z" and "energy" reproduce the audited per-sample -> orbit-mean math
    verbatim (circularity through the ``_circularity_scale`` table of the same
    potential). "omega_z" is the FFT peak frequency of z(t) over each orbit's
    full sampled span, potential-model-free. "jr_phi"/"jz_phi"/"jphi_phi" are
    orbit means of Staeckel actions evaluated per sample in the azimuthal m=0
    average of the true triaxial potential (CylSpline mmax=0; the round-2
    oracle pinned it to the numerical phi-average at 1.5e-4 with exact
    circular-orbit actions). "jz_over_jtot_phi" is the orbit-mean vertical
    fraction mean(Jz) / mean(Jr + Jz + |Jphi|) in that same basis, identical
    to the round-2 exploration's f_z_phi column. Only requested variables
    are computed.
    """

    started = time.perf_counter()
    is_seed = np.isin(library.seed_index, successful)
    seed_samples = library.phase_space[is_seed]
    seed_owner = library.seed_index[is_seed]
    order = np.argsort(successful)
    column = np.searchsorted(successful[order], seed_owner)
    counts = np.bincount(column, minlength=successful.size)
    values: dict[str, np.ndarray] = {}

    if "lam_z" in names or "energy" in names:
        e_circ, lz_circ = _circularity_scale(potential)
        lz = seed_samples[:, 0] * seed_samples[:, 4] - seed_samples[:, 1] * seed_samples[:, 3]
        energy = 0.5 * np.sum(seed_samples[:, 3:] ** 2, axis=1) + np.asarray(
            potential.potential(seed_samples[:, :3]), dtype=float
        )
        energy = np.clip(energy, e_circ[0], e_circ[-1])
        if "energy" in names:
            orbit_energy = np.zeros(successful.size)
            np.add.at(orbit_energy, column, energy)
            values["energy"] = orbit_energy / counts
        if "lam_z" in names:
            lambda_z = lz / np.maximum(np.interp(energy, e_circ, lz_circ), 1e-12)
            orbit_lambda = np.zeros(successful.size)
            np.add.at(orbit_lambda, column, lambda_z)
            values["lam_z"] = orbit_lambda / counts

    if "omega_z" in names:
        if not np.all(counts == counts[0]):
            raise ValueError("omega_z needs an equal number of samples per orbit")
        if not np.all(np.diff(column) >= 0):
            raise ValueError("omega_z needs library samples grouped by seed in seed order")
        samples_per_orbit = int(counts[0])
        z_series = seed_samples[:, 2].reshape(successful.size, samples_per_orbit)
        times_grid = library.time[is_seed].reshape(successful.size, samples_per_orbit)
        dt_rows = np.median(np.diff(times_grid, axis=1), axis=1)
        values["omega_z"] = dominant_frequency(z_series, dt_rows)

    if any(name in names for name in ("jr_phi", "jz_phi", "jphi_phi", "jz_over_jtot_phi")):
        import agama

        phi_potential = agama.Potential(type="CylSpline", density=potential, mmax=0)
        finder = agama.ActionFinder(phi_potential)
        actions = np.asarray(finder(seed_samples), dtype=float)
        action_sums = np.zeros((successful.size, 4))
        np.add.at(action_sums, column, np.column_stack(
            (actions[:, 0], actions[:, 1], actions[:, 2], np.abs(actions[:, 2]))
        ))
        action_means = action_sums / counts[:, None]
        for index, name in enumerate(("jr_phi", "jz_phi", "jphi_phi")):
            if name in names:
                values[name] = action_means[:, index]
        if "jz_over_jtot_phi" in names:
            values["jz_over_jtot_phi"] = action_means[:, 1] / np.maximum(
                action_means[:, 0] + action_means[:, 1] + action_means[:, 3], 1e-12
            )

    missing = [name for name in names if name not in values]
    if missing:
        raise ValueError(f"grouping variables were not computed: {', '.join(missing)}")
    return values, time.perf_counter() - started


def assemble_bundled_solution(response, problem, bundle_weights, assignments, target_normalized, error_normalized, solve_seconds):
    """Build one production ``WeightSolution`` from a bundled NNLS solution.

    Semantics mirror ``solve_density_weights``: ``inner_objective`` is the
    data term F plus the regularization penalty (reported separately in the
    case JSON), ``kkt_residual`` is the normalized reduced-space KKT, and the
    full-space KKT of the original per-orbit problem is returned alongside
    as a diagnostic only. The same ``assignments`` array must be passed here
    that was passed to ``solve_bundled_weights``.
    """

    from halo_mw_lmc.weights import _primal_kkt_residual

    successful = response.successful_seed_index
    active_columns = np.flatnonzero(problem.active_columns)
    seed_weights = map_bundle_weights_to_seeds(
        bundle_weights, assignments, successful, active_columns, response.seed_count,
    )
    active_weights = seed_weights[successful[active_columns]]
    finite_nonnegative = bool(np.all(np.isfinite(active_weights)) and np.all(active_weights >= 0))
    total = float(np.sum(seed_weights))
    squared = float(np.dot(seed_weights, seed_weights))
    density_residual = problem.design @ active_weights - problem.observed
    data_term = float(np.dot(density_residual, density_residual))
    regularization_penalty = problem.regularization * squared
    reduced_problem = _reduced_augmented_problem(problem, assignments)
    reduced_raw, reduced_normalized = _primal_kkt_residual(reduced_problem, bundle_weights)
    from scipy.sparse import csr_matrix
    from halo_mw_lmc.weights import _problem_fingerprint
    reduced_fingerprint = _problem_fingerprint(
        csr_matrix(reduced_problem.design), reduced_problem.observed, 0.0,
    )
    full_raw, full_normalized = _primal_kkt_residual(problem, active_weights)
    converged = bool(
        finite_nonnegative and total > 0 and reduced_normalized <= REDUCED_KKT_TOLERANCE
    )
    solution_kkt = reduced_normalized if np.isfinite(reduced_normalized) else full_raw
    from halo_mw_lmc.weights import WeightSolution

    return WeightSolution(
        seed_weights=seed_weights,
        model_density=response.model_density(seed_weights),
        target_density=target_normalized,
        target_error=error_normalized,
        inner_objective=data_term + regularization_penalty,
        regularization_penalty=regularization_penalty,
        effective_orbit_count=total**2 / squared if squared > 0 else 0.0,
        maximum_weight_fraction=float(np.max(seed_weights)) / total if total > 0 else 0.0,
        active_orbit_count=int(np.count_nonzero(seed_weights > max(float(np.max(seed_weights)) * 1e-12, 0.0))),
        converged=converged,
        status=0 if converged else 1,
        message="dense NNLS on bundled columns" if converged else "bundled NNLS failed reduced-KKT tolerance",
        iterations=0,
        optimality=reduced_raw if np.isfinite(reduced_raw) else full_raw,
        solver_backend="bundled_dense_nnls",
        kkt_residual=solution_kkt,
        solve_wall_seconds=solve_seconds,
        solver_cost=0.5 * (data_term + regularization_penalty),
        problem_fingerprint=problem.fingerprint,
    ), {"reduced_raw": reduced_raw, "reduced_normalized": reduced_normalized, "full_raw": full_raw, "full_normalized": full_normalized, "reduced_fingerprint": reduced_fingerprint}


def save_attempt(directory: Path, name: str, case: dict, arrays: dict) -> None:
    """Write one case directory: case.json + evaluation.npz."""

    case_dir = directory / name
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "case.json").write_text(json.dumps(case, indent=2, default=float) + "\n")
    np.savez(case_dir / "evaluation.npz", **arrays)


def load_attempt(directory: Path, name: str) -> tuple[dict, dict]:
    """Read back one case directory written by ``save_attempt``."""

    case_dir = directory / name
    case = json.loads((case_dir / "case.json").read_text())
    with np.load(case_dir / "evaluation.npz") as payload:
        arrays = {key: payload[key] for key in payload.files}
    return case, arrays


# -----------------------------------------------------------------------
# Stage-2 driver: full / bundled / random on one frozen library.
# -----------------------------------------------------------------------


def main(config_path: str) -> None:
    global REDUCED_KKT_TOLERANCE
    started_total = time.perf_counter()
    config_file = Path(config_path).resolve()
    with open(config_file, "rb") as handle:
        experiment = tomllib.load(handle)
    run_config_path = REPO / experiment["run_config"]
    frozen_path = REPO / experiment["frozen_cache"]
    frozen_provenance_path = REPO / experiment["frozen_provenance"]
    output_dir = REPO / experiment["output_dir"]
    n_lambda = int(experiment["n_lambda"])
    n_energy = int(experiment["n_energy"])
    grouping_variables = resolve_grouping_variables(experiment)
    random_seed = int(experiment["random_seed"])
    if int(experiment["repeat_runs"]) != 1:
        raise ValueError("this driver implements the single-measurement round only (repeat_runs = 1)")
    REDUCED_KKT_TOLERANCE = float(experiment.get("reduced_kkt_tolerance", REDUCED_KKT_TOLERANCE))

    from scipy.sparse import csr_matrix

    try:
        import agama  # noqa: F401
    except ImportError:
        # AGAMA is vendored at Agama-master/ (exact casing); the audited
        # prototype scripts put it on sys.path the same way.
        sys.path.insert(0, str(REPO / "Agama-master"))

    from halo_mw_lmc.config import load_run_configuration, resolve_model
    from halo_mw_lmc.density import build_orbit_density_response, density_fit_mask
    from halo_mw_lmc.evaluate import (
        _require_external_response_matches_library,
        evaluate_orbit_library,
        score_orbit_weights,
    )
    from halo_mw_lmc.orbits import OrbitLibrary
    from halo_mw_lmc.prepare import prepare_model_data
    from halo_mw_lmc.potential import (
        ZHU_2026_BEST_FIT,
        ZhuHaloParameters,
        build_potential_from_parameters,
    )
    from halo_mw_lmc.weights import _build_weight_problem, _normalized_target

    # ---- input boundary: run config, prepared data, frozen library ----
    run_configuration = load_run_configuration(run_config_path)
    model = resolve_model(run_configuration["recipe"])
    prepared = prepare_model_data(
        run_configuration["data"]["catalog"],
        run_configuration["data"]["target_density"],
        model,
    )
    config = prepared.config

    frozen = np.load(frozen_path)
    library = OrbitLibrary(
        seed_index=frozen["library_seed_index"],
        time=frozen["library_time"],
        phase_space=frozen["library_phase_space"],
    )
    # The frozen cache stores the 4-sector response matrix; the nphi1 response
    # is rebuilt from the same library on the configured grid and then
    # validated against that library, so all three methods share one response.
    started = time.perf_counter()
    response = build_orbit_density_response(
        library, config["density_grid"], seed_count=prepared.initial_conditions.shape[0],
    )
    response_seconds = time.perf_counter() - started
    _require_external_response_matches_library(response, library, prepared)
    print(f"density grid {config['density_grid'].shape}, response {response.matrix.shape} "
          f"nnz={response.matrix.nnz} built in {response_seconds:.2f}s")

    # ---- full case: production entry, outer timer ----
    started = time.perf_counter()
    evaluation_full = evaluate_orbit_library(library, prepared, response=response)
    full_total = time.perf_counter() - started
    full_solution = evaluation_full.weight_solution

    # ---- grouping: per-seed mean lambda_z and energy in the frozen potential ----
    frozen_provenance = json.loads(frozen_provenance_path.read_text())
    recorded_parameters = frozen_provenance.get("potential_parameters")
    if recorded_parameters:
        grouping_note = "potential parameters taken from the frozen-run provenance record"
        parameters = ZhuHaloParameters(
            rho0=recorded_parameters["rho0"], log_rs=recorded_parameters["log_rs"],
            phalo=recorded_parameters["phalo"], qhalo=recorded_parameters["qhalo"],
            gamma=recorded_parameters["gamma"],
        )
    else:
        grouping_note = (
            "frozen provenance records no potential parameters; grouping uses "
            "ZHU_2026_BEST_FIT with log_rs=log10(70), which differs from the "
            "production analytic anchor log_rs=1.845; valid only for comparison "
            "within this one frozen library"
        )
        parameters = ZhuHaloParameters(
            rho0=ZHU_2026_BEST_FIT["rho0"], log_rs=ZHU_2026_BEST_FIT["log_rs"],
            phalo=ZHU_2026_BEST_FIT["phalo"], qhalo=ZHU_2026_BEST_FIT["qhalo"],
            gamma=ZHU_2026_BEST_FIT["gamma"],
        )
    potential = build_potential_from_parameters(parameters)
    successful = response.successful_seed_index
    orbit_variables, grouping_seconds = compute_orbit_variables(
        grouping_variables, library, successful, potential,
    )
    print(f"grouping variables {grouping_variables} in {grouping_seconds:.2f}s "
          f"(n bins {n_lambda}x{n_energy})")

    # ---- shared inner problem: identical normalization path as production ----
    fit_keys = {
        key: config["density_fit"][key]
        for key in ("min_abs_z", "min_spherical_radius", "max_spherical_radius", "require_positive_data")
    }
    mask = density_fit_mask(prepared.target_density, prepared.target_error, config["density_grid"], **fit_keys)
    target_normalized, error_normalized = _normalized_target(
        prepared.target_density, prepared.target_error, mask, response,
        config["weight_model"]["target_normalization"],
    )
    regularization = float(config["weight_model"]["regularization_strength"])
    problem = _build_weight_problem(response, target_normalized, error_normalized, mask, regularization)
    if problem.fingerprint != full_solution.problem_fingerprint:
        raise ValueError("bundled inner problem does not match the production full-solve problem")
    design = problem.design
    active_columns = np.flatnonzero(problem.active_columns)

    def run_bundled_case(name, case_assignments):
        started_case = time.perf_counter()
        started_solve = time.perf_counter()
        u = solve_bundled_weights(design, problem.observed, case_assignments, regularization)
        solve_seconds = time.perf_counter() - started_solve
        started_reconstruct = time.perf_counter()
        solution, kkt = assemble_bundled_solution(
            response, problem, u, case_assignments, target_normalized, error_normalized, solve_seconds,
        )
        reconstruct_seconds = time.perf_counter() - started_reconstruct
        started_score = time.perf_counter()
        evaluation = score_orbit_weights(library, prepared, solution, response=response)
        score_seconds = time.perf_counter() - started_score
        case = {
            "method": name,
            "solver_backend": solution.solver_backend,
            "objective_velocity": evaluation.objective_velocity,
            "selected_objective": evaluation.selected_objective,
            "density_chi2_per_bin": evaluation.density_chi2_per_bin,
            "density_gate_passed": bool(evaluation.density_gate_passed),
            "density_scale": float(evaluation.density.scale),
            "weight_sum": evaluation.weight_sum,
            "effective_orbit_count": solution.effective_orbit_count,
            "maximum_weight_fraction": solution.maximum_weight_fraction,
            "active_orbit_count": solution.active_orbit_count,
            "bundle_count": int(np.max(case_assignments)) + 1,
            "active_bundles": int(np.count_nonzero(u > 0)),
            "inner_objective": solution.inner_objective,
            "data_term_F": solution.inner_objective - solution.regularization_penalty,
            "regularization_penalty": solution.regularization_penalty,
            "kkt_reduced_raw": kkt["reduced_raw"],
            "kkt_reduced_normalized": kkt["reduced_normalized"],
            "kkt_full_space_raw": kkt["full_raw"],
            "kkt_full_space_normalized": kkt["full_normalized"],
            "kkt_tolerance_reduced": REDUCED_KKT_TOLERANCE,
            "iterations": None,
            "converged": bool(solution.converged),
            "message": solution.message,
            "problem_fingerprint": problem.fingerprint,
            "reduced_problem_fingerprint": kkt["reduced_fingerprint"],
            "assignments_fingerprint": _assignments_fingerprint(case_assignments),
            "timings_seconds": {
                "grouping": grouping_seconds,
                "grouping_shared_across_cases": True,
                "matrix": None,
                "matrix_note": "reduced/augmented matrix construction happens inside solve_bundled_weights and is not separable without changing the frozen core function",
                "solve": solve_seconds,
                "reconstruct": reconstruct_seconds,
                "score": score_seconds,
                "total_outer": time.perf_counter() - started_case,
                "end_to_end_seconds": None,  # filled below with the shared response build
                "end_to_end_note": "end_to_end_seconds = shared response build + shared grouping + per-method total_outer; artifact writing and plotting are excluded",
            },
        }
        arrays = {
            "seed_weights": solution.seed_weights,
            "orbit_weights": response.sample_weights(solution.seed_weights, library),
            "model_density": evaluation.density.raw_model_density,
            "residual": evaluation.density.residual,
            "fit_mask": evaluation.density.fit_mask,
        }
        return case, arrays, evaluation, u

    assignments, k_total = quantile_bundle_grid(
        orbit_variables[grouping_variables[0]][active_columns],
        orbit_variables[grouping_variables[1]][active_columns],
        n_lambda, n_energy,
    )
    bundled_case, bundled_arrays, evaluation_bundled, u_bundled = run_bundled_case("bundled", assignments)

    rng = np.random.default_rng(random_seed)
    random_assignments = rng.permutation(assignments)
    # A permutation cannot change the value multiset; this assertion only
    # guards a future switch to non-permutation randomization.
    if not np.array_equal(
        np.sort(np.bincount(assignments, minlength=k_total)),
        np.sort(np.bincount(random_assignments, minlength=k_total)),
    ):
        raise ValueError("random membership does not preserve the physical bundle member counts")
    random_case, random_arrays, evaluation_random, u_random = run_bundled_case("random", random_assignments)
    for case in (bundled_case, random_case):
        case["timings_seconds"]["end_to_end_seconds"] = (
            response_seconds + grouping_seconds + case["timings_seconds"]["total_outer"]
        )

    # ---- full case JSON ----
    full_case = {
        "method": "full",
        "solver_backend": full_solution.solver_backend,
        "objective_velocity": evaluation_full.objective_velocity,
        "selected_objective": evaluation_full.selected_objective,
        "density_chi2_per_bin": evaluation_full.density_chi2_per_bin,
        "density_gate_passed": bool(evaluation_full.density_gate_passed),
        "density_scale": float(evaluation_full.density.scale),
        "weight_sum": evaluation_full.weight_sum,
        "effective_orbit_count": full_solution.effective_orbit_count,
        "maximum_weight_fraction": full_solution.maximum_weight_fraction,
        "active_orbit_count": full_solution.active_orbit_count,
        "inner_objective": full_solution.inner_objective,
        "data_term_F": full_solution.inner_objective - full_solution.regularization_penalty,
        "regularization_penalty": full_solution.regularization_penalty,
        "kkt_full_space_normalized": full_solution.kkt_residual,
        "iterations": int(full_solution.iterations),
        "converged": bool(full_solution.converged),
        "message": full_solution.message,
        "problem_fingerprint": full_solution.problem_fingerprint,
        "timings_seconds": {
            "input_response_build": response_seconds,
            "grouping": None,
            "matrix": None,
            "solve": full_solution.solve_wall_seconds,
            "reconstruct": None,
            "score": None,
            "total_outer": full_total,
            "end_to_end_seconds": response_seconds + full_total,
            "end_to_end_note": "end_to_end_seconds = shared response build + shared grouping (none for full) + per-method total_outer; artifact writing and plotting are excluded",
            "note": "outer timer covers the whole evaluate_orbit_library call; solve is the production solve_density_weights backend timer; other stages are not separable inside the production entry",
        },
    }
    full_arrays = {
        "seed_weights": full_solution.seed_weights,
        "orbit_weights": response.sample_weights(full_solution.seed_weights, library),
        "model_density": evaluation_full.density.raw_model_density,
        "residual": evaluation_full.density.residual,
        "fit_mask": evaluation_full.density.fit_mask,
    }

    # ---- artifacts ----
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = {
        "experiment": experiment,
        "run_config": str(run_config_path),
        "recipe": model["name"] if "name" in model else run_config_path.name,
        "density_grid_shape": list(config["density_grid"].shape),
        "density_fit": config["density_fit"],
        "weight_model": config["weight_model"],
        "objective": config["objective"],
        "include_velocity": config["include_velocity"],
        "grouping_note": grouping_note,
        "reduced_kkt_tolerance": REDUCED_KKT_TOLERANCE,
    }
    (output_dir / "resolved_config.json").write_text(json.dumps(resolved, indent=2, default=float) + "\n")

    import scipy

    try:
        from threadpoolctl import threadpool_info
        threads = threadpool_info()
    except ImportError:
        threads = "threadpoolctl: unavailable"
    git_head = "unavailable"
    git_dirty = None
    try:
        import subprocess
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True)
        if head.returncode == 0:
            git_head = head.stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True)
        if status.returncode == 0:
            git_dirty = bool(status.stdout.strip())
    except OSError:
        pass
    historical_path = REPO / experiment["historical_reference"] if experiment.get("historical_reference") else None
    provenance = {
        "script_sha256": _sha256(Path(__file__).resolve()),
        "config_sha256": _sha256(config_file),
        "frozen_cache_sha256": _sha256(frozen_path),
        "target_sha256": _sha256(prepared.density_path),
        "git_head": git_head,
        "git_dirty": git_dirty,
        "frozen_provenance": frozen_provenance,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "threadpool_info": threads,
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str) + "\n")

    np.savez(
        output_dir / "bundles.npz",
        assignments=assignments,
        random_assignments=random_assignments,
        member_count=np.bincount(assignments, minlength=k_total),
        random_member_count=np.bincount(random_assignments, minlength=k_total),
        u_bundled=u_bundled,
        u_random=u_random,
        seed_weights_bundled=bundled_arrays["seed_weights"],
        seed_weights_random=random_arrays["seed_weights"],
        orbit_var_first=orbit_variables[grouping_variables[0]],
        orbit_var_second=orbit_variables[grouping_variables[1]],
        successful_seed_index=successful,
        target_normalized=target_normalized,
        error_normalized=error_normalized,
        fit_mask=mask,
    )
    save_attempt(output_dir, "full", full_case, full_arrays)
    save_attempt(output_dir, "bundled", bundled_case, bundled_arrays)
    save_attempt(output_dir, "random", random_case, random_arrays)

    historical = None
    if historical_path is not None and historical_path.exists():
        historical = {
            "source": str(historical_path),
            "sha256": _sha256(historical_path),
            "reference_only": True,
            "values": json.loads(historical_path.read_text()),
        }
    open_questions = None
    if historical is not None:
        old_full = historical["values"].get("variants", {}).get("full_lsq_linear", {}).get("objective_velocity")
        production_reference = historical["values"].get("production_reference", {}).get("objective")
        open_questions = {
            "full_reproduction_difference": {
                "new_full_minus_historical_full": None if old_full is None else full_case["objective_velocity"] - old_full,
                "new_full_minus_production_reference": None if production_reference is None else full_case["objective_velocity"] - production_reference,
                "cause": "undetermined (thread nondeterminism vs other sources); deferred to plan stage 3 controlled repeats",
            },
        }
    comparison = {
        "methods": {
            "full": {key: full_case[key] for key in (
                "objective_velocity", "density_chi2_per_bin", "density_gate_passed",
                "weight_sum", "active_orbit_count", "timings_seconds",
            )},
            "bundled": {key: bundled_case[key] for key in (
                "objective_velocity", "density_chi2_per_bin", "density_gate_passed",
                "weight_sum", "active_orbit_count", "timings_seconds",
            )},
            "random": {key: random_case[key] for key in (
                "objective_velocity", "density_chi2_per_bin", "density_gate_passed",
                "weight_sum", "active_orbit_count", "timings_seconds",
            )},
        },
        "notes": {
            "bundled_is_restricted": "bundled/random solve the reduced bundle-coadded problem; their convergence and gates refer to that restricted approximation, not the production per-orbit solver",
            "scoring": "all three methods scored through halo_mw_lmc.evaluate.score_orbit_weights on the same response and target",
            "historical_reference": "values from the pre-repair audit are reference-only and enter no computation",
            "timing_definition": "report end-to-end speedup only from end_to_end_seconds (= shared response build + shared grouping + per-method total_outer); total_outer alone omits grouping and overstates the bundled speedup",
        },
        "open_questions": open_questions,
        "historical_reference": historical,
    }
    (output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2, default=float) + "\n")

    for name, case in (("full", full_case), ("bundled", bundled_case), ("random", random_case)):
        solve = case["timings_seconds"]["solve"]
        print(f"{name:8s}: J={case['objective_velocity']:.3f} "
              f"chi2/bin={case['density_chi2_per_bin']:.4f} gate={case['density_gate_passed']} "
              f"solve={solve if solve is not None else float('nan'):.2f}s "
              f"total={case['timings_seconds']['total_outer']:.2f}s")
    print(f"artifacts: {output_dir} (wall {time.perf_counter() - started_total:.1f}s)")


if __name__ == "__main__":
    default_config = REPO / "configs" / "benchmarks" / "nphi1_bundling.toml"
    main(sys.argv[1] if len(sys.argv) > 1 else str(default_config))
