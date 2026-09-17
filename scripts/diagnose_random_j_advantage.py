#!/usr/bin/env python3
"""Diagnose why random bundling beats physical bundling on the velocity J.

Companion diagnostic of ``docs/nphi1_bundling_repair_plan.md`` §7, on the
same frozen library, (lambda_z, E) 32x32 partition, shared scoring.  The
velocity J is an outer objective the inner NNLS never sees, so the random
advantage has four candidate sources: seed luck, the direction of the
equal-weight subspace, the particular density re-solve inside that
subspace, or the constraint strength.  Four separable probes:

1. seed robustness -- many independent random permutations of the physical
   member-count multiset, one bundled solve + score each;
2. projection -- the full solve's weights mapped onto the bundle subspace by
   member means (no solver involved), for the physical and one random
   assignment: separates the subspace direction from the re-solve;
3. alpha interpolation -- J along w(alpha) = (1-alpha) w_full + alpha
   w_projection for both subspaces: marginal J cost of moving to equal
   weights;
4. bundle-count sweep -- 8x8 .. 64x64 physical vs random: constraint
   strength dependence of the gap.

Every case is scored through ``halo_mw_lmc.evaluate.score_orbit_weights``
with the identical response and target; gates and solver statuses are
recorded but this script draws no production conclusion.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from benchmark_nphi1_bundling import (
    _sha256,
    compute_orbit_variables,
    map_bundle_weights_to_seeds,
    project_weights_to_bundles,
    quantile_bundle_grid,
    resolve_grouping_variables,
    solve_bundled_weights,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def main(config_path: str) -> None:
    import tomllib

    started_total = time.perf_counter()
    config_file = Path(config_path).resolve()
    with open(config_file, "rb") as handle:
        experiment = tomllib.load(handle)
    output_dir = REPO / experiment["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    grouping_variables = resolve_grouping_variables(experiment)
    n_first = int(experiment["n_lambda"])
    n_second = int(experiment["n_energy"])
    random_seeds = [int(seed) for seed in experiment["random_seeds"]]
    sweep_bins = [(int(a), int(b)) for a, b in experiment["sweep_bins"]]
    sweep_seed = int(experiment["sweep_random_seed"])
    alphas = [float(alpha) for alpha in experiment["projection_alphas"]]
    projection_seed = int(experiment["projection_random_seed"])

    try:
        import agama  # noqa: F401
    except ImportError:
        sys.path.insert(0, str(REPO / "Agama-master"))

    from halo_mw_lmc.config import load_run_configuration, resolve_model
    from halo_mw_lmc.density import build_orbit_density_response, density_fit_mask
    from halo_mw_lmc.evaluate import evaluate_orbit_library, score_orbit_weights
    from halo_mw_lmc.orbits import OrbitLibrary
    from halo_mw_lmc.prepare import prepare_model_data
    from halo_mw_lmc.potential import ZHU_2026_BEST_FIT, ZhuHaloParameters, build_potential_from_parameters
    from halo_mw_lmc.weights import WeightSolution, _build_weight_problem, _normalized_target

    # ---- input boundary: identical to the benchmark driver ----
    run_configuration = load_run_configuration(REPO / experiment["run_config"])
    model = resolve_model(run_configuration["recipe"])
    prepared = prepare_model_data(
        run_configuration["data"]["catalog"],
        run_configuration["data"]["target_density"],
        model,
    )
    config = prepared.config
    frozen = np.load(REPO / experiment["frozen_cache"])
    library = OrbitLibrary(
        seed_index=frozen["library_seed_index"],
        time=frozen["library_time"],
        phase_space=frozen["library_phase_space"],
    )
    response = build_orbit_density_response(
        library, config["density_grid"], seed_count=prepared.initial_conditions.shape[0],
    )
    print(f"response {response.matrix.shape} nnz={response.matrix.nnz}")

    evaluation_full = evaluate_orbit_library(library, prepared, response=response)
    full_solution = evaluation_full.weight_solution
    j_full = float(evaluation_full.objective_velocity)
    print(f"full: J={j_full:.3f} active={full_solution.active_orbit_count} "
          f"N_eff={full_solution.effective_orbit_count:.0f}")

    frozen_provenance = json.loads((REPO / experiment["frozen_provenance"]).read_text())
    recorded = frozen_provenance.get("potential_parameters")
    if recorded:
        parameters = ZhuHaloParameters(
            rho0=recorded["rho0"], log_rs=recorded["log_rs"], phalo=recorded["phalo"],
            qhalo=recorded["qhalo"], gamma=recorded["gamma"],
        )
    else:
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
    print(f"grouping variables {grouping_variables} in {grouping_seconds:.2f}s")

    # ---- shared inner problem ----
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
        raise ValueError("diagnostic inner problem does not match the production full-solve problem")
    active_columns = np.flatnonzero(problem.active_columns)
    w_full_active = full_solution.seed_weights[successful[active_columns]]

    def score_seed_weights(seed_weights, label):
        """Uniform J/chi2/statistics scoring for one full seed-weight vector."""

        weights_active = seed_weights[successful[active_columns]]
        residual = problem.design @ weights_active - problem.observed
        data_term = float(residual @ residual)
        squared = float(np.dot(seed_weights, seed_weights))
        total = float(np.sum(seed_weights))
        solution = WeightSolution(
            seed_weights=seed_weights,
            model_density=response.model_density(seed_weights),
            target_density=target_normalized,
            target_error=error_normalized,
            inner_objective=data_term + regularization * squared,
            regularization_penalty=regularization * squared,
            effective_orbit_count=total**2 / squared if squared > 0 else 0.0,
            maximum_weight_fraction=float(np.max(seed_weights)) / total if total > 0 else 0.0,
            active_orbit_count=int(np.count_nonzero(seed_weights > max(float(np.max(seed_weights)) * 1e-12, 0.0))),
            converged=True,
            status=0,
            message=f"diagnostic case {label}",
            iterations=0,
            optimality=0.0,
            solver_backend=f"diagnostic:{label}",
            kkt_residual=float("nan"),
            solve_wall_seconds=0.0,
            solver_cost=0.5 * (data_term + regularization * squared),
            problem_fingerprint=problem.fingerprint,
        )
        started = time.perf_counter()
        evaluation = score_orbit_weights(library, prepared, solution, response=response)
        return {
            "label": label,
            "objective_velocity": float(evaluation.objective_velocity),
            "delta_j": float(evaluation.objective_velocity) - j_full,
            "density_chi2_per_bin": float(evaluation.density_chi2_per_bin),
            "density_gate_passed": bool(evaluation.density_gate_passed),
            "weight_sum": float(evaluation.weight_sum),
            "effective_orbit_count": solution.effective_orbit_count,
            "maximum_weight_fraction": solution.maximum_weight_fraction,
            "active_orbit_count": solution.active_orbit_count,
            "data_term_F": data_term,
            "score_seconds": time.perf_counter() - started,
        }

    def solve_case(assignments, label):
        started = time.perf_counter()
        u = solve_bundled_weights(problem.design, problem.observed, assignments, regularization)
        seed_weights = map_bundle_weights_to_seeds(u, assignments, successful, active_columns, response.seed_count)
        case = score_seed_weights(seed_weights, label)
        case["solve_seconds"] = time.perf_counter() - started
        case["bundle_count"] = int(np.max(assignments)) + 1
        case["active_bundles"] = int(np.count_nonzero(u > 0))
        return case, u, seed_weights

    # ---- probe 1: physical reference + random seed robustness ----
    var_first = orbit_variables[grouping_variables[0]][active_columns]
    var_second = orbit_variables[grouping_variables[1]][active_columns]
    assignments_physical, _ = quantile_bundle_grid(var_first, var_second, n_first, n_second)
    physical_case, u_physical, w_physical = solve_case(assignments_physical, "physical_32x32")
    print(f"physical: dJ={physical_case['delta_j']:+.2f} chi2/bin={physical_case['density_chi2_per_bin']:.4f} "
          f"N_eff={physical_case['effective_orbit_count']:.0f}")

    random_cases = []
    for seed in random_seeds:
        rng = np.random.default_rng(seed)
        assignments_random = rng.permutation(assignments_physical)
        case, _, _ = solve_case(assignments_random, f"random_seed{seed}")
        random_cases.append(case)
        print(f"random seed {seed}: dJ={case['delta_j']:+.2f} chi2/bin={case['density_chi2_per_bin']:.4f} "
              f"N_eff={case['effective_orbit_count']:.0f}")

    # ---- probe 2: subspace projections of the full solution (no re-solve) ----
    u_projection_physical = project_weights_to_bundles(w_full_active, assignments_physical)
    w_projection_physical = map_bundle_weights_to_seeds(
        u_projection_physical, assignments_physical, successful, active_columns, response.seed_count,
    )
    rng_projection = np.random.default_rng(projection_seed)
    assignments_projection_random = rng_projection.permutation(assignments_physical)
    u_projection_random = project_weights_to_bundles(w_full_active, assignments_projection_random)
    w_projection_random = map_bundle_weights_to_seeds(
        u_projection_random, assignments_projection_random, successful, active_columns, response.seed_count,
    )
    projection_physical_case = score_seed_weights(w_projection_physical, "projection_physical")
    projection_random_case = score_seed_weights(w_projection_random, "projection_random")
    print(f"projection physical: dJ={projection_physical_case['delta_j']:+.2f} "
          f"chi2/bin={projection_physical_case['density_chi2_per_bin']:.4f}")
    print(f"projection random  : dJ={projection_random_case['delta_j']:+.2f} "
          f"chi2/bin={projection_random_case['density_chi2_per_bin']:.4f}")

    # ---- probe 3: alpha interpolation full -> projection ----
    interpolation = {"alphas": alphas, "physical": [], "random": []}
    for alpha in alphas:
        for side, w_projection in (("physical", w_projection_physical), ("random", w_projection_random)):
            weights = (1.0 - alpha) * full_solution.seed_weights + alpha * w_projection
            case = score_seed_weights(weights, f"interp_{side}_alpha{alpha}")
            interpolation[side].append({
                "alpha": alpha, "delta_j": case["delta_j"],
                "density_chi2_per_bin": case["density_chi2_per_bin"],
                "effective_orbit_count": case["effective_orbit_count"],
            })
    for side in ("physical", "random"):
        curve = ", ".join(f"a={p['alpha']:.2f}:dJ={p['delta_j']:+.1f}" for p in interpolation[side])
        print(f"interpolation {side}: {curve}")

    # ---- probe 4: bundle-count sweep, physical vs random ----
    sweep = []
    for n_a, n_b in sweep_bins:
        assignments_sweep, _ = quantile_bundle_grid(var_first, var_second, n_a, n_b)
        case_physical, _, _ = solve_case(assignments_sweep, f"sweep_physical_{n_a}x{n_b}")
        rng_sweep = np.random.default_rng(sweep_seed)
        case_random, _, _ = solve_case(rng_sweep.permutation(assignments_sweep), f"sweep_random_{n_a}x{n_b}")
        sweep.append({
            "bins": [n_a, n_b], "bundles": n_a * n_b,
            "physical": case_physical, "random": case_random,
        })
        print(f"sweep {n_a}x{n_b}: physical dJ={case_physical['delta_j']:+.2f} "
              f"random dJ={case_random['delta_j']:+.2f} "
              f"(N_eff {case_physical['effective_orbit_count']:.0f} vs {case_random['effective_orbit_count']:.0f})")

    # ---- artifacts ----
    summary = {
        "grouping_variables": grouping_variables,
        "bins": [n_first, n_second],
        "j_full": j_full,
        "full_case": {
            "active_orbit_count": full_solution.active_orbit_count,
            "effective_orbit_count": full_solution.effective_orbit_count,
            "density_chi2_per_bin": float(evaluation_full.density_chi2_per_bin),
        },
        "physical_case": physical_case,
        "random_cases": random_cases,
        "projections": {"physical": projection_physical_case, "random": projection_random_case},
        "interpolation": interpolation,
        "sweep": sweep,
        "potential_parameters": parameters.as_dict(),
        "notes": {
            "projection": "member-mean projection of the full solve onto the equal-weight subspace; no solver involved",
            "scoring": "all cases scored through halo_mw_lmc.evaluate.score_orbit_weights on the same response and target",
            "scope": "single frozen library, single measurement per case; method diagnostics only",
        },
        "provenance": {
            "script_sha256": _sha256(Path(__file__).resolve()),
            "config_sha256": _sha256(config_file),
            "frozen_cache_sha256": _sha256(REPO / experiment["frozen_cache"]),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    np.savez(
        output_dir / "weights.npz",
        seed_weights_full=full_solution.seed_weights,
        seed_weights_physical=w_physical,
        seed_weights_projection_physical=w_projection_physical,
        seed_weights_projection_random=w_projection_random,
        assignments_physical=assignments_physical,
        assignments_projection_random=assignments_projection_random,
    )

    # ---- figures ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    random_delta = [case["delta_j"] for case in random_cases]
    axes[0].hist(random_delta, bins=8, color="#4C72B0", label=f"random seeds (n={len(random_delta)})")
    axes[0].axvline(physical_case["delta_j"], color="crimson", lw=2, label="physical")
    axes[0].axvline(0.0, color="k", lw=1, ls="--", label="full")
    axes[0].set_xlabel(r"$\Delta J$ = $J$ - $J_{\rm full}$")
    axes[0].set_title("probe 1: random seed spread vs physical")
    axes[0].legend(fontsize=8)

    for side, color in (("physical", "crimson"), ("random", "#4C72B0")):
        values = [point["delta_j"] for point in interpolation[side]]
        axes[1].plot(alphas, values, "o-", color=color, label=f"{side} projection")
    axes[1].axhline(0.0, color="k", lw=1, ls="--")
    axes[1].set_xlabel(r"$\alpha$  (w = (1-$\alpha$) w$_{full}$ + $\alpha$ w$_{proj}$)")
    axes[1].set_ylabel(r"$\Delta J$")
    axes[1].xaxis.set_major_locator(MaxNLocator(6))
    axes[1].set_title("probe 2+3: cost of moving to equal weights")
    axes[1].legend(fontsize=8)

    bundles = [point["bundles"] for point in sweep]
    axes[2].plot(bundles, [point["physical"]["delta_j"] for point in sweep], "o-", color="crimson", label="physical")
    axes[2].plot(bundles, [point["random"]["delta_j"] for point in sweep], "s-", color="#4C72B0", label=f"random (seed {sweep_seed})")
    axes[2].axhline(0.0, color="k", lw=1, ls="--")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("bundle count")
    axes[2].set_ylabel(r"$\Delta J$")
    axes[2].set_title("probe 4: constraint-strength sweep")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "random_j_advantage.png", dpi=150)
    plt.close(fig)
    print(f"[done] {output_dir} (wall {time.perf_counter() - started_total:.1f}s)")


if __name__ == "__main__":
    default_config = REPO / "configs" / "benchmarks" / "nphi1_random_diagnosis.toml"
    main(sys.argv[1] if len(sys.argv) > 1 else str(default_config))


