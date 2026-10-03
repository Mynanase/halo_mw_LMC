#!/usr/bin/env python3
"""Distribution review of the (Jz/Jtot, E) candidate bundling basis.

Motivation: the round-2 corner-actions figure shows the orbit-mean vertical
action fraction Jz/Jtot (phi-average Staeckel actions) is close to rank-
uncorrelated with the orbit energy.  This review audits, without any new
integration or solver call:

1. variable space on the nphi4 active-orbit subset: pair correlations of the
   four candidate bases, marginals / pile-ups of Jz/Jtot, the joint
   (Jz/Jtot, E) distribution with the quantile-warp of the partition grid,
   and the uniform split-half conservation metric for Jz/Jtot (halves of Jr,
   Jz from the round-2 table; the Jphi half reconstructed from half lambda_z
   times Lz_circ(half energy), exact up to the within-orbit variation of
   Lz_circ, which the round-2 E conservation bounds at 5e-7);
2. the density design matrix under equal-weight bundling: per candidate basis
   and bundle count, the unweighted column-mean distortion
   D = 1 - sum_k n_k ||mean column||^2 / sum_j ||a_j||^2 (0 = member columns
   are positive-collinear, 1 = averaging destroys the column), the per-bundle
   coherence c_k, and the member-mean projection of the saved full-solve
   weights (chi2/bin and relative design residual, no re-solve).

Everything reads frozen artifacts: the round-2 invariant table, the frozen
paper-best library, the nphi4 run configuration and the saved full-solve
seed weights.  Companion exploration of docs/nphi1_bundling_repair_plan.md;
touches no production artifact.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from benchmark_nphi1_bundling import _circularity_scale, project_weights_to_bundles, quantile_bundle_grid

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import spearmanr

INVARIANT_TABLE = REPO / ".agent-local/benchmarks/orbit_invariants_round2/orbit_invariants_round2.npz"
FROZEN_CACHE = REPO / ".agent-local/benchmarks/solver_settings_sweep/frozen_paper_best.npz"
FROZEN_PROVENANCE = REPO / ".agent-local/benchmarks/solver_settings_sweep/freeze.json"
FULL_WEIGHTS = REPO / ".agent-local/benchmarks/nphi4_repair/full/evaluation.npz"
FULL_CASE = REPO / ".agent-local/benchmarks/nphi4_repair/full/case.json"
RUN_CONFIG = REPO / "configs/runs/density_solved_benchmark.toml"
OUTPUT = REPO / ".agent-local/benchmarks/orbit_fz_energy_review"

BASES = {
    "lam_z_energy": ("mean_lam_z", "mean_energy", ("$\\langle\\lambda_z\\rangle$", "$\\langle E\\rangle$ [km$^2$/s$^2$]")),
    "energy_omega_z": ("mean_energy", "mean_omega_z", ("$\\langle E\\rangle$ [km$^2$/s$^2$]", "$\\omega_z$")),
    "jr_jz": ("mean_jr_phi", "mean_jz_phi", ("$J_r$ ($\\phi$-avg)", "$J_z$ ($\\phi$-avg)")),
    "fz_energy": ("mean_f_z_phi", "mean_energy", ("$J_z/J_{\\rm tot}$ ($\\phi$-avg)", "$\\langle E\\rangle$ [km$^2$/s$^2$]")),
}


def resolve_grouping_parameters(frozen_provenance_path: Path):
    from halo_mw_lmc.potential import ZHU_2026_BEST_FIT, ZhuHaloParameters

    recorded = json.loads(frozen_provenance_path.read_text()).get("potential_parameters")
    if recorded:
        return ZhuHaloParameters(
            rho0=recorded["rho0"], log_rs=recorded["log_rs"],
            phalo=recorded["phalo"], qhalo=recorded["qhalo"], gamma=recorded["gamma"],
        )
    return ZhuHaloParameters(
        rho0=ZHU_2026_BEST_FIT["rho0"], log_rs=ZHU_2026_BEST_FIT["log_rs"],
        phalo=ZHU_2026_BEST_FIT["phalo"], qhalo=ZHU_2026_BEST_FIT["qhalo"],
        gamma=ZHU_2026_BEST_FIT["gamma"],
    )


def build_design_problem():
    """Rebuild the nphi4 shared inner problem exactly as the stage-2 driver does."""

    from halo_mw_lmc.config import load_run_configuration, resolve_model
    from halo_mw_lmc.density import build_orbit_density_response, density_fit_mask
    from halo_mw_lmc.evaluate import _require_external_response_matches_library
    from halo_mw_lmc.orbits import OrbitLibrary
    from halo_mw_lmc.prepare import prepare_model_data
    from halo_mw_lmc.weights import _build_weight_problem, _normalized_target

    run_configuration = load_run_configuration(RUN_CONFIG)
    model = resolve_model(run_configuration["recipe"])
    prepared = prepare_model_data(
        run_configuration["data"]["catalog"],
        run_configuration["data"]["target_density"],
        model,
    )
    config = prepared.config
    frozen = np.load(FROZEN_CACHE)
    library = OrbitLibrary(
        seed_index=frozen["library_seed_index"],
        time=frozen["library_time"],
        phase_space=frozen["library_phase_space"],
    )
    response = build_orbit_density_response(
        library, config["density_grid"], seed_count=prepared.initial_conditions.shape[0],
    )
    _require_external_response_matches_library(response, library, prepared)
    fit_keys = {
        key: config["density_fit"][key]
        for key in ("min_abs_z", "min_spherical_radius", "max_spherical_radius", "require_positive_data")
    }
    mask = density_fit_mask(prepared.target_density, prepared.target_error, config["density_grid"], **fit_keys)
    target_normalized, error_normalized = _normalized_target(
        prepared.target_density, prepared.target_error, mask, response,
        config["weight_model"]["target_normalization"],
    )
    problem = _build_weight_problem(
        response, target_normalized, error_normalized, mask,
        float(config["weight_model"]["regularization_strength"]),
    )
    return problem, response


def equal_weight_distortion(design_dense, assignments, k_total):
    """D, coherence stats and populated-bundle count of an equal-weight partition.

    D = 1 - sum_k n_k ||mean column_k||^2 / sum_j ||a_j||^2 is the relative
    squared error of replacing every member column by its unweighted bundle
    mean; per-bundle coherence c_k = n_k ||mean_k||^2 / sum_{j in k} ||a_j||^2
    in (0, 1] equals 1 iff the member columns are positive-collinear.
    """

    column_sq = np.sum(design_dense ** 2, axis=0)
    total = float(np.sum(column_sq))
    member = np.bincount(assignments, minlength=k_total).astype(float)
    bundle_sq = np.zeros(k_total)
    np.add.at(bundle_sq, assignments, column_sq)
    sums = np.zeros((k_total, design_dense.shape[0]))
    np.add.at(sums, assignments, design_dense.T)
    populated = member > 0
    mean_sq = np.zeros(k_total)
    mean_sq[populated] = np.sum(sums[populated] ** 2, axis=1) / member[populated] ** 2
    coherence = np.zeros(k_total)
    coherence[populated] = member[populated] * mean_sq[populated] / bundle_sq[populated]
    return {
        "distortion": 1.0 - float(np.sum(member * mean_sq)) / total,
        "coherence_median": float(np.median(coherence[populated])),
        "coherence_member_weighted_mean": float(np.sum(member[populated] * coherence[populated]) / np.sum(member)),
        "populated_bundles": int(np.count_nonzero(populated)),
        "mean_members_per_populated": float(np.mean(member[populated])),
    }


def projection_of_full_weights(design_dense, observed, w_full_active, assignments):
    """Member-mean projection of the saved full weights; no solver involved."""

    u_projection = project_weights_to_bundles(w_full_active, assignments)
    w_projection = u_projection[assignments]
    model_full = design_dense @ w_full_active
    return {
        "projection_chi2_per_bin": float(np.mean((design_dense @ w_projection - observed) ** 2)),
        "projection_design_relative_error": float(
            np.linalg.norm(model_full - design_dense @ w_projection) / np.linalg.norm(model_full)
        ),
    }


def main() -> None:
    started = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    try:
        import agama  # noqa: F401
    except ImportError:
        sys.path.insert(0, str(REPO / "Agama-master"))

    from halo_mw_lmc.potential import build_potential_from_parameters

    # ---- shared inner problem and full-solve weights (input boundary) ----
    problem, response = build_design_problem()
    active_columns = np.flatnonzero(problem.active_columns)
    design_dense = np.asarray(problem.design.todense(), dtype=float)
    observed = problem.observed
    print(f"design {design_dense.shape} (fit-mask rows x active orbits), ||b||={np.linalg.norm(observed):.4g}")

    with np.load(INVARIANT_TABLE) as table:
        seed_index = table["seed_index"]
        variables = {name: table[name] for name in table.files if name.startswith("mean_")}
        halves = {name: table[name] for name in table.files if name.startswith("half_")}
    if not np.array_equal(seed_index, response.successful_seed_index):
        raise ValueError("round-2 invariant table rows do not match the response successful-seed order")

    with np.load(FULL_WEIGHTS) as payload:
        full_seed_weights = payload["seed_weights"]
    active_mask = problem.active_columns
    w_full_active = full_seed_weights[seed_index[active_mask]]
    chi2_full = float(np.mean((design_dense @ w_full_active - observed) ** 2))
    full_case = json.loads(FULL_CASE.read_text())
    print(f"oracle: design-space chi2/bin of saved full weights = {chi2_full:.6f} "
          f"(scored case.json value {full_case['density_chi2_per_bin']:.6f})")
    if not np.isclose(chi2_full, full_case["density_chi2_per_bin"], rtol=1e-6):
        raise ValueError("full-weight design-space chi2 does not reproduce the scored case value")

    # ---- 1. variable-space audit on the active subset ----
    active_variables = {name: values[active_mask] for name, values in variables.items()}
    fz, energy = active_variables["mean_f_z_phi"], active_variables["mean_energy"]

    correlations = {}
    for base, (first, second, _) in BASES.items():
        correlations[base] = float(spearmanr(active_variables[first], active_variables[second]).statistic)
    correlations["fz_vs_lam_z"] = float(spearmanr(fz, active_variables["mean_lam_z"]).statistic)
    correlations["fz_vs_ecc"] = float(spearmanr(fz, active_variables["mean_ecc"]).statistic)
    correlations["fz_vs_r_apo"] = float(spearmanr(fz, active_variables["mean_r_apo"]).statistic)
    correlations["E_vs_r_apo"] = float(spearmanr(energy, active_variables["mean_r_apo"]).statistic)
    print("[spearman on active orbits] " + ", ".join(f"{k}={v:+.3f}" for k, v in correlations.items()))

    pileup = {
        "fz_percentiles": {f"p{q}": float(np.percentile(fz, q)) for q in (1, 5, 25, 50, 75, 95, 99)},
        "fz_frac_lt_0.02": float(np.mean(fz < 0.02)),
        "fz_frac_lt_0.05": float(np.mean(fz < 0.05)),
        "fz_frac_gt_0.8": float(np.mean(fz > 0.8)),
        "energy_percentiles": {f"p{q}": float(np.percentile(energy, q)) for q in (1, 25, 50, 75, 99)},
        "jtot_min": float(variables["mean_jtot_phi"].min()),
    }

    # split-half conservation on the active subset; Jphi halves from
    # half lambda_z x Lz_circ(half energy), see module docstring.
    parameters = resolve_grouping_parameters(FROZEN_PROVENANCE)
    potential = build_potential_from_parameters(parameters)
    e_circ, lz_circ = _circularity_scale(potential)
    rows = np.flatnonzero(active_mask)
    half_fz = np.zeros((rows.size, 2))
    for side in (0, 1):
        jr = halves["half_jr_phi"][rows, side]
        jz = halves["half_jz_phi"][rows, side]
        lz = halves["half_lam_z"][rows, side] * np.interp(
            np.clip(halves["half_energy"][rows, side], e_circ[0], e_circ[-1]), e_circ, lz_circ
        )
        half_fz[:, side] = jz / np.maximum(jr + jz + np.abs(lz), 1e-12)

    def split_half_noise_signal(values, half):
        noise = float(np.median(np.abs(half[:, 0] - half[:, 1])))
        signal = float(np.percentile(values, 75) - np.percentile(values, 25))
        return {"noise_over_signal": noise / signal if signal > 0 else np.inf,
                "median_abs_half_difference": noise, "inter_orbit_iqr": signal}

    conservation = {
        "E": split_half_noise_signal(energy, halves["half_energy"][rows]),
        "Jr (phi-avg)": split_half_noise_signal(active_variables["mean_jr_phi"], halves["half_jr_phi"][rows]),
        "Jz (phi-avg)": split_half_noise_signal(active_variables["mean_jz_phi"], halves["half_jz_phi"][rows]),
        "Jz/Jtot (phi-avg)": split_half_noise_signal(fz, half_fz),
    }
    print("[split-half noise/signal on active orbits]")
    for name, entry in conservation.items():
        print(f"    {name:>18s}  {entry['noise_over_signal']:9.3g}")

    # ---- figures: marginals, joint distribution, quantile warp, geometry ----
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes[0, 0].hist(fz, bins=90, color="#4C72B0")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("$J_z/J_{\\rm tot}$ ($\\phi$-avg)")
    axes[0, 0].set_title("active-orbit marginal: broad plateau, no pile-up")
    axes[0, 1].hist(energy, bins=90, color="#4C72B0")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("$\\langle E\\rangle$ [km$^2$/s$^2$]")
    axes[0, 1].set_title("energy marginal")
    axes[0, 2].hist2d(fz, energy, bins=90, cmin=1, norm=LogNorm(), cmap="viridis")
    axes[0, 2].text(0.03, 0.96, f"$\\rho={correlations['fz_energy']:+.3f}$", transform=axes[0, 2].transAxes,
                    va="top", color="white",
                    bbox={"facecolor": "black", "alpha": 0.55, "pad": 1.5, "edgecolor": "none"})
    axes[0, 2].set_xlabel("$J_z/J_{\\rm tot}$")
    axes[0, 2].set_ylabel("$\\langle E\\rangle$")
    axes[0, 2].set_title("joint distribution")

    panel = axes[1, 0]
    panel.hist2d(fz, energy, bins=90, cmin=1, norm=LogNorm(), cmap="viridis")
    for edge in np.quantile(fz, np.linspace(0.0, 1.0, 65))[::8]:
        panel.axvline(edge, color="white", lw=0.4, alpha=0.7)
    for edge in np.quantile(energy, np.linspace(0.0, 1.0, 65))[::8]:
        panel.axhline(edge, color="white", lw=0.4, alpha=0.7)
    panel.set_xlabel("$J_z/J_{\\rm tot}$")
    panel.set_ylabel("$\\langle E\\rangle$ [km$^2$/s$^2$]")
    panel.set_title("64x64 quantile grid (every 8th edge shown)")

    axes[1, 1].hist2d(fz, energy, bins=60, weights=active_variables["mean_r_apo"], cmin=1, cmap="viridis")
    axes[1, 1].set_xlabel("$J_z/J_{\\rm tot}$")
    axes[1, 1].set_ylabel("$\\langle E\\rangle$")
    axes[1, 1].set_title("weighted by $r_{\\rm apo}$: E rows fix the radial size")
    axes[1, 2].hist2d(fz, energy, bins=60, weights=active_variables["mean_ecc"], cmin=1, cmap="viridis")
    axes[1, 2].set_xlabel("$J_z/J_{\\rm tot}$")
    axes[1, 2].set_ylabel("$\\langle E\\rangle$")
    axes[1, 2].set_title("weighted by $e$: fz columns trade radial/vertical")
    fig.suptitle("(Jz/Jtot, E) candidate bundling basis - active-orbit distribution review (nphi4)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT / "fz_energy_distribution.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
    for ax, (base, (first, second, (x_label, y_label))) in zip(axes, BASES.items()):
        x_values, y_values = active_variables[first], active_variables[second]
        ax.hist2d(x_values, y_values, bins=80, cmin=1, norm=LogNorm(), cmap="viridis")
        rho = float(spearmanr(x_values, y_values).statistic)
        ax.text(0.03, 0.96, f"$\\rho={rho:+.3f}$", transform=ax.transAxes, va="top", color="white",
                bbox={"facecolor": "black", "alpha": 0.55, "pad": 1.5, "edgecolor": "none"})
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel(y_label, fontsize=9)
        ax.set_title(base, fontsize=10)
    fig.suptitle("candidate 2D bases, active orbits", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT / "base_comparison_joint.png", dpi=150)
    plt.close(fig)

    # ---- 2. equal-weight design distortion per basis and bundle count ----
    audit = {}
    for bins_per_axis in (32, 64):
        for base, (first, second, _) in BASES.items():
            assignments, k_total = quantile_bundle_grid(
                active_variables[first], active_variables[second], bins_per_axis, bins_per_axis,
            )
            entry = {
                "bundle_grid": f"{bins_per_axis}x{bins_per_axis}",
                **equal_weight_distortion(design_dense, assignments, k_total),
                **projection_of_full_weights(design_dense, observed, w_full_active, assignments),
            }
            audit[f"{base}_{bins_per_axis}x{bins_per_axis}"] = entry

            rng = np.random.default_rng(0)
            random_assignments = rng.permutation(assignments)
            audit[f"{base}_{bins_per_axis}x{bins_per_axis}_random"] = {
                "bundle_grid": f"{bins_per_axis}x{bins_per_axis} (random)",
                **equal_weight_distortion(design_dense, random_assignments, k_total),
                **projection_of_full_weights(design_dense, observed, w_full_active, random_assignments),
            }

    print("[equal-weight design audit: distortion D (lower = member columns stay collinear)]")
    for key, entry in audit.items():
        print(f"    {key:>36s}  D={entry['distortion']:.4f}  coh_med={entry['coherence_median']:.4f}  "
              f"populated={entry['populated_bundles']:5d}  proj_chi2/bin={entry['projection_chi2_per_bin']:8.3f}")

    fig, ax = plt.subplots(figsize=(10, 4.8))
    palette = {"lam_z_energy": "#4C72B0", "energy_omega_z": "#DD8452", "jr_jz": "#55A868", "fz_energy": "#C44E52"}
    labels, values, colors, hatches = [], [], [], []
    for bins_per_axis in (32, 64):
        for base in BASES:
            for suffix in ("", "_random"):
                labels.append(f"{base} {bins_per_axis}x{bins_per_axis}" + (" random" if suffix else ""))
                values.append(audit[f"{base}_{bins_per_axis}x{bins_per_axis}{suffix}"]["distortion"])
                colors.append(palette[base] if not suffix else "0.75")
                hatches.append(None if not suffix else "///")
    bars = ax.bar(range(len(values)), values, color=colors)
    for bar, hatch in zip(bars, hatches):
        if hatch:
            bar.set_hatch(hatch)
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("equal-weight column-mean distortion D")
    ax.set_title("density design matrix under equal-weight bundling (hatched = random control, seed 0)")
    fig.tight_layout()
    fig.savefig(OUTPUT / "distortion_by_base.png", dpi=150)
    plt.close(fig)

    summary = {
        "design_shape": list(design_dense.shape),
        "active_orbits": int(active_mask.sum()),
        "oracle_full_chi2_per_bin": chi2_full,
        "spearman_active": correlations,
        "pileup_and_ranges": pileup,
        "split_half_active": conservation,
        "equal_weight_audit": audit,
        "notes": {
            "jphi_halves": "reconstructed from half lambda_z x Lz_circ(half energy); exact up to within-orbit Lz_circ variation bounded by the round-2 E conservation 5e-7",
            "distortion": "D = 1 - sum_k n_k ||mean column||^2 / sum_j ||a_j||^2 on the error-normalized density design; member columns replaced by unweighted bundle means",
            "projection": "member-mean projection of the saved full-solve seed weights; no solver involved; chi2/bin on the same fit-mask rows as the scored cases",
            "population": "quantile partition computed on active-orbit variable values only, mirroring the stage-2 driver",
        },
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"[done] review artifacts in {OUTPUT} (wall {time.perf_counter() - started:.1f}s)")


if __name__ == "__main__":
    main()
