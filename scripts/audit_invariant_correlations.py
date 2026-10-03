#!/usr/bin/env python3
"""Section-14 audit: orbit-invariant correlations and multi-axis bundle candidates.

Read-only (no solver): loads the round-2 AGAMA invariant table plus the frozen
library, computes the xy principal-axis angle, reports (i) the Spearman rank
correlation matrix across all per-orbit invariants on active orbits, (ii)
split-half stability for every table invariant, and (iii) the corrected (n_k
weighted) equal-weight distortion D and the full-weights projection chi2/bin
of candidate product-quantile partitions -- the 2D (f_z,E) bundle-count curve
between 2304 and 4096, and 3D (f_z,E,X) grids over every low-correlation
third-axis candidate X, at 2304/3072/4096 total bundles. Interpretation bounds
as in SS6-SS13: single frozen library, projection chi2 is a pessimistic bound,
no production claim.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from audit_third_invariant import bundle_grid, xy_angle_statistics  # noqa: E402
from review_fz_energy_basis import (  # noqa: E402
    FROZEN_CACHE,
    INVARIANT_TABLE,
    build_design_problem,
    equal_weight_distortion,
    projection_of_full_weights,
)

OUTPUT = REPO / ".agent-local/benchmarks/invariant_correlations"
FULL_WEIGHTS = REPO / ".agent-local/benchmarks/nphi4_repair_2d_fz_energy/full/evaluation.npz"

# Third-axis candidates evaluated as grids, beyond the (f_z,E) main plane.
THIRD_AXIS_KEYS = [
    "mean_lam_z", "mean_lz", "mean_l_mod", "mean_jr_phi", "mean_jphi_phi",
    "mean_jz_phi", "mean_omega_r", "mean_omega_phi", "mean_omega_z",
    "mean_ratio_phi_r", "mean_ratio_rz", "mean_cos_i", "mean_ecc",
    "mean_r_peri", "mean_f_r_phi",
]
SHAPES_2304 = [(16, 12, 12), (12, 12, 16)]
SHAPES_3072 = [(16, 16, 12)]
SHAPES_4096 = [(16, 16, 16)]


def main() -> None:
    started = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    with np.load(INVARIANT_TABLE) as table:
        invariants = {key: table[key] for key in table.files if key.startswith("mean_")}
        halves = {key[5:]: table[key] for key in table.files if key.startswith("half_")}
    with np.load(FULL_WEIGHTS) as payload:
        full_seed_weights = payload["seed_weights"]
    problem, response = build_design_problem()
    seeds = response.successful_seed_index
    active = np.flatnonzero(problem.active_columns)
    design_dense = np.asarray(problem.design.todense(), dtype=float)
    w_full_active = full_seed_weights[seeds[active]]
    print(f"design {design_dense.shape}; active orbits {active.size}")

    with np.load(FROZEN_CACHE) as frozen:
        successful = frozen["successful_seed_index"]
        sample_count = frozen["sample_count"]
        if not np.array_equal(np.repeat(successful, sample_count), frozen["library_seed_index"]):
            raise ValueError("frozen library samples are not contiguous per successful seed")
        phase = frozen["library_phase_space"].reshape(successful.size, int(sample_count[0]), 6)
    angle_all, wrapped_half_difference = xy_angle_statistics(phase)
    invariants["mean_xy_angle"] = angle_all
    halves["xy_angle"] = wrapped_half_difference

    fz = invariants["mean_f_z_phi"][active]
    energy = invariants["mean_energy"][active]
    values = {name: column[active] for name, column in invariants.items()}

    # ---- (i) Spearman rank correlation matrix on active orbits ----
    names = sorted(values)
    matrix = np.zeros((len(names), len(names)))
    for i, left in enumerate(names):
        for j in range(i, len(names)):
            rho = float(spearmanr(values[left], values[names[j]]).statistic)
            matrix[i, j] = matrix[j, i] = rho
    header = "invariant," + ",".join(names)
    lines = [header] + [
        names[i] + "," + ",".join(f"{matrix[i, j]:+.4f}" for j in range(len(names)))
        for i in range(len(names))
    ]
    (OUTPUT / "correlation_matrix.csv").write_text("\n".join(lines) + "\n")

    fz_index, e_index = names.index("mean_f_z_phi"), names.index("mean_energy")
    third_report = {}
    for name in THIRD_AXIS_KEYS + ["mean_xy_angle"]:
        column = values[name]
        spread = float(np.percentile(column, 99) - np.percentile(column, 1))
        half = halves.get(name)
        if half is not None and name != "mean_xy_angle":
            difference = np.abs(half[active, 0] - half[active, 1])
            noise = float(np.median(difference))
            stability = {"median_half_abs_difference": noise, "spread_p01_p99": spread,
                         "noise_over_spread": noise / spread if spread > 0 else float("inf")}
        elif name == "mean_xy_angle":
            angle_deg = np.degrees(wrapped_half_difference[active])
            stability = {"median_abs_half_difference_deg": float(np.median(np.abs(angle_deg))),
                         "fraction_within_10deg": float(np.mean(np.abs(angle_deg) < 10.0))}
        else:
            stability = {"median_half_abs_difference": None, "spread_p01_p99": spread}
        third_report[name] = {
            "spearman_with_fz": float(matrix[names.index(name), fz_index]),
            "spearman_with_energy": float(matrix[names.index(name), e_index]),
            "stability": stability,
        }

    # ---- (iii) candidate partitions: D + projection chi2 ----
    candidates = []

    def record(label, variables, bins):
        assignments, k_total = bundle_grid(variables, bins)
        distortion = equal_weight_distortion(design_dense, assignments, k_total)
        projection = projection_of_full_weights(design_dense, problem.observed, w_full_active, assignments)
        candidates.append({
            "candidate": label, "shape": "x".join(str(b) for b in bins), "bundles": int(k_total),
            "populated": int(distortion["populated_bundles"]),
            "D": float(distortion["distortion"]),
            "coherence_median": float(distortion["coherence_median"]),
            "projection_chi2_per_bin": float(projection["projection_chi2_per_bin"]),
        })

    for n in (48, 52, 55, 60, 64, 68):
        record(f"fz_e_{n}x{n}", [fz, energy], [n, n])
    for key in THIRD_AXIS_KEYS + ["mean_xy_angle"]:
        column = values[key]
        for shape in SHAPES_2304 + SHAPES_3072 + SHAPES_4096:
            record(f"fz_e_{key[5:]}_{shape[0]}x{shape[1]}x{shape[2]}", [fz, energy, column], list(shape))
    # Unequal third-axis allocations around 4096 bundles for the best
    # integrators (SS14 second pass): finer main plane vs deeper third axis.
    for key, shapes in {
        "mean_omega_phi": [(32, 16, 8), (16, 32, 8), (24, 24, 8), (20, 20, 10), (32, 32, 4), (16, 16, 20)],
        "mean_cos_i": [(24, 24, 8), (32, 16, 8)],
        "mean_xy_angle": [(24, 24, 8)],
    }.items():
        for shape in shapes:
            record(f"fz_e_{key[5:]}_{shape[0]}x{shape[1]}x{shape[2]}", [fz, energy, values[key]], list(shape))

    summary = {
        "active_orbits": int(active.size),
        "third_axis": third_report,
        "candidates": candidates,
        "wall_seconds": time.perf_counter() - started,
        "notes": "D corrected (n_k-weighted); projection chi2 is the pessimistic no-resolve bound; single frozen library, seed 0",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["candidate,shape,bundles,populated,D,coherence_median,projection_chi2_per_bin"]
    lines += [f"{c['candidate']},{c['shape']},{c['bundles']},{c['populated']},{c['D']:.4f},"
              f"{c['coherence_median']:.3f},{c['projection_chi2_per_bin']:.3f}" for c in candidates]
    (OUTPUT / "candidates.csv").write_text("\n".join(lines) + "\n")

    print("\nthird-axis report (rank correlation with f_z / E, stability):")
    for name, report in third_report.items():
        print(f"  {name:18s} rho_fz={report['spearman_with_fz']:+.3f} rho_E={report['spearman_with_energy']:+.3f} "
              f"{report['stability']}")
    print("\ncandidates sorted by D:")
    for row in sorted(candidates, key=lambda c: c["D"])[:24]:
        print(f"  {row['candidate']:34s} {row['shape']:>10s} populated={row['populated']:4d} "
              f"D={row['D']:.4f} proj={row['projection_chi2_per_bin']:7.2f}")
    print(f"artifacts: {OUTPUT} (wall {time.perf_counter() - started:.1f}s)")


if __name__ == "__main__":
    main()
