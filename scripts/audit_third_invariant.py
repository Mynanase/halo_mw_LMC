#!/usr/bin/env python3
"""Is the 1024-bundle gate failure a variable problem or a count problem?

The §10 (Jz/Jtot, E) basis passes the nphi4 density gate only at 4096
bundles (~1.4 active orbits per populated bundle, barely binding), while
1024 bundles fail for every 2D basis.  Before adding a third invariant this
audit measures, without any solver:

1. the bundle-count curve of the 2D (Jz/Jtot, E) basis at 32/40/48/55/64
   bins per axis -- maybe the gate already passes between 1024 and 4096 and
   no third variable is needed;
2. third-axis candidates at ~1024 total bundles: lambda_z (azimuthal
   circulation, rank-uncorrelated with both Jz/Jtot and E) and a new xy
   principal-axis angle per orbit (orientation of the orbit's xy occupancy,
   the phi-structure degree of freedom the 4-sector grid actually sees);
   both are also audited for within-orbit stability (split-half);
3. a response-similarity oracle: plain k-means (k = 1024 and 2304) on the
   raw error-normalized response columns, minimizing the same within-bundle
   distortion D that equal-weight bundling pays.  Its D and full-weight
   projection chi2 bound what ANY orbit grouping at that bundle count could
   achieve -- if the oracle cannot get below the 2D/3D variables' D by much,
   the gate binds on bundle count, not on variable choice.

Metrics per partition: distortion D, populated bundles, full-weight
member-mean projection chi2/bin (no re-solve) and the median cosine of 2000
sampled same-bundle column pairs.  Reads frozen artifacts only (frozen
library, round-2 invariant table, §10 bundles.npz, nphi4 run configuration).
Companion of docs/nphi1_bundling_repair_plan.md §11; outputs to
.agent-local/benchmarks/third_invariant_audit/.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from review_fz_energy_basis import (
    FROZEN_CACHE,
    INVARIANT_TABLE,
    build_design_problem,
    equal_weight_distortion,
    projection_of_full_weights,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

BUNDLES_64 = REPO / ".agent-local/benchmarks/nphi4_repair_2d_fz_energy/bundles.npz"
FULL_WEIGHTS = REPO / ".agent-local/benchmarks/nphi4_repair/full/evaluation.npz"
OUTPUT = REPO / ".agent-local/benchmarks/third_invariant_audit"


def bundle_grid(variables, bins):
    """Product quantile partition over N variables; 2D call equals the driver's grid."""

    assignments = np.zeros(variables[0].size, dtype=np.int64)
    total = 1
    for values, n in zip(variables, bins):
        edges = np.quantile(values, np.linspace(0.0, 1.0, n + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        index = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, n - 1)
        assignments = assignments * n + index
        total *= n
    return assignments, total


def xy_angle_statistics(phase):
    """Per-orbit xy-occupancy principal-axis angle (period pi) plus split-half stability."""

    orbit_count = phase.shape[0]
    angle = np.zeros(orbit_count)
    wrapped_half_difference = np.zeros(orbit_count)
    middle = phase.shape[1] // 2
    for start in range(0, orbit_count, 1500):
        block = phase[start:start + 1500]
        x, y = block[..., 0], block[..., 1]

        def axis_angle(x_values, y_values):
            cx = np.mean((x_values - x_values.mean(axis=1, keepdims=True)) ** 2, axis=1)
            cy = np.mean((y_values - y_values.mean(axis=1, keepdims=True)) ** 2, axis=1)
            cxy = np.mean(
                (x_values - x_values.mean(axis=1, keepdims=True))
                * (y_values - y_values.mean(axis=1, keepdims=True)), axis=1,
            )
            return 0.5 * np.arctan2(2.0 * cxy, cx - cy)

        rows = slice(start, start + block.shape[0])
        angle[rows] = axis_angle(x, y)
        first, second = axis_angle(x[:, :middle], y[:, :middle]), axis_angle(x[:, middle:], y[:, middle:])
        wrapped_half_difference[rows] = np.angle(np.exp(1j * (first - second))) / np.pi
    return angle, wrapped_half_difference


def kmeans_columns(points, k, seed=0, max_iterations=60, tolerance=1e-9):
    """Plain k-means (kmeans++ init, Lloyd) on raw columns; minimizes D's objective."""

    rng = np.random.default_rng(seed)
    count = points.shape[0]
    centers = points[rng.integers(count)].reshape(1, -1).copy()
    min_distance = np.full(count, np.inf)
    while centers.shape[0] < k:
        distance = np.sum((points - centers[-1]) ** 2, axis=1)
        min_distance = np.minimum(min_distance, distance)
        min_distance[~np.isfinite(min_distance)] = 0.0
        total = min_distance.sum()
        if total <= 0:
            raise ValueError("kmeans++ ran out of distinct points")
        centers = np.vstack([centers, points[rng.choice(count, p=min_distance / total)]])
    for _ in range(max_iterations):
        assignments = np.argmin(
            np.sum(points ** 2, axis=1)[:, None] - 2.0 * (points @ centers.T) + np.sum(centers ** 2, axis=1)[None, :],
            axis=1,
        )
        new_centers = np.zeros_like(centers)
        counts = np.bincount(assignments, minlength=k).astype(float)
        np.add.at(new_centers, assignments, points)
        empty = counts == 0
        if np.any(empty):
            farthest = np.argmax(np.sum((points - centers[assignments]) ** 2, axis=1))
            new_centers[empty] = points[farthest]
            counts[empty] = 1.0
        new_centers /= counts[:, None]
        shift = float(np.max(np.sum((new_centers - centers) ** 2, axis=1)))
        centers = new_centers
        if shift < tolerance:
            break
    return assignments


def sampled_pair_cosines(design_dense, assignments, k_total, rng, sample=2000):
    """Median cosine of sampled same-bundle column pairs (direction alignment)."""

    members = {int(bundle): np.flatnonzero(assignments == bundle)
               for bundle in np.unique(assignments) if np.count_nonzero(assignments == bundle) >= 2}
    keys = rng.choice(np.array(sorted(members)), size=min(sample, len(members)), replace=len(members) < sample)
    cosines = []
    norms = np.linalg.norm(design_dense, axis=0)
    for key in keys:
        group = members[int(key)]
        if group.size < 2:
            continue
        left, right = rng.choice(group, size=2, replace=False)
        cosines.append(float(design_dense[:, left] @ design_dense[:, right] / (norms[left] * norms[right])))
    return float(np.median(cosines)) if cosines else float("nan")


def main() -> None:
    started = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # ---- input boundary ----
    with np.load(BUNDLES_64) as bundles:
        seeds = bundles["successful_seed_index"]
        fz_all = bundles["orbit_var_first"]
        energy_all = bundles["orbit_var_second"]
    with np.load(INVARIANT_TABLE) as table:
        if not np.array_equal(seeds, table["seed_index"]):
            raise ValueError("round-2 invariant table rows do not match the experiment seed order")
        lam_all = table["mean_lam_z"]
    with np.load(FROZEN_CACHE) as frozen:
        successful = frozen["successful_seed_index"]
        sample_count = frozen["sample_count"]
        if not np.array_equal(np.repeat(successful, sample_count), frozen["library_seed_index"]):
            raise ValueError("frozen library samples are not contiguous per successful seed")
        phase = frozen["library_phase_space"].reshape(successful.size, int(sample_count[0]), 6)
    with np.load(FULL_WEIGHTS) as payload:
        full_seed_weights = payload["seed_weights"]

    problem, _ = build_design_problem()
    active = np.flatnonzero(problem.active_columns)
    design_dense = np.asarray(problem.design.todense(), dtype=float)
    observed = problem.observed
    w_full_active = full_seed_weights[seeds[active]]
    print(f"design {design_dense.shape}; active orbits {active.size}")

    fz, energy, lam = fz_all[active], energy_all[active], lam_all[active]

    # ---- third-axis candidates: lambda_z needs no computation; xy angle ----
    angle_all, wrapped_half_difference = xy_angle_statistics(phase)
    angle = angle_all[active]
    angle_deg = np.degrees(wrapped_half_difference[active])
    third_axis_diagnostics = {
        "lam_z_rank_correlation_with_fz_and_E": [
            float(spearmanr(lam, fz).statistic), float(spearmanr(lam, energy).statistic),
        ],
        "xy_angle_rank_correlation_with_fz_and_E": [
            float(spearmanr(np.degrees(angle), fz).statistic), float(spearmanr(np.degrees(angle), energy).statistic),
        ],
        "xy_angle_split_half_median_abs_deg": float(np.median(np.abs(angle_deg))),
        "xy_angle_split_half_fraction_within_10deg": float(np.mean(np.abs(angle_deg) < 10.0)),
        "xy_angle_deg_percentiles": {f"p{q}": float(np.degrees(np.percentile(angle, q))) for q in (5, 25, 50, 75, 95)},
    }
    print("[third-axis diagnostics] " + json.dumps(third_axis_diagnostics))

    # ---- partitions to audit ----
    candidates = {
        "fz_e_32x32": ([fz, energy], [32, 32]),
        "fz_e_40x40": ([fz, energy], [40, 40]),
        "fz_e_48x48": ([fz, energy], [48, 48]),
        "fz_e_55x55": ([fz, energy], [55, 55]),
        "fz_e_64x64": ([fz, energy], [64, 64]),
        "fz_e_lamz_16x16x4": ([fz, energy, lam], [16, 16, 4]),
        "fz_e_angle_16x16x4": ([fz, energy, angle], [16, 16, 4]),
        "fz_e_lamz_angle_16x8x8": ([fz, energy, lam, angle], [16, 8, 8]),
    }

    rng = np.random.default_rng(0)
    audit = {}
    for name, (variables, bins) in candidates.items():
        assignments, k_total = bundle_grid(variables, bins)
        audit[name] = {
            "bundle_grid": "x".join(str(n) for n in bins),
            "bundle_count": k_total,
            **equal_weight_distortion(design_dense, assignments, k_total),
            **projection_of_full_weights(design_dense, observed, w_full_active, assignments),
            "sampled_same_bundle_cosine_median": sampled_pair_cosines(design_dense, assignments, k_total, rng),
        }
        random_assignments = rng.permutation(assignments)
        audit[f"{name}_random"] = {
            "bundle_grid": audit[name]["bundle_grid"] + " (random)",
            "bundle_count": k_total,
            **equal_weight_distortion(design_dense, random_assignments, k_total),
            "sampled_same_bundle_cosine_median": sampled_pair_cosines(design_dense, random_assignments, k_total, rng),
        }

    # ---- oracle: k-means on raw response columns at 1024 and 2304 ----
    points = design_dense.T.copy()
    for k in (1024, 2304):
        started_kmeans = time.perf_counter()
        assignments = kmeans_columns(points, k, seed=0)
        audit[f"oracle_kmeans_{k}"] = {
            "bundle_grid": f"k-means k={k}",
            "bundle_count": int(assignments.max()) + 1,
            **equal_weight_distortion(design_dense, assignments, int(assignments.max()) + 1),
            **projection_of_full_weights(design_dense, observed, w_full_active, assignments),
            "sampled_same_bundle_cosine_median": sampled_pair_cosines(design_dense, assignments, k, rng),
            "kmeans_seconds": time.perf_counter() - started_kmeans,
        }

    print("[audit: lower D / lower proj chi2 = better equal-weight basis]")
    for key, entry in audit.items():
        print(f"    {key:>28s}  k={entry['bundle_count']:5d}  D={entry['distortion']:.4f}  "
              f"populated={entry['populated_bundles']:5d}  proj_chi2/bin={entry['projection_chi2_per_bin']:7.3f}  "
              f"cos_med={entry.get('sampled_same_bundle_cosine_median', float('nan')):.3f}")

    # ---- figure: bundle-count curve with oracle and candidates ----
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for axis, metric, label in ((axes[0], "distortion", "equal-weight column-mean distortion D"),
                                (axes[1], "projection_chi2_per_bin", "full-weight projection chi2/bin (no re-solve)")):
        for name, color, marker in (("fz_e_32x32", "#4C72B0", "o"), ("fz_e_40x40", "#4C72B0", "o"),
                                    ("fz_e_48x48", "#4C72B0", "o"), ("fz_e_55x55", "#4C72B0", "o"),
                                    ("fz_e_64x64", "#4C72B0", "o")):
            if name in audit:
                axis.scatter(audit[name]["bundle_count"], audit[name][metric], color=color, marker=marker, zorder=3)
        for name, color, marker in (("fz_e_32x32_random", "0.6"), ("fz_e_64x64_random", "0.6")):
            if name in audit:
                axis.scatter(audit[name]["bundle_count"], audit[name][metric], color=color, marker="x", zorder=3)
        for name, color in (("fz_e_lamz_16x16x4", "#DD8452"), ("fz_e_angle_16x16x4", "#55A868"),
                            ("fz_e_lamz_angle_16x8x8", "#8172B3")):
            axis.scatter(audit[name]["bundle_count"], audit[name][metric], color=color, marker="s", zorder=3)
        for k in (1024, 2304):
            axis.scatter(audit[f"oracle_kmeans_{k}"]["bundle_count"], audit[f"oracle_kmeans_{k}"][metric],
                         color="#C44E52", marker="*", s=140, zorder=4)
        axis.axhline(2.0 if metric == "projection_chi2_per_bin" else 0.0, color="crimson", lw=0.8, ls="--")
        axis.set_xscale("log")
        axis.set_xlabel("bundles")
        axis.set_ylabel(label)
    handles = [
        plt.Line2D([], [], color="#4C72B0", marker="o", ls="", label="(fz,E) 2D"),
        plt.Line2D([], [], color="0.6", marker="x", ls="", label="random"),
        plt.Line2D([], [], color="#DD8452", marker="s", ls="", label="3D +lam_z 16x16x4"),
        plt.Line2D([], [], color="#55A868", marker="s", ls="", label="3D +xy angle 16x16x4"),
        plt.Line2D([], [], color="#8172B3", marker="s", ls="", label="4D +both 16x8x8"),
        plt.Line2D([], [], color="#C44E52", marker="*", ls="", label="oracle k-means"),
    ]
    axes[1].legend(handles=handles, fontsize=8)
    fig.suptitle("bundle-count curve, third-invariant candidates, and the response-similarity oracle", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT / "third_invariant_audit.png", dpi=150)
    plt.close(fig)

    summary = {
        "design_shape": list(design_dense.shape),
        "third_axis_diagnostics": third_axis_diagnostics,
        "audit": audit,
        "notes": {
            "oracle": "plain k-means (kmeans++ init, Lloyd, raw columns) minimizes the same within-bundle distortion D that equal-weight bundling pays; its numbers bound any variable-based grouping at that count",
            "xy_angle": "principal axis of the orbit's xy occupancy covariance; period pi; split-half wrapped difference in degrees",
            "projection": "member-mean projection of the saved full-solve weights; solver-free but pessimistic (§10: re-solve recovered chi2 1.61 from projection 5.41 at 4096)",
        },
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"[done] audit artifacts in {OUTPUT} (wall {time.perf_counter() - started:.1f}s)")


if __name__ == "__main__":
    main()
