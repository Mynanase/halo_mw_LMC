#!/usr/bin/env python3
"""Orbit-portrait audit of the (Jz/Jtot, E) plane: near points, similar orbits?

The §10 review measured the (Jz/Jtot, E) basis in variable space and on the
density design matrix; this audit looks at the orbits themselves.  For a
spread of anchors in quantile-rank space (u = rank of Jz/Jtot, v = rank of E
over the nphi4 active orbits) it takes the pair of active orbits nearest the
anchor that the 64x64 (Jz/Jtot, E) quantile partition actually bundles
together, and draws their meridional-plane portraits (R = sqrt(x^2+y^2) vs
z, mirrored to -R, equal aspect) side by side plus an overlay, with an
xy-projection overlay for the phi side of the 4-sector response grid.
Control rows probe the separating power the portraits should reveal: same
energy rank but far apart in Jz/Jtot, same Jz/Jtot but far apart in energy,
and close under the audited (lambda_z, E) basis while far apart in Jz/Jtot
(a pair the old basis bundles into one 64x64 cell but the new one separates).

Each row is annotated with the raw variable differences, same-bundle flags
under both 64x64 partitions, per-orbit spatial summaries (r_apo, r_peri,
|z|max, z_rms) and the cosine similarity of the two orbits' error-normalized
density response columns.  A non-cherry-picked companion samples 400
same-bundle pairs, adjacent-column same-row pairs and random pairs and
compares their summary differences and response cosines.

Two index families are kept explicit throughout: successful-seed-relative
rows for trajectories, portraits and orbit tables; active-column-relative
indices for ranks, bundle assignments, design columns and response cosines.

Reads frozen artifacts only (frozen library, round-2 invariant table, the
§10 experiment's bundles.npz, nphi4 run configuration); no integration, no
solver.  Companion of docs/nphi1_bundling_repair_plan.md §10; outputs to
.agent-local/benchmarks/orbit_fz_energy_review/.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from benchmark_nphi1_bundling import quantile_bundle_grid
from review_fz_energy_basis import FROZEN_CACHE, INVARIANT_TABLE, build_design_problem

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

BUNDLES_64 = REPO / ".agent-local/benchmarks/nphi4_repair_2d_fz_energy/bundles.npz"
OUTPUT = REPO / ".agent-local/benchmarks/orbit_fz_energy_review"

NEAR_ANCHORS = [(0.20, 0.20), (0.50, 0.20), (0.80, 0.20), (0.20, 0.80), (0.50, 0.80), (0.80, 0.80)]


def rank01(values):
    """Quantile-rank positions in [0, 1]; the partition's own metric."""

    ranks = np.empty(values.size, dtype=float)
    ranks[np.argsort(values, kind="stable")] = np.arange(values.size, dtype=float)
    return ranks / max(values.size - 1, 1)


def orbit_spatial_summaries(phase):
    """Per-orbit (r_apo, r_peri, z_max, z_rms, R_rms) from 10-period samples."""

    orbit_count = phase.shape[0]
    summaries = {name: np.zeros(orbit_count) for name in ("r_apo", "r_peri", "z_max", "z_rms", "R_rms")}
    for start in range(0, orbit_count, 1500):
        block = phase[start:start + 1500]
        radius = np.sqrt(np.sum(block[..., :3] ** 2, axis=2))
        rows = slice(start, start + block.shape[0])
        summaries["r_apo"][rows] = np.percentile(radius, 99.5, axis=1)
        summaries["r_peri"][rows] = np.percentile(radius, 0.5, axis=1)
        summaries["z_max"][rows] = np.percentile(np.abs(block[..., 2]), 99.5, axis=1)
        summaries["z_rms"][rows] = np.sqrt(np.mean(block[..., 2] ** 2, axis=1))
        summaries["R_rms"][rows] = np.sqrt(np.mean(np.sum(block[..., :2] ** 2, axis=2), axis=1))
    return summaries


def portrait(ax, position, color, label, limits):
    """Meridional-plane scatter of one orbit, mirrored to -R, equal aspect."""

    cylindrical = np.sqrt(np.sum(position[:, :2] ** 2, axis=1))
    height = position[:, 2]
    ax.scatter(cylindrical, height, s=1.0, color=color, alpha=0.30, linewidths=0)
    ax.scatter(-cylindrical, height, s=1.0, color=color, alpha=0.30, linewidths=0)
    ax.set_xlim(-limits[0], limits[0])
    ax.set_ylim(-limits[1], limits[1])
    ax.set_aspect("equal")
    ax.set_title(label, fontsize=8)


def pair_row(axis_row, row_a, row_b, phase, seeds, summaries, fz_all, energy_all, row_title, extra):
    """One row of three panels for two successful-seed rows: A, B, overlay."""

    limits = (1.05 * max(summaries["r_apo"][row_a], summaries["r_apo"][row_b]),
              1.05 * max(summaries["z_max"][row_a], summaries["z_max"][row_b]))
    portrait(axis_row[0], phase[row_a], "#4C72B0",
             f"seed {seeds[row_a]}  fz={fz_all[row_a]:.3f}  E={energy_all[row_a]/1e3:.1f}k", limits)
    portrait(axis_row[1], phase[row_b], "#C44E52",
             f"seed {seeds[row_b]}  fz={fz_all[row_b]:.3f}  E={energy_all[row_b]/1e3:.1f}k", limits)
    portrait(axis_row[2], phase[row_a], "#4C72B0", "", limits)
    cylindrical_b = np.sqrt(np.sum(phase[row_b][:, :2] ** 2, axis=1))
    axis_row[2].scatter(cylindrical_b, phase[row_b][:, 2], s=1.0, color="#C44E52", alpha=0.30, linewidths=0)
    axis_row[2].scatter(-cylindrical_b, phase[row_b][:, 2], s=1.0, color="#C44E52", alpha=0.30, linewidths=0)
    axis_row[2].set_title(extra, fontsize=8)
    axis_row[0].set_ylabel(row_title, fontsize=9)
    for axis in axis_row:
        axis.set_xlabel("R [kpc]", fontsize=8)


def describe_pair(i, j, active, seeds, fz, energy, assign_fz, assign_lam, summaries_active, extra=None):
    """JSON record of one active-relative pair; trajectories use seed rows."""

    entry = {
        "pair_active": [int(i), int(j)],
        "pair_seed_rows": [int(active[i]), int(active[j])],
        "seeds": [int(seeds[active[i]]), int(seeds[active[j]])],
        "fz": [float(fz[i]), float(fz[j])],
        "energy": [float(energy[i]), float(energy[j])],
        "same_fz_energy_bundle_64": bool(assign_fz[i] == assign_fz[j]),
        "same_lam_energy_bundle_64": bool(assign_lam[i] == assign_lam[j]),
        "summaries": {name: [float(values[i]), float(values[j])] for name, values in summaries_active.items()},
    }
    if extra:
        entry.update(extra)
    return entry


def main() -> None:
    started = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # ---- input boundary: variables, partition, design columns, trajectories ----
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
        expected = np.repeat(successful, sample_count)
        if not np.array_equal(frozen["library_seed_index"], expected):
            raise ValueError("frozen library samples are not contiguous per successful seed")
        phase = frozen["library_phase_space"].reshape(successful.size, int(sample_count[0]), 6)
    if not np.array_equal(seeds, successful):
        raise ValueError("experiment bundle table and frozen library disagree on successful seeds")

    problem, _ = build_design_problem()
    active = np.flatnonzero(problem.active_columns)
    design_dense = np.asarray(problem.design.todense(), dtype=float)
    print(f"design {design_dense.shape}; active orbits {active.size} of {seeds.size}")

    fz, energy, lam = fz_all[active], energy_all[active], lam_all[active]
    u_fz, v_e, u_lam = rank01(fz), rank01(energy), rank01(lam)
    assign_fz, _ = quantile_bundle_grid(fz, energy, 64, 64)
    assign_lam, _ = quantile_bundle_grid(lam, energy, 64, 64)
    fz_bin, e_bin = assign_fz // 64, assign_fz % 64

    column_norm = np.linalg.norm(design_dense, axis=0)

    def response_cosine(i, j):
        return float(design_dense[:, i] @ design_dense[:, j] / (column_norm[i] * column_norm[j]))

    summaries = orbit_spatial_summaries(phase)
    summaries_active = {name: values[active] for name, values in summaries.items()}

    # ---- near pairs: first two same-bundle orbits met walking out from the anchor ----
    def nearest_same_bundle_pair(anchor_u, anchor_v):
        walk = np.argsort((u_fz - anchor_u) ** 2 + (v_e - anchor_v) ** 2)
        seen: dict[int, int] = {}
        for index in walk:
            bundle = assign_fz[index]
            if bundle in seen:
                return seen[bundle], int(index)
            seen[bundle] = int(index)
        raise ValueError("no bundle with two members reachable from the anchor")

    near_pairs = []
    for anchor_u, anchor_v in NEAR_ANCHORS:
        i, j = nearest_same_bundle_pair(anchor_u, anchor_v)
        near_pairs.append(describe_pair(
            i, j, active, seeds, fz, energy, assign_fz, assign_lam, summaries_active,
            extra={
                "anchor": [anchor_u, anchor_v],
                "rank_distance": float(np.hypot(u_fz[i] - u_fz[j], v_e[i] - v_e[j])),
                "response_cosine": response_cosine(i, j),
            },
        ))

    # ---- control pairs ----
    i_a = int(np.argmin(np.where(u_fz < 0.12, np.abs(v_e - 0.30), np.inf)))
    j_a = int(np.argmin(np.where(u_fz > 0.88, np.abs(v_e - v_e[i_a]), np.inf)))
    i_b = int(np.argmin(np.where(v_e < 0.12, np.abs(u_fz - 0.50), np.inf)))
    j_b = int(np.argmin(np.where(v_e > 0.88, np.abs(u_fz - u_fz[i_b]), np.inf)))
    # The lambda_z control needs a coarsening where a cell actually holds many
    # orbits (64x64 cells average ~1.4 members and cannot mix anything): use
    # the 16x16 (lam_z, E) partition and take its most fz-diverse cell.
    assign_lam16, _ = quantile_bundle_grid(lam, energy, 16, 16)
    most_diverse, diversity = None, -1.0
    for cell in np.unique(assign_lam16):
        group = np.flatnonzero(assign_lam16 == cell)
        if group.size < 2:
            continue
        low, high = group[np.argmin(u_fz[group])], group[np.argmax(u_fz[group])]
        spread = float(abs(u_fz[high] - u_fz[low]))
        if spread > diversity:
            most_diverse, diversity = (int(low), int(high)), spread
    if most_diverse is None or diversity < 0.3:
        raise ValueError("no 16x16 (lam_z, E) cell with fz-diverse members found for the control")
    controls = [
        ("same E, far fz", i_a, j_a),
        ("same fz, far E", i_b, j_b),
        ("one 16x16 (lam_z,E) cell, far fz", *most_diverse),
    ]
    control_entries = [
        describe_pair(i, j, active, seeds, fz, energy, assign_fz, assign_lam, summaries_active,
                      extra={"label": label, "response_cosine": response_cosine(i, j)})
        for label, i, j in controls
    ]

    # ---- figures ----
    fig, axes = plt.subplots(len(near_pairs), 3, figsize=(12.5, 3.6 * len(near_pairs)))
    for entry, axis_row in zip(near_pairs, axes):
        row_a, row_b = entry["pair_seed_rows"]
        pair_row(axis_row, row_a, row_b, phase, seeds, summaries, fz_all, energy_all,
                 row_title=f"anchor ({entry['anchor'][0]:.2f}, {entry['anchor'][1]:.2f})"
                           + (" near-duplicate" if entry["response_cosine"] > 0.999 else ""),
                 extra=(f"dfz={entry['fz'][1]-entry['fz'][0]:+.3f}  dE={entry['energy'][1]-entry['energy'][0]:+.0f}\n"
                        f"cos(resp)={entry['response_cosine']:.2f}  dz_rms={abs(entry['summaries']['z_rms'][1]-entry['summaries']['z_rms'][0]):.2f} kpc"))
    fig.suptitle("(Jz/Jtot, E): nearest same-bundle orbit pairs - meridional portraits (nphi4 active orbits)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT / "pair_portraits_near.png", dpi=125)
    plt.close(fig)

    fig, axes = plt.subplots(len(controls), 3, figsize=(12.5, 3.6 * len(controls)))
    for entry, axis_row in zip(control_entries, axes):
        row_a, row_b = entry["pair_seed_rows"]
        pair_row(axis_row, row_a, row_b, phase, seeds, summaries, fz_all, energy_all,
                 row_title=entry["label"].split(",")[0],
                 extra=(f"dfz={entry['fz'][1]-entry['fz'][0]:+.3f}  dE={entry['energy'][1]-entry['energy'][0]:+.0f}\n"
                        f"cos(resp)={entry['response_cosine']:.2f}  fz-cell64 same: {entry['same_fz_energy_bundle_64']}"))
    fig.suptitle("control pairs: the axes must separate what they claim to separate", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT / "pair_portraits_controls.png", dpi=125)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.4))
    for entry, ax in zip(near_pairs, axes.ravel()):
        row_a, row_b = entry["pair_seed_rows"]
        ax.scatter(phase[row_a][:, 0], phase[row_a][:, 1], s=1.0, color="#4C72B0", alpha=0.30, linewidths=0)
        ax.scatter(phase[row_b][:, 0], phase[row_b][:, 1], s=1.0, color="#C44E52", alpha=0.30, linewidths=0)
        limit = 1.05 * max(summaries["r_apo"][row_a], summaries["r_apo"][row_b])
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal")
        ax.set_title(f"anchor ({entry['anchor'][0]:.2f}, {entry['anchor'][1]:.2f})  cos(resp)={entry['response_cosine']:.2f}", fontsize=9)
        ax.set_xlabel("x [kpc]", fontsize=8)
        ax.set_ylabel("y [kpc]", fontsize=8)
    fig.suptitle("(Jz/Jtot, E) near pairs - xy projection overlay (blue / red)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT / "pair_portraits_near_xy.png", dpi=125)
    plt.close(fig)

    # ---- non-cherry-picked companion: 400 pairs per category ----
    rng = np.random.default_rng(0)
    bundle_order = np.argsort(assign_fz, kind="stable")
    bundle_bounds = np.searchsorted(assign_fz[bundle_order], np.arange(int(assign_fz.max()) + 2))
    members = {
        bundle: bundle_order[bundle_bounds[bundle]:bundle_bounds[bundle + 1]]
        for bundle in range(int(assign_fz.max()) + 1)
        if bundle_bounds[bundle + 1] - bundle_bounds[bundle] >= 2
    }

    def sample_pairs(category):
        if category == "same_bundle":
            keys = rng.choice(np.array(sorted(members)), size=400, replace=True)
            return [tuple(rng.choice(members[int(key)], size=2, replace=False)) for key in keys]
        if category == "same_row_adjacent_column":
            pairs = []
            while len(pairs) < 400:
                i = int(rng.integers(active.size))
                candidates = np.flatnonzero((fz_bin == fz_bin[i] + 1) & (e_bin == e_bin[i]))
                if candidates.size:
                    pairs.append((i, int(candidates[int(rng.integers(candidates.size))])))
            return pairs
        return [tuple(rng.integers(0, active.size, size=2)) for _ in range(400)]

    def pair_statistics(pairs):
        left = np.array([i for i, _ in pairs])
        right = np.array([j for _, j in pairs])
        result = {"response_cosine_median": float(np.median([response_cosine(i, j) for i, j in zip(left, right)]))}
        for name in ("r_apo", "z_max", "z_rms"):
            values_l, values_r = summaries_active[name][left], summaries_active[name][right]
            scale = np.maximum((values_l + values_r) / 2, 1e-9)
            result[f"median_abs_delta_{name}"] = float(np.median(np.abs(values_l - values_r)))
            result[f"median_rel_delta_{name}"] = float(np.median(np.abs(values_l - values_r) / scale))
        return result

    statistics = {category: pair_statistics(sample_pairs(category))
                  for category in ("same_bundle", "same_row_adjacent_column", "random")}

    mixed = [tuple(rng.integers(0, active.size, size=2)) for _ in range(800)]
    distances = np.array([np.hypot(u_fz[i] - u_fz[j], v_e[i] - v_e[j]) for i, j in mixed])
    z_deltas = np.array([abs(summaries_active["z_rms"][i] - summaries_active["z_rms"][j]) for i, j in mixed])
    rank_distance_vs_z_delta = float(spearmanr(distances, z_deltas).statistic)

    print("[near same-bundle pairs: portraits]")
    for entry in near_pairs:
        print(f"    anchor ({entry['anchor'][0]:.2f},{entry['anchor'][1]:.2f}): seeds {entry['seeds']} "
              f"dfz={entry['fz'][1]-entry['fz'][0]:+.4f} dE={entry['energy'][1]-entry['energy'][0]:+.1f} "
              f"cos={entry['response_cosine']:.3f} "
              f"dz_rms={abs(entry['summaries']['z_rms'][1]-entry['summaries']['z_rms'][0]):.2f}kpc")
    print("[controls]")
    for entry in control_entries:
        print(f"    {entry['label']:34s}: dfz={entry['fz'][1]-entry['fz'][0]:+.3f} dE={entry['energy'][1]-entry['energy'][0]:+.0f} "
              f"cos={entry['response_cosine']:.3f} fz_cell64_same={entry['same_fz_energy_bundle_64']} "
              f"z_rms=[{entry['summaries']['z_rms'][0]:.1f},{entry['summaries']['z_rms'][1]:.1f}]kpc")
    print("[400-pair statistics]")
    for category, entry in statistics.items():
        print(f"    {category:26s}: cos_med={entry['response_cosine_median']:.3f} "
              f"rel_dRapo={entry['median_rel_delta_r_apo']:.3f} rel_dzmax={entry['median_rel_delta_z_max']:.3f} "
              f"rel_dzrms={entry['median_rel_delta_z_rms']:.3f}")
    print(f"[spearman] rank-distance vs |dz_rms| over 800 random pairs: {rank_distance_vs_z_delta:+.3f}")

    summary = {
        "active_orbits": int(active.size),
        "near_pairs": near_pairs,
        "controls": control_entries,
        "pair_statistics_400": statistics,
        "spearman_rank_distance_vs_abs_delta_z_rms_800_pairs": rank_distance_vs_z_delta,
        "notes": {
            "portrait": "meridional plane (R vs z, mirrored to -R, equal aspect) of 10-period samples; per-row shared axis limits",
            "near_pair_selection": "walk outward from each anchor in (rank fz, rank E) until two orbits of the same 64x64 (fz,E) quantile bundle appear",
            "pair_statistics": "same_bundle = both members in one 64x64 cell; same_row_adjacent_column = one column apart at fixed energy row; random = unrestricted",
        },
    }
    (OUTPUT / "pair_portraits_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"[done] portraits and statistics in {OUTPUT} (wall {time.perf_counter() - started:.1f}s)")


if __name__ == "__main__":
    main()
