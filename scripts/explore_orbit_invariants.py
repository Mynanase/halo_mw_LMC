#!/usr/bin/env python3
"""Exploratory orbit-invariant diagnostics for bundling-variable selection.

Round 2.  Reads the frozen paper-best orbit library (no new integration, no
solver) and computes, per orbit, the classical invariants used or considered
for bundling -- mean energy, Lz / |L|, circularity lambda_z, orbital
inclination, pericenter/apocenter, eccentricity -- plus Staeckel-fudge actions
(Jr, Jz, Jphi) evaluated on every orbit sample in TWO axisymmetric stand-ins
for the triaxial Zhu et al. (2026) potential:

* ``swap``  -- the round-1 ad-hoc component swap (Ferrers bulge replaced by a
  same-mass spherical Plummer, every remaining p set to 1);
* ``phi``   -- the azimuthal Fourier m=0 average of the true triaxial
  potential, built as ``agama.Potential(type='CylSpline', density=true_pot,
  mmax=0)``.  CylSpline performs a genuine azimuthal Fourier decomposition of
  the source and Poisson's equation decouples per Fourier mode, so the m=0
  term is mathematically the phi-average of the true potential; only spline
  interpolation error remains (checked against a direct numerical average).

A third, potential-model-free basis is added: per-orbit fundamental
frequencies (omega_r, omega_z, omega_phi) estimated by FFT peak finding on
the sampled time series r(t), z(t) and the de-trended unwrapped phi(t).
Frequencies are observables of the true orbit in the true potential and
involve no potential approximation at all; their precision is limited by the
finite 10-period span.

Conservation-vs-discriminability is measured uniformly for every candidate
variable as the median split-half difference of orbit means divided by the
inter-orbit interquartile range.  Outputs (figures + npz + json) answer:
which variables are well conserved, which pairs are close to orthogonal
(|Spearman rho| near zero), how much the phi-average improves on the swap
proxy, and whether the frequency basis is a viable proxy-free bundling set.
Companion exploration of docs/nphi1_bundling_repair_plan.md; touches no
production artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
# Reuse the reviewed grouping math: the circularity scale so the lambda_z here
# is exactly the variable the (lambda_z, E) bundling partitions on, and the
# shared FFT frequency estimator now owned by the benchmark driver.
from benchmark_nphi1_bundling import _circularity_scale, dominant_frequency

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
from scipy.stats import spearmanr

ORBIT_BLOCK = 1500

# Sample-level columns reduced to orbit means (suffix _swap / _phi for the
# two action potentials); frequency columns come from the time series pass.
TABLE_COLUMNS = [
    "energy", "lam_z", "lz", "l_mod", "cos_i",
    "ecc", "r_apo", "r_peri", "r_mean",
    "jr_swap", "jz_swap", "jphi_swap",
    "jr_phi", "jz_phi", "jphi_phi", "jtot_phi", "f_r_phi", "f_z_phi",
    "omega_r", "omega_z", "omega_phi", "ratio_rz", "ratio_phi_r",
]
INTRAOFF_COLUMNS = [
    "energy", "lam_z", "lz", "l_mod", "jr_swap", "jz_swap", "jr_phi", "jz_phi",
]
SPLITHALF_COLUMNS = [
    "energy", "lam_z", "l_mod", "jr_swap", "jz_swap", "jr_phi", "jz_phi",
    "omega_r", "omega_z", "omega_phi", "ratio_rz", "ratio_phi_r",
]

FREQUENCY_CORNER = ["energy", "omega_r", "omega_z", "omega_phi", "ratio_rz", "ratio_phi_r", "ecc"]
CORRELATION_SET = [
    "energy", "lam_z", "cos_i", "ecc", "r_apo", "l_mod",
    "jr_phi", "jz_phi", "jphi_phi",
    "omega_r", "omega_z", "omega_phi", "ratio_rz", "ratio_phi_r",
]
LABELS = {
    "energy": "$\\langle E\\rangle$ [km$^2$/s$^2$]",
    "lam_z": "$\\langle\\lambda_z\\rangle$",
    "lz": "$\\langle L_z\\rangle$ [kpc km/s]",
    "l_mod": "$\\langle |L|\\rangle$ [kpc km/s]",
    "cos_i": "$\\langle L_z/|L|\\rangle$",
    "ecc": "$e$",
    "r_apo": "$r_{\\rm apo}$ [kpc]",
    "r_peri": "$r_{\\rm peri}$ [kpc]",
    "r_mean": "$\\langle r\\rangle$ [kpc]",
    "jr_swap": "$J_r$ (swap proxy)",
    "jz_swap": "$J_z$ (swap proxy)",
    "jphi_swap": "$J_\\phi$ (swap proxy)",
    "jr_phi": "$J_r$ ($\\phi$-avg)",
    "jz_phi": "$J_z$ ($\\phi$-avg)",
    "jphi_phi": "$J_\\phi$ ($\\phi$-avg)",
    "jtot_phi": "$J_{\\rm tot}$ ($\\phi$-avg)",
    "f_r_phi": "$J_r/J_{\\rm tot}$ ($\\phi$-avg)",
    "f_z_phi": "$J_z/J_{\\rm tot}$ ($\\phi$-avg)",
    "omega_r": "$\\omega_r$",
    "omega_z": "$\\omega_z$",
    "omega_phi": "$\\omega_\\phi$",
    "ratio_rz": "$\\omega_r/\\omega_z$",
    "ratio_phi_r": "$\\omega_\\phi/\\omega_r$",
}
SPLITHALF_LABELS = {
    "energy": "E", "lam_z": "$\\lambda_z$", "l_mod": "|L|",
    "jr_swap": "$J_r$ (swap)", "jz_swap": "$J_z$ (swap)",
    "jr_phi": "$J_r$ ($\\phi$-avg)", "jz_phi": "$J_z$ ($\\phi$-avg)",
    "omega_r": "$\\omega_r$", "omega_z": "$\\omega_z$", "omega_phi": "$\\omega_\\phi$",
    "ratio_rz": "$\\omega_r/\\omega_z$", "ratio_phi_r": "$\\omega_\\phi/\\omega_r$",
}


def build_component_swap_proxy(parameters):
    """Round-1 ad-hoc axisymmetric stand-in: Plummer for Ferrers, p->1 elsewhere."""

    import agama

    from halo_mw_lmc.potential import zhu_2026_component_parameters

    components = []
    for component in zhu_2026_component_parameters(
        rho0=parameters.rho0, log_rs=parameters.log_rs,
        phalo=parameters.phalo, qhalo=parameters.qhalo, gamma=parameters.gamma,
    ):
        component = dict(component)
        if component["type"] == "Ferrers":
            components.append({
                "type": "Plummer",
                "mass": component["mass"],
                "scaleRadius": component["scaleRadius"],
            })
        elif "p" in component:
            component["p"] = 1.0
            components.append(component)
        else:
            components.append(component)
    return agama.Potential(*components)


def build_phi_averaged_potential(true_potential):
    """Azimuthal m=0 Fourier average of the true triaxial potential."""

    import agama

    try:
        return agama.Potential(type="CylSpline", density=true_potential, mmax=0), "CylSpline mmax=0"
    except Exception as error:
        print(f"[setup] CylSpline construction failed ({error}); falling back to Multipole mmax=0")
        return agama.Potential(type="Multipole", density=true_potential, mmax=0), "Multipole mmax=0"


def check_phi_average_oracle(true_potential, averaged_potential):
    """The m=0 potential must equal the numerical phi-average; circles stay exact."""

    max_relative = 0.0
    for radius, height in [(5., 0.), (8., 4.), (20., 0.), (20., 10.), (30., 15.), (45., 25.), (60., 0.)]:
        phi = np.linspace(0.0, 2.0 * np.pi, 145)[:-1]
        points = np.column_stack([
            radius * np.cos(phi), radius * np.sin(phi), np.full(phi.size, height),
        ])
        average = float(np.mean(true_potential.potential(points)))
        value = float(averaged_potential.potential(np.array([[radius, 0.0, height]]))[0])
        max_relative = max(max_relative, abs(value - average) / abs(average))

    import agama

    finder = agama.ActionFinder(averaged_potential)
    worst_circular = 0.0
    for radius in (8.0, 20.0, 40.0):
        force = np.asarray(averaged_potential.force(np.array([[radius, 0.0, 0.0]])))[0]
        v_circ = np.sqrt(radius * abs(force[0]))
        actions = np.asarray(finder(np.array([[radius, 0.0, 0.0, 0.0, v_circ, 0.0]])))[0]
        worst_circular = max(worst_circular, abs(actions[0]), abs(actions[1]),
                             abs(actions[2] - radius * v_circ))
    return {
        "phi_average_max_relative_difference": max_relative,
        "circular_orbit_max_absolute_error": worst_circular,
    }


def resolve_grouping_parameters(frozen_provenance_path: Path):
    """Same potential-parameter resolution as the repair experiment grouping."""

    from halo_mw_lmc.potential import ZHU_2026_BEST_FIT, ZhuHaloParameters

    if frozen_provenance_path.exists():
        recorded = json.loads(frozen_provenance_path.read_text()).get("potential_parameters")
        if recorded:
            note = "potential parameters taken from the frozen-run provenance record"
            return ZhuHaloParameters(
                rho0=recorded["rho0"], log_rs=recorded["log_rs"],
                phalo=recorded["phalo"], qhalo=recorded["qhalo"], gamma=recorded["gamma"],
            ), note
    note = (
        "frozen provenance records no potential parameters; using ZHU_2026_BEST_FIT "
        "with log_rs=log10(70), the same fallback the repair experiment used for grouping"
    )
    return ZhuHaloParameters(
        rho0=ZHU_2026_BEST_FIT["rho0"], log_rs=ZHU_2026_BEST_FIT["log_rs"],
        phalo=ZHU_2026_BEST_FIT["phalo"], qhalo=ZHU_2026_BEST_FIT["qhalo"],
        gamma=ZHU_2026_BEST_FIT["gamma"],
    ), note


def invariant_pass(phase, true_potential, e_circ, lz_circ, action_finders):
    """Pass A: per-sample invariants and actions reduced to per-orbit statistics."""

    orbit_count = phase.shape[0]
    means = {name: np.zeros(orbit_count) for name in TABLE_COLUMNS}
    stds = {name: np.zeros(orbit_count) for name in INTRAOFF_COLUMNS}
    half = {  # split-half orbit means for the uniform conservation metric
        name: np.zeros((orbit_count, 2)) for name in SPLITHALF_COLUMNS if not name.startswith(("omega", "ratio"))
    }
    middle = phase.shape[1] // 2
    for start in range(0, orbit_count, ORBIT_BLOCK):
        block = phase[start:start + ORBIT_BLOCK]
        pos, vel = block[..., :3], block[..., 3:]
        radius = np.sqrt(np.sum(pos ** 2, axis=2))
        energy = 0.5 * np.sum(vel ** 2, axis=2) + np.asarray(
            true_potential.potential(pos.reshape(-1, 3)), dtype=float
        ).reshape(block.shape[:2])
        lz = pos[..., 0] * vel[..., 1] - pos[..., 1] * vel[..., 0]
        l_mod = np.sqrt(np.sum(np.cross(pos, vel) ** 2, axis=2))
        energy_clipped = np.clip(energy, e_circ[0], e_circ[-1])
        lam_z = lz / np.maximum(np.interp(energy_clipped, e_circ, lz_circ), 1e-12)
        actions = {
            label: np.asarray(finder(block.reshape(-1, 6)), dtype=float).reshape(block.shape[:2] + (3,))
            for label, finder in action_finders.items()
        }
        jtot = actions["phi"][..., 0] + actions["phi"][..., 1] + np.abs(actions["phi"][..., 2])

        rows = slice(start, start + block.shape[0])
        means["energy"][rows] = energy.mean(axis=1)
        means["lam_z"][rows] = lam_z.mean(axis=1)
        means["lz"][rows] = lz.mean(axis=1)
        means["l_mod"][rows] = l_mod.mean(axis=1)
        means["cos_i"][rows] = (lz / np.maximum(l_mod, 1e-12)).mean(axis=1)
        means["r_mean"][rows] = radius.mean(axis=1)
        means["r_peri"][rows] = np.percentile(radius, 0.5, axis=1)
        means["r_apo"][rows] = np.percentile(radius, 99.5, axis=1)
        means["ecc"][rows] = (means["r_apo"][rows] - means["r_peri"][rows]) / np.maximum(
            means["r_apo"][rows] + means["r_peri"][rows], 1e-12
        )
        for label in ("swap", "phi"):
            for index, name in enumerate(("jr", "jz", "jphi")):
                means[f"{name}_{label}"][rows] = actions[label][..., index].mean(axis=1)
                std_name = f"{name}_{label}"
                if std_name in stds:
                    stds[std_name][rows] = actions[label][..., index].std(axis=1)
        means["jtot_phi"][rows] = jtot.mean(axis=1)
        safe_jtot = np.maximum(means["jtot_phi"][rows], 1e-12)
        means["f_r_phi"][rows] = means["jr_phi"][rows] / safe_jtot
        means["f_z_phi"][rows] = means["jz_phi"][rows] / safe_jtot
        stds["energy"][rows] = energy.std(axis=1)
        stds["lam_z"][rows] = lam_z.std(axis=1)
        stds["lz"][rows] = lz.std(axis=1)
        stds["l_mod"][rows] = l_mod.std(axis=1)

        halves = ((slice(None), slice(0, middle)), (slice(None), slice(middle, None)))
        for name in half:
            source = {
                "energy": energy, "lam_z": lam_z, "l_mod": l_mod,
                "jr_swap": actions["swap"][..., 0], "jz_swap": actions["swap"][..., 1],
                "jr_phi": actions["phi"][..., 0], "jz_phi": actions["phi"][..., 1],
            }[name]
            for side, half_slice in enumerate(halves):
                half[name][rows, side] = source[half_slice].mean(axis=1)
    return means, stds, half


def mean_rate(times, phi):
    """Least-squares slope of the unwrapped phi(t) in cycles per time unit."""

    phi_unwrapped = np.unwrap(phi, axis=1)
    time_centered = times - times.mean(axis=1, keepdims=True)
    numerator = np.sum(time_centered * (phi_unwrapped - phi_unwrapped.mean(axis=1, keepdims=True)), axis=1)
    return numerator / (2.0 * np.pi) / np.sum(time_centered ** 2, axis=1)


def frequency_pass(phase, times):
    """Pass B: FFT frequencies of r(t), z(t) and the phi(t) circulation rate.

    Each orbit is integrated over its own 10 periods, so the sample step dt
    is read per orbit from the median of the within-orbit time differences.
    """

    def estimate(block_phase, block_times):
        pos = block_phase[..., :3]
        radius = np.sqrt(np.sum(pos ** 2, axis=2))
        phi = np.arctan2(pos[..., 1], pos[..., 0])
        dt_rows = np.median(np.diff(block_times, axis=1), axis=1)
        return {
            "omega_r": dominant_frequency(radius, dt_rows),
            "omega_z": dominant_frequency(pos[..., 2], dt_rows),
            "omega_phi": mean_rate(block_times, phi),
        }

    orbit_count = phase.shape[0]
    full = {name: np.zeros(orbit_count) for name in ("omega_r", "omega_z", "omega_phi")}
    halves = {name: np.zeros((orbit_count, 2)) for name in ("omega_r", "omega_z", "omega_phi")}
    middle = phase.shape[1] // 2
    for start in range(0, orbit_count, ORBIT_BLOCK):
        block_phase = phase[start:start + ORBIT_BLOCK]
        block_times = times[start:start + ORBIT_BLOCK]
        rows = slice(start, start + block_phase.shape[0])
        estimates = estimate(block_phase, block_times)
        for name in full:
            full[name][rows] = estimates[name]
            halves[name][rows, 0] = estimate(block_phase[:, :middle], block_times[:, :middle])[name]
            halves[name][rows, 1] = estimate(block_phase[:, middle:], block_times[:, middle:])[name]
    result = dict(full)
    result["ratio_rz"] = full["omega_r"] / np.maximum(np.abs(full["omega_z"]), 1e-9)
    result["ratio_phi_r"] = full["omega_phi"] / np.maximum(np.abs(full["omega_r"]), 1e-9)
    halves["ratio_rz"] = np.column_stack([
        halves["omega_r"][:, side] / np.maximum(np.abs(halves["omega_z"][:, side]), 1e-9) for side in (0, 1)
    ])
    halves["ratio_phi_r"] = np.column_stack([
        halves["omega_phi"][:, side] / np.maximum(np.abs(halves["omega_r"][:, side]), 1e-9) for side in (0, 1)
    ])
    return result, halves


def frequency_self_check():
    """Inline oracle: the estimator must recover a known synthetic frequency."""

    n, dt = 1000, 0.0274
    t = np.arange(n) * dt
    rng = np.random.default_rng(0)
    series = np.stack([
        np.sin(2 * np.pi * 3.7 * t + rng.uniform(0, 2 * np.pi)) + 0.3 * rng.standard_normal(n)
        for _ in range(50)
    ])
    recovered = dominant_frequency(series, dt)
    return float(np.median(np.abs(recovered - 3.7) / 3.7))


def corner_figure(table, variables, path: Path, title: str):
    """Lower-triangle pairs plot with per-panel Spearman annotation."""

    n = len(variables)
    fig, axes = plt.subplots(n, n, figsize=(2.2 * n, 2.2 * n))
    for row in range(n):
        for col in range(n):
            ax = axes[row, col]
            y_name, x_name = variables[row], variables[col]
            if row == col:
                ax.hist(table[y_name], bins=90, color="#4C72B0")
                ax.set_yscale("log")
            elif col < row:
                ax.hist2d(table[x_name], table[y_name], bins=70, cmin=1, norm=LogNorm(), cmap="viridis")
                rho = spearmanr(table[x_name], table[y_name]).statistic
                ax.text(
                    0.03, 0.96, f"$\\rho={rho:+.2f}$", transform=ax.transAxes,
                    va="top", ha="left", fontsize=8, color="white",
                    bbox={"facecolor": "black", "alpha": 0.55, "pad": 1.5, "edgecolor": "none"},
                )
            else:
                ax.set_visible(False)
                continue
            if row == n - 1:
                ax.set_xlabel(LABELS[x_name], fontsize=9)
            if col == 0 and row > 0:
                ax.set_ylabel(LABELS[y_name], fontsize=9)
            ax.tick_params(labelsize=7)
            if row != col:
                ax.xaxis.set_major_locator(MaxNLocator(4))
                ax.yaxis.set_major_locator(MaxNLocator(4))
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def correlation_heatmap(table, variables, path: Path):
    n = len(variables)
    matrix = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            rho = spearmanr(table[variables[i]], table[variables[j]]).statistic
            matrix[i, j] = matrix[j, i] = rho
    fig, ax = plt.subplots(figsize=(1.0 * n + 3, 1.0 * n))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n), [LABELS[v] for v in variables], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n), [LABELS[v] for v in variables], fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{matrix[i, j]:+.2f}", ha="center", va="center", fontsize=6.5)
    fig.colorbar(image, ax=ax, shrink=0.8, label="Spearman $\\rho$")
    ax.set_title("Spearman rank correlation of orbit-mean invariants (round 2)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return matrix


def lambda_focus_figure(table, path: Path):
    lam = table["lam_z"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(lam, bins=200, color="#4C72B0")
    axes[0].set_yscale("log")
    axes[0].axvline(0.0, color="k", lw=0.8)
    axes[0].set_xlabel(LABELS["lam_z"])
    axes[0].set_title("orbit-mean circularity piles up near zero")
    axes[1].hist2d(table["energy"], lam, bins=80, cmin=1, norm=LogNorm(), cmap="viridis")
    axes[1].xaxis.set_major_locator(MaxNLocator(5))
    axes[1].set_xlabel(LABELS["energy"])
    axes[1].set_ylabel(LABELS["lam_z"])
    axes[1].set_title("circularity vs energy")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def proxy_comparison_figure(table, path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for index, name in enumerate(("jr", "jz", "jphi")):
        swap, phi = table[f"{name}_swap"], table[f"{name}_phi"]
        top, bottom = axes[0, index], axes[1, index]
        top.hist2d(swap, phi, bins=80, cmin=1, norm=LogNorm(), cmap="viridis")
        limit = max(float(np.percentile(np.abs(phi), 99.5)), float(np.percentile(np.abs(swap), 99.5)))
        top.plot([-limit, limit], [-limit, limit], color="crimson", lw=1.0)
        top.set_xlim(-0.05 * limit, limit)
        top.set_ylim(-0.05 * limit, limit)
        top.set_xlabel(LABELS[f"{name}_swap"], fontsize=9)
        top.set_ylabel(LABELS[f"{name}_phi"], fontsize=9)
        rho = spearmanr(swap, phi).statistic
        top.set_title(f"$\\rho={rho:+.3f}$", fontsize=10)
        relative = (phi - swap) / np.maximum(np.abs(swap), 1e-9)
        bottom.hist(np.clip(relative, -1.0, 1.0), bins=120, color="#4C72B0")
        bottom.axvline(0.0, color="k", lw=0.8)
        bottom.set_xlabel(f"relative difference ({LABELS[f'{name}_phi']} - swap)/|swap|", fontsize=9)
        bottom.set_yscale("log")
    axes[0, 0].set_title("")
    axes[0, 0].text(0.03, 0.96, f"$\\rho={spearmanr(table['jr_swap'], table['jr_phi']).statistic:+.3f}$",
                    transform=axes[0, 0].transAxes, va="top", color="white", fontsize=10,
                    bbox={"facecolor": "black", "alpha": 0.55, "pad": 1.5, "edgecolor": "none"})
    fig.suptitle("swap proxy vs azimuthal-average actions, same orbits", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def conservation_figure(half_means, table, path: Path):
    """Uniform split-half noise/signal for every candidate bundling variable."""

    names, ratios = [], []
    for name in SPLITHALF_COLUMNS:
        noise = float(np.median(np.abs(half_means[name][:, 0] - half_means[name][:, 1])))
        signal = float(np.percentile(table[name], 75) - np.percentile(table[name], 25))
        names.append(SPLITHALF_LABELS[name])
        ratios.append(noise / signal if signal > 0 else np.inf)
    order = np.argsort(ratios)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh([names[i] for i in order], [ratios[i] for i in order], color="#4C72B0")
    ax.axvline(1.0, color="crimson", lw=1.0, ls="--", label="noise = signal")
    ax.set_xscale("log")
    ax.set_xlabel("median |mean(1st half) - mean(2nd half)| / inter-orbit IQR")
    ax.set_title("conservation vs discriminability (uniform split-half metric)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return dict(zip(names, (float(r) for r in ratios)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache", type=Path,
        default=REPO / ".agent-local/benchmarks/solver_settings_sweep/frozen_paper_best.npz",
    )
    parser.add_argument(
        "--frozen-provenance", type=Path,
        default=REPO / ".agent-local/benchmarks/solver_settings_sweep/freeze.json",
    )
    parser.add_argument(
        "--output", type=Path, default=REPO / ".agent-local/benchmarks/orbit_invariants_round2",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    import agama

    from halo_mw_lmc.potential import build_potential_from_parameters

    parameters, parameter_note = resolve_grouping_parameters(args.frozen_provenance)
    print(f"[setup] potential for E/lambda_z: {parameters.as_dict()}")
    print(f"[setup] {parameter_note}")
    agama.setUnits(length=1, velocity=1, mass=1)
    true_potential = build_potential_from_parameters(parameters)
    phi_potential, phi_method = build_phi_averaged_potential(true_potential)
    swap_potential = build_component_swap_proxy(parameters)
    oracle = check_phi_average_oracle(true_potential, phi_potential)
    print(f"[oracle] phi-average {phi_method}: max rel diff vs numerical average "
          f"{oracle['phi_average_max_relative_difference']:.2e}, "
          f"circular-orbit max abs error {oracle['circular_orbit_max_absolute_error']:.2e}")
    action_finders = {"swap": agama.ActionFinder(swap_potential), "phi": agama.ActionFinder(phi_potential)}

    with np.load(args.cache) as frozen:
        seed_index = frozen["successful_seed_index"]
        sample_count = frozen["sample_count"]
        expected = np.repeat(seed_index, sample_count)
        if not np.array_equal(frozen["library_seed_index"], expected):
            raise ValueError("frozen library samples are not contiguous per successful seed")
        samples_per_orbit = int(sample_count[0])
        phase = frozen["library_phase_space"].reshape(seed_index.size, samples_per_orbit, 6)
        times = frozen["library_time"].reshape(seed_index.size, samples_per_orbit)
    steps = np.diff(times, axis=1)
    dt_rows = np.median(steps, axis=1)
    within_orbit = np.max(np.abs(steps - dt_rows[:, None]), axis=1) / dt_rows
    if float(within_orbit.max()) > 0.05:
        raise ValueError("orbit sample times are not uniform within orbits; frequency pass needs a per-orbit uniform grid")
    print(f"[setup] {seed_index.size} orbits x {samples_per_orbit} samples, per-orbit dt "
          f"{dt_rows.min():.4g}..{dt_rows.max():.4g} (each orbit spans its own 10 periods, "
          f"max within-orbit step jitter {within_orbit.max():.1%})")

    check_error = frequency_self_check()
    print(f"[oracle] frequency estimator recovers synthetic 3.7/T to {check_error:.2e} median relative error")

    e_circ, lz_circ = _circularity_scale(true_potential)
    started = time.perf_counter()
    means, stds, half_means = invariant_pass(phase, true_potential, e_circ, lz_circ, action_finders)
    frequencies, frequency_halves = frequency_pass(phase, times)
    means.update(frequencies)
    half_means.update(frequency_halves)
    print(f"[compute] invariant + frequency passes in {time.perf_counter() - started:.1f} s")

    np.savez(
        args.output / "orbit_invariants_round2.npz",
        seed_index=seed_index,
        **{f"mean_{name}": values for name, values in means.items()},
        **{f"std_{name}": values for name, values in stds.items()},
        **{f"half_{name}": values for name, values in half_means.items()},
    )

    matrix = correlation_heatmap(means, CORRELATION_SET, args.output / "correlation_spearman.png")
    corner_figure(means, FREQUENCY_CORNER, args.output / "corner_frequency.png",
                  "frequency basis vs energy and eccentricity")
    corner_figure(means, ["energy", "jr_phi", "jz_phi", "jphi_phi", "jtot_phi", "f_r_phi", "f_z_phi", "ecc"],
                  args.output / "corner_actions_phi_avg.png",
                  "actions in the azimuthal-average potential")
    lambda_focus_figure(means, args.output / "lambda_z_focus.png")
    proxy_comparison_figure(means, args.output / "proxy_comparison.png")
    ratios = conservation_figure(half_means, means, args.output / "conservation.png")

    lam = means["lam_z"]
    summary = {
        "orbit_count": int(seed_index.size),
        "potential_parameters": parameters.as_dict(),
        "potential_note": parameter_note,
        "phi_average": {
            "method": phi_method,
            "note": "Staeckel-fudge actions in the azimuthal m=0 average of the true triaxial potential; Jphi equals Lz by construction",
            "oracle": oracle,
        },
        "swap_proxy_note": "round-1 ad-hoc proxy: spherical Plummer for the Ferrers bulge, p->1 elsewhere",
        "frequency_note": {
            "method": "Hann-windowed FFT peak with log-parabola interpolation on r(t), z(t); phi(t) circulation rate from the unwrapped slope",
            "units": "cycles per time unit (time unit = kpc/(km/s), ~0.978 Gyr)",
            "resolution_limit": f"~1 period over {times[0, -1]:.1f} time units of span; split-half consistency quantifies the actual stability",
            "self_check_relative_error": check_error,
        },
        "lambda_z_pileup": {
            "fraction_abs_lt_0.02": float(np.mean(np.abs(lam) < 0.02)),
            "fraction_abs_lt_0.05": float(np.mean(np.abs(lam) < 0.05)),
            "fraction_abs_lt_0.10": float(np.mean(np.abs(lam) < 0.10)),
        },
        "split_half_noise_over_signal": ratios,
        "spearman": {
            CORRELATION_SET[i]: {
                CORRELATION_SET[j]: float(matrix[i, j]) for j in range(len(CORRELATION_SET))
            } for i in range(len(CORRELATION_SET))
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1))

    print("[split-half noise/signal, smaller is better]")
    for name, ratio in sorted(ratios.items(), key=lambda item: item[1]):
        print(f"    {name:>16s}  {ratio:9.3g}")
    print("[lambda_z pile-up] "
          + ", ".join(f"|λ|<{k}: {v:.1%}" for k, v in
                      (("<0.02", summary["lambda_z_pileup"]["fraction_abs_lt_0.02"]),
                       ("<0.05", summary["lambda_z_pileup"]["fraction_abs_lt_0.05"]),
                       ("<0.10", summary["lambda_z_pileup"]["fraction_abs_lt_0.10"]))))
    print(f"[done] round-2 figures and tables written to {args.output}")


if __name__ == "__main__":
    main()
