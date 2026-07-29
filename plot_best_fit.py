#!/usr/bin/env python3
"""Render fit diagnostics for the best sample in ``model_skopt/sample.dat``.

Re-evaluates the best trial potential with ``plot=True`` so that the existing
``plot_model_diagnostics`` routine writes the full phi-resolved density and
velocity plot set, and draws a standalone convergence-trajectory figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from halo_mw_lmc.config import ZhuComparisonConfig
from skopt_oint_lamost_4phi import (
    evaluate_prepared_model,
    prepare_fixed_weight_data,
)


def _best_row(sample_file: Path) -> dict:
    data = np.genfromtxt(sample_file, names=True)
    best = data[np.argmin(data["objective"])]
    return {
        "iteration": int(best["iteration"]),
        "qhalo": float(best["qhalo"]),
        "phalo": float(best["phalo"]),
        "rho0": float(best["rho0"]),
        "rho0_plus_2logrs": float(best["rho0_plus_2logrs"]),
        "gamma": float(best["gamma"]),
        "objective": float(best["objective"]),
        "chi2": float(best["chi2"]),
        "density_scale": float(best["density_scale"]),
        "chi2_by_phi": [float(best[f"chi2_phi{i}"]) for i in range(4)],
    }


def _plot_convergence(sample_file: Path, output: Path) -> None:
    import matplotlib.pyplot as plt

    data = np.genfromtxt(sample_file, names=True)
    iteration = data["iteration"]
    objective = data["objective"]
    cmin = np.minimum.accumulate(objective)

    figure, axis = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    axis.scatter(iteration, objective, s=6, alpha=0.35, label="trial objective")
    axis.plot(iteration, cmin, color="red", linewidth=1.5, label="best-so-far")
    best_i = int(iteration[np.argmin(objective)])
    axis.axvline(best_i, color="0.5", linestyle="--", alpha=0.7)
    axis.set_xlabel("iteration")
    axis.set_ylabel("objective (-log likelihood)")
    axis.set_yscale("log")
    axis.grid(alpha=0.2)
    axis.legend(loc="upper right")
    axis.set_title(
        f"Bayesian-optimization trajectory ({len(objective)} trials; "
        f"best at iter {best_i})"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-path", type=Path, default=Path.cwd())
    parser.add_argument("--model", default="model_skopt")
    parser.add_argument("--catalog", type=Path, default=Path(
        "data_for_model/lamost_dr8_SFlast_cut4_4phi/halo_clean_N.txt"
    ))
    parser.add_argument("--nphi", type=int, default=4)
    parser.add_argument("--n-rz", type=int, default=25)
    parser.add_argument("--rz-max", type=float, default=50.0)
    parser.add_argument("--orbit-samples", type=int, default=1000)
    parser.add_argument("--include-velocity", action="store_true")
    parser.add_argument(
        "--minimum-seed-count", type=int, default=1,
    )
    args = parser.parse_args(argv)

    base = args.base_path.expanduser().resolve()
    output_dir = base / args.model
    sample_file = output_dir / "sample.dat"
    if not sample_file.exists():
        raise SystemExit(f"no sample file found at {sample_file}")

    best = _best_row(sample_file)
    log_rs = round((best["rho0_plus_2logrs"] - best["rho0"]) / 2, 3)
    print(
        f"best trial: iter={best['iteration']} objective={best['objective']:.4f} "
        f"chi2={best['chi2']:.2f} density_scale={best['density_scale']:.4f}"
    )
    print(
        f"qhalo={best['qhalo']:.3f} phalo={best['phalo']:.3f} "
        f"rho0={best['rho0']:.3f} log_rs={log_rs:.3f} gamma={best['gamma']:.3f}"
    )
    print(f"chi2_by_phi = {best['chi2_by_phi']}")

    config = ZhuComparisonConfig.legacy_4phi(
        n_phi=args.nphi,
        n_rz=args.n_rz,
        rz_max=args.rz_max,
        orbit_samples_per_orbit=args.orbit_samples,
        include_velocity=args.include_velocity,
        minimum_seed_count=args.minimum_seed_count,
    )
    catalog = (
        args.catalog.expanduser().resolve()
        if args.catalog.is_absolute()
        else (base / args.catalog).resolve()
    )
    prepared = prepare_fixed_weight_data(
        base, catalog, observed_density_file=None, comparison_config=config,
    )

    evaluation = evaluate_prepared_model(
        base,
        args.model,
        best["rho0"],
        log_rs,
        best["phalo"],
        best["qhalo"],
        0.0,
        0.0,
        best["gamma"],
        prepared,
        plot=True,
    )
    diagnostics_dir = output_dir / "diagnostics" / (
        f"rho0{best['rho0']:.3f}_rs{log_rs:.3f}_p{best['phalo']:.3f}"
        f"_q{best['qhalo']:.3f}_gamma{best['gamma']:.3f}"
    )
    print(f"wrote diagnostics into {diagnostics_dir}")

    convergence_path = output_dir / "convergence.pdf"
    _plot_convergence(sample_file, convergence_path)
    print(f"wrote convergence figure to {convergence_path}")

    print(
        f"recomputed density: chi2={evaluation.density.chi2:.4f} "
        f"scale={evaluation.density.scale:.4f} "
        f"chi2_by_phi={evaluation.density.chi2_by_phi.tolist()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())