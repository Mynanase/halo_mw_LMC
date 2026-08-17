#!/usr/bin/env python3
"""Render fit diagnostics for the best sample in ``model_skopt/sample.dat``.

Re-evaluates the best trial potential with ``plot=True`` so that the existing
``plot_model_diagnostics`` routine writes the full phi-resolved density and
velocity plot set, and draws a standalone convergence-trajectory figure.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from halo_mw_lmc.config import ZhuComparisonConfig
from halo_mw_lmc.plotting import plot_model_diagnostics
from halo_mw_lmc.samples import SampleFileError, best_sample, load_sample_table
from skopt_oint_lamost_4phi import (
    evaluate_prepared_model,
    prepare_fixed_weight_data,
)


def _warn_if_weight_source_mismatch(output_dir: Path) -> None:
    """Flag diagnostics that cannot exactly replay an older optimizer run."""

    config_path = output_dir / "run_config.json"
    if not config_path.exists():
        warnings.warn(
            "run_config.json is missing; cannot verify the optimizer's orbit-weight source",
            stacklevel=2,
        )
        return
    try:
        run_config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        warnings.warn(f"cannot read {config_path}: {exc}", stacklevel=2)
        return
    orbit_weights = run_config.get("orbit_weights", {})
    if orbit_weights.get("source") != "catalogue_column":
        warnings.warn(
            "this run predates catalogue-supplied orbit weights; the saved objective "
            "used target-derived weights, while this diagnostic re-evaluation uses "
            "halo_clean_N.txt['w'] and is therefore not an exact replay",
            stacklevel=2,
        )


def _best_row(data: np.ndarray, nphi: int) -> dict:
    best = best_sample(data)
    scalar_names = (
        "iteration",
        "qhalo",
        "phalo",
        "rho0",
        "rho0_plus_2logrs",
        "gamma",
        "objective",
        "chi2",
        "density_scale",
    )
    values = {name: float(best[name]) for name in scalar_names}
    chi2_by_phi = [float(best[f"chi2_phi{i}"]) for i in range(nphi)]
    if not np.all(np.isfinite([*values.values(), *chi2_by_phi])):
        raise SampleFileError("best sample contains non-finite diagnostic values")
    return {
        **values,
        "iteration": int(values["iteration"]),
        "chi2_by_phi": chi2_by_phi,
    }


def _plot_convergence(data: np.ndarray, output: Path) -> None:
    import matplotlib.pyplot as plt

    iteration = np.asarray(data["iteration"], dtype=float)
    objective = np.asarray(data["objective"], dtype=float)
    finite = np.isfinite(iteration) & np.isfinite(objective)
    iteration = iteration[finite]
    objective = objective[finite]
    if objective.size == 0:
        raise SampleFileError(
            "sample file contains no finite iteration/objective pairs"
        )
    cmin = np.minimum.accumulate(objective)

    figure, axis = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    axis.scatter(iteration, objective, s=6, alpha=0.35, label="trial objective")
    axis.plot(iteration, cmin, color="red", linewidth=1.5, label="best-so-far")
    best_i = int(iteration[np.argmin(objective)])
    axis.axvline(best_i, color="0.5", linestyle="--", alpha=0.7)
    axis.set_xlabel("iteration")
    axis.set_ylabel("objective (-log likelihood)")
    axis.set_yscale("log" if np.all(objective > 0) else "symlog")
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
    parser.add_argument(
        "--density",
        type=Path,
        default=None,
        help=(
            "flattened target density file; required for a non-default "
            "R-z-phi grid"
        ),
    )
    parser.add_argument("--nphi", type=int, default=4)
    parser.add_argument("--n-rz", type=int, default=25)
    parser.add_argument("--rz-max", type=float, default=50.0)
    parser.add_argument("--orbit-samples", type=int, default=1000)
    parser.add_argument("--include-velocity", action="store_true")
    parser.add_argument(
        "--velocity-plot-bin-factor",
        type=int,
        default=3,
        help=(
            "combine this many adjacent fitting bins in velocity plots only "
            "(default: 3, about 24 km/s)"
        ),
    )
    args = parser.parse_args(argv)
    if min(
        args.nphi,
        args.n_rz,
        args.orbit_samples,
        args.velocity_plot_bin_factor,
    ) < 1:
        raise SystemExit(
            "bin counts, orbit samples, and the velocity plot bin factor "
            "must be positive"
        )
    if (
        args.density is None
        and (args.nphi != 4 or args.n_rz != 25 or args.rz_max != 50.0)
    ):
        raise SystemExit("--density is required for a non-default R-z-phi grid")

    base = args.base_path.expanduser().resolve()
    output_dir = base / args.model
    sample_file = output_dir / "sample.dat"
    if not sample_file.exists():
        raise SystemExit(f"no sample file found at {sample_file}")
    _warn_if_weight_source_mismatch(output_dir)

    required_columns = (
        "iteration",
        "qhalo",
        "phalo",
        "rho0",
        "rho0_plus_2logrs",
        "gamma",
        "objective",
        "chi2",
        "density_scale",
        *(f"chi2_phi{i}" for i in range(args.nphi)),
    )
    try:
        samples = load_sample_table(
            sample_file,
            required_columns=required_columns,
        )
        best = _best_row(samples, args.nphi)
    except SampleFileError as exc:
        raise SystemExit(f"invalid sample file: {exc}") from exc
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
    )
    catalog = (
        args.catalog.expanduser().resolve()
        if args.catalog.is_absolute()
        else (base / args.catalog).resolve()
    )
    density = (
        args.density.expanduser().resolve()
        if args.density is not None and args.density.is_absolute()
        else (base / args.density).resolve()
        if args.density is not None
        else None
    )
    if not catalog.exists():
        raise SystemExit(f"catalogue not found: {catalog}")
    if density is not None and not density.exists():
        raise SystemExit(f"target density file not found: {density}")
    prepared = prepare_fixed_weight_data(
        base, catalog, observed_density_file=density, comparison_config=config,
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
        plot=False,
    )
    diagnostics_dir = output_dir / "diagnostics" / (
        f"rho0{best['rho0']:.3f}_rs{log_rs:.3f}_p{best['phalo']:.3f}"
        f"_q{best['qhalo']:.3f}_gamma{best['gamma']:.3f}"
    )
    plot_model_diagnostics(
        evaluation.density,
        evaluation.velocity_distributions,
        diagnostics_dir,
        velocity_bin_factor=args.velocity_plot_bin_factor,
    )
    print(f"wrote diagnostics into {diagnostics_dir}")

    convergence_path = output_dir / "convergence.pdf"
    try:
        _plot_convergence(samples, convergence_path)
    except SampleFileError as exc:
        raise SystemExit(f"could not plot convergence: {exc}") from exc
    print(f"wrote convergence figure to {convergence_path}")

    print(
        f"recomputed density: chi2={evaluation.density.chi2:.4f} "
        f"scale={evaluation.density.scale:.4f} "
        f"chi2_by_phi={evaluation.density.chi2_by_phi.tolist()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
