#!/usr/bin/env python3
"""Optimize the Zhu empirical-orbit model on an ``(R,z,phi)`` grid."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from halo_mw_lmc.config import ZhuComparisonConfig
from halo_mw_lmc.potentials import (
    ZHU_2026_BEST_FIT,
    ZHU_2026_LOCAL_SEARCH_BOUNDS,
    ZHU_2026_POTENTIAL_NAME,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-path",
        type=Path,
        default=Path.cwd(),
        help="project/data root (default: current directory)",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path(
            "data_for_model/lamost_dr8_SFlast_cut4_4phi/halo_clean_N.txt"
        ),
        help="6D seed-star catalogue, relative to --base-path unless absolute",
    )
    parser.add_argument(
        "--density",
        type=Path,
        default=None,
        help=(
            "flattened nu_target(R,z,phi) file; required when the grid is not "
            "the historical 25x25x4 setup"
        ),
    )
    parser.add_argument("--model", default="model_skopt", help="output directory name")
    parser.add_argument("--nphi", type=int, default=4)
    parser.add_argument("--n-rz", type=int, default=25)
    parser.add_argument("--rz-max", type=float, default=50.0)
    parser.add_argument("--orbit-samples", type=int, default=1000)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument(
        "--qhalo-min",
        type=float,
        default=ZHU_2026_LOCAL_SEARCH_BOUNDS["qhalo"][0],
    )
    parser.add_argument(
        "--qhalo-max",
        type=float,
        default=ZHU_2026_LOCAL_SEARCH_BOUNDS["qhalo"][1],
    )
    parser.add_argument(
        "--phalo-min",
        type=float,
        default=ZHU_2026_LOCAL_SEARCH_BOUNDS["phalo"][0],
    )
    parser.add_argument(
        "--phalo-max",
        type=float,
        default=ZHU_2026_LOCAL_SEARCH_BOUNDS["phalo"][1],
    )
    parser.add_argument(
        "--rho0-min",
        type=float,
        default=ZHU_2026_LOCAL_SEARCH_BOUNDS["rho0"][0],
        help="lower bound for log10 halo density normalization",
    )
    parser.add_argument(
        "--rho0-max",
        type=float,
        default=ZHU_2026_LOCAL_SEARCH_BOUNDS["rho0"][1],
        help="upper bound for log10 halo density normalization",
    )
    parser.add_argument(
        "--rho0-plus-2logrs-min",
        type=float,
        default=ZHU_2026_LOCAL_SEARCH_BOUNDS["rho0_plus_2logrs"][0],
        help="lower bound for rho0+2logrs",
    )
    parser.add_argument(
        "--rho0-plus-2logrs-max",
        type=float,
        default=ZHU_2026_LOCAL_SEARCH_BOUNDS["rho0_plus_2logrs"][1],
        help="upper bound for rho0+2logrs",
    )
    parser.add_argument(
        "--gamma-min",
        type=float,
        default=ZHU_2026_LOCAL_SEARCH_BOUNDS["gamma"][0],
    )
    parser.add_argument(
        "--gamma-max",
        type=float,
        default=ZHU_2026_LOCAL_SEARCH_BOUNDS["gamma"][1],
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="optimizer random seed (default: 0)",
    )
    parser.add_argument(
        "--include-velocity",
        action="store_true",
        help="include Zhu's three per-star velocity likelihood terms for r>=8 kpc",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help=(
            "for each new best-so-far trial, write phi-resolved density maps, "
            "flattening profiles, and (with --include-velocity) velocity PDFs"
        ),
    )
    parser.add_argument(
        "--velocity-plot-bin-factor",
        type=int,
        default=3,
        help=(
            "combine this many adjacent fitting bins in velocity plots only "
            "(default: 3, about 24 km/s)"
        ),
    )
    return parser


def _resolve(base: Path, path: Path | None) -> Path | None:
    if path is None:
        return None
    return path.expanduser().resolve() if path.is_absolute() else (base / path).resolve()


def _source_provenance() -> dict[str, object]:
    repository = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": bool(status.strip())}


def _paper_best_optimizer_point() -> list[float]:
    best = ZHU_2026_BEST_FIT
    return [
        best["qhalo"],
        best["phalo"],
        best["rho0"],
        best["rho0"] + 2 * best["log_rs"],
        best["gamma"],
    ]


def _catalogue_weight_audit(initial, weights, grid) -> dict[str, np.ndarray]:
    """Summarize fixed catalogue weights globally and on the density grid."""

    radius = np.hypot(initial[:, 0], initial[:, 1])
    z = initial[:, 2]
    phi = np.arctan2(initial[:, 1], initial[:, 0])
    seed_counts = grid.histogram(radius, z, phi)
    cell_weight_sum = grid.histogram(radius, z, phi, weights=weights)
    cell_weight_sq_sum = grid.histogram(radius, z, phi, weights=weights**2)
    initial_density = np.divide(
        cell_weight_sum,
        grid.volumes,
        out=np.zeros_like(cell_weight_sum),
        where=grid.volumes > 0,
    )

    ir, iz, iphi, in_grid = grid.bin_indices(radius, z, phi)
    cell_max_weight = np.zeros(grid.shape, dtype=float)
    np.maximum.at(
        cell_max_weight,
        (ir[in_grid], iz[in_grid], iphi[in_grid]),
        weights[in_grid],
    )
    cell_effective_seed_count = np.divide(
        cell_weight_sum**2,
        cell_weight_sq_sum,
        out=np.zeros_like(cell_weight_sum),
        where=cell_weight_sq_sum > 0,
    )
    cell_max_weight_fraction = np.divide(
        cell_max_weight,
        cell_weight_sum,
        out=np.zeros_like(cell_weight_sum),
        where=cell_weight_sum > 0,
    )

    total_weight = float(np.sum(weights))
    weight_sq_sum = float(np.sum(weights**2))
    quantile_levels = np.array([0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0])
    return {
        "weights": weights,
        "seed_counts": seed_counts,
        "cell_weight_sum": cell_weight_sum,
        "cell_weight_sq_sum": cell_weight_sq_sum,
        "cell_effective_seed_count": cell_effective_seed_count,
        "cell_max_weight": cell_max_weight,
        "cell_max_weight_fraction": cell_max_weight_fraction,
        "initial_catalog_density": initial_density,
        "quantile_levels": quantile_levels,
        "weight_quantiles": np.quantile(weights, quantile_levels),
        "positive_seed_count": np.asarray(np.count_nonzero(weights > 0)),
        "in_grid_seed_count": np.asarray(np.count_nonzero(in_grid)),
        "total_weight": np.asarray(total_weight),
        "in_grid_weight": np.asarray(float(np.sum(weights[in_grid]))),
        "effective_seed_count": np.asarray(
            total_weight**2 / weight_sq_sum if weight_sq_sum > 0 else 0.0
        ),
        "max_weight_fraction": np.asarray(
            float(np.max(weights)) / total_weight if total_weight > 0 else 0.0
        ),
    }


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if min(
        args.nphi,
        args.n_rz,
        args.orbit_samples,
        args.iterations,
        args.velocity_plot_bin_factor,
    ) < 1:
        raise SystemExit(
            "bin counts, orbit samples, iterations, and the velocity plot "
            "bin factor must be positive"
        )
    if (
        args.density is None
        and (args.nphi != 4 or args.n_rz != 25 or args.rz_max != 50.0)
    ):
        raise SystemExit("--density is required for a non-default R-z-phi grid")

    configured_bounds = {
        "qhalo": (args.qhalo_min, args.qhalo_max),
        "phalo": (args.phalo_min, args.phalo_max),
        "rho0": (args.rho0_min, args.rho0_max),
        "rho0-plus-2logrs": (
            args.rho0_plus_2logrs_min,
            args.rho0_plus_2logrs_max,
        ),
        "gamma": (args.gamma_min, args.gamma_max),
    }
    for name, (lower, upper) in configured_bounds.items():
        if lower >= upper:
            raise SystemExit(
                f"--{name}-min must be strictly less than --{name}-max"
            )

    try:
        from skopt import Optimizer
        from skopt.space import Real
    except ImportError as exc:
        raise SystemExit(
            "scikit-optimize is required to run the optimizer; install the 'inference' extra"
        ) from exc
    from skopt_oint_lamost_4phi import (
        evaluate_prepared_model,
        prepare_fixed_weight_data,
    )

    base = args.base_path.expanduser().resolve()
    catalog = _resolve(base, args.catalog)
    density = _resolve(base, args.density)
    assert catalog is not None
    if not catalog.exists():
        raise SystemExit(f"catalogue not found: {catalog}")
    if density is not None and not density.exists():
        raise SystemExit(f"target density file not found: {density}")

    config = ZhuComparisonConfig.legacy_4phi(
        n_phi=args.nphi,
        n_rz=args.n_rz,
        rz_max=args.rz_max,
        orbit_samples_per_orbit=args.orbit_samples,
        include_velocity=args.include_velocity,
    )
    output_dir = base / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_file = output_dir / "sample.dat"

    parameter_space = [
        Real(args.qhalo_min, args.qhalo_max, name="qhalo"),
        Real(args.phalo_min, args.phalo_max, name="phalo"),
        Real(args.rho0_min, args.rho0_max, name="rho0"),
        Real(
            args.rho0_plus_2logrs_min,
            args.rho0_plus_2logrs_max,
            name="rho0_plus_2logrs",
        ),
        Real(args.gamma_min, args.gamma_max, name="gamma"),
    ]
    optimizer = Optimizer(parameter_space, random_state=args.random_state)
    paper_best_point = _paper_best_optimizer_point()
    evaluate_paper_best_first = all(
        dimension.low <= value <= dimension.high
        for dimension, value in zip(parameter_space, paper_best_point)
    )
    phi_columns = " ".join(f"chi2_phi{i}" for i in range(args.nphi))
    velocity_columns = ""
    if args.include_velocity:
        velocity_columns = " " + " ".join(
            f"lnL_{component}_phi{iphi}"
            for component in ("vr", "vphi", "vtheta")
            for iphi in range(args.nphi)
        )
    expected_header = (
        "# iteration qhalo phalo rho0 rho0_plus_2logrs gamma "
        f"objective chi2 density_scale successful_orbits "
        f"{phi_columns}{velocity_columns}"
    )
    if not sample_file.exists():
        sample_file.write_text(expected_header + "\n")
    else:
        with sample_file.open() as stream:
            existing_header = stream.readline().rstrip("\n")
            output_has_samples = any(
                line.strip() and not line.lstrip().startswith("#") for line in stream
            )
        if existing_header != expected_header:
            raise SystemExit(
                f"existing sample schema does not match this run: {sample_file}; "
                "choose a new --model directory"
            )
        if output_has_samples:
            raise SystemExit(
                f"{sample_file} already contains samples; cold-start runs require "
                "a new --model directory"
            )

    run_config = {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **_source_provenance(),
        "paths": {
            "base": str(base),
            "catalog": str(catalog),
            "density": str(density) if density is not None else None,
            "output": str(output_dir),
        },
        "grid": {
            "n_phi": args.nphi,
            "n_rz": args.n_rz,
            "rz_max": args.rz_max,
            "orbit_samples_per_orbit": args.orbit_samples,
            "velocity_fit_min_radius": config.velocity_fit_min_radius,
        },
        "orbit_weights": {
            "source": "catalogue_column",
            "column": "w",
            "fixed_across_trial_potentials": True,
        },
        "optimizer": {
            "iterations": args.iterations,
            "random_state": args.random_state,
            "parameter_bounds": {
                dimension.name: [dimension.low, dimension.high]
                for dimension in parameter_space
            },
        },
        "potential": {
            "name": ZHU_2026_POTENTIAL_NAME,
            "reference": (
                "Zhu et al. (2026), A vertically orientated dark matter halo "
                "marks a flip of the Galactic disc, equations 6-8"
            ),
            "representative_best_fit": ZHU_2026_BEST_FIT,
            "paper_best_evaluated_first": evaluate_paper_best_first,
        },
        "include_velocity": args.include_velocity,
        "plot_new_best": args.plot,
        "velocity_plot_bin_factor": args.velocity_plot_bin_factor,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n"
    )

    prepared = prepare_fixed_weight_data(
        base,
        catalog,
        observed_density_file=density,
        comparison_config=config,
    )
    weight_audit = _catalogue_weight_audit(
        prepared.initial_conditions,
        prepared.seed_weights,
        config.density_grid,
    )
    np.savez_compressed(
        output_dir / "fixed_seed_weights.npz",
        **weight_audit,
        target_density=prepared.target_density,
        target_error=prepared.target_error,
        r_edges=config.density_grid.r_edges,
        z_edges=config.density_grid.z_edges,
        phi_edges=config.density_grid.phi_edges,
        weight_source=np.asarray("catalogue_column"),
        weight_column=np.asarray("w"),
        catalog_path=np.asarray(str(prepared.catalog_path)),
        density_path=np.asarray(str(prepared.density_path)),
    )
    max_cell_fraction = float(np.max(weight_audit["cell_max_weight_fraction"]))
    print(
        "fixed catalogue weights: "
        f"{weight_audit['positive_seed_count'].item()}/"
        f"{prepared.seed_weights.size} positive, "
        f"effective N={weight_audit['effective_seed_count'].item():.1f}, "
        f"global max fraction={weight_audit['max_weight_fraction'].item():.4f}, "
        f"max cell fraction={max_cell_fraction:.4f}"
    )

    best_objective = np.inf
    for iteration in range(args.iterations):
        suggested = (
            paper_best_point
            if iteration == 0 and evaluate_paper_best_first
            else optimizer.ask()
        )
        evaluated = [round(float(value), 3) for value in suggested]
        qhalo, phalo, rho0, rho0_plus_2logrs, gamma = evaluated
        log_rs = round((rho0_plus_2logrs - rho0) / 2, 3)
        evaluation = evaluate_prepared_model(
            base,
            args.model,
            rho0,
            log_rs,
            phalo,
            qhalo,
            0.0,
            0.0,
            gamma,
            prepared,
            plot=False,
        )
        objective = -evaluation.log_likelihood
        # Keep the surrogate coordinates identical to the evaluated and
        # persisted coordinates.
        optimizer.tell(evaluated, objective)
        if args.plot and objective < best_objective:
            from halo_mw_lmc.plotting import plot_model_diagnostics

            tag = (
                f"rho0{rho0:.3f}_rs{log_rs:.3f}_p{phalo:.3f}_q{qhalo:.3f}"
                f"_gamma{gamma:.3f}"
            )
            plot_model_diagnostics(
                evaluation.density,
                evaluation.velocity_distributions,
                output_dir / "diagnostics" / tag,
                velocity_bin_factor=args.velocity_plot_bin_factor,
            )
        best_objective = min(best_objective, objective)

        chi2_phi = " ".join(
            f"{value:.8e}" for value in evaluation.density.chi2_by_phi
        )
        velocity_phi = ""
        if args.include_velocity:
            velocity_phi = " " + " ".join(
                f"{value:.8e}"
                for component in ("vr", "vphi", "vtheta")
                for value in evaluation.velocity_loglike_by_phi[component]
            )
        with sample_file.open("a") as stream:
            stream.write(
                f"{iteration:d} {qhalo:.3f} {phalo:.3f} {rho0:.3f} "
                f"{rho0_plus_2logrs:.3f} {gamma:.3f} {objective:.8e} "
                f"{evaluation.density.chi2:.8e} {evaluation.density.scale:.8e} "
                f"{evaluation.successful_orbits:d} {chi2_phi}{velocity_phi}\n"
            )
        print(
            f"iteration={iteration} objective={objective:.6g} "
            f"chi2_phi={evaluation.density.chi2_by_phi.tolist()}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
