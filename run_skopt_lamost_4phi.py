#!/usr/bin/env python3
"""Optimize the Zhu empirical-orbit model on an ``(R,z,phi)`` grid."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from halo_mw_lmc.config import ZhuComparisonConfig


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
    parser.add_argument(
        "--minimum-seed-count",
        type=int,
        default=1,
        help="minimum seed stars required before a target cell receives weight",
    )
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--rho0-min", type=float, default=5.0, help="lower bound for rho0")
    parser.add_argument("--rho0-max", type=float, default=8.0, help="upper bound for rho0")
    parser.add_argument(
        "--rho0-plus-2logrs-min",
        type=float,
        default=9.3,
        help="lower bound for rho0+2logrs",
    )
    parser.add_argument(
        "--rho0-plus-2logrs-max",
        type=float,
        default=10.3,
        help="upper bound for rho0+2logrs",
    )
    parser.add_argument(
        "--warm-start",
        type=Path,
        default=None,
        help="previous sample.dat to replay into the optimizer before new ask/tell loop",
    )
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument(
        "--include-velocity",
        action="store_true",
        help="also include Zhu's three per-star velocity likelihood terms",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help=(
            "for each new best-so-far trial, write phi-resolved density maps, "
            "flattening profiles, and (with --include-velocity) velocity PDFs"
        ),
    )
    return parser


def _resolve(base: Path, path: Path | None) -> Path | None:
    if path is None:
        return None
    return path.expanduser().resolve() if path.is_absolute() else (base / path).resolve()


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if min(
        args.nphi,
        args.n_rz,
        args.orbit_samples,
        args.minimum_seed_count,
        args.iterations,
    ) < 1:
        raise SystemExit("bin counts, orbit samples, and iterations must be positive")
    if (
        args.density is None
        and (args.nphi != 4 or args.n_rz != 25 or args.rz_max != 50.0)
    ):
        raise SystemExit("--density is required for a non-default R-z-phi grid")

    if args.rho0_min >= args.rho0_max:
        raise SystemExit("--rho0-min must be strictly less than --rho0-max")
    if args.rho0_plus_2logrs_min >= args.rho0_plus_2logrs_max:
        raise SystemExit(
            "--rho0-plus-2logrs-min must be strictly less than --rho0-plus-2logrs-max"
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
        minimum_seed_count=args.minimum_seed_count,
    )
    output_dir = base / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_file = output_dir / "sample.dat"
    prepared = prepare_fixed_weight_data(
        base,
        catalog,
        observed_density_file=density,
        comparison_config=config,
    )
    fixed = prepared.representative_weights
    np.savez_compressed(
        output_dir / "fixed_weights_rzphi.npz",
        weights=fixed.weights,
        seed_counts=fixed.seed_counts,
        cell_weight=fixed.cell_weight,
        supported_cells=fixed.supported_cells,
        target_density=prepared.target_density,
        target_error=prepared.target_error,
        assigned_mass=np.asarray(fixed.assigned_mass),
        positive_target_mass=np.asarray(fixed.positive_target_mass),
        unsupported_positive_mass=np.asarray(fixed.unsupported_positive_mass),
        supported_mass_fraction=np.asarray(fixed.supported_mass_fraction),
        r_edges=config.density_grid.r_edges,
        z_edges=config.density_grid.z_edges,
        phi_edges=config.density_grid.phi_edges,
        catalog_path=np.asarray(str(prepared.catalog_path)),
        density_path=np.asarray(str(prepared.density_path)),
    )
    print(
        "fixed R-z-phi weights: "
        f"{fixed.weighted_seed_count}/{fixed.in_grid_seed_count} in-grid seeds weighted, "
        f"supported target mass={fixed.supported_mass_fraction:.4f}"
    )

    parameter_space = [
        Real(0.5, 1.5, name="qhalo"),
        Real(0.1, 1.5, name="phalo"),
        Real(args.rho0_min, args.rho0_max, name="rho0"),
        Real(
            args.rho0_plus_2logrs_min,
            args.rho0_plus_2logrs_max,
            name="rho0_plus_2logrs",
        ),
        Real(0.1, 2.9, name="gamma"),
    ]
    optimizer = Optimizer(parameter_space, random_state=args.random_state)
    if args.warm_start is not None:
        warm_path = _resolve(base, args.warm_start)
        if not warm_path.exists():
            raise SystemExit(f"warm-start sample not found: {warm_path}")
        warm = np.genfromtxt(warm_path, names=True)
        Xi, yi = [], []
        for row in warm:
            params = [
                float(row["qhalo"]),
                float(row["phalo"]),
                float(row["rho0"]),
                float(row["rho0_plus_2logrs"]),
                float(row["gamma"]),
            ]
            if not (args.rho0_min <= params[2] <= args.rho0_max):
                continue
            if not (
                args.rho0_plus_2logrs_min <= params[3] <= args.rho0_plus_2logrs_max
            ):
                continue
            Xi.append(params)
            yi.append(float(row["objective"]))
        if Xi:
            optimizer.tell(Xi, yi)
        print(
            f"warm-start: replayed {len(Xi)}/{len(warm)} prior samples "
            f"into the optimizer (within rho0=[{args.rho0_min},{args.rho0_max}], "
            f"rho0+2logrs=[{args.rho0_plus_2logrs_min},{args.rho0_plus_2logrs_max}])"
        )
    best_objective = np.inf
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
        if existing_header != expected_header:
            raise SystemExit(
                f"existing sample schema does not match this run: {sample_file}; "
                "choose a new --model directory"
            )

    for iteration in range(args.iterations):
        suggested = optimizer.ask()
        qhalo, phalo, rho0, rho0_plus_2logrs, gamma = [
            round(value, 3) for value in suggested
        ]
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
        optimizer.tell(suggested, objective)
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
