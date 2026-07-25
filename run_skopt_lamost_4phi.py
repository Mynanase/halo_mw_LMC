#!/usr/bin/env python3
"""Optimize the Zhu empirical-orbit model on an ``(R,z,phi)`` grid."""

from __future__ import annotations

import argparse
from pathlib import Path

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
            "flattened observed density file; required when the grid is not "
            "the historical 25x25x4 setup"
        ),
    )
    parser.add_argument("--model", default="model_skopt", help="output directory name")
    parser.add_argument("--nphi", type=int, default=4)
    parser.add_argument("--n-rz", type=int, default=25)
    parser.add_argument("--rz-max", type=float, default=50.0)
    parser.add_argument("--orbit-samples", type=int, default=1000)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument(
        "--include-velocity",
        action="store_true",
        help="also include Zhu's three per-star velocity likelihood terms",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="write a three-row phi diagnostic PDF for every trial",
    )
    return parser


def _resolve(base: Path, path: Path | None) -> Path | None:
    if path is None:
        return None
    return path.expanduser().resolve() if path.is_absolute() else (base / path).resolve()


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if min(args.nphi, args.n_rz, args.orbit_samples, args.iterations) < 1:
        raise SystemExit("bin counts, orbit samples, and iterations must be positive")
    if (
        args.density is None
        and (args.nphi != 4 or args.n_rz != 25 or args.rz_max != 50.0)
    ):
        raise SystemExit("--density is required for a non-default R-z-phi grid")

    try:
        from skopt import Optimizer
        from skopt.space import Real
    except ImportError as exc:
        raise SystemExit(
            "scikit-optimize is required to run the optimizer; install the 'inference' extra"
        ) from exc
    from skopt_oint_lamost_4phi import evaluate_one_model

    base = args.base_path.expanduser().resolve()
    catalog = _resolve(base, args.catalog)
    density = _resolve(base, args.density)
    assert catalog is not None
    if not catalog.exists():
        raise SystemExit(f"catalogue not found: {catalog}")
    if density is not None and not density.exists():
        raise SystemExit(f"observed density file not found: {density}")

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
        Real(0.5, 1.5, name="qhalo"),
        Real(0.1, 1.5, name="phalo"),
        Real(5.0, 8.0, name="rho0"),
        Real(9.3, 10.3, name="rho0_plus_2logrs"),
        Real(0.1, 3.0, name="gamma"),
    ]
    optimizer = Optimizer(parameter_space, random_state=args.random_state)
    if not sample_file.exists():
        phi_columns = " ".join(f"chi2_phi{i}" for i in range(args.nphi))
        velocity_columns = ""
        if args.include_velocity:
            velocity_columns = " " + " ".join(
                f"lnL_{component}_phi{iphi}"
                for component in ("vr", "vphi", "vtheta")
                for iphi in range(args.nphi)
            )
        sample_file.write_text(
            "# iteration qhalo phalo rho0 rho0_plus_2logrs gamma "
            f"objective chi2 {phi_columns}{velocity_columns}\n"
        )

    for iteration in range(args.iterations):
        suggested = optimizer.ask()
        qhalo, phalo, rho0, rho0_plus_2logrs, gamma = [
            round(value, 3) for value in suggested
        ]
        log_rs = round((rho0_plus_2logrs - rho0) / 2, 3)
        evaluation = evaluate_one_model(
            base,
            args.model,
            rho0,
            log_rs,
            phalo,
            qhalo,
            0.0,
            0.0,
            gamma,
            catalog,
            observed_density_file=density,
            comparison_config=config,
            plot=args.plot,
        )
        objective = -evaluation.log_likelihood
        optimizer.tell(suggested, objective)

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
                f"{evaluation.density.chi2:.8e} {chi2_phi}{velocity_phi}\n"
            )
        print(
            f"iteration={iteration} objective={objective:.6g} "
            f"chi2_phi={evaluation.density.chi2_by_phi.tolist()}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
