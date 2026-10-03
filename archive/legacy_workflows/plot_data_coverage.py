#!/usr/bin/env python3
"""Plot data-only projections and occupancy diagnostics for the 6D catalogue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from halo_mw_lmc.data_coverage import (
    PHASE_SPACE_COLUMNS,
    build_data_coverage,
    plot_all_data_coverage,
)
from halo_mw_lmc.grids import CylindricalGrid


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
        help="ASCII 6D catalogue, relative to --base-path unless absolute",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_coverage"),
        help="diagnostic output directory, relative to --base-path unless absolute",
    )
    parser.add_argument("--nphi", type=int, default=4)
    parser.add_argument("--n-rz", type=int, default=25)
    parser.add_argument("--rz-max", type=float, default=50.0)
    parser.add_argument("--z-min", type=float, default=0.0)
    parser.add_argument("--z-max", type=float, default=50.0)
    parser.add_argument(
        "--velocity-limit",
        type=float,
        default=600.0,
        help="symmetric display limit for velocity projections [km/s]",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=20_000,
        help="maximum raw points overlaid in each panel",
    )
    parser.add_argument("--random-state", type=int, default=0)
    return parser


def _resolve(base: Path, path: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else (base / path).resolve()


def _float_column(values) -> np.ndarray:
    return np.ma.asarray(values, dtype=float).filled(np.nan)


def _read_initial_conditions(path: Path) -> np.ndarray:
    try:
        from astropy.table import Table
    except ImportError:
        try:
            table = np.genfromtxt(path, names=True, ndmin=1)
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError(f"could not read ASCII catalogue {path}: {exc}") from exc
        names = set(table.dtype.names or ())
        missing = sorted(set(PHASE_SPACE_COLUMNS) - names)
        if missing:
            raise ValueError(
                "catalogue is missing required columns: " + ", ".join(missing)
            )
        return np.column_stack(
            [_float_column(table[name]) for name in PHASE_SPACE_COLUMNS]
        )

    table = Table.read(path, format="ascii")
    missing = sorted(set(PHASE_SPACE_COLUMNS) - set(table.colnames))
    if missing:
        raise ValueError(
            "catalogue is missing required columns: " + ", ".join(missing)
        )
    return np.column_stack(
        [_float_column(table[name]) for name in PHASE_SPACE_COLUMNS]
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if min(args.nphi, args.n_rz, args.max_points) < 1:
        raise SystemExit("--nphi, --n-rz, and --max-points must be positive")
    if args.rz_max <= 0 or args.velocity_limit <= 0:
        raise SystemExit("--rz-max and --velocity-limit must be positive")
    if args.z_min >= args.z_max:
        raise SystemExit("--z-min must be strictly less than --z-max")

    base = args.base_path.expanduser().resolve()
    catalog_path = _resolve(base, args.catalog)
    output_directory = _resolve(base, args.output_dir)
    if not catalog_path.exists():
        raise SystemExit(f"catalogue not found: {catalog_path}")

    try:
        initial_conditions = _read_initial_conditions(catalog_path)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    grid = CylindricalGrid.uniform(
        n_r=args.n_rz,
        r_range=(0.0, args.rz_max),
        n_z=args.n_rz,
        z_range=(args.z_min, args.z_max),
        n_phi=args.nphi,
    )
    coverage = build_data_coverage(
        initial_conditions,
        rzphi_grid=grid,
    )

    written = plot_all_data_coverage(
        coverage,
        output_directory,
        spatial_limit=args.rz_max,
        velocity_limit=args.velocity_limit,
        maximum_points=args.max_points,
        random_state=args.random_state,
    )
    summary = coverage.summary()
    summary_document = {
        "catalog_path": str(catalog_path),
        "density_interpretation": (
            "raw catalogue sampling density; no selection-function correction"
        ),
        "configuration": {
            "n_phi": args.nphi,
            "n_rz": args.n_rz,
            "r_range_kpc": [0.0, args.rz_max],
            "z_range_kpc": [args.z_min, args.z_max],
            "velocity_display_limit_km_s": args.velocity_limit,
            "random_state": args.random_state,
        },
        **summary,
    }
    summary_path = output_directory / "coverage_summary.json"
    summary_path.write_text(
        json.dumps(summary_document, indent=2, sort_keys=True) + "\n"
    )
    counts_path = output_directory / "coverage_counts.npz"
    np.savez_compressed(
        counts_path,
        rzphi_counts=coverage.rzphi_counts,
        rzphi_sampling_density=coverage.rzphi_sampling_density,
        r_edges=coverage.rzphi_grid.r_edges,
        z_edges=coverage.rzphi_grid.z_edges,
        phi_edges=coverage.phi_edges,
        rtheta_phi_counts=coverage.rtheta_phi_counts,
        rtheta_phi_sampling_density=coverage.rtheta_phi_sampling_density,
        spherical_radius_edges=coverage.spherical_radius_edges,
        theta_edges=coverage.theta_edges,
    )

    print(
        f"complete 6D rows: {summary['complete_6d_rows']:,}/"
        f"{summary['input_rows']:,} ({summary['complete_6d_fraction']:.1%})"
    )
    print(
        "empty spatial cells: "
        f"R-z-phi={summary['rzphi']['empty_fraction']:.1%}, "
        f"r-theta-phi={summary['rtheta_phi']['empty_fraction']:.1%}"
    )
    for path in [*written, summary_path, counts_path]:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
