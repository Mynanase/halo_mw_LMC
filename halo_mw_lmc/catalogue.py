"""Adapters for six-dimensional seed catalogues with optional fixed weights."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from .coverage import PHASE_SPACE_COLUMNS
from .weights import catalogue_seed_weights


VELOCITY_ERROR_COLUMNS = {
    "vr": "vr_err",
    "vphi": "vphi_err",
    "vtheta": "vthe_err",
}


@dataclass(frozen=True)
class SeedCatalogue:
    """Core-ready arrays read from one survey catalogue."""

    initial_conditions: np.ndarray
    seed_weights: np.ndarray | None
    velocity_errors: Mapping[str, np.ndarray]


def read_phase_space_catalogue(path: str | Path) -> np.ndarray:
    """Return ``(x,y,z,vx,vy,vz)`` columns as a ``float64 (N,6)`` array."""

    columns = read_named_columns(path, PHASE_SPACE_COLUMNS)
    return np.column_stack([columns[name] for name in PHASE_SPACE_COLUMNS])


def read_seed_catalogue(
    path: str | Path,
    *,
    include_velocity: bool,
    require_weights: bool = True,
) -> SeedCatalogue:
    """Read phase space, optional fixed weights, and velocity errors."""

    required = [*PHASE_SPACE_COLUMNS]
    if require_weights:
        required.append("w")
    if include_velocity:
        required.extend(VELOCITY_ERROR_COLUMNS.values())
    columns = read_named_columns(path, required)
    initial = np.column_stack([columns[name] for name in PHASE_SPACE_COLUMNS])
    weights = catalogue_seed_weights(columns["w"]) if require_weights else None
    errors = {component: np.asarray(columns[column], dtype=float) for component, column in VELOCITY_ERROR_COLUMNS.items()} if include_velocity else {}
    return SeedCatalogue(initial_conditions=np.asarray(initial, dtype=float), seed_weights=weights, velocity_errors=errors)


from pathlib import Path
from typing import Iterable

import numpy as np


def read_named_columns(
    path: str | Path,
    required: Iterable[str],
) -> dict[str, np.ndarray]:
    """Read selected named columns without exposing an Astropy table downstream."""

    source = Path(path)
    required_names = tuple(required)
    try:
        from astropy.table import Table
    except ImportError:
        try:
            table = np.genfromtxt(source, names=True, ndmin=1)
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError(f"could not read ASCII table {source}: {exc}") from exc
        available = set(table.dtype.names or ())
        missing = sorted(set(required_names) - available)
        if missing:
            raise ValueError(
                "ASCII table is missing required columns: " + ", ".join(missing)
            )
        return {
            name: np.ma.asarray(table[name], dtype=float).filled(np.nan)
            for name in required_names
        }

    try:
        table = Table.read(source, format="ascii")
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"could not read ASCII table {source}: {exc}") from exc
    missing = sorted(set(required_names) - set(table.colnames))
    if missing:
        raise ValueError(
            "ASCII table is missing required columns: " + ", ".join(missing)
        )
    return {
        name: np.ma.asarray(table[name], dtype=float).filled(np.nan)
        for name in required_names
    }


from pathlib import Path

import numpy as np

from .grids import CylindricalGrid


def read_target_density(
    path: str | Path,
    grid: CylindricalGrid,
) -> tuple[np.ndarray, np.ndarray]:
    """Return density and error in the core ``(R,z,phi)`` axis order.

    The source ASCII file is flattened from the historical ``(z,R,phi)``
    array.  The explicit transpose is the only place that legacy convention
    enters the active pipeline.
    """

    source = Path(path)
    if source.suffix.lower() == ".npz":
        required = ("target_density", "target_error", "r_edges", "z_edges", "phi_edges")
        try:
            with np.load(source, allow_pickle=False) as archive:
                missing = [name for name in required if name not in archive]
                if missing:
                    raise ValueError("target NPZ is missing arrays: " + ", ".join(missing))
                for name, expected in (("r_edges", grid.r_edges), ("z_edges", grid.z_edges), ("phi_edges", grid.phi_edges)):
                    actual = np.asarray(archive[name], dtype=float)
                    if actual.shape != expected.shape or not np.allclose(actual, expected):
                        raise ValueError(f"target NPZ {name} do not match the configured grid")
                density = np.asarray(archive["target_density"], dtype=float).copy()
                error = np.asarray(archive["target_error"], dtype=float).copy()
        except OSError as exc:
            raise ValueError(f"could not read target NPZ {source}: {exc}") from exc
        for name, values in (("target_density", density), ("target_error", error)):
            if values.shape != grid.shape:
                raise ValueError(f"target NPZ {name} has shape {values.shape}; expected {grid.shape}")
        return density, error

    historical_grid = grid.shape == (25, 25, 4) and np.allclose(grid.r_edges, np.linspace(0.0, 50.0, 26)) and np.allclose(grid.z_edges, np.linspace(0.0, 50.0, 26)) and np.allclose(grid.phi_edges, np.linspace(-np.pi, np.pi, 5))
    if not historical_grid:
        raise ValueError(
            "metadata-free ASCII target densities are restricted to the historical "
            "25x25x4 grid (R,z=0..50 kpc, phi=-pi..pi); use an NPZ with "
            "target_density, target_error, and grid edges for a custom grid"
        )

    columns = read_named_columns(source, ("den", "den_srr"))
    n_r, n_z, n_phi = grid.shape
    source_shape = (n_z, n_r, n_phi)
    expected = int(np.prod(source_shape))
    for name, values in columns.items():
        if values.size != expected:
            raise ValueError(
                f"column {name!r} contains {values.size} values; expected "
                f"{expected} for source shape {source_shape}"
            )
    density_zrphi = columns["den"].reshape(source_shape)
    error_zrphi = columns["den_srr"].reshape(source_shape)
    return np.transpose(density_zrphi, (1, 0, 2)), np.transpose(error_zrphi, (1, 0, 2))
