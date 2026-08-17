"""Reader for the historical flattened Zhu target-density product."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.grids import CylindricalGrid
from .ascii import read_named_columns


def _is_historical_ascii_grid(grid: CylindricalGrid) -> bool:
    return (
        grid.shape == (25, 25, 4)
        and np.allclose(grid.r_edges, np.linspace(0.0, 50.0, 26))
        and np.allclose(grid.z_edges, np.linspace(0.0, 50.0, 26))
        and np.allclose(grid.phi_edges, np.linspace(-np.pi, np.pi, 5))
    )


def _read_npz_target(
    path: Path,
    grid: CylindricalGrid,
) -> tuple[np.ndarray, np.ndarray]:
    required = (
        "target_density",
        "target_error",
        "r_edges",
        "z_edges",
        "phi_edges",
    )
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = [name for name in required if name not in archive]
            if missing:
                raise ValueError(
                    "target NPZ is missing arrays: " + ", ".join(missing)
                )
            for name, expected in (
                ("r_edges", grid.r_edges),
                ("z_edges", grid.z_edges),
                ("phi_edges", grid.phi_edges),
            ):
                actual = np.asarray(archive[name], dtype=float)
                if actual.shape != expected.shape or not np.allclose(actual, expected):
                    raise ValueError(
                        f"target NPZ {name} do not match the configured grid"
                    )
            density = np.asarray(archive["target_density"], dtype=float).copy()
            error = np.asarray(archive["target_error"], dtype=float).copy()
    except OSError as exc:
        raise ValueError(f"could not read target NPZ {path}: {exc}") from exc
    for name, values in (("target_density", density), ("target_error", error)):
        if values.shape != grid.shape:
            raise ValueError(
                f"target NPZ {name} has shape {values.shape}; expected {grid.shape}"
            )
    return density, error


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
        return _read_npz_target(source, grid)
    if not _is_historical_ascii_grid(grid):
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
    return (
        np.transpose(density_zrphi, (1, 0, 2)),
        np.transpose(error_zrphi, (1, 0, 2)),
    )
