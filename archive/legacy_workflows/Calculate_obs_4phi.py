"""Backward-compatible wrappers for azimuth-resolved model histograms.

New code should import :mod:`halo_mw_lmc.density` and
:mod:`halo_mw_lmc.velocity` directly.  These functions retain the historical
names and array order expected by the original scripts.
"""

from __future__ import annotations

import numpy as np

from halo_mw_lmc.density import orbit_density
from halo_mw_lmc.grids import CylindricalGrid
from halo_mw_lmc.velocity import (
    SphericalVelocityGrid,
    conditional_velocity_histogram,
)


def calculate_RzSB_4phi(dt, nRz, Rzmax, nphi):
    """Return density in the legacy ``(z, R, phi)`` axis order."""

    grid = CylindricalGrid.uniform(
        n_r=nRz,
        r_range=(0.0, Rzmax),
        n_z=nRz,
        z_range=(0.0, Rzmax),
        n_phi=nphi,
    )
    density_rzphi = orbit_density(
        dt["x"],
        dt["y"],
        dt["z"],
        dt["w"],
        grid,
    )
    radius, z, _ = grid.center_mesh
    # Historical callers expect z to be the first array dimension.
    return (
        np.transpose(z, (1, 0, 2)),
        np.transpose(radius, (1, 0, 2)),
        np.transpose(density_rzphi, (1, 0, 2)),
    )


def _velocity_histograms(dt, rbd, tbd, pbd, vbd, *, weighted):
    grid = SphericalVelocityGrid(
        radius_edges=np.asarray(rbd, dtype=float),
        theta_edges=np.deg2rad(np.asarray(tbd, dtype=float)),
        phi_edges=np.deg2rad(np.asarray(pbd, dtype=float)),
        velocity_edges=np.asarray(vbd, dtype=float),
    )
    weights = dt["w"] if weighted else None
    arguments = (
        np.asarray(dt["r3d"], dtype=float),
        np.asarray(dt["theta"], dtype=float),
        np.asarray(dt["phi"], dtype=float),
    )
    vr, _ = conditional_velocity_histogram(
        *arguments, np.asarray(dt["vr"], dtype=float), grid, weights=weights
    )
    vphi, _ = conditional_velocity_histogram(
        *arguments, np.asarray(dt["v_phi"], dtype=float), grid, weights=weights
    )
    vtheta, _ = conditional_velocity_histogram(
        *arguments, np.asarray(dt["v_the"], dtype=float), grid, weights=weights
    )
    return vr, vphi, vtheta


def calculate_vhist_weight_4phi(dt, rbd, tbd, pbd, vbd):
    return _velocity_histograms(dt, rbd, tbd, pbd, vbd, weighted=True)


def calculate_vhist_4phi(dt, rbd, tbd, pbd, vbd):
    return _velocity_histograms(dt, rbd, tbd, pbd, vbd, weighted=False)
