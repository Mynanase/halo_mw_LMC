"""Small, dependency-free Galactocentric coordinate transforms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SphericalPhaseSpace:
    radius: FloatArray
    theta: FloatArray
    phi: FloatArray
    radial_velocity: FloatArray
    azimuthal_velocity: FloatArray
    polar_velocity: FloatArray


def cartesian_to_spherical_phase_space(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    vx: ArrayLike,
    vy: ArrayLike,
    vz: ArrayLike,
) -> SphericalPhaseSpace:
    """Transform Cartesian phase space using Zhu's velocity convention.

    ``theta`` is latitude above the Galactic plane, and ``v_theta`` points
    toward increasing theta.
    """

    x, y, z, vx, vy, vz = np.broadcast_arrays(
        *[np.asarray(values, dtype=float) for values in (x, y, z, vx, vy, vz)]
    )
    radius = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(y, x)
    theta = np.arctan2(z, np.hypot(x, y))

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    radial_velocity = (
        vx * cos_phi * cos_theta
        + vy * sin_phi * cos_theta
        + vz * sin_theta
    )
    azimuthal_velocity = -vx * sin_phi + vy * cos_phi
    polar_velocity = (
        -vx * cos_phi * sin_theta
        - vy * sin_phi * sin_theta
        + vz * cos_theta
    )
    return SphericalPhaseSpace(
        radius=radius,
        theta=theta,
        phi=phi,
        radial_velocity=radial_velocity,
        azimuthal_velocity=azimuthal_velocity,
        polar_velocity=polar_velocity,
    )
