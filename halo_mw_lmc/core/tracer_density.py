"""Analytic stellar-halo tracer-density models and grid quadrature."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .grids import CylindricalGrid


FloatArray = NDArray[np.float64]
DensityFunction = Callable[[ArrayLike, ArrayLike, ArrayLike], FloatArray]


@dataclass(frozen=True)
class DesiKGiantsDensityModel:
    """DESI year-1 K-giant triaxial broken-power-law density shape.

    The parameters are transcribed from ``Desi/3D_density_profile.py``. The
    source's first fitted value is not used by its density function, so this
    class intentionally represents only a relative density shape.
    """

    p0: float = 9.13810522e-1
    q0: float = 6.32550491e-1
    phi0_rad: float = 9.45689142e-1
    theta0_rad: float = 1.52348376e-1
    p_coefficients: tuple[float, float, float] = (
        -5.28876970e-3,
        6.39679000e-5,
        -2.55300000e-7,
    )
    q_coefficients: tuple[float, float, float] = (
        5.41388730e-3,
        -9.93104000e-5,
        4.86200000e-7,
    )
    phi_coefficients: tuple[float, float, float] = (
        -7.97878894e-2,
        1.25673290e-3,
        -6.18220000e-6,
    )
    theta_coefficients: tuple[float, float, float] = (
        -4.52960955e-2,
        3.52501300e-4,
        -3.74000000e-8,
    )
    break_radii_kpc: tuple[float, float] = (15.8041, 77.1921)
    slopes: tuple[float, float, float] = (1.2795, 3.4636, 5.189)
    unused_fit_offset: float = -2.39106294e2

    @staticmethod
    def _polynomial(
        radius: FloatArray,
        intercept: float,
        coefficients: tuple[float, float, float],
    ) -> FloatArray:
        linear, quadratic, cubic = coefficients
        return intercept + radius * (
            linear + radius * (quadratic + radius * cubic)
        )

    def parameter_document(self) -> dict[str, object]:
        """Return a JSON/NPZ-safe record of the transcribed model state."""

        return {
            "p0": self.p0,
            "q0": self.q0,
            "phi0_rad": self.phi0_rad,
            "theta0_rad": self.theta0_rad,
            "p_coefficients": self.p_coefficients,
            "q_coefficients": self.q_coefficients,
            "phi_coefficients": self.phi_coefficients,
            "theta_coefficients": self.theta_coefficients,
            "break_radii_kpc": self.break_radii_kpc,
            "slopes": self.slopes,
            "unused_fit_offset": self.unused_fit_offset,
        }

    def __call__(
        self,
        x_kpc: ArrayLike,
        y_kpc: ArrayLike,
        z_kpc: ArrayLike,
    ) -> FloatArray:
        """Evaluate the relative density in Galactocentric Cartesian space."""

        x, y, z = np.broadcast_arrays(
            np.asarray(x_kpc, dtype=float),
            np.asarray(y_kpc, dtype=float),
            np.asarray(z_kpc, dtype=float),
        )
        spherical_radius = np.sqrt(x * x + y * y + z * z)
        p = self._polynomial(spherical_radius, self.p0, self.p_coefficients)
        q = self._polynomial(spherical_radius, self.q0, self.q_coefficients)
        phi = self._polynomial(
            spherical_radius,
            self.phi0_rad,
            self.phi_coefficients,
        )
        theta = self._polynomial(
            spherical_radius,
            self.theta0_rad,
            self.theta_coefficients,
        )
        if np.any(np.isfinite(p) & (p <= 0)) or np.any(np.isfinite(q) & (q <= 0)):
            raise ValueError("DESI density axis ratios became non-positive")

        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rotated_x = (
            x * cos_phi * cos_theta
            - y * sin_phi
            + z * cos_phi * sin_theta
        )
        rotated_y = (
            x * sin_phi * cos_theta
            + y * cos_phi
            + z * sin_phi * sin_theta
        )
        rotated_z = -x * sin_theta + z * cos_theta
        ellipsoidal_radius = np.sqrt(
            rotated_x * rotated_x
            + rotated_y * rotated_y / (p * p)
            + rotated_z * rotated_z / (q * q)
        )

        first_break, second_break = self.break_radii_kpc
        inner_slope, middle_slope, outer_slope = self.slopes
        density = np.empty_like(ellipsoidal_radius, dtype=float)
        inner = ellipsoidal_radius < first_break
        middle = (
            (ellipsoidal_radius >= first_break)
            & (ellipsoidal_radius < second_break)
        )
        outer = ellipsoidal_radius >= second_break
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            density[inner] = (
                ellipsoidal_radius[inner] / first_break
            ) ** (-inner_slope)
            density[middle] = (
                ellipsoidal_radius[middle] / first_break
            ) ** (-middle_slope)
            density[outer] = (
                second_break / first_break
            ) ** (-middle_slope) * (
                ellipsoidal_radius[outer] / second_break
            ) ** (-outer_slope)
        return density


DESI_YEAR1_KGIANTS_DENSITY = DesiKGiantsDensityModel()


def cell_average_cylindrical_density(
    density_function: DensityFunction,
    grid: CylindricalGrid,
    *,
    quadrature_order: int = 4,
) -> FloatArray:
    """Volume-average a Cartesian density over every ``(R,z,phi)`` cell.

    Gauss--Legendre nodes are applied to ``u=R^2/2``, ``z``, and ``phi``.
    This absorbs the cylindrical Jacobian because ``du = R dR``.
    """

    if (
        isinstance(quadrature_order, bool)
        or not isinstance(quadrature_order, (int, np.integer))
        or quadrature_order < 1
    ):
        raise ValueError("quadrature_order must be a positive integer")
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    normalized_weights = weights / 2.0

    def mapped_nodes(edges: FloatArray) -> FloatArray:
        lower = edges[:-1, None]
        upper = edges[1:, None]
        return 0.5 * (lower + upper) + 0.5 * (upper - lower) * nodes

    radial_u_edges = 0.5 * grid.r_edges**2
    radius = np.sqrt(2.0 * mapped_nodes(radial_u_edges))[
        :, None, None, :, None, None
    ]
    z = mapped_nodes(grid.z_edges)[None, :, None, None, :, None]
    phi = mapped_nodes(grid.phi_edges)[None, None, :, None, None, :]
    x = radius * np.cos(phi)
    y = radius * np.sin(phi)
    density = np.asarray(density_function(x, y, z), dtype=float)
    expected_shape = (
        *grid.shape,
        quadrature_order,
        quadrature_order,
        quadrature_order,
    )
    if density.shape != expected_shape:
        density = np.broadcast_to(density, expected_shape)
    if not np.all(np.isfinite(density)) or np.any(density <= 0):
        raise ValueError(
            "density function must be finite and positive at all quadrature nodes"
        )
    result = np.einsum(
        "abcdef,d,e,f->abc",
        density,
        normalized_weights,
        normalized_weights,
        normalized_weights,
        optimize=True,
    )
    return np.asarray(result, dtype=float)
