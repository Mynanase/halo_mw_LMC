"""Density construction and scoring for empirical orbit superposition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import DensityFitSettings
from .grids import CylindricalGrid


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class DensityComparison:
    """Products of a globally normalized ``(R, z, phi)`` comparison."""

    data_density: FloatArray
    data_error: FloatArray
    raw_model_density: FloatArray
    model_density: FloatArray
    residual: FloatArray
    fit_mask: BoolArray
    normalization_mask: BoolArray
    scale: float
    chi2: float
    chi2_by_phi: FloatArray
    valid_bins_by_phi: NDArray[np.int64]
    grid: CylindricalGrid


@dataclass(frozen=True)
class DensityShellDiagnostics:
    """Density residual statistics in radial shells and azimuth sectors."""

    radius_edges: FloatArray
    chi2_by_shell: FloatArray
    valid_bins_by_shell: NDArray[np.int64]
    chi2_by_shell_phi: FloatArray
    valid_bins_by_shell_phi: NDArray[np.int64]

    @property
    def chi2_per_bin_by_shell(self) -> FloatArray:
        return np.divide(
            self.chi2_by_shell,
            self.valid_bins_by_shell,
            out=np.full_like(self.chi2_by_shell, np.inf, dtype=float),
            where=self.valid_bins_by_shell > 0,
        )

    @property
    def chi2_per_bin_by_shell_phi(self) -> FloatArray:
        return np.divide(
            self.chi2_by_shell_phi,
            self.valid_bins_by_shell_phi,
            out=np.full_like(self.chi2_by_shell_phi, np.inf, dtype=float),
            where=self.valid_bins_by_shell_phi > 0,
        )


def density_shell_diagnostics(
    comparison: DensityComparison,
    radius_edges: ArrayLike,
) -> DensityShellDiagnostics:
    """Aggregate fitted residuals in left-closed, right-open radial shells."""

    edges = np.asarray(radius_edges, dtype=float)
    if (
        edges.ndim != 1
        or edges.size < 2
        or not np.all(np.isfinite(edges))
        or np.any(np.diff(edges) <= 0)
    ):
        raise ValueError("density shell edges must be finite and strictly increasing")

    radius, z, _ = comparison.grid.center_mesh
    spherical_radius = np.hypot(radius, z)
    squared = np.where(comparison.fit_mask, comparison.residual**2, 0.0)
    n_shell = edges.size - 1
    n_phi = comparison.grid.shape[-1]
    chi2_by_shell_phi = np.zeros((n_shell, n_phi), dtype=float)
    valid_by_shell_phi = np.zeros((n_shell, n_phi), dtype=np.int64)
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        shell = (spherical_radius >= lower) & (spherical_radius < upper)
        shell_fit = comparison.fit_mask & shell
        chi2_by_shell_phi[index] = np.sum(
            np.where(shell_fit, squared, 0.0),
            axis=(0, 1),
        )
        valid_by_shell_phi[index] = np.sum(
            shell_fit,
            axis=(0, 1),
            dtype=np.int64,
        )

    return DensityShellDiagnostics(
        radius_edges=edges.copy(),
        chi2_by_shell=np.sum(chi2_by_shell_phi, axis=1),
        valid_bins_by_shell=np.sum(valid_by_shell_phi, axis=1, dtype=np.int64),
        chi2_by_shell_phi=chi2_by_shell_phi,
        valid_bins_by_shell_phi=valid_by_shell_phi,
    )


def orbit_density(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    weights: ArrayLike,
    grid: CylindricalGrid,
    *,
    sample_divisor: float = 1.0,
) -> FloatArray:
    """Convert equal-time orbit samples into a cylindrical number density.

    Each sample carries its parent orbit's fixed catalogue weight. The caller
    supplies the divisor used to distribute that weight over samples in the
    analysed domain.
    """

    if sample_divisor <= 0:
        raise ValueError("sample_divisor must be positive")
    x_values, y_values, z_values, weight_values = np.broadcast_arrays(
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
        np.asarray(z, dtype=float),
        np.asarray(weights, dtype=float),
    )
    if np.any(np.isfinite(weight_values) & (weight_values < 0)):
        raise ValueError("orbit weights must be non-negative")
    radius = np.hypot(x_values, y_values)
    phi = np.arctan2(y_values, x_values)
    mass = grid.histogram(
        radius,
        z_values,
        phi,
        weights=weight_values / sample_divisor,
    )
    return np.divide(
        mass,
        grid.volumes,
        out=np.zeros_like(mass),
        where=grid.volumes > 0,
    )


def _shape_checked(values: ArrayLike, grid: CylindricalGrid, name: str) -> FloatArray:
    result = np.asarray(values, dtype=float)
    if result.shape != grid.shape:
        raise ValueError(f"{name} has shape {result.shape}; expected {grid.shape} (R, z, phi)")
    return result


def density_fit_mask(
    data_density: ArrayLike,
    data_error: ArrayLike,
    grid: CylindricalGrid,
    settings: DensityFitSettings | None = None,
) -> BoolArray:
    """Return the data-defined density constraint mask without a trial model."""

    settings = settings or DensityFitSettings()
    data = _shape_checked(data_density, grid, "data_density")
    error = _shape_checked(data_error, grid, "data_error")
    radius, z, _ = grid.center_mesh
    spherical_radius = np.hypot(radius, z)
    valid = np.isfinite(data) & np.isfinite(error) & (error > 0)
    if settings.require_positive_data:
        valid &= data > 0
    result = (
        valid
        & (np.abs(z) >= settings.min_abs_z)
        & (spherical_radius >= settings.min_spherical_radius)
        & (spherical_radius < settings.max_spherical_radius)
    )
    if not np.any(result):
        raise ValueError("no valid density bins remain after applying the fit mask")
    return np.asarray(result, dtype=bool)


def compare_density(
    data_density: ArrayLike,
    data_error: ArrayLike,
    model_density: ArrayLike,
    grid: CylindricalGrid,
    settings: DensityFitSettings | None = None,
) -> DensityComparison:
    """Normalize as configured and compute the Zhu density chi-square.

    When a scale is fitted it is global across phi, so relative over/under-
    density between azimuth bins remains part of the likelihood. Density-solved
    orbit weights use ``normalization='none'`` and therefore keep scale one.
    """

    settings = settings or DensityFitSettings()
    data = _shape_checked(data_density, grid, "data_density")
    error = _shape_checked(data_error, grid, "data_error")
    model = _shape_checked(model_density, grid, "model_density")

    radius, z, _ = grid.center_mesh
    spherical_radius = np.hypot(radius, z)
    valid = np.isfinite(data) & np.isfinite(error) & (error > 0) & np.isfinite(model) & (model >= 0)
    if settings.require_positive_data:
        valid &= data > 0

    normalization_mask = valid & (
        spherical_radius >= settings.normalization_min_radius
    )
    fit_mask = (
        valid
        & (np.abs(z) >= settings.min_abs_z)
        & (spherical_radius >= settings.min_spherical_radius)
        & (spherical_radius < settings.max_spherical_radius)
    )
    if not np.any(fit_mask):
        raise ValueError("no valid density bins remain after applying the fit mask")

    if settings.normalization == "none":
        scale = 1.0
    elif settings.normalization == "volume":
        if not np.any(normalization_mask):
            raise ValueError("no valid bins remain for volume normalization")
        data_mass = np.sum(data[normalization_mask] * grid.volumes[normalization_mask])
        model_mass = np.sum(model[normalization_mask] * grid.volumes[normalization_mask])
        if not np.isfinite(model_mass) or model_mass <= 0:
            raise ValueError("the model has no positive mass in the normalization region")
        scale = float(data_mass / model_mass)
    else:
        numerator = np.sum(data[fit_mask] * model[fit_mask] / error[fit_mask] ** 2)
        denominator = np.sum(model[fit_mask] ** 2 / error[fit_mask] ** 2)
        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError("the model cannot be normalized in the fit region")
        scale = float(numerator / denominator)

    scaled_model = scale * model
    residual = np.full(grid.shape, np.nan, dtype=float)
    residual[fit_mask] = (data[fit_mask] - scaled_model[fit_mask]) / error[fit_mask]
    squared = np.where(fit_mask, residual**2, 0.0)
    chi2_by_phi = np.sum(squared, axis=(0, 1))
    valid_by_phi = np.sum(fit_mask, axis=(0, 1), dtype=np.int64)

    return DensityComparison(
        data_density=data,
        data_error=error,
        raw_model_density=model,
        model_density=scaled_model,
        residual=residual,
        fit_mask=fit_mask,
        normalization_mask=normalization_mask,
        scale=scale,
        chi2=float(np.sum(chi2_by_phi)),
        chi2_by_phi=np.asarray(chi2_by_phi, dtype=float),
        valid_bins_by_phi=np.asarray(valid_by_phi, dtype=np.int64),
        grid=grid,
    )


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
        phi = self._polynomial(spherical_radius, self.phi0_rad, self.phi_coefficients)
        theta = self._polynomial(spherical_radius, self.theta0_rad, self.theta_coefficients)
        if np.any(np.isfinite(p) & (p <= 0)) or np.any(np.isfinite(q) & (q <= 0)):
            raise ValueError("DESI density axis ratios became non-positive")

        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rotated_x = x * cos_phi * cos_theta - y * sin_phi + z * cos_phi * sin_theta
        rotated_y = x * sin_phi * cos_theta + y * cos_phi + z * sin_phi * sin_theta
        rotated_z = -x * sin_theta + z * cos_theta
        ellipsoidal_radius = np.sqrt(rotated_x * rotated_x + rotated_y * rotated_y / (p * p) + rotated_z * rotated_z / (q * q))

        first_break, second_break = self.break_radii_kpc
        inner_slope, middle_slope, outer_slope = self.slopes
        density = np.empty_like(ellipsoidal_radius, dtype=float)
        inner = ellipsoidal_radius < first_break
        middle = (ellipsoidal_radius >= first_break) & (ellipsoidal_radius < second_break)
        outer = ellipsoidal_radius >= second_break
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            density[inner] = (ellipsoidal_radius[inner] / first_break) ** (-inner_slope)
            density[middle] = (ellipsoidal_radius[middle] / first_break) ** (-middle_slope)
            density[outer] = (second_break / first_break) ** (-middle_slope) * (ellipsoidal_radius[outer] / second_break) ** (-outer_slope)
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
    expected_shape = (*grid.shape, quadrature_order, quadrature_order, quadrature_order)
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


from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .grids import CylindricalGrid
from .orbits import OrbitLibrary

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix
else:
    csr_matrix = Any


FloatArray = NDArray[np.float64]
RESPONSE_BUILD_CHUNK_SIZE = 1_000_000


@dataclass(frozen=True)
class OrbitDensityResponse:
    """Sparse map from successful seed-orbit weights to grid density.

    Rows use C-order flattened ``(R,z,phi)`` cells. Columns are ordered by
    ``successful_seed_index``. Each column contains the orbit's equal-time
    occupancy fraction divided by cell volume.
    """

    matrix: csr_matrix
    successful_seed_index: NDArray[np.int64]
    sample_count: NDArray[np.int64]
    grid: CylindricalGrid
    seed_count: int

    def __post_init__(self) -> None:
        expected = (int(np.prod(self.grid.shape)), self.successful_seed_index.size)
        if getattr(self.matrix, "shape", None) != expected:
            raise ValueError(
                f"response matrix has shape {getattr(self.matrix, 'shape', None)}; "
                f"expected {expected}"
            )
        if self.sample_count.shape != self.successful_seed_index.shape:
            raise ValueError("sample_count must align with successful_seed_index")
        if np.any(self.sample_count <= 0):
            raise ValueError("every successful orbit must have at least one sample")

    def model_density(self, seed_weights: ArrayLike) -> FloatArray:
        """Apply full-catalogue seed weights and return ``(R,z,phi)`` density."""

        weights = np.asarray(seed_weights, dtype=float)
        if weights.shape != (self.seed_count,):
            raise ValueError(
                f"seed_weights has shape {weights.shape}; expected {(self.seed_count,)}"
            )
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("seed_weights must be finite and non-negative")
        result = np.asarray(
            self.matrix @ weights[self.successful_seed_index],
            dtype=float,
        ).reshape(self.grid.shape)
        return result

    def sample_weights(self, seed_weights: ArrayLike, library: OrbitLibrary) -> FloatArray:
        """Distribute each orbit's total weight over its finite time samples."""

        weights = np.asarray(seed_weights, dtype=float)
        if weights.shape != (self.seed_count,):
            raise ValueError(
                f"seed_weights has shape {weights.shape}; expected {(self.seed_count,)}"
            )
        counts_by_seed = np.zeros(self.seed_count, dtype=np.int64)
        counts_by_seed[self.successful_seed_index] = self.sample_count
        counts = counts_by_seed[library.seed_index]
        if np.any(counts <= 0):
            raise ValueError("orbit library contains a seed absent from the response")
        return weights[library.seed_index] / counts


@dataclass(frozen=True)
class OrbitSupportAudit:
    """Overlap between density-fit response and velocity spatial support."""

    density_supported_orbit_count: int
    velocity_supported_orbit_count: int
    zero_density_response_velocity_orbit_count: int
    zero_density_response_velocity_sample_count: int
    zero_density_response_velocity_weight_sum: float

    @property
    def zero_density_response_velocity_orbit_fraction(self) -> float:
        if self.velocity_supported_orbit_count <= 0:
            return 0.0
        return (
            self.zero_density_response_velocity_orbit_count
            / self.velocity_supported_orbit_count
        )


def build_orbit_density_response(
    library: OrbitLibrary,
    grid: CylindricalGrid,
    *,
    seed_count: int,
) -> OrbitDensityResponse:
    """Build a CSR response from a flattened equal-time orbit library."""

    try:
        from scipy.sparse import coo_matrix, csr_matrix as make_csr_matrix
    except ImportError as exc:
        raise RuntimeError(
            "SciPy is required for density-solved orbit weights"
        ) from exc

    if seed_count < 1:
        raise ValueError("seed_count must be positive")
    seed_index = np.asarray(library.seed_index, dtype=np.int64)
    if seed_index.ndim != 1 or seed_index.shape[0] != library.phase_space.shape[0]:
        raise ValueError("orbit seed indices must align with phase-space samples")
    if np.any(seed_index < 0) or np.any(seed_index >= seed_count):
        raise ValueError("orbit seed indices fall outside the seed catalogue")

    successful = np.unique(seed_index)
    sample_count = np.bincount(seed_index, minlength=seed_count)[successful]
    column_by_seed = np.full(seed_count, -1, dtype=np.int64)
    column_by_seed[successful] = np.arange(successful.size, dtype=np.int64)

    response_shape = (int(np.prod(grid.shape)), successful.size)
    occupancy = make_csr_matrix(response_shape, dtype=float)
    for start in range(0, seed_index.size, RESPONSE_BUILD_CHUNK_SIZE):
        stop = min(start + RESPONSE_BUILD_CHUNK_SIZE, seed_index.size)
        phase_space = library.phase_space[start:stop]
        radius = np.hypot(phase_space[:, 0], phase_space[:, 1])
        phi = np.arctan2(phase_space[:, 1], phase_space[:, 0])
        ir, iz, iphi, in_grid = grid.bin_indices(radius, phase_space[:, 2], phi)
        if not np.any(in_grid):
            continue
        flat_cell = np.ravel_multi_index((ir[in_grid], iz[in_grid], iphi[in_grid]), grid.shape)
        columns = column_by_seed[seed_index[start:stop][in_grid]]
        chunk = coo_matrix((np.ones(flat_cell.size, dtype=float), (flat_cell, columns)), shape=response_shape).tocsr()
        occupancy = occupancy + chunk
    occupancy.sum_duplicates()
    inverse_samples = 1.0 / sample_count.astype(float)
    inverse_volume = 1.0 / grid.volumes.reshape(-1)
    matrix = occupancy.multiply(inverse_samples[None, :])
    matrix = matrix.multiply(inverse_volume[:, None]).tocsr()
    return OrbitDensityResponse(
        matrix=matrix,
        successful_seed_index=successful,
        sample_count=np.asarray(sample_count, dtype=np.int64),
        grid=grid,
        seed_count=seed_count,
    )
