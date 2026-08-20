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


def _base_validity(
    data: FloatArray,
    error: FloatArray,
    model: FloatArray,
    *,
    require_positive_data: bool,
) -> BoolArray:
    valid = (
        np.isfinite(data)
        & np.isfinite(error)
        & (error > 0)
        & np.isfinite(model)
        & (model >= 0)
    )
    if require_positive_data:
        valid &= data > 0
    return valid


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
    valid = _base_validity(
        data,
        error,
        model,
        require_positive_data=settings.require_positive_data,
    )

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
        numerator = np.sum(
            data[fit_mask] * model[fit_mask] / error[fit_mask] ** 2
        )
        denominator = np.sum(model[fit_mask] ** 2 / error[fit_mask] ** 2)
        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError("the model cannot be normalized in the fit region")
        scale = float(numerator / denominator)

    scaled_model = scale * model
    residual = np.full(grid.shape, np.nan, dtype=float)
    residual[fit_mask] = (
        data[fit_mask] - scaled_model[fit_mask]
    ) / error[fit_mask]
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
