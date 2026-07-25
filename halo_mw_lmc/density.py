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

    Each sample carries its parent orbit's representative weight.  The caller
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
        raise ValueError("orbit representative weights must be non-negative")
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


def compare_density(
    data_density: ArrayLike,
    data_error: ArrayLike,
    model_density: ArrayLike,
    grid: CylindricalGrid,
    settings: DensityFitSettings | None = None,
) -> DensityComparison:
    """Fit one model amplitude and compute the Zhu density chi-square.

    The scale is global across phi.  Therefore relative over/under-density
    between azimuth bins remains part of the likelihood.
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
