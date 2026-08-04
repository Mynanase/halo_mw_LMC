"""Azimuth-resolved velocity histograms and Zhu-style likelihoods."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def _edges(values: ArrayLike, name: str) -> FloatArray:
    result = np.asarray(values, dtype=float)
    if (
        result.ndim != 1
        or result.size < 2
        or not np.all(np.isfinite(result))
        or np.any(np.diff(result) <= 0)
    ):
        raise ValueError(f"{name} must be one-dimensional and strictly increasing")
    return result.copy()


def wrap_periodic(values: ArrayLike, origin: float, period: float = 2 * np.pi) -> FloatArray:
    values = np.asarray(values, dtype=float)
    return (values - origin) % period + origin


@dataclass(frozen=True)
class SphericalVelocityGrid:
    """Spatial and velocity edges; angles are in radians."""

    radius_edges: FloatArray
    theta_edges: FloatArray
    phi_edges: FloatArray
    velocity_edges: FloatArray

    def __post_init__(self) -> None:
        radius = _edges(self.radius_edges, "radius_edges")
        theta = _edges(self.theta_edges, "theta_edges")
        phi = _edges(self.phi_edges, "phi_edges")
        velocity = _edges(self.velocity_edges, "velocity_edges")
        if not np.isclose(phi[-1] - phi[0], 2 * np.pi):
            raise ValueError("phi_edges must span exactly 2π")
        object.__setattr__(self, "radius_edges", radius)
        object.__setattr__(self, "theta_edges", theta)
        object.__setattr__(self, "phi_edges", phi)
        object.__setattr__(self, "velocity_edges", velocity)

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return tuple(
            edges.size - 1
            for edges in (
                self.radius_edges,
                self.theta_edges,
                self.phi_edges,
                self.velocity_edges,
            )
        )

    @property
    def velocity_centers(self) -> FloatArray:
        return 0.5 * (self.velocity_edges[:-1] + self.velocity_edges[1:])

    def wrap_phi(self, phi: ArrayLike) -> FloatArray:
        return wrap_periodic(phi, self.phi_edges[0])


@dataclass(frozen=True)
class VelocityHistogramSummary:
    """A conditional velocity histogram and its spatial-cell occupancy."""

    probability: FloatArray
    uncertainty: FloatArray
    occupancy: FloatArray


@dataclass(frozen=True)
class VelocityDistributionComparison:
    """Observed and model velocity distributions on one common grid."""

    component: str
    grid: SphericalVelocityGrid
    data_probability: FloatArray
    data_uncertainty: FloatArray
    data_occupancy: FloatArray
    model_probability: FloatArray
    model_occupancy: FloatArray


def conditional_velocity_histogram(
    radius: ArrayLike,
    theta: ArrayLike,
    phi: ArrayLike,
    velocity: ArrayLike,
    grid: SphericalVelocityGrid,
    *,
    weights: ArrayLike | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Return ``p(v | r, theta, phi)`` as probability masses and occupancies."""

    radius, theta, phi, velocity = np.broadcast_arrays(
        np.asarray(radius, dtype=float),
        np.asarray(theta, dtype=float),
        grid.wrap_phi(phi),
        np.asarray(velocity, dtype=float),
    )
    valid = np.isfinite(radius) & np.isfinite(theta) & np.isfinite(phi) & np.isfinite(velocity)
    histogram_weights = None
    if weights is not None:
        histogram_weights = np.broadcast_to(np.asarray(weights, dtype=float), radius.shape)
        if np.any(np.isfinite(histogram_weights) & (histogram_weights < 0)):
            raise ValueError("orbit weights must be non-negative")
        valid &= np.isfinite(histogram_weights)
        histogram_weights = histogram_weights[valid]

    samples = np.column_stack(
        (radius[valid], theta[valid], phi[valid], velocity[valid])
    )
    histogram, _ = np.histogramdd(
        samples,
        bins=(
            grid.radius_edges,
            grid.theta_edges,
            grid.phi_edges,
            grid.velocity_edges,
        ),
        weights=histogram_weights,
    )
    occupancy = np.sum(histogram, axis=-1)
    probability = np.divide(
        histogram,
        occupancy[..., None],
        out=np.zeros_like(histogram, dtype=float),
        where=occupancy[..., None] > 0,
    )
    return probability, occupancy


def multinomial_histogram_uncertainty(
    probability: ArrayLike,
    occupancy: ArrayLike,
) -> FloatArray:
    """Return the diagonal multinomial 1-sigma uncertainty of each bin."""

    probability = np.asarray(probability, dtype=float)
    occupancy = np.asarray(occupancy, dtype=float)
    if probability.shape[:-1] != occupancy.shape:
        raise ValueError("probability and occupancy shapes are inconsistent")
    variance = np.divide(
        probability * np.clip(1.0 - probability, 0.0, None),
        occupancy[..., None],
        out=np.zeros_like(probability),
        where=occupancy[..., None] > 0,
    )
    return np.sqrt(variance)


def velocity_log_likelihood(
    radius: ArrayLike,
    theta: ArrayLike,
    phi: ArrayLike,
    observed_velocity: ArrayLike,
    velocity_error: ArrayLike,
    model_probability: ArrayLike,
    grid: SphericalVelocityGrid,
    *,
    probability_floor: float = 1e-300,
    minimum_radius: float | None = None,
) -> tuple[float, FloatArray, NDArray[np.int64]]:
    """Convolve model velocity bins with each star's Gaussian uncertainty.

    Returns the natural-log likelihood in each phi bin and the number of stars
    used there.  Stars outside the configured spatial grid are excluded.
    Observed stars in an empty model cell receive the probability floor, so a
    trial potential cannot improve merely by failing to cover the data.
    ``minimum_radius`` restores the paper/legacy exclusion of the incomplete
    inner orbit library (8 kpc in the production configuration).
    """

    model = np.asarray(model_probability, dtype=float)
    if model.shape != grid.shape:
        raise ValueError(f"model_probability has shape {model.shape}; expected {grid.shape}")
    if probability_floor <= 0:
        raise ValueError("probability_floor must be positive")
    if minimum_radius is not None and (
        not np.isfinite(minimum_radius) or minimum_radius < 0
    ):
        raise ValueError("minimum_radius must be finite and non-negative")

    radius, theta, phi, observed_velocity, velocity_error = np.broadcast_arrays(
        np.asarray(radius, dtype=float),
        np.asarray(theta, dtype=float),
        grid.wrap_phi(phi),
        np.asarray(observed_velocity, dtype=float),
        np.asarray(velocity_error, dtype=float),
    )
    radius = radius.ravel()
    theta = theta.ravel()
    phi = phi.ravel()
    observed_velocity = observed_velocity.ravel()
    velocity_error = velocity_error.ravel()
    valid = (
        np.isfinite(radius)
        & np.isfinite(theta)
        & np.isfinite(phi)
        & np.isfinite(observed_velocity)
        & np.isfinite(velocity_error)
        & (velocity_error > 0)
    )
    if minimum_radius is not None:
        valid &= radius >= minimum_radius

    ir = np.searchsorted(grid.radius_edges, radius, side="right") - 1
    it = np.searchsorted(grid.theta_edges, theta, side="right") - 1
    ip = np.searchsorted(grid.phi_edges, phi, side="right") - 1
    nr, nt, nphi, _ = grid.shape
    valid &= (ir >= 0) & (ir < nr) & (it >= 0) & (it < nt) & (ip >= 0) & (ip < nphi)

    loglike_by_phi = np.zeros(nphi, dtype=float)
    used_by_phi = np.zeros(nphi, dtype=np.int64)
    centers = grid.velocity_centers
    normalizer = np.sqrt(2 * np.pi)

    for index in np.flatnonzero(valid):
        cell_probability = model[ir[index], it[index], ip[index]]
        mass = np.sum(cell_probability)
        phi_index = ip[index]
        if not np.isfinite(mass) or mass <= 0:
            loglike_by_phi[phi_index] += np.log(probability_floor)
            used_by_phi[phi_index] += 1
            continue
        sigma = velocity_error[index]
        kernel = np.exp(-0.5 * ((observed_velocity[index] - centers) / sigma) ** 2)
        likelihood = np.sum(cell_probability * kernel) / (normalizer * sigma * mass)
        loglike_by_phi[phi_index] += np.log(max(float(likelihood), probability_floor))
        used_by_phi[phi_index] += 1

    return float(np.sum(loglike_by_phi)), loglike_by_phi, used_by_phi
