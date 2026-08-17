"""Coordinate grids shared by the data and orbit models.

All angles in the core package are in radians.  The cylindrical histogram
axis order is always ``(R, z, phi)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def _validated_edges(values: ArrayLike, name: str) -> FloatArray:
    edges = np.asarray(values, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError(f"{name} must be a one-dimensional array with at least two edges")
    if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise ValueError(f"{name} must contain finite, strictly increasing values")
    return edges.copy()


@dataclass(frozen=True)
class CylindricalGrid:
    """A full-azimuth grid with explicit cylindrical cell volumes.

    ``phi_edges`` must cover one complete period.  Values outside the chosen
    interval are wrapped before binning, so catalogues using ``[0, 2π)`` and
    orbit integrations using ``[-π, π)`` can be compared safely.
    """

    r_edges: FloatArray
    z_edges: FloatArray
    phi_edges: FloatArray

    def __post_init__(self) -> None:
        r_edges = _validated_edges(self.r_edges, "r_edges")
        z_edges = _validated_edges(self.z_edges, "z_edges")
        phi_edges = _validated_edges(self.phi_edges, "phi_edges")
        if r_edges[0] < 0:
            raise ValueError("r_edges cannot include negative cylindrical radii")
        if not np.isclose(phi_edges[-1] - phi_edges[0], 2 * np.pi):
            raise ValueError("phi_edges must span exactly 2π radians")
        object.__setattr__(self, "r_edges", r_edges)
        object.__setattr__(self, "z_edges", z_edges)
        object.__setattr__(self, "phi_edges", phi_edges)

    @classmethod
    def uniform(
        cls,
        *,
        n_r: int = 25,
        r_range: tuple[float, float] = (0.0, 50.0),
        n_z: int = 25,
        z_range: tuple[float, float] = (0.0, 50.0),
        n_phi: int = 4,
        phi_origin: float = -np.pi,
    ) -> "CylindricalGrid":
        if min(n_r, n_z, n_phi) < 1:
            raise ValueError("all bin counts must be positive")
        return cls(
            r_edges=np.linspace(*r_range, n_r + 1),
            z_edges=np.linspace(*z_range, n_z + 1),
            phi_edges=np.linspace(phi_origin, phi_origin + 2 * np.pi, n_phi + 1),
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return (
            self.r_edges.size - 1,
            self.z_edges.size - 1,
            self.phi_edges.size - 1,
        )

    @property
    def centers(self) -> tuple[FloatArray, FloatArray, FloatArray]:
        return tuple(
            0.5 * (edges[:-1] + edges[1:])
            for edges in (self.r_edges, self.z_edges, self.phi_edges)
        )

    @property
    def center_mesh(self) -> tuple[FloatArray, FloatArray, FloatArray]:
        return np.meshgrid(*self.centers, indexing="ij")

    @property
    def volumes(self) -> FloatArray:
        """Exact cell volumes ``½(R₂²-R₁²) Δz Δphi`` in kpc³."""

        annulus = 0.5 * np.diff(self.r_edges**2)
        dz = np.diff(self.z_edges)
        dphi = np.diff(self.phi_edges)
        return annulus[:, None, None] * dz[None, :, None] * dphi[None, None, :]

    def wrap_phi(self, phi: ArrayLike) -> FloatArray:
        phi_values = np.asarray(phi, dtype=float)
        origin = self.phi_edges[0]
        return (phi_values - origin) % (2 * np.pi) + origin

    def bin_indices(
        self,
        radius: ArrayLike,
        z: ArrayLike,
        phi: ArrayLike,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], NDArray[np.bool_]]:
        """Return integer ``(R,z,phi)`` indices and an in-grid mask."""

        radius_values, z_values, phi_values = np.broadcast_arrays(
            np.asarray(radius, dtype=float),
            np.asarray(z, dtype=float),
            self.wrap_phi(phi),
        )
        ir = np.searchsorted(self.r_edges, radius_values, side="right") - 1
        iz = np.searchsorted(self.z_edges, z_values, side="right") - 1
        iphi = np.searchsorted(self.phi_edges, phi_values, side="right") - 1
        valid = (
            np.isfinite(radius_values)
            & np.isfinite(z_values)
            & np.isfinite(phi_values)
            & (ir >= 0)
            & (ir < self.shape[0])
            & (iz >= 0)
            & (iz < self.shape[1])
            & (iphi >= 0)
            & (iphi < self.shape[2])
        )
        return (
            np.asarray(ir, dtype=np.int64),
            np.asarray(iz, dtype=np.int64),
            np.asarray(iphi, dtype=np.int64),
            np.asarray(valid, dtype=bool),
        )

    def histogram(
        self,
        radius: ArrayLike,
        z: ArrayLike,
        phi: ArrayLike,
        *,
        weights: ArrayLike | None = None,
    ) -> FloatArray:
        radius_values, z_values, phi_values = np.broadcast_arrays(
            np.asarray(radius, dtype=float),
            np.asarray(z, dtype=float),
            self.wrap_phi(phi),
        )
        valid = np.isfinite(radius_values) & np.isfinite(z_values) & np.isfinite(phi_values)

        weight_values: FloatArray | None = None
        if weights is not None:
            weight_values = np.broadcast_to(
                np.asarray(weights, dtype=float), radius_values.shape
            )
            valid &= np.isfinite(weight_values)
            weight_values = weight_values[valid]

        samples = np.column_stack(
            (radius_values[valid], z_values[valid], phi_values[valid])
        )
        histogram, _ = np.histogramdd(
            samples,
            bins=(self.r_edges, self.z_edges, self.phi_edges),
            weights=weight_values,
        )
        return np.asarray(histogram, dtype=float)
