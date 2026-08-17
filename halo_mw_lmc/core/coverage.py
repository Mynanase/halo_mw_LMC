"""Array-only coverage diagnostics for six-dimensional catalogues."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .grids import CylindricalGrid
from .phase_space import SphericalPhaseSpace, cartesian_to_spherical_phase_space


FloatArray = NDArray[np.float64]

PHASE_SPACE_COLUMNS = ("x_gc", "y_gc", "z_gc", "vx_gc", "vy_gc", "vz_gc")
DEFAULT_SPHERICAL_RADIUS_EDGES = np.array(
    [4, 6, 8, 10, 12, 15, 20, 30, 50],
    dtype=float,
)
DEFAULT_THETA_EDGES = np.deg2rad(
    np.array([0, 15, 30, 45, 60, 90], dtype=float)
)


def _validated_edges(values: ArrayLike, name: str) -> FloatArray:
    edges = np.asarray(values, dtype=float)
    if (
        edges.ndim != 1
        or edges.size < 2
        or not np.all(np.isfinite(edges))
        or np.any(np.diff(edges) <= 0)
    ):
        raise ValueError(f"{name} must be finite and strictly increasing")
    return edges.copy()


def _occupancy_statistics(counts: np.ndarray) -> dict[str, Any]:
    flat = np.asarray(counts, dtype=float).ravel()
    occupied = flat[flat > 0]
    total_cells = int(flat.size)
    empty_cells = int(np.count_nonzero(flat == 0))
    return {
        "total_cells": total_cells,
        "occupied_cells": int(occupied.size),
        "empty_cells": empty_cells,
        "empty_fraction": empty_cells / total_cells if total_cells else 0.0,
        "cells_with_1_to_5_stars": int(
            np.count_nonzero((flat >= 1) & (flat <= 5))
        ),
        "cells_with_1_to_10_stars": int(
            np.count_nonzero((flat >= 1) & (flat <= 10))
        ),
        "minimum_nonempty_count": int(np.min(occupied)) if occupied.size else 0,
        "median_nonempty_count": (
            float(np.median(occupied)) if occupied.size else 0.0
        ),
        "maximum_count": int(np.max(occupied)) if occupied.size else 0,
    }


@dataclass(frozen=True)
class DataCoverage:
    """Finite catalogue rows and occupancy on the two fitting grids."""

    input_rows: int
    column_finite_rows: tuple[int, ...]
    position_finite_rows: int
    positions: FloatArray
    initial_conditions: FloatArray
    phase_space: SphericalPhaseSpace
    cylindrical_radius: FloatArray
    rzphi_grid: CylindricalGrid
    rzphi_counts: FloatArray
    spherical_radius_edges: FloatArray
    theta_edges: FloatArray
    phi_edges: FloatArray
    rtheta_phi_counts: FloatArray

    @property
    def complete_phase_space_rows(self) -> int:
        return int(self.initial_conditions.shape[0])

    @property
    def spherical_cell_volumes(self) -> FloatArray:
        radial = np.diff(self.spherical_radius_edges**3) / 3.0
        latitude = np.diff(np.sin(self.theta_edges))
        azimuth = np.diff(self.phi_edges)
        return (
            radial[:, None, None]
            * latitude[None, :, None]
            * azimuth[None, None, :]
        )

    @property
    def rzphi_sampling_density(self) -> FloatArray:
        return np.divide(
            self.rzphi_counts,
            self.rzphi_grid.volumes,
            out=np.zeros_like(self.rzphi_counts),
            where=self.rzphi_grid.volumes > 0,
        )

    @property
    def rtheta_phi_sampling_density(self) -> FloatArray:
        volumes = self.spherical_cell_volumes
        return np.divide(
            self.rtheta_phi_counts,
            volumes,
            out=np.zeros_like(self.rtheta_phi_counts),
            where=volumes > 0,
        )

    def summary(self) -> dict[str, Any]:
        complete = self.complete_phase_space_rows
        return {
            "input_rows": self.input_rows,
            "finite_rows_by_column": {
                name: count
                for name, count in zip(PHASE_SPACE_COLUMNS, self.column_finite_rows)
            },
            "finite_position_rows": self.position_finite_rows,
            "complete_6d_rows": complete,
            "complete_6d_fraction": (
                complete / self.input_rows if self.input_rows else 0.0
            ),
            "rzphi": {
                "shape": list(self.rzphi_counts.shape),
                "in_grid_rows": int(np.sum(self.rzphi_counts)),
                "outside_grid_rows": complete - int(np.sum(self.rzphi_counts)),
                "in_grid_fraction": (
                    float(np.sum(self.rzphi_counts)) / complete if complete else 0.0
                ),
                "rows_by_phi": [
                    int(value) for value in np.sum(self.rzphi_counts, axis=(0, 1))
                ],
                **_occupancy_statistics(self.rzphi_counts),
            },
            "rtheta_phi": {
                "shape": list(self.rtheta_phi_counts.shape),
                "in_grid_rows": int(np.sum(self.rtheta_phi_counts)),
                "outside_grid_rows": complete
                - int(np.sum(self.rtheta_phi_counts)),
                "in_grid_fraction": (
                    float(np.sum(self.rtheta_phi_counts)) / complete
                    if complete
                    else 0.0
                ),
                "rows_by_phi": [
                    int(value)
                    for value in np.sum(self.rtheta_phi_counts, axis=(0, 1))
                ],
                **_occupancy_statistics(self.rtheta_phi_counts),
            },
        }


def build_data_coverage(
    initial_conditions: ArrayLike,
    *,
    rzphi_grid: CylindricalGrid | None = None,
    spherical_radius_edges: ArrayLike = DEFAULT_SPHERICAL_RADIUS_EDGES,
    theta_edges: ArrayLike = DEFAULT_THETA_EDGES,
) -> DataCoverage:
    """Measure raw catalogue coverage without selection-function correction."""

    initial = np.asarray(initial_conditions, dtype=float)
    if initial.ndim != 2 or initial.shape[1] != 6:
        raise ValueError("initial_conditions must have shape (N, 6)")

    grid = rzphi_grid or CylindricalGrid.uniform()
    radius_edges = _validated_edges(
        spherical_radius_edges,
        "spherical_radius_edges",
    )
    latitude_edges = _validated_edges(theta_edges, "theta_edges")
    if latitude_edges[0] < -np.pi / 2 or latitude_edges[-1] > np.pi / 2:
        raise ValueError("theta_edges must lie within [-pi/2, pi/2]")

    finite_by_column = tuple(
        int(np.count_nonzero(np.isfinite(initial[:, index])))
        for index in range(initial.shape[1])
    )
    position_mask = np.all(np.isfinite(initial[:, :3]), axis=1)
    complete_mask = np.all(np.isfinite(initial), axis=1)
    positions = initial[position_mask, :3]
    complete = initial[complete_mask]
    if complete.shape[0] == 0:
        raise ValueError("catalogue contains no complete finite 6D rows")

    phase = cartesian_to_spherical_phase_space(
        *[complete[:, index] for index in range(6)]
    )
    cylindrical_radius = np.hypot(complete[:, 0], complete[:, 1])
    rzphi_counts = grid.histogram(
        cylindrical_radius,
        complete[:, 2],
        phase.phi,
    )

    wrapped_phi = grid.wrap_phi(phase.phi)
    rtheta_phi_counts, _ = np.histogramdd(
        np.column_stack((phase.radius, phase.theta, wrapped_phi)),
        bins=(radius_edges, latitude_edges, grid.phi_edges),
    )
    return DataCoverage(
        input_rows=int(initial.shape[0]),
        column_finite_rows=finite_by_column,
        position_finite_rows=int(np.count_nonzero(position_mask)),
        positions=positions,
        initial_conditions=complete,
        phase_space=phase,
        cylindrical_radius=cylindrical_radius,
        rzphi_grid=grid,
        rzphi_counts=np.asarray(rzphi_counts, dtype=float),
        spherical_radius_edges=radius_edges,
        theta_edges=latitude_edges,
        phi_edges=grid.phi_edges.copy(),
        rtheta_phi_counts=np.asarray(rtheta_phi_counts, dtype=float),
    )
