"""Fixed Zhu representative weights on an ``(R,z,phi)`` target grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .grids import CylindricalGrid


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RepresentativeWeightResult:
    """Fixed seed-orbit weights and their spatial support diagnostics."""

    weights: FloatArray
    seed_counts: NDArray[np.int64]
    cell_weight: FloatArray
    supported_cells: NDArray[np.bool_]
    target_mass: FloatArray
    assigned_mass: float
    positive_target_mass: float
    unsupported_positive_mass: float
    weighted_seed_count: int
    in_grid_seed_count: int

    @property
    def supported_mass_fraction(self) -> float:
        if self.positive_target_mass <= 0:
            return 0.0
        return self.assigned_mass / self.positive_target_mass


def representative_weights_from_target(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    target_density: ArrayLike,
    grid: CylindricalGrid,
    *,
    minimum_seed_count: int = 1,
) -> RepresentativeWeightResult:
    """Assign fixed orbit weights from a three-dimensional target density.

    For a spatial cell ``j=(R,z,phi)`` containing ``N_j`` seed stars,

    ``w_i = nu_target[j] * V[j] / N_j``.

    Consequently, summing the seed weights in any supported cell exactly
    reproduces its target tracer mass.  The result is fixed before trial
    potentials are evaluated.
    """

    if minimum_seed_count < 1:
        raise ValueError("minimum_seed_count must be positive")
    x_values, y_values, z_values = np.broadcast_arrays(
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
        np.asarray(z, dtype=float),
    )
    if x_values.ndim != 1:
        raise ValueError("seed coordinates must be one-dimensional arrays")

    target = np.asarray(target_density, dtype=float)
    if target.shape != grid.shape:
        raise ValueError(
            f"target_density has shape {target.shape}; expected {grid.shape} (R,z,phi)"
        )
    if np.any(np.isfinite(target) & (target < 0)):
        raise ValueError("target_density cannot contain negative values")

    radius = np.hypot(x_values, y_values)
    phi = np.arctan2(y_values, x_values)
    ir, iz, iphi, in_grid = grid.bin_indices(radius, z_values, phi)
    seed_counts_float = grid.histogram(radius, z_values, phi)
    seed_counts = np.asarray(np.rint(seed_counts_float), dtype=np.int64)

    finite_target = np.isfinite(target) & (target >= 0)
    target_mass = np.where(finite_target, target * grid.volumes, 0.0)
    supported = finite_target & (seed_counts >= minimum_seed_count)
    cell_weight = np.divide(
        target_mass,
        seed_counts,
        out=np.zeros_like(target_mass),
        where=supported,
    )

    weights = np.zeros(x_values.size, dtype=float)
    valid_seed = in_grid & supported[
        np.clip(ir, 0, grid.shape[0] - 1),
        np.clip(iz, 0, grid.shape[1] - 1),
        np.clip(iphi, 0, grid.shape[2] - 1),
    ]
    weights[valid_seed] = cell_weight[
        ir[valid_seed],
        iz[valid_seed],
        iphi[valid_seed],
    ]

    positive_target = finite_target & (target > 0)
    assigned_mass = float(np.sum(target_mass[supported]))
    positive_mass = float(np.sum(target_mass[positive_target]))
    unsupported_mass = float(np.sum(target_mass[positive_target & ~supported]))
    return RepresentativeWeightResult(
        weights=weights,
        seed_counts=seed_counts,
        cell_weight=cell_weight,
        supported_cells=supported,
        target_mass=target_mass,
        assigned_mass=assigned_mass,
        positive_target_mass=positive_mass,
        unsupported_positive_mass=unsupported_mass,
        weighted_seed_count=int(np.count_nonzero(weights > 0)),
        in_grid_seed_count=int(np.count_nonzero(in_grid)),
    )
