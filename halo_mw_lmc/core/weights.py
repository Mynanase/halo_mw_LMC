"""Fixed catalogue weights and experimental target-derived weights."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .grids import CylindricalGrid


FloatArray = NDArray[np.float64]


def catalogue_seed_weights(values: ArrayLike) -> FloatArray:
    """Validate and copy the fixed per-star weights supplied by the catalogue."""

    weights = np.asarray(values, dtype=float)
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError("catalogue weights must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(weights)):
        raise ValueError("catalogue weights must all be finite")
    if np.any(weights < 0):
        raise ValueError("catalogue weights must be non-negative")
    if not np.any(weights > 0):
        raise ValueError("catalogue weights must contain at least one positive value")
    return weights.copy()


def catalogue_weight_audit(
    initial_conditions: ArrayLike,
    weights: ArrayLike,
    grid: CylindricalGrid,
) -> dict[str, np.ndarray]:
    """Summarize fixed catalogue weights globally and per density cell."""

    initial = np.asarray(initial_conditions, dtype=float)
    weight_values = catalogue_seed_weights(weights)
    if initial.ndim != 2 or initial.shape != (weight_values.size, 6):
        raise ValueError("initial_conditions must have shape (N, 6) matching weights")
    radius = np.hypot(initial[:, 0], initial[:, 1])
    z = initial[:, 2]
    phi = np.arctan2(initial[:, 1], initial[:, 0])
    seed_counts = grid.histogram(radius, z, phi)
    cell_weight_sum = grid.histogram(radius, z, phi, weights=weight_values)
    cell_weight_sq_sum = grid.histogram(
        radius,
        z,
        phi,
        weights=weight_values**2,
    )
    initial_density = np.divide(
        cell_weight_sum,
        grid.volumes,
        out=np.zeros_like(cell_weight_sum),
        where=grid.volumes > 0,
    )

    ir, iz, iphi, in_grid = grid.bin_indices(radius, z, phi)
    cell_max_weight = np.zeros(grid.shape, dtype=float)
    np.maximum.at(
        cell_max_weight,
        (ir[in_grid], iz[in_grid], iphi[in_grid]),
        weight_values[in_grid],
    )
    cell_effective_seed_count = np.divide(
        cell_weight_sum**2,
        cell_weight_sq_sum,
        out=np.zeros_like(cell_weight_sum),
        where=cell_weight_sq_sum > 0,
    )
    cell_max_weight_fraction = np.divide(
        cell_max_weight,
        cell_weight_sum,
        out=np.zeros_like(cell_weight_sum),
        where=cell_weight_sum > 0,
    )

    total_weight = float(np.sum(weight_values))
    weight_sq_sum = float(np.sum(weight_values**2))
    quantile_levels = np.array([0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0])
    return {
        "weights": weight_values,
        "seed_counts": seed_counts,
        "cell_weight_sum": cell_weight_sum,
        "cell_weight_sq_sum": cell_weight_sq_sum,
        "cell_effective_seed_count": cell_effective_seed_count,
        "cell_max_weight": cell_max_weight,
        "cell_max_weight_fraction": cell_max_weight_fraction,
        "initial_catalog_density": initial_density,
        "quantile_levels": quantile_levels,
        "weight_quantiles": np.quantile(weight_values, quantile_levels),
        "positive_seed_count": np.asarray(np.count_nonzero(weight_values > 0)),
        "in_grid_seed_count": np.asarray(np.count_nonzero(in_grid)),
        "total_weight": np.asarray(total_weight),
        "in_grid_weight": np.asarray(float(np.sum(weight_values[in_grid]))),
        "effective_seed_count": np.asarray(
            total_weight**2 / weight_sq_sum if weight_sq_sum > 0 else 0.0
        ),
        "max_weight_fraction": np.asarray(
            float(np.max(weight_values)) / total_weight if total_weight > 0 else 0.0
        ),
    }


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
    """Assign experimental orbit weights from a three-dimensional target density.

    The production pipeline does not use this function. It is retained for
    isolated comparisons of target-derived and catalogue-supplied weights.

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
