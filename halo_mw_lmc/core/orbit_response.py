"""Sparse three-dimensional density responses for equal-time orbit samples."""

from __future__ import annotations

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
        ir, iz, iphi, in_grid = grid.bin_indices(
            radius,
            phase_space[:, 2],
            phi,
        )
        if not np.any(in_grid):
            continue
        flat_cell = np.ravel_multi_index(
            (ir[in_grid], iz[in_grid], iphi[in_grid]),
            grid.shape,
        )
        columns = column_by_seed[seed_index[start:stop][in_grid]]
        chunk = coo_matrix(
            (
                np.ones(flat_cell.size, dtype=float),
                (flat_cell, columns),
            ),
            shape=response_shape,
        ).tocsr()
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
