"""Adapters for six-dimensional seed catalogues with optional fixed weights."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from ..core.coverage import PHASE_SPACE_COLUMNS
from ..core.weights import catalogue_seed_weights
from .ascii import read_named_columns


VELOCITY_ERROR_COLUMNS = {
    "vr": "vr_err",
    "vphi": "vphi_err",
    "vtheta": "vthe_err",
}


@dataclass(frozen=True)
class SeedCatalogue:
    """Core-ready arrays read from one survey catalogue."""

    initial_conditions: np.ndarray
    seed_weights: np.ndarray | None
    velocity_errors: Mapping[str, np.ndarray]


def read_phase_space_catalogue(path: str | Path) -> np.ndarray:
    """Return ``(x,y,z,vx,vy,vz)`` columns as a ``float64 (N,6)`` array."""

    columns = read_named_columns(path, PHASE_SPACE_COLUMNS)
    return np.column_stack([columns[name] for name in PHASE_SPACE_COLUMNS])


def read_seed_catalogue(
    path: str | Path,
    *,
    include_velocity: bool,
    require_weights: bool = True,
) -> SeedCatalogue:
    """Read phase space, optional fixed weights, and velocity errors."""

    required = [*PHASE_SPACE_COLUMNS]
    if require_weights:
        required.append("w")
    if include_velocity:
        required.extend(VELOCITY_ERROR_COLUMNS.values())
    columns = read_named_columns(path, required)
    initial = np.column_stack([columns[name] for name in PHASE_SPACE_COLUMNS])
    weights = catalogue_seed_weights(columns["w"]) if require_weights else None
    if weights is not None and weights.shape[0] != initial.shape[0]:
        raise ValueError("catalogue phase-space and weight columns have different lengths")
    errors = (
        {
            component: np.asarray(columns[column], dtype=float)
            for component, column in VELOCITY_ERROR_COLUMNS.items()
        }
        if include_velocity
        else {}
    )
    return SeedCatalogue(
        initial_conditions=np.asarray(initial, dtype=float),
        seed_weights=weights,
        velocity_errors=errors,
    )
