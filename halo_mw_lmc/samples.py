"""Utilities for reading and validating optimizer sample files."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Sequence

import numpy as np


class SampleFileError(ValueError):
    """Raised when an optimizer sample file cannot be used safely."""


def _require_columns(data: np.ndarray, columns: Sequence[str]) -> None:
    names = set(data.dtype.names or ())
    missing = [name for name in columns if name not in names]
    if missing:
        raise SampleFileError(
            "sample file is missing required columns: " + ", ".join(missing)
        )


def load_sample_table(
    path: str | Path,
    *,
    required_columns: Sequence[str] = (),
) -> np.ndarray:
    """Read a non-empty ``sample.dat`` as a one-dimensional structured array."""

    sample_path = Path(path)
    if not sample_path.exists():
        raise SampleFileError(f"sample file not found: {sample_path}")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            data = np.genfromtxt(sample_path, names=True, ndmin=1)
    except (OSError, TypeError, ValueError) as exc:
        raise SampleFileError(
            f"could not read sample file {sample_path}: {exc}"
        ) from exc

    if data.dtype.names is None:
        raise SampleFileError(f"sample file has no named header: {sample_path}")
    _require_columns(data, required_columns)
    if data.size == 0:
        raise SampleFileError(f"sample file contains no samples: {sample_path}")
    return data


def best_sample(data: np.ndarray) -> np.void:
    """Return the row with the smallest finite objective."""

    _require_columns(data, ("objective",))
    try:
        objective = np.asarray(data["objective"], dtype=float)
    except (TypeError, ValueError) as exc:
        raise SampleFileError("objective column is not numeric") from exc
    finite = np.flatnonzero(np.isfinite(objective))
    if finite.size == 0:
        raise SampleFileError("sample file contains no finite objective values")
    return data[finite[np.argmin(objective[finite])]]
