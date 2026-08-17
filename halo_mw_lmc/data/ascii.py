"""Small adapter for named-column ASCII tables used by the survey products."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def read_named_columns(
    path: str | Path,
    required: Iterable[str],
) -> dict[str, np.ndarray]:
    """Read selected named columns without exposing an Astropy table downstream."""

    source = Path(path)
    required_names = tuple(required)
    try:
        from astropy.table import Table
    except ImportError:
        try:
            table = np.genfromtxt(source, names=True, ndmin=1)
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError(f"could not read ASCII table {source}: {exc}") from exc
        available = set(table.dtype.names or ())
        missing = sorted(set(required_names) - available)
        if missing:
            raise ValueError(
                "ASCII table is missing required columns: " + ", ".join(missing)
            )
        return {
            name: np.ma.asarray(table[name], dtype=float).filled(np.nan)
            for name in required_names
        }

    try:
        table = Table.read(source, format="ascii")
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"could not read ASCII table {source}: {exc}") from exc
    missing = sorted(set(required_names) - set(table.colnames))
    if missing:
        raise ValueError(
            "ASCII table is missing required columns: " + ", ".join(missing)
        )
    return {
        name: np.ma.asarray(table[name], dtype=float).filled(np.nan)
        for name in required_names
    }
