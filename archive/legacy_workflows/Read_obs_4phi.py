"""Readers for historical flattened ``4phi`` products."""

from __future__ import annotations

import numpy as np
from astropy import table


def _reshape_column(data, column, shape):
    values = np.asarray(data[column], dtype=float)
    expected = int(np.prod(shape))
    if values.size != expected:
        raise ValueError(
            f"column {column!r} contains {values.size} values; "
            f"expected {expected} for shape {shape}"
        )
    return values.reshape(shape)


def Read_obsSB_v2_4phi(dtfile, nRz, nphi):
    """Read density arrays in the legacy ``(z, R, phi)`` order."""

    data = table.Table.read(dtfile, format="ascii")
    shape = (nRz, nRz, nphi)
    return (
        _reshape_column(data, "den", shape),
        _reshape_column(data, "den_srr", shape),
        _reshape_column(data, "R2d", shape),
        _reshape_column(data, "z2d", shape),
    )


def Read_obsSB_4phi(dtfile, nRz, nphi):
    data = table.Table.read(dtfile, format="ascii")
    shape = (nRz, nRz, nphi)
    return (
        _reshape_column(data, "den", shape),
        _reshape_column(data, "den_srr", shape),
    )


def Read_obsvhist_4phi(dtfile, nr, ntheta, nphi, nv):
    data = table.Table.read(dtfile, format="ascii")
    shape = (nr, ntheta, nphi, nv)
    components = []
    for value_column, error_column in (
        ("vr_mean", "vr_err"),
        ("vphi_mean", "vphi_err"),
        ("vtheta_mean", "vtheta_err"),
    ):
        values = _reshape_column(data, value_column, shape)
        errors = _reshape_column(data, error_column, shape)
        norm = np.sum(values, axis=-1, keepdims=True)
        values = np.divide(
            values,
            norm,
            out=np.zeros_like(values),
            where=norm > 0,
        )
        errors = np.divide(
            errors,
            norm,
            out=np.zeros_like(errors),
            where=norm > 0,
        )
        components.extend((values, errors))
    return tuple(components)
