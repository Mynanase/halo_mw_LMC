"""Compatibility layer for the historical ``int_ag`` function."""

from __future__ import annotations

import numpy as np

from halo_mw_lmc.orbits import integrate_agama_orbits


def int_ag(ic, pot, t_p, tra, omg, per=False):
    """Integrate AGAMA orbits and return the original dictionary layout.

    New code should use :func:`halo_mw_lmc.orbits.integrate_agama_orbits`.
    Per-orbit apo/pericentre diagnostics from the old experimental branch are
    intentionally not exposed because they assumed a fragile trajectory shape.
    """

    if per:
        raise NotImplementedError(
            "per-orbit diagnostics are not part of the Zhu comparison pipeline"
        )
    pattern_speed = None if omg is False else float(omg)
    library = integrate_agama_orbits(
        ic,
        pot,
        periods=float(t_p),
        samples_per_orbit=int(tra),
        pattern_speed=pattern_speed,
    )
    phase = library.phase_space
    kinetic_energy = 0.5 * np.sum(phase[:, 3:] ** 2, axis=1)
    potential_energy = np.asarray(pot.potential(phase[:, :3]), dtype=float)
    if pattern_speed is None:
        jacobi_term = 0.0
    else:
        angular_momentum_z = phase[:, 0] * phase[:, 4] - phase[:, 1] * phase[:, 3]
        jacobi_term = pattern_speed * angular_momentum_z
    return {
        "orb_tl": library.seed_index,
        "t": library.time,
        "x": phase[:, 0],
        "y": phase[:, 1],
        "z": phase[:, 2],
        "vx": phase[:, 3],
        "vy": phase[:, 4],
        "vz": phase[:, 5],
        "E": potential_energy + kinetic_energy - jacobi_term,
    }
