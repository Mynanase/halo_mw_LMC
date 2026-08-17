"""AGAMA adapter for equal-time empirical orbit libraries."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class OrbitLibrary:
    """Flattened equal-time samples and their parent catalogue rows."""

    seed_index: NDArray[np.int64]
    time: FloatArray
    phase_space: FloatArray

    @property
    def x(self) -> FloatArray:
        return self.phase_space[:, 0]

    @property
    def y(self) -> FloatArray:
        return self.phase_space[:, 1]

    @property
    def z(self) -> FloatArray:
        return self.phase_space[:, 2]

    @property
    def vx(self) -> FloatArray:
        return self.phase_space[:, 3]

    @property
    def vy(self) -> FloatArray:
        return self.phase_space[:, 4]

    @property
    def vz(self) -> FloatArray:
        return self.phase_space[:, 5]

    @property
    def successful_seed_index(self) -> NDArray[np.int64]:
        return np.unique(self.seed_index)


def integrate_agama_orbits(
    initial_conditions: ArrayLike,
    potential,
    *,
    periods: float = 10.0,
    samples_per_orbit: int = 1000,
    pattern_speed: float | None = None,
    quiet: bool = True,
) -> OrbitLibrary:
    """Integrate catalogue phase points for a fixed number of circular periods."""

    initial = np.asarray(initial_conditions, dtype=float)
    if initial.ndim != 2 or initial.shape[1] != 6:
        raise ValueError("initial_conditions must have shape (N, 6)")
    if periods <= 0 or samples_per_orbit < 1:
        raise ValueError("periods and samples_per_orbit must be positive")

    circular_period = np.asarray(potential.Tcirc(initial), dtype=float)
    valid_seed = np.flatnonzero(np.isfinite(circular_period) & (circular_period > 0))
    if valid_seed.size == 0:
        raise ValueError("AGAMA found no initial conditions with a finite circular period")

    kwargs = {
        "ic": initial,
        "potential": potential,
        "time": periods * circular_period,
        "trajsize": samples_per_orbit,
    }
    if pattern_speed is not None:
        kwargs["Omega"] = pattern_speed

    if quiet:
        with Path("/dev/null").open("w") as sink:
            with redirect_stdout(sink), redirect_stderr(sink):
                raw_orbits = __import__("agama").orbit(**kwargs)
    else:
        raw_orbits = __import__("agama").orbit(**kwargs)

    times: list[FloatArray] = []
    trajectories: list[FloatArray] = []
    parent_rows: list[NDArray[np.int64]] = []
    for seed in valid_seed:
        time = np.asarray(raw_orbits[seed][0], dtype=float)
        trajectory = np.asarray(raw_orbits[seed][1], dtype=float)
        if (
            trajectory.ndim != 2
            or trajectory.shape[1] != 6
            or time.size != trajectory.shape[0]
            or trajectory.shape[0] == 0
        ):
            continue
        finite = np.all(np.isfinite(trajectory), axis=1) & np.isfinite(time)
        if not np.any(finite):
            continue
        times.append(time[finite])
        trajectories.append(trajectory[finite])
        parent_rows.append(np.full(np.sum(finite), seed, dtype=np.int64))

    if not trajectories:
        raise ValueError("AGAMA returned no finite orbit samples")
    return OrbitLibrary(
        seed_index=np.concatenate(parent_rows),
        time=np.concatenate(times),
        phase_space=np.concatenate(trajectories, axis=0),
    )
