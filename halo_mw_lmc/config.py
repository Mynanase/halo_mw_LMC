"""Configuration objects for the Zhu-style comparison."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .grids import CylindricalGrid
from .velocity import SphericalVelocityGrid


def _default_velocity_grid() -> SphericalVelocityGrid:
    return SphericalVelocityGrid(
        radius_edges=np.array([4, 6, 8, 10, 12, 15, 20, 30, 50], dtype=float),
        theta_edges=np.deg2rad(np.array([0, 15, 30, 45, 60, 90], dtype=float)),
        phi_edges=np.linspace(-np.pi, np.pi, 5),
        velocity_edges=np.linspace(-800, 800, 202),
    )


@dataclass(frozen=True)
class DensityFitSettings:
    """Masks and normalization used in the density term.

    One *global* scale is fitted over every azimuth bin.  This is essential:
    fitting a separate scale in each phi bin would erase the azimuthal signal
    that the extension is intended to measure.
    """

    min_abs_z: float = 2.0
    min_spherical_radius: float = 15.0
    max_spherical_radius: float = 40.0
    normalization_min_radius: float = 10.0
    require_positive_data: bool = True
    normalization: str = "volume"

    def __post_init__(self) -> None:
        if self.min_abs_z < 0:
            raise ValueError("min_abs_z cannot be negative")
        if self.min_spherical_radius >= self.max_spherical_radius:
            raise ValueError("the spherical-radius fit interval is empty")
        if self.normalization not in {"volume", "weighted_least_squares", "none"}:
            raise ValueError(
                "normalization must be 'volume', 'weighted_least_squares', or 'none'"
            )


@dataclass(frozen=True)
class ZhuComparisonConfig:
    """Numerical choices for orbit sampling and the data/model comparison."""

    density_grid: CylindricalGrid = field(default_factory=CylindricalGrid.uniform)
    density_fit: DensityFitSettings = field(default_factory=DensityFitSettings)
    velocity_grid: SphericalVelocityGrid = field(default_factory=_default_velocity_grid)
    include_velocity: bool = False
    velocity_fit_min_radius: float = 8.0
    orbit_samples_per_orbit: int = 1000
    # The legacy run compares z>0 after sampling a full orbit.  Dividing by
    # Nsample/2 preserves its amplitude; a subsequent single global scale makes
    # the exact constant irrelevant to the azimuthal shape.
    orbit_sample_divisor: float = 500.0

    def __post_init__(self) -> None:
        if self.orbit_samples_per_orbit < 1:
            raise ValueError("orbit_samples_per_orbit must be positive")
        if (
            not np.isfinite(self.velocity_fit_min_radius)
            or self.velocity_fit_min_radius < 0
        ):
            raise ValueError("velocity_fit_min_radius must be finite and non-negative")
        if self.orbit_sample_divisor <= 0:
            raise ValueError("orbit_sample_divisor must be positive")

    @classmethod
    def legacy_4phi(
        cls,
        *,
        n_phi: int = 4,
        n_rz: int = 25,
        rz_max: float = 50.0,
        orbit_samples_per_orbit: int = 1000,
        include_velocity: bool = False,
    ) -> "ZhuComparisonConfig":
        phi_edges = np.linspace(-np.pi, np.pi, n_phi + 1)
        return cls(
            density_grid=CylindricalGrid.uniform(
                n_r=n_rz,
                r_range=(0.0, rz_max),
                n_z=n_rz,
                z_range=(0.0, rz_max),
                n_phi=n_phi,
            ),
            velocity_grid=SphericalVelocityGrid(
                radius_edges=np.array(
                    [4, 6, 8, 10, 12, 15, 20, 30, 50], dtype=float
                ),
                theta_edges=np.deg2rad(
                    np.array([0, 15, 30, 45, 60, 90], dtype=float)
                ),
                phi_edges=phi_edges,
                velocity_edges=np.linspace(-800, 800, 202),
            ),
            include_velocity=include_velocity,
            velocity_fit_min_radius=8.0,
            orbit_samples_per_orbit=orbit_samples_per_orbit,
            orbit_sample_divisor=orbit_samples_per_orbit / 2,
        )
