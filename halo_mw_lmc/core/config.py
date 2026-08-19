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
class WeightModelSettings:
    """How orbit weights are supplied or profiled for each trial potential."""

    mode: str = "catalogue_fixed"
    solver: str | None = None
    target_normalization: str | None = None
    regularization: str | None = None
    regularization_strength: float = 0.0
    max_iter: int = 20000
    # Inner LSMR tolerance of the TRF least-squares solver. A tight fixed
    # value makes each outer iteration accurate; scipy's "auto" rule keeps
    # the inner solve coarse while the optimality residual is large, which
    # stalls outer convergence for thousands of iterations. None keeps the
    # scipy "auto" behaviour for backward compatibility.
    lsmr_tol: float | None = 1e-6

    def __post_init__(self) -> None:
        if self.mode not in {"catalogue_fixed", "density_solved"}:
            raise ValueError(
                "weight mode must be 'catalogue_fixed' or 'density_solved'"
            )
        if self.mode == "catalogue_fixed":
            if any(
                value is not None
                for value in (
                    self.solver,
                    self.target_normalization,
                    self.regularization,
                )
            ) or self.regularization_strength != 0:
                raise ValueError(
                    "catalogue_fixed weights cannot define solver options"
                )
            return
        if self.solver != "lsq_linear":
            raise ValueError("density_solved currently requires solver='lsq_linear'")
        if self.target_normalization not in {"absolute", "unit_mass"}:
            raise ValueError(
                "target_normalization must be 'absolute' or 'unit_mass'"
            )
        if self.regularization != "l2":
            raise ValueError("density_solved currently requires regularization='l2'")
        if (
            not np.isfinite(self.regularization_strength)
            or self.regularization_strength < 0
        ):
            raise ValueError("regularization_strength must be finite and non-negative")
        if self.max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if self.lsmr_tol is not None and (
            not np.isfinite(self.lsmr_tol) or self.lsmr_tol <= 0
        ):
            raise ValueError("lsmr_tol must be None or a positive finite number")


@dataclass(frozen=True)
class ObjectiveSettings:
    """Outer objective after profiling any trial-specific orbit weights."""

    mode: str = "density_velocity"
    density_max_chi2_per_bin: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"velocity_only", "density_velocity"}:
            raise ValueError(
                "objective mode must be 'velocity_only' or 'density_velocity'"
            )
        if self.mode == "velocity_only":
            if (
                self.density_max_chi2_per_bin is None
                or not np.isfinite(self.density_max_chi2_per_bin)
                or self.density_max_chi2_per_bin <= 0
            ):
                raise ValueError(
                    "velocity_only requires a positive density chi2-per-bin limit"
                )
        elif self.density_max_chi2_per_bin is not None:
            raise ValueError(
                "density_velocity does not use a density chi2-per-bin gate"
            )


@dataclass(frozen=True)
class DensityFitSettings:
    """Masks and normalization used in the density term.

    Fixed-weight fits use one *global* scale over every azimuth bin; fitting a
    separate scale in each phi bin would erase the azimuthal signal. Profiled
    density weights instead use ``normalization='none'`` because their weights
    already set the density amplitude.
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
    velocity_probability_floor: float = 1e-300
    orbit_periods: float = 10.0
    orbit_samples_per_orbit: int = 1000
    # The legacy run compares z>0 after sampling a full orbit.  Dividing by
    # Nsample/2 preserves its amplitude; a subsequent single global scale makes
    # the exact constant irrelevant to the azimuthal shape.
    orbit_sample_divisor: float = 500.0
    weight_model: WeightModelSettings = field(default_factory=WeightModelSettings)
    objective: ObjectiveSettings = field(default_factory=ObjectiveSettings)

    def __post_init__(self) -> None:
        if self.orbit_samples_per_orbit < 1:
            raise ValueError("orbit_samples_per_orbit must be positive")
        if (
            not np.isfinite(self.velocity_fit_min_radius)
            or self.velocity_fit_min_radius < 0
        ):
            raise ValueError("velocity_fit_min_radius must be finite and non-negative")
        if (
            not np.isfinite(self.velocity_probability_floor)
            or self.velocity_probability_floor <= 0
        ):
            raise ValueError("velocity_probability_floor must be finite and positive")
        if not np.isfinite(self.orbit_periods) or self.orbit_periods <= 0:
            raise ValueError("orbit_periods must be finite and positive")
        if self.orbit_sample_divisor <= 0:
            raise ValueError("orbit_sample_divisor must be positive")
        if self.weight_model.mode == "density_solved":
            if self.density_fit.normalization != "none":
                raise ValueError(
                    "density_solved weights require density normalization='none'"
                )
            if not self.include_velocity:
                raise ValueError(
                    "density_solved weights require velocity likelihoods"
                )
        if self.objective.mode == "velocity_only" and not self.include_velocity:
            raise ValueError("velocity_only objective requires velocity likelihoods")

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
