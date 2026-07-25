"""Evaluate one Zhu-style empirical orbit model with explicit phi bins.

The public ``int_one_model`` signature remains compatible with the historical
optimizer.  The implementation now separates orbit integration, cylindrical
binning, density scoring, and optional velocity scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from halo_mw_lmc.config import ZhuComparisonConfig
from halo_mw_lmc.density import DensityComparison, compare_density, orbit_density
from halo_mw_lmc.orbits import OrbitLibrary, integrate_agama_orbits
from halo_mw_lmc.phase_space import cartesian_to_spherical_phase_space
from halo_mw_lmc.plotting import plot_density_comparison
from halo_mw_lmc.velocity import (
    conditional_velocity_histogram,
    velocity_log_likelihood,
)
from halo_mw_lmc.weights import (
    RepresentativeWeightResult,
    representative_weights_from_target,
)


@dataclass(frozen=True)
class PreparedFixedWeightData:
    """Catalogue and fixed weights shared by every trial potential."""

    catalog: object
    initial_conditions: np.ndarray
    target_density: np.ndarray
    target_error: np.ndarray
    representative_weights: RepresentativeWeightResult
    config: ZhuComparisonConfig
    catalog_path: Path
    density_path: Path


@dataclass(frozen=True)
class ModelEvaluation:
    """Density and velocity terms returned by a single trial potential."""

    density: DensityComparison
    velocity_loglike: Mapping[str, float]
    velocity_loglike_by_phi: Mapping[str, np.ndarray]
    velocity_stars_by_phi: Mapping[str, np.ndarray]
    successful_orbits: int
    representative_weights: RepresentativeWeightResult

    @property
    def log_likelihood(self) -> float:
        return -0.5 * self.density.chi2 + float(sum(self.velocity_loglike.values()))


def _require_columns(data, names) -> None:
    missing = sorted(set(names) - set(data.colnames))
    if missing:
        raise ValueError(f"catalogue is missing required columns: {', '.join(missing)}")


def _build_potential(rho0, rs, phalo, qhalo, gamma, alpha_halo, beta_halo):
    if phalo <= 0 or qhalo <= 0:
        raise ValueError("halo axis ratios p and q must be positive")
    if not np.isclose(alpha_halo, 0) or not np.isclose(beta_halo, 0):
        raise NotImplementedError(
            "the current AGAMA Spheroid is axis-aligned; halo rotation angles "
            "must be zero until the rotated Multipole implementation is enabled"
        )

    import agama

    agama.setUnits(length=1, velocity=1, mass=1)
    disk = agama.Potential(
        type="Disk",
        mass=10**10.5,
        scaleRadius=3,
        scaleHeight=-0.4,
        innerCutoffRadius=0,
        sersicIndex=1,
    )
    bulge = agama.Potential(
        type="Spheroid",
        mass=10**10.2,
        alpha=1,
        gamma=0,
        beta=1.8,
        scaleRadius=0.2,
        outerCutoffRadius=1.8,
        cutoffStrength=2,
    )
    halo = agama.Potential(
        type="Spheroid",
        rho0=10**rho0,
        alpha=1,
        gamma=gamma,
        beta=3,
        scaleRadius=10**rs,
        p=phalo,
        q=qhalo,
        outerCutoffRadius=500,
        cutoffStrength=5,
    )
    return agama.Potential(disk, bulge, halo)


def _initial_conditions(data) -> np.ndarray:
    names = ("x_gc", "y_gc", "z_gc", "vx_gc", "vy_gc", "vz_gc")
    _require_columns(data, names)
    return np.column_stack([np.asarray(data[name], dtype=float) for name in names])


def _default_density_path(base_path: Path, config: ZhuComparisonConfig) -> Path:
    n_r, n_z, n_phi = config.density_grid.shape
    if n_r != n_z or n_r != 25 or n_phi != 4:
        raise ValueError(
            "a target density file is required for a grid other than the "
            "historical 25x25x4 product"
        )
    return (
        base_path
        / "data_for_model"
        / "lamost_dr8_SFlast_cut4_4phi"
        / "w4_SB_Rz_254phi_err.txt"
    )


def _read_density(path: str | Path, config: ZhuComparisonConfig):
    """Read historical ``(z,R,phi)`` data and return ``(R,z,phi)`` arrays."""

    from Read_obs_4phi import Read_obsSB_4phi

    n_r, n_z, n_phi = config.density_grid.shape
    if n_r != n_z:
        raise ValueError(
            "the legacy ASCII density format has no axis metadata and therefore "
            "requires equal R and z bin counts"
        )
    density_zrphi, error_zrphi = Read_obsSB_4phi(path, n_r, n_phi)
    return (
        np.transpose(density_zrphi, (1, 0, 2)),
        np.transpose(error_zrphi, (1, 0, 2)),
    )


def _resolve_input_path(base: Path, path: str | Path) -> Path:
    result = Path(path).expanduser()
    return result.resolve() if result.is_absolute() else (base / result).resolve()


def prepare_fixed_weight_data(
    base_path,
    dtfile,
    *,
    observed_density_file=None,
    comparison_config: ZhuComparisonConfig | None = None,
) -> PreparedFixedWeightData:
    """Read inputs and compute one fixed ``(R,z,phi)`` weight per seed orbit."""

    config = comparison_config or ZhuComparisonConfig.legacy_4phi()
    base = Path(base_path).expanduser().resolve()
    catalog_path = _resolve_input_path(base, dtfile)
    density_path = (
        _resolve_input_path(base, observed_density_file)
        if observed_density_file is not None
        else _default_density_path(base, config)
    )

    from astropy import table

    data = table.Table.read(catalog_path, format="ascii")
    initial = _initial_conditions(data)
    target_density, target_error = _read_density(density_path, config)
    fixed_weights = representative_weights_from_target(
        initial[:, 0],
        initial[:, 1],
        initial[:, 2],
        target_density,
        config.density_grid,
        minimum_seed_count=config.minimum_seed_count,
    )
    if fixed_weights.weighted_seed_count == 0:
        raise ValueError(
            "the target density and seed catalogue have no positively weighted "
            "cells in common"
        )
    return PreparedFixedWeightData(
        catalog=data,
        initial_conditions=initial,
        target_density=target_density,
        target_error=target_error,
        representative_weights=fixed_weights,
        config=config,
        catalog_path=catalog_path,
        density_path=density_path,
    )


def _score_velocities(
    data,
    library: OrbitLibrary,
    orbit_weights: np.ndarray,
    config: ZhuComparisonConfig,
):
    model_phase = cartesian_to_spherical_phase_space(
        library.x,
        library.y,
        library.z,
        library.vx,
        library.vy,
        library.vz,
    )
    catalogue_phase = cartesian_to_spherical_phase_space(
        *[
            np.asarray(data[name], dtype=float)
            for name in ("x_gc", "y_gc", "z_gc", "vx_gc", "vy_gc", "vz_gc")
        ]
    )
    error_columns = {
        "vr": "vr_err",
        "vphi": "vphi_err",
        "vtheta": "vthe_err",
    }
    observed_velocity = {
        "vr": catalogue_phase.radial_velocity,
        "vphi": catalogue_phase.azimuthal_velocity,
        "vtheta": catalogue_phase.polar_velocity,
    }
    model_velocity = {
        "vr": model_phase.radial_velocity,
        "vphi": model_phase.azimuthal_velocity,
        "vtheta": model_phase.polar_velocity,
    }
    _require_columns(data, error_columns.values())

    total: dict[str, float] = {}
    by_phi: dict[str, np.ndarray] = {}
    stars_by_phi: dict[str, np.ndarray] = {}
    for name in ("vr", "vphi", "vtheta"):
        probability, _ = conditional_velocity_histogram(
            model_phase.radius,
            model_phase.theta,
            model_phase.phi,
            model_velocity[name],
            config.velocity_grid,
            weights=orbit_weights,
        )
        loglike, component_by_phi, used_by_phi = velocity_log_likelihood(
            catalogue_phase.radius,
            catalogue_phase.theta,
            catalogue_phase.phi,
            observed_velocity[name],
            np.asarray(data[error_columns[name]], dtype=float),
            probability,
            config.velocity_grid,
        )
        total[name] = loglike
        by_phi[name] = component_by_phi
        stars_by_phi[name] = used_by_phi
    return total, by_phi, stars_by_phi


def evaluate_prepared_model(
    base_path,
    model,
    rho0,
    rs,
    phalo,
    qhalo,
    alpha_halo,
    beta_halo,
    gamma,
    prepared: PreparedFixedWeightData,
    *,
    plot=False,
) -> ModelEvaluation:
    """Build and score one trial potential using precomputed fixed weights.

    ``rho0`` and ``rs`` retain the historical log10 parameterization.
    """

    config = prepared.config
    base = Path(base_path).expanduser().resolve()
    potential = _build_potential(
        rho0,
        rs,
        phalo,
        qhalo,
        gamma,
        alpha_halo,
        beta_halo,
    )
    library = integrate_agama_orbits(
        prepared.initial_conditions,
        potential,
        periods=10,
        samples_per_orbit=config.orbit_samples_per_orbit,
    )
    orbit_weights = prepared.representative_weights.weights[library.seed_index]
    model_density = orbit_density(
        library.x,
        library.y,
        library.z,
        orbit_weights,
        config.density_grid,
        sample_divisor=config.orbit_sample_divisor,
    )
    density = compare_density(
        prepared.target_density,
        prepared.target_error,
        model_density,
        config.density_grid,
        config.density_fit,
    )

    velocity_loglike: Mapping[str, float] = {}
    velocity_by_phi: Mapping[str, np.ndarray] = {}
    velocity_stars: Mapping[str, np.ndarray] = {}
    if config.include_velocity:
        velocity_loglike, velocity_by_phi, velocity_stars = _score_velocities(
            prepared.catalog,
            library,
            orbit_weights,
            config,
        )

    result = ModelEvaluation(
        density=density,
        velocity_loglike=velocity_loglike,
        velocity_loglike_by_phi=velocity_by_phi,
        velocity_stars_by_phi=velocity_stars,
        successful_orbits=library.successful_seed_index.size,
        representative_weights=prepared.representative_weights,
    )
    if plot:
        tag = (
            f"rho0{rho0:.3f}_rs{rs:.3f}_p{phalo:.3f}_q{qhalo:.3f}"
            f"_gamma{gamma:.3f}"
        )
        plot_density_comparison(
            density,
            base / model / "density_rzphi" / f"{tag}.pdf",
        )
    return result


def evaluate_one_model(
    base_path,
    model,
    rho0,
    rs,
    phalo,
    qhalo,
    alpha_halo,
    beta_halo,
    gamma,
    dtfile,
    *,
    observed_density_file=None,
    comparison_config: ZhuComparisonConfig | None = None,
    plot=False,
) -> ModelEvaluation:
    """Compatibility wrapper that prepares fixed weights and evaluates a model."""

    prepared = prepare_fixed_weight_data(
        base_path,
        dtfile,
        observed_density_file=observed_density_file,
        comparison_config=comparison_config,
    )
    return evaluate_prepared_model(
        base_path,
        model,
        rho0,
        rs,
        phalo,
        qhalo,
        alpha_halo,
        beta_halo,
        gamma,
        prepared,
        plot=plot,
    )


def int_one_model(
    base_path,
    model,
    rho0,
    rs,
    phalo,
    qhalo,
    alpha_halo,
    beta_halo,
    gamma,
    dtfile,
    *,
    observed_density_file=None,
    comparison_config: ZhuComparisonConfig | None = None,
    plot=False,
):
    """Compatibility entry point returning the total natural log-likelihood."""

    return evaluate_one_model(
        base_path,
        model,
        rho0,
        rs,
        phalo,
        qhalo,
        alpha_halo,
        beta_halo,
        gamma,
        dtfile,
        observed_density_file=observed_density_file,
        comparison_config=comparison_config,
        plot=plot,
    ).log_likelihood


def loglike_cal(
    rs_ori,
    theta_ori,
    phi_ori,
    v_ori,
    verr_ori,
    vr_m,
    nr,
    nr_s,
    ntheta,
    ntheta_s,
    nv,
    rbd,
    tbd,
    pbd,
    vbd,
):
    """Compatibility wrapper for the former per-star velocity likelihood."""

    from halo_mw_lmc.velocity import SphericalVelocityGrid

    grid = SphericalVelocityGrid(
        radius_edges=np.asarray(rbd[nr_s : nr + 1], dtype=float),
        theta_edges=np.deg2rad(np.asarray(tbd[ntheta_s : ntheta + 1], dtype=float)),
        phi_edges=np.deg2rad(np.asarray(pbd, dtype=float)),
        velocity_edges=np.asarray(vbd, dtype=float),
    )
    model = np.asarray(vr_m, dtype=float)[nr_s:nr, ntheta_s:ntheta, :, :]
    return velocity_log_likelihood(
        rs_ori,
        theta_ori,
        phi_ori,
        v_ori,
        verr_ori,
        model,
        grid,
    )[0]
