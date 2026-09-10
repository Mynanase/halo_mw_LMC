"""Pure model-evaluation workflow with no file, plotting, or CLI concerns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ..core.density import (
    DensityComparison,
    DensityShellDiagnostics,
    compare_density,
    density_shell_diagnostics,
    orbit_density,
)
from ..core.orbit_response import (
    OrbitDensityResponse,
    OrbitSupportAudit,
    build_orbit_density_response,
)
from ..core.orbits import OrbitLibrary, integrate_agama_orbits
from ..core.phase_space import (
    SphericalPhaseSpace,
    cartesian_to_spherical_phase_space,
)
from ..core.potentials import ZhuHaloParameters, build_potential_from_parameters
from ..core.velocity import (
    VelocityDistributionComparison,
    conditional_velocity_histogram,
    velocity_log_likelihood,
)
from ..core.weight_solver import WeightSolution, solve_density_weights
from .preparation import PreparedModelData


INVALID_TRIAL_PENALTY = 1e30


@dataclass(frozen=True)
class ModelEvaluation:
    """Numerical products returned by one trial potential."""

    density: DensityComparison
    velocity_loglike: Mapping[str, float]
    velocity_loglike_by_phi: Mapping[str, np.ndarray]
    velocity_stars_by_phi: Mapping[str, np.ndarray]
    velocity_distributions: Mapping[str, VelocityDistributionComparison]
    successful_orbits: int
    weight_mode: str
    weight_solution: WeightSolution
    objective_mode: str
    density_max_chi2_per_bin: float | None
    density_shells: DensityShellDiagnostics | None = None
    density_shell_phi_max_chi2_per_bin: float | None = None
    orbit_support_audit: OrbitSupportAudit | None = None

    @property
    def log_likelihood(self) -> float:
        return -0.5 * self.density.chi2 + float(sum(self.velocity_loglike.values()))

    @property
    def density_chi2_per_bin(self) -> float:
        count = int(np.count_nonzero(self.density.fit_mask))
        return self.density.chi2 / count if count else np.inf

    @property
    def seed_orbits(self) -> int:
        """Number of catalogue seeds presented to the orbit integrator."""

        return int(self.weight_solution.seed_weights.size)

    @property
    def failed_orbits(self) -> int:
        """Seeds without a finite integrated orbit in this trial."""

        return self.seed_orbits - self.successful_orbits

    @property
    def weight_sum(self) -> float:
        """Total trial weight, including zero slots for failed seeds."""

        return float(np.sum(self.weight_solution.seed_weights))

    @property
    def velocity_negative_log_likelihood(self) -> float:
        return -float(sum(self.velocity_loglike.values()))

    @property
    def objective_velocity(self) -> float:
        return self.velocity_negative_log_likelihood

    @property
    def objective_density_velocity(self) -> float:
        return 0.5 * self.density.chi2 + self.velocity_negative_log_likelihood

    @property
    def density_shell_phi_gate_passed(self) -> bool:
        if self.density_shells is None:
            return True
        limit = self.density_shell_phi_max_chi2_per_bin
        if limit is None:
            return False
        counts = self.density_shells.valid_bins_by_shell_phi
        values = self.density_shells.chi2_per_bin_by_shell_phi
        return bool(
            np.all(counts > 0)
            and np.all(np.isfinite(values))
            and np.all(values <= float(limit))
        )

    @property
    def density_worst_shell_phi_chi2_per_bin(self) -> float:
        if self.density_shells is None:
            return np.nan
        values = self.density_shells.chi2_per_bin_by_shell_phi
        return float(np.max(values)) if values.size else np.inf

    @property
    def density_worst_shell_phi_index(self) -> tuple[int, int] | None:
        if self.density_shells is None:
            return None
        values = self.density_shells.chi2_per_bin_by_shell_phi
        if values.size == 0:
            return None
        return tuple(
            int(value)
            for value in np.unravel_index(np.argmax(values), values.shape)
        )

    @property
    def density_gate_passed(self) -> bool:
        limit = self.density_max_chi2_per_bin
        return bool(
            limit is not None
            and np.isfinite(self.density_chi2_per_bin)
            and self.density_chi2_per_bin <= float(limit)
            and self.density_shell_phi_gate_passed
        )

    @property
    def selected_objective(self) -> float:
        fallback = (
            self.density_chi2_per_bin
            if np.isfinite(self.density_chi2_per_bin)
            else 0.0
        )
        if not self.weight_solution.converged:
            return INVALID_TRIAL_PENALTY + fallback
        if self.objective_mode == "velocity_only":
            if not self.density_gate_passed:
                worst = self.density_worst_shell_phi_chi2_per_bin
                violation = max(fallback, worst if np.isfinite(worst) else 0.0)
                return INVALID_TRIAL_PENALTY + violation
            return self.objective_velocity
        return self.objective_density_velocity


def _orbit_support_audit(
    response: OrbitDensityResponse,
    library: OrbitLibrary,
    density: DensityComparison,
    seed_weights: np.ndarray,
    prepared: PreparedModelData,
    model_phase: SphericalPhaseSpace,
) -> OrbitSupportAudit:
    """Measure velocity-supported orbits that have no fitted density response."""

    row_mask = density.fit_mask.reshape(-1)
    density_response = np.asarray(
        response.matrix[row_mask].sum(axis=0) > 0,
        dtype=bool,
    ).ravel()
    density_seed = np.zeros(response.seed_count, dtype=bool)
    density_seed[response.successful_seed_index] = density_response

    grid = prepared.config.velocity_grid
    wrapped_phi = grid.wrap_phi(model_phase.phi)
    velocity_sample = (
        np.isfinite(model_phase.radius)
        & np.isfinite(model_phase.theta)
        & np.isfinite(wrapped_phi)
        & (model_phase.radius >= prepared.config.velocity_fit_min_radius)
        & (model_phase.radius < grid.radius_edges[-1])
        & (model_phase.theta >= grid.theta_edges[0])
        & (model_phase.theta < grid.theta_edges[-1])
        & (wrapped_phi >= grid.phi_edges[0])
        & (wrapped_phi < grid.phi_edges[-1])
    )
    velocity_seed = np.zeros(response.seed_count, dtype=bool)
    velocity_seed[np.unique(library.seed_index[velocity_sample])] = True
    unsupported_seed = velocity_seed & ~density_seed
    unsupported_sample = velocity_sample & unsupported_seed[library.seed_index]
    solved_seed_weights = np.asarray(seed_weights, dtype=float)
    return OrbitSupportAudit(
        density_supported_orbit_count=int(np.count_nonzero(density_seed)),
        velocity_supported_orbit_count=int(np.count_nonzero(velocity_seed)),
        zero_density_response_velocity_orbit_count=int(
            np.count_nonzero(unsupported_seed)
        ),
        zero_density_response_velocity_sample_count=int(
            np.count_nonzero(unsupported_sample)
        ),
        zero_density_response_velocity_weight_sum=float(
            np.sum(solved_seed_weights[unsupported_seed])
        ),
    )


def _score_velocities(
    prepared: PreparedModelData,
    library: OrbitLibrary,
    orbit_weights: np.ndarray,
    *,
    model_phase: SphericalPhaseSpace | None = None,
):
    config = prepared.config
    catalogue_phase = prepared.catalog_phase_space
    if model_phase is None:
        model_phase = cartesian_to_spherical_phase_space(
            library.x,
            library.y,
            library.z,
            library.vx,
            library.vy,
            library.vz,
        )
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

    total: dict[str, float] = {}
    by_phi: dict[str, np.ndarray] = {}
    stars_by_phi: dict[str, np.ndarray] = {}
    distributions: dict[str, VelocityDistributionComparison] = {}
    for name in ("vr", "vphi", "vtheta"):
        observed = prepared.observed_velocity_histograms[name]
        model_probability, model_occupancy = conditional_velocity_histogram(
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
            prepared.catalogue.velocity_errors[name],
            model_probability,
            config.velocity_grid,
            probability_floor=config.velocity_probability_floor,
            minimum_radius=config.velocity_fit_min_radius,
        )
        total[name] = loglike
        by_phi[name] = component_by_phi
        stars_by_phi[name] = used_by_phi
        distributions[name] = VelocityDistributionComparison(
            component=name,
            grid=config.velocity_grid,
            data_probability=observed.probability,
            data_uncertainty=observed.uncertainty,
            data_occupancy=observed.occupancy,
            model_probability=model_probability,
            model_occupancy=model_occupancy,
        )
    return total, by_phi, stars_by_phi, distributions


def _require_external_response_matches_library(
    response: OrbitDensityResponse,
    library: OrbitLibrary,
    prepared: PreparedModelData,
) -> None:
    """Reject a caller-supplied response that does not describe this library.

    The solver-budget experiment reuses one frozen orbit library across solver
    methods, so a supplied response must come from exactly that library: same
    density grid, same successful-seed column mapping, and the same finite
    sample counts per seed.
    """

    grid = prepared.config.density_grid
    if response.grid.shape != grid.shape:
        raise ValueError(
            f"external response grid shape {response.grid.shape} does not match "
            f"the configured density grid {grid.shape}"
        )
    for name in ("r_edges", "z_edges", "phi_edges"):
        if not np.allclose(getattr(response.grid, name), getattr(grid, name)):
            raise ValueError(
                f"external response grid {name} does not match the configured "
                "density grid"
            )
    seed_count = int(prepared.initial_conditions.shape[0])
    if response.seed_count != seed_count:
        raise ValueError(
            f"external response seed_count {response.seed_count} does not match "
            f"the prepared catalogue size {seed_count}"
        )
    seed_index = np.asarray(library.seed_index, dtype=np.int64)
    successful = np.unique(seed_index)
    if not np.array_equal(
        np.asarray(response.successful_seed_index, dtype=np.int64), successful
    ):
        raise ValueError(
            "external response column mapping does not match the orbit library "
            "successful seeds"
        )
    raw_counts = np.asarray(response.sample_count, dtype=float)
    if raw_counts.ndim != 1 or not np.all(np.isfinite(raw_counts)):
        raise ValueError("external response sample counts must be finite")
    if not np.all(raw_counts == np.floor(raw_counts)) or np.any(raw_counts < 0.0):
        raise ValueError("external response sample counts must be non-negative integers")
    sample_count = raw_counts.astype(np.int64)
    expected_counts = np.bincount(seed_index, minlength=seed_count)[successful]
    if not np.array_equal(sample_count, expected_counts):
        raise ValueError(
            "external response sample counts do not match the orbit library "
            "provenance"
        )
    matrix = response.matrix
    if getattr(matrix, "shape", None) != (
        int(np.prod(grid.shape)),
        successful.size,
    ):
        raise ValueError(
            "external response matrix shape does not match the density grid and "
            "successful seeds"
        )
    if not np.all(np.isfinite(matrix.data)):
        raise ValueError("external response matrix contains non-finite entries")


def evaluate_orbit_library(
    library: OrbitLibrary,
    prepared: PreparedModelData,
    *,
    response: OrbitDensityResponse | None = None,
) -> ModelEvaluation:
    """Score one already-integrated orbit library without any external effects.

    ``response`` lets a caller reuse a frozen density response for this exact
    library instead of rebuilding it; the entry validates grid, column mapping,
    finite sample counts, and provenance.  Everything after integration is
    shared with :func:`evaluate_prepared_model`.
    """

    config = prepared.config
    if config.weight_model.mode == "density_solved":
        if response is None:
            response = build_orbit_density_response(
                library,
                config.density_grid,
                seed_count=prepared.initial_conditions.shape[0],
            )
        else:
            _require_external_response_matches_library(response, library, prepared)
        weight_solution = solve_density_weights(
            response,
            prepared.target_density,
            prepared.target_error,
            config.density_fit,
            config.weight_model,
        )
        model_density = weight_solution.model_density
        density_target = weight_solution.target_density
        density_error = weight_solution.target_error
        orbit_weights = response.sample_weights(
            weight_solution.seed_weights,
            library,
        )
    else:
        if response is not None:
            raise ValueError(
                "catalogue_fixed scoring does not use an orbit density response"
            )
        fixed_weights = prepared.seed_weights
        orbit_weights = fixed_weights[library.seed_index]
        model_density = orbit_density(
            library.x,
            library.y,
            library.z,
            orbit_weights,
            config.density_grid,
            sample_divisor=config.orbit_sample_divisor,
        )
        density_target = prepared.target_density
        density_error = prepared.target_error
        total_weight = float(np.sum(fixed_weights))
        weight_square_sum = float(np.dot(fixed_weights, fixed_weights))
        weight_solution = WeightSolution(
            seed_weights=fixed_weights.copy(),
            model_density=model_density,
            target_density=density_target.copy(),
            target_error=density_error.copy(),
            inner_objective=0.0,
            regularization_penalty=0.0,
            effective_orbit_count=(
                total_weight**2 / weight_square_sum
                if weight_square_sum > 0
                else 0.0
            ),
            maximum_weight_fraction=(
                float(np.max(fixed_weights)) / total_weight
                if total_weight > 0
                else 0.0
            ),
            active_orbit_count=int(np.count_nonzero(fixed_weights > 0)),
            converged=True,
            status=0,
            message="catalogue-fixed weights",
            solver_backend="catalogue_fixed",
            kkt_residual=0.0,
        )
    density = compare_density(
        density_target,
        density_error,
        model_density,
        config.density_grid,
        config.density_fit,
    )
    shell_diagnostics = None
    if config.objective.density_shell_edges is not None:
        shell_diagnostics = density_shell_diagnostics(
            density,
            config.objective.density_shell_edges,
        )

    model_phase = None
    if config.include_velocity:
        model_phase = cartesian_to_spherical_phase_space(
            library.x,
            library.y,
            library.z,
            library.vx,
            library.vy,
            library.vz,
        )

    support_audit = None
    if config.weight_model.mode == "density_solved":
        if model_phase is None:
            raise ValueError("density-solved support audit requires velocity phase space")
        support_audit = _orbit_support_audit(
            response,
            library,
            density,
            weight_solution.seed_weights,
            prepared,
            model_phase,
        )

    velocity_loglike: Mapping[str, float] = {}
    velocity_by_phi: Mapping[str, np.ndarray] = {}
    velocity_stars: Mapping[str, np.ndarray] = {}
    velocity_distributions: Mapping[str, VelocityDistributionComparison] = {}
    if config.include_velocity:
        (
            velocity_loglike,
            velocity_by_phi,
            velocity_stars,
            velocity_distributions,
        ) = _score_velocities(
            prepared,
            library,
            orbit_weights,
            model_phase=model_phase,
        )

    return ModelEvaluation(
        density=density,
        velocity_loglike=velocity_loglike,
        velocity_loglike_by_phi=velocity_by_phi,
        velocity_stars_by_phi=velocity_stars,
        velocity_distributions=velocity_distributions,
        successful_orbits=library.successful_seed_index.size,
        weight_mode=config.weight_model.mode,
        weight_solution=weight_solution,
        objective_mode=config.objective.mode,
        density_max_chi2_per_bin=config.objective.density_max_chi2_per_bin,
        density_shells=shell_diagnostics,
        density_shell_phi_max_chi2_per_bin=(
            config.objective.density_shell_phi_max_chi2_per_bin
        ),
        orbit_support_audit=support_audit,
    )


def evaluate_prepared_model(
    parameters: ZhuHaloParameters,
    prepared: PreparedModelData,
) -> ModelEvaluation:
    """Build, integrate, and score one trial without any external side effects."""

    config = prepared.config
    potential = build_potential_from_parameters(parameters)
    library = integrate_agama_orbits(
        prepared.initial_conditions,
        potential,
        periods=config.orbit_periods,
        samples_per_orbit=config.orbit_samples_per_orbit,
    )
    return evaluate_orbit_library(library, prepared)
