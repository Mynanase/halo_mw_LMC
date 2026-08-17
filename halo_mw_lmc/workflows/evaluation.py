"""Pure model-evaluation workflow with no file, plotting, or CLI concerns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ..core.density import DensityComparison, compare_density, orbit_density
from ..core.orbit_response import build_orbit_density_response
from ..core.orbits import OrbitLibrary, integrate_agama_orbits
from ..core.phase_space import cartesian_to_spherical_phase_space
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
    def selected_objective(self) -> float:
        if not self.weight_solution.converged:
            return INVALID_TRIAL_PENALTY + self.density_chi2_per_bin
        if self.objective_mode == "velocity_only":
            limit = float(self.density_max_chi2_per_bin)
            if self.density_chi2_per_bin > limit:
                return INVALID_TRIAL_PENALTY + self.density_chi2_per_bin
            return self.objective_velocity
        return self.objective_density_velocity


def _score_velocities(
    prepared: PreparedModelData,
    library: OrbitLibrary,
    orbit_weights: np.ndarray,
):
    config = prepared.config
    catalogue_phase = prepared.catalog_phase_space
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
    if config.weight_model.mode == "density_solved":
        response = build_orbit_density_response(
            library,
            config.density_grid,
            seed_count=prepared.initial_conditions.shape[0],
        )
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
        )
    density = compare_density(
        density_target,
        density_error,
        model_density,
        config.density_grid,
        config.density_fit,
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
        ) = _score_velocities(prepared, library, orbit_weights)

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
    )
