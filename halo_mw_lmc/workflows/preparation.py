"""Translate survey files into immutable arrays shared by every trial."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np

from ..core.config import ZhuComparisonConfig
from ..core.phase_space import (
    SphericalPhaseSpace,
    cartesian_to_spherical_phase_space,
)
from ..core.velocity import (
    VelocityHistogramSummary,
    conditional_velocity_histogram,
    multinomial_histogram_uncertainty,
)
from ..data.catalogue import SeedCatalogue, read_seed_catalogue
from ..data.density_target import read_target_density


@dataclass(frozen=True)
class PreparedModelData:
    """Core-ready catalogue arrays and fixed observations for all trials."""

    catalogue: SeedCatalogue
    target_density: np.ndarray
    target_error: np.ndarray
    config: ZhuComparisonConfig
    catalog_path: Path
    density_path: Path
    catalog_phase_space: SphericalPhaseSpace
    observed_velocity_histograms: Mapping[
        str,
        VelocityHistogramSummary,
    ] = field(default_factory=dict)

    @property
    def initial_conditions(self) -> np.ndarray:
        return self.catalogue.initial_conditions

    @property
    def seed_weights(self) -> np.ndarray:
        if self.catalogue.seed_weights is None:
            raise ValueError("this model configuration has no fixed catalogue weights")
        return self.catalogue.seed_weights


def _observed_velocity_histograms(
    phase_space: SphericalPhaseSpace,
    config: ZhuComparisonConfig,
) -> dict[str, VelocityHistogramSummary]:
    velocities = {
        "vr": phase_space.radial_velocity,
        "vphi": phase_space.azimuthal_velocity,
        "vtheta": phase_space.polar_velocity,
    }
    result: dict[str, VelocityHistogramSummary] = {}
    for name, values in velocities.items():
        probability, occupancy = conditional_velocity_histogram(
            phase_space.radius,
            phase_space.theta,
            phase_space.phi,
            values,
            config.velocity_grid,
        )
        result[name] = VelocityHistogramSummary(
            probability=probability,
            uncertainty=multinomial_histogram_uncertainty(
                probability,
                occupancy,
            ),
            occupancy=occupancy,
        )
    return result


def prepare_model_data(
    catalog_path: str | Path,
    density_path: str | Path,
    comparison_config: ZhuComparisonConfig,
) -> PreparedModelData:
    """Read inputs once, before any trial potential is evaluated."""

    catalog_source = Path(catalog_path).expanduser().resolve()
    density_source = Path(density_path).expanduser().resolve()
    catalogue = read_seed_catalogue(
        catalog_source,
        include_velocity=comparison_config.include_velocity,
        require_weights=(
            comparison_config.weight_model.mode == "catalogue_fixed"
        ),
    )
    phase_space = cartesian_to_spherical_phase_space(
        *[catalogue.initial_conditions[:, index] for index in range(6)]
    )
    observed_histograms = (
        _observed_velocity_histograms(phase_space, comparison_config)
        if comparison_config.include_velocity
        else {}
    )
    target_density, target_error = read_target_density(
        density_source,
        comparison_config.density_grid,
    )
    return PreparedModelData(
        catalogue=catalogue,
        target_density=target_density,
        target_error=target_error,
        config=comparison_config,
        catalog_path=catalog_source,
        density_path=density_source,
        catalog_phase_space=phase_space,
        observed_velocity_histograms=observed_histograms,
    )


# Compatibility aliases for callers that explicitly exercise the fixed mode.
PreparedFixedWeightData = PreparedModelData


def prepare_fixed_weight_data(
    catalog_path: str | Path,
    density_path: str | Path,
    comparison_config: ZhuComparisonConfig,
) -> PreparedModelData:
    if comparison_config.weight_model.mode != "catalogue_fixed":
        raise ValueError("prepare_fixed_weight_data requires catalogue_fixed mode")
    return prepare_model_data(catalog_path, density_path, comparison_config)
