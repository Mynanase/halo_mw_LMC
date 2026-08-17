"""Stable array-only numerical core for orbit-superposition halo models."""

from .config import (
    DensityFitSettings,
    ObjectiveSettings,
    WeightModelSettings,
    ZhuComparisonConfig,
)
from .density import DensityComparison, compare_density, density_fit_mask, orbit_density
from .grids import CylindricalGrid
from .orbit_response import OrbitDensityResponse, build_orbit_density_response
from .tracer_density import (
    DESI_YEAR1_KGIANTS_DENSITY,
    DesiKGiantsDensityModel,
    cell_average_cylindrical_density,
)
from .weight_solver import WeightSolution, solve_density_weights
from .weights import catalogue_seed_weights

__all__ = [
    "CylindricalGrid",
    "DensityComparison",
    "DensityFitSettings",
    "DESI_YEAR1_KGIANTS_DENSITY",
    "DesiKGiantsDensityModel",
    "ObjectiveSettings",
    "OrbitDensityResponse",
    "WeightModelSettings",
    "WeightSolution",
    "ZhuComparisonConfig",
    "catalogue_seed_weights",
    "cell_average_cylindrical_density",
    "compare_density",
    "build_orbit_density_response",
    "density_fit_mask",
    "orbit_density",
    "solve_density_weights",
]
