"""Tools for Zhu-style empirical orbit-superposition modelling."""

from .core.config import (
    DensityFitSettings,
    ObjectiveSettings,
    WeightModelSettings,
    ZhuComparisonConfig,
)
from .core.density import DensityComparison, compare_density, orbit_density
from .core.grids import CylindricalGrid
from .core.tracer_density import (
    DESI_YEAR1_KGIANTS_DENSITY,
    DesiKGiantsDensityModel,
    cell_average_cylindrical_density,
)
from .core.weights import (
    RepresentativeWeightResult,
    catalogue_seed_weights,
    representative_weights_from_target,
)

__all__ = [
    "CylindricalGrid",
    "DESI_YEAR1_KGIANTS_DENSITY",
    "DesiKGiantsDensityModel",
    "DensityComparison",
    "DensityFitSettings",
    "ObjectiveSettings",
    "RepresentativeWeightResult",
    "WeightModelSettings",
    "ZhuComparisonConfig",
    "catalogue_seed_weights",
    "cell_average_cylindrical_density",
    "compare_density",
    "orbit_density",
    "representative_weights_from_target",
]

__version__ = "0.3.0"
