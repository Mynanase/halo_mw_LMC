"""Strict TOML configuration for Zhu orbit-superposition model runs.

The numerical core receives typed settings and never reads configuration
files.  This module owns the operational boundary: it loads one run file and
its referenced reusable recipe, resolves paths relative to the file that
declares them, and constructs the core :class:`ZhuComparisonConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import tomllib
from typing import Any, Mapping

import numpy as np

from halo_mw_lmc.core.config import (
    DensityFitSettings,
    ObjectiveSettings,
    WeightModelSettings,
    ZhuComparisonConfig,
)
from halo_mw_lmc.core.grids import CylindricalGrid
from halo_mw_lmc.core.potentials import (
    ZHU_2026_BEST_FIT,
    ZHU_2026_POTENTIAL_NAME,
)
from halo_mw_lmc.core.velocity import SphericalVelocityGrid


CONFIGURATION_SCHEMA_VERSION = 1
SYNTHETIC_DENSITY_MODEL_NAMES = ("desi_year1_kgiants_3d",)
SEARCH_PARAMETER_NAMES = (
    "qhalo",
    "phalo",
    "rho0",
    "rho0_plus_2logrs",
    "gamma",
)


class ConfigurationError(ValueError):
    """Raised when a run or recipe TOML document is invalid."""


@dataclass(frozen=True)
class PotentialConfiguration:
    recipe: str


@dataclass(frozen=True)
class DensityGridConfiguration:
    n_r: int
    r_range_kpc: tuple[float, float]
    n_z: int
    z_range_kpc: tuple[float, float]
    n_phi: int
    phi_origin_deg: float

    def build(self) -> CylindricalGrid:
        return CylindricalGrid.uniform(
            n_r=self.n_r,
            r_range=self.r_range_kpc,
            n_z=self.n_z,
            z_range=self.z_range_kpc,
            n_phi=self.n_phi,
            phi_origin=np.deg2rad(self.phi_origin_deg),
        )


@dataclass(frozen=True)
class DensityFitConfiguration:
    min_abs_z_kpc: float
    min_radius_kpc: float
    max_radius_kpc: float
    normalization_min_radius_kpc: float
    require_positive_target: bool
    normalization: str

    def build(self) -> DensityFitSettings:
        return DensityFitSettings(
            min_abs_z=self.min_abs_z_kpc,
            min_spherical_radius=self.min_radius_kpc,
            max_spherical_radius=self.max_radius_kpc,
            normalization_min_radius=self.normalization_min_radius_kpc,
            require_positive_data=self.require_positive_target,
            normalization=self.normalization,
        )


@dataclass(frozen=True)
class VelocityFitConfiguration:
    enabled: bool
    min_radius_kpc: float
    probability_floor: float
    radius_edges_kpc: tuple[float, ...]
    theta_edges_deg: tuple[float, ...]
    velocity_range_km_s: tuple[float, float]
    velocity_bins: int

    def build_grid(self, phi_edges: np.ndarray) -> SphericalVelocityGrid:
        return SphericalVelocityGrid(
            radius_edges=np.asarray(self.radius_edges_kpc, dtype=float),
            theta_edges=np.deg2rad(np.asarray(self.theta_edges_deg, dtype=float)),
            phi_edges=np.asarray(phi_edges, dtype=float),
            velocity_edges=np.linspace(
                self.velocity_range_km_s[0],
                self.velocity_range_km_s[1],
                self.velocity_bins + 1,
            ),
        )


@dataclass(frozen=True)
class OrbitConfiguration:
    periods: float
    samples_per_orbit: int
    sample_weight_divisor: str | None

    @property
    def resolved_sample_weight_divisor(self) -> float:
        if self.sample_weight_divisor == "half_samples":
            return self.samples_per_orbit / 2.0
        if self.sample_weight_divisor is None:
            # Density-solved weights use each orbit's actual finite sample
            # count. This compatibility value is never used in that mode.
            return float(self.samples_per_orbit)
        raise ConfigurationError(
            "unsupported orbit sample_weight_divisor: "
            f"{self.sample_weight_divisor!r}"
        )


@dataclass(frozen=True)
class SearchBounds:
    qhalo: tuple[float, float]
    phalo: tuple[float, float]
    rho0: tuple[float, float]
    rho0_plus_2logrs: tuple[float, float]
    gamma: tuple[float, float]

    def as_dict(self) -> dict[str, tuple[float, float]]:
        return {
            name: getattr(self, name)
            for name in SEARCH_PARAMETER_NAMES
        }


@dataclass(frozen=True)
class SearchConfiguration:
    initial_point: str
    round_decimals: int
    bounds: SearchBounds


@dataclass(frozen=True)
class RecipeConfiguration:
    source_path: Path
    schema_version: int
    name: str
    potential: PotentialConfiguration
    density_grid: DensityGridConfiguration
    density_fit: DensityFitConfiguration
    velocity_fit: VelocityFitConfiguration
    weight_model: WeightModelSettings
    objective: ObjectiveSettings
    orbits: OrbitConfiguration
    search: SearchConfiguration

    @property
    def orbit_periods(self) -> float:
        return self.orbits.periods

    def to_comparison_config(self) -> ZhuComparisonConfig:
        density_grid = self.density_grid.build()
        return ZhuComparisonConfig(
            density_grid=density_grid,
            density_fit=self.density_fit.build(),
            velocity_grid=self.velocity_fit.build_grid(density_grid.phi_edges),
            include_velocity=self.velocity_fit.enabled,
            velocity_fit_min_radius=self.velocity_fit.min_radius_kpc,
            velocity_probability_floor=self.velocity_fit.probability_floor,
            orbit_periods=self.orbits.periods,
            orbit_samples_per_orbit=self.orbits.samples_per_orbit,
            orbit_sample_divisor=self.orbits.resolved_sample_weight_divisor,
            weight_model=self.weight_model,
            objective=self.objective,
        )


@dataclass(frozen=True)
class RunIdentityConfiguration:
    id: str
    output_dir: Path


@dataclass(frozen=True)
class DataConfiguration:
    catalog: Path
    target_density: Path


@dataclass(frozen=True)
class OptimizerConfiguration:
    iterations: int
    random_seed: int


@dataclass(frozen=True)
class ReportConfiguration:
    velocity_bin_factor: int


@dataclass(frozen=True)
class CoverageConfiguration:
    output_dir: Path
    maximum_points: int
    velocity_limit_km_s: float
    random_seed: int


@dataclass(frozen=True)
class RunConfiguration:
    source_path: Path
    schema_version: int
    recipe: RecipeConfiguration
    run: RunIdentityConfiguration
    data: DataConfiguration
    optimizer: OptimizerConfiguration
    report: ReportConfiguration
    coverage: CoverageConfiguration

    @property
    def run_id(self) -> str:
        return self.run.id

    @property
    def output_dir(self) -> Path:
        return self.run.output_dir

    @property
    def orbit_periods(self) -> float:
        return self.recipe.orbit_periods

    @property
    def search_bounds(self) -> dict[str, tuple[float, float]]:
        return self.recipe.search.bounds.as_dict()

    @property
    def round_decimals(self) -> int:
        return self.recipe.search.round_decimals

    @property
    def iterations(self) -> int:
        return self.optimizer.iterations

    @property
    def random_seed(self) -> int:
        return self.optimizer.random_seed

    def to_comparison_config(self) -> ZhuComparisonConfig:
        return self.recipe.to_comparison_config()


@dataclass(frozen=True)
class SyntheticDensityConfiguration:
    """Resolved configuration for one analytic target-density artifact."""

    source_path: Path
    schema_version: int
    recipe: RecipeConfiguration
    model_name: str
    model_source: Path
    quadrature_order: int
    validation_order: int
    fractional_uncertainty: float
    output_path: Path

    @property
    def grid(self) -> CylindricalGrid:
        return self.recipe.density_grid.build()


def _read_toml(path: str | Path, context: str) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    try:
        with source.open("rb") as stream:
            document = tomllib.load(stream)
    except FileNotFoundError as exc:
        raise ConfigurationError(f"{context} file not found: {source}") from exc
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigurationError(f"could not read {context} file {source}: {exc}") from exc
    if not isinstance(document, dict):
        raise ConfigurationError(f"{context} document must be a TOML table")
    return source, document


def _require_exact_fields(
    table: Mapping[str, Any],
    expected: set[str],
    context: str,
) -> None:
    actual = set(table)
    unknown = sorted(actual - expected)
    missing = sorted(expected - actual)
    if unknown:
        raise ConfigurationError(
            f"unknown field(s) in {context}: {', '.join(unknown)}"
        )
    if missing:
        raise ConfigurationError(
            f"missing required field(s) in {context}: {', '.join(missing)}"
        )


def _table(document: Mapping[str, Any], name: str, context: str) -> Mapping[str, Any]:
    value = document[name]
    if not isinstance(value, dict):
        raise ConfigurationError(f"{context}.{name} must be a TOML table")
    return value


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{context} must be a non-empty string")
    return value


def _boolean(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{context} must be a boolean")
    return value


def _integer(value: Any, context: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{context} must be an integer")
    if minimum is not None and value < minimum:
        raise ConfigurationError(f"{context} must be at least {minimum}")
    return value


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{context} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigurationError(f"{context} must be finite")
    return result


def _positive_number(value: Any, context: str) -> float:
    result = _number(value, context)
    if result <= 0:
        raise ConfigurationError(f"{context} must be positive")
    return result


def _pair(value: Any, context: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ConfigurationError(f"{context} must contain exactly two numbers")
    lower = _number(value[0], f"{context}[0]")
    upper = _number(value[1], f"{context}[1]")
    if lower >= upper:
        raise ConfigurationError(f"{context} must be strictly increasing")
    return lower, upper


def _edges(value: Any, context: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) < 2:
        raise ConfigurationError(f"{context} must contain at least two numbers")
    result = tuple(_number(item, f"{context}[{index}]") for index, item in enumerate(value))
    if any(left >= right for left, right in zip(result, result[1:])):
        raise ConfigurationError(f"{context} must be strictly increasing")
    return result


def _resolved_path(value: Any, source: Path, context: str) -> Path:
    raw = Path(_string(value, context)).expanduser()
    return raw.resolve() if raw.is_absolute() else (source.parent / raw).resolve()


def _schema_version(value: Any, context: str) -> int:
    version = _integer(value, context, minimum=1)
    if version != CONFIGURATION_SCHEMA_VERSION:
        raise ConfigurationError(
            f"unsupported {context}: {version}; expected {CONFIGURATION_SCHEMA_VERSION}"
        )
    return version


def load_recipe_configuration(path: str | Path) -> RecipeConfiguration:
    """Load and validate a reusable scientific recipe TOML file."""

    source, document = _read_toml(path, "recipe configuration")
    _require_exact_fields(
        document,
        {
            "schema_version",
            "name",
            "potential",
            "density_grid",
            "density_fit",
            "velocity_fit",
            "weight_model",
            "objective",
            "orbits",
            "search",
        },
        "recipe",
    )

    potential_table = _table(document, "potential", "recipe")
    _require_exact_fields(potential_table, {"recipe"}, "recipe.potential")
    potential_name = _string(
        potential_table["recipe"],
        "recipe.potential.recipe",
    )
    if potential_name != ZHU_2026_POTENTIAL_NAME:
        raise ConfigurationError(
            f"unsupported potential recipe: {potential_name!r}; "
            f"expected {ZHU_2026_POTENTIAL_NAME!r}"
        )
    potential = PotentialConfiguration(recipe=potential_name)

    grid_table = _table(document, "density_grid", "recipe")
    _require_exact_fields(
        grid_table,
        {
            "n_r",
            "r_range_kpc",
            "n_z",
            "z_range_kpc",
            "n_phi",
            "phi_origin_deg",
        },
        "recipe.density_grid",
    )
    r_range = _pair(grid_table["r_range_kpc"], "recipe.density_grid.r_range_kpc")
    if r_range[0] < 0:
        raise ConfigurationError(
            "recipe.density_grid.r_range_kpc cannot include negative radii"
        )
    density_grid = DensityGridConfiguration(
        n_r=_integer(grid_table["n_r"], "recipe.density_grid.n_r", minimum=1),
        r_range_kpc=r_range,
        n_z=_integer(grid_table["n_z"], "recipe.density_grid.n_z", minimum=1),
        z_range_kpc=_pair(
            grid_table["z_range_kpc"], "recipe.density_grid.z_range_kpc"
        ),
        n_phi=_integer(
            grid_table["n_phi"], "recipe.density_grid.n_phi", minimum=1
        ),
        phi_origin_deg=_number(
            grid_table["phi_origin_deg"], "recipe.density_grid.phi_origin_deg"
        ),
    )

    density_fit_table = _table(document, "density_fit", "recipe")
    _require_exact_fields(
        density_fit_table,
        {
            "min_abs_z_kpc",
            "min_radius_kpc",
            "max_radius_kpc",
            "normalization_min_radius_kpc",
            "require_positive_target",
            "normalization",
        },
        "recipe.density_fit",
    )
    min_abs_z = _number(
        density_fit_table["min_abs_z_kpc"], "recipe.density_fit.min_abs_z_kpc"
    )
    min_radius = _number(
        density_fit_table["min_radius_kpc"], "recipe.density_fit.min_radius_kpc"
    )
    max_radius = _number(
        density_fit_table["max_radius_kpc"], "recipe.density_fit.max_radius_kpc"
    )
    normalization_min_radius = _number(
        density_fit_table["normalization_min_radius_kpc"],
        "recipe.density_fit.normalization_min_radius_kpc",
    )
    if min_abs_z < 0 or min_radius < 0 or normalization_min_radius < 0:
        raise ConfigurationError("density-fit radii cannot be negative")
    if min_radius >= max_radius:
        raise ConfigurationError(
            "recipe.density_fit radius interval must be strictly increasing"
        )
    normalization = _string(
        density_fit_table["normalization"], "recipe.density_fit.normalization"
    )
    if normalization not in {"volume", "weighted_least_squares", "none"}:
        raise ConfigurationError(
            "recipe.density_fit.normalization must be 'volume', "
            "'weighted_least_squares', or 'none'"
        )
    density_fit = DensityFitConfiguration(
        min_abs_z_kpc=min_abs_z,
        min_radius_kpc=min_radius,
        max_radius_kpc=max_radius,
        normalization_min_radius_kpc=normalization_min_radius,
        require_positive_target=_boolean(
            density_fit_table["require_positive_target"],
            "recipe.density_fit.require_positive_target",
        ),
        normalization=normalization,
    )

    velocity_table = _table(document, "velocity_fit", "recipe")
    _require_exact_fields(
        velocity_table,
        {
            "enabled",
            "min_radius_kpc",
            "probability_floor",
            "radius_edges_kpc",
            "theta_edges_deg",
            "velocity_range_km_s",
            "velocity_bins",
        },
        "recipe.velocity_fit",
    )
    velocity_min_radius = _number(
        velocity_table["min_radius_kpc"], "recipe.velocity_fit.min_radius_kpc"
    )
    if velocity_min_radius < 0:
        raise ConfigurationError("recipe.velocity_fit.min_radius_kpc cannot be negative")
    radius_edges = _edges(
        velocity_table["radius_edges_kpc"], "recipe.velocity_fit.radius_edges_kpc"
    )
    if radius_edges[0] < 0:
        raise ConfigurationError(
            "recipe.velocity_fit.radius_edges_kpc cannot include negative radii"
        )
    theta_edges = _edges(
        velocity_table["theta_edges_deg"], "recipe.velocity_fit.theta_edges_deg"
    )
    if theta_edges[0] < -90 or theta_edges[-1] > 90:
        raise ConfigurationError(
            "recipe.velocity_fit.theta_edges_deg must lie within [-90, 90]"
        )
    velocity_fit = VelocityFitConfiguration(
        enabled=_boolean(velocity_table["enabled"], "recipe.velocity_fit.enabled"),
        min_radius_kpc=velocity_min_radius,
        probability_floor=_positive_number(
            velocity_table["probability_floor"],
            "recipe.velocity_fit.probability_floor",
        ),
        radius_edges_kpc=radius_edges,
        theta_edges_deg=theta_edges,
        velocity_range_km_s=_pair(
            velocity_table["velocity_range_km_s"],
            "recipe.velocity_fit.velocity_range_km_s",
        ),
        velocity_bins=_integer(
            velocity_table["velocity_bins"],
            "recipe.velocity_fit.velocity_bins",
            minimum=1,
        ),
    )

    weight_table = _table(document, "weight_model", "recipe")
    weight_mode = _string(weight_table.get("mode"), "recipe.weight_model.mode")
    if weight_mode == "catalogue_fixed":
        _require_exact_fields(weight_table, {"mode"}, "recipe.weight_model")
        weight_model = WeightModelSettings(mode=weight_mode)
    elif weight_mode == "density_solved":
        _require_exact_fields(
            weight_table,
            {
                "mode",
                "solver",
                "target_normalization",
                "regularization",
                "regularization_strength",
                "max_iter",
                "lsmr_tol",
            },
            "recipe.weight_model",
        )
        try:
            weight_model = WeightModelSettings(
                mode=weight_mode,
                solver=_string(
                    weight_table["solver"], "recipe.weight_model.solver"
                ),
                target_normalization=_string(
                    weight_table["target_normalization"],
                    "recipe.weight_model.target_normalization",
                ),
                regularization=_string(
                    weight_table["regularization"],
                    "recipe.weight_model.regularization",
                ),
                regularization_strength=_number(
                    weight_table["regularization_strength"],
                    "recipe.weight_model.regularization_strength",
                ),
                max_iter=_integer(
                    weight_table["max_iter"],
                    "recipe.weight_model.max_iter",
                    minimum=1,
                ),
                lsmr_tol=_positive_number(
                    weight_table["lsmr_tol"],
                    "recipe.weight_model.lsmr_tol",
                ),
            )
        except ValueError as exc:
            raise ConfigurationError(str(exc)) from exc
    else:
        raise ConfigurationError(
            "recipe.weight_model.mode must be 'catalogue_fixed' or "
            "'density_solved'"
        )

    objective_table = _table(document, "objective", "recipe")
    objective_mode = _string(objective_table.get("mode"), "recipe.objective.mode")
    if objective_mode == "velocity_only":
        _require_exact_fields(
            objective_table,
            {"mode", "density_max_chi2_per_bin"},
            "recipe.objective",
        )
        limit = _positive_number(
            objective_table["density_max_chi2_per_bin"],
            "recipe.objective.density_max_chi2_per_bin",
        )
    elif objective_mode == "density_velocity":
        _require_exact_fields(objective_table, {"mode"}, "recipe.objective")
        limit = None
    else:
        raise ConfigurationError(
            "recipe.objective.mode must be 'velocity_only' or 'density_velocity'"
        )
    objective = ObjectiveSettings(
        mode=objective_mode,
        density_max_chi2_per_bin=limit,
    )

    if weight_model.mode == "density_solved":
        if density_fit.normalization != "none":
            raise ConfigurationError(
                "density_solved weights require density_fit.normalization='none'"
            )
        if not velocity_fit.enabled:
            raise ConfigurationError(
                "density_solved weights require velocity_fit.enabled=true"
            )

    orbit_table = _table(document, "orbits", "recipe")
    expected_orbit_fields = {"periods", "samples_per_orbit"}
    if weight_model.mode == "catalogue_fixed":
        expected_orbit_fields.add("sample_weight_divisor")
    _require_exact_fields(orbit_table, expected_orbit_fields, "recipe.orbits")
    divisor_policy = None
    if weight_model.mode == "catalogue_fixed":
        divisor_policy = _string(
            orbit_table["sample_weight_divisor"],
            "recipe.orbits.sample_weight_divisor",
        )
        if divisor_policy != "half_samples":
            raise ConfigurationError(
                "recipe.orbits.sample_weight_divisor must be 'half_samples'"
            )
    orbits = OrbitConfiguration(
        periods=_positive_number(orbit_table["periods"], "recipe.orbits.periods"),
        samples_per_orbit=_integer(
            orbit_table["samples_per_orbit"],
            "recipe.orbits.samples_per_orbit",
            minimum=1,
        ),
        sample_weight_divisor=divisor_policy,
    )

    search_table = _table(document, "search", "recipe")
    _require_exact_fields(
        search_table,
        {"initial_point", "round_decimals", "bounds"},
        "recipe.search",
    )
    initial_point = _string(
        search_table["initial_point"], "recipe.search.initial_point"
    )
    if initial_point not in {"paper_best", "optimizer"}:
        raise ConfigurationError(
            "recipe.search.initial_point must be 'paper_best' or 'optimizer'"
        )
    round_decimals = _integer(
        search_table["round_decimals"],
        "recipe.search.round_decimals",
        minimum=0,
    )
    bounds_table = _table(search_table, "bounds", "recipe.search")
    _require_exact_fields(
        bounds_table,
        set(SEARCH_PARAMETER_NAMES),
        "recipe.search.bounds",
    )
    bounds = SearchBounds(
        **{
            name: _pair(bounds_table[name], f"recipe.search.bounds.{name}")
            for name in SEARCH_PARAMETER_NAMES
        }
    )
    if bounds.qhalo[0] <= 0 or bounds.phalo[0] <= 0:
        raise ConfigurationError("qhalo and phalo search bounds must be positive")
    if bounds.gamma[0] < 0 or bounds.gamma[1] >= 3:
        raise ConfigurationError("gamma search bounds must satisfy 0 <= gamma < 3")
    for name, interval in bounds.as_dict().items():
        for endpoint in interval:
            if not math.isclose(
                endpoint,
                round(endpoint, round_decimals),
                rel_tol=0.0,
                abs_tol=10 ** (-(round_decimals + 10)),
            ):
                raise ConfigurationError(
                    f"recipe.search.bounds.{name} endpoints must be representable "
                    f"with round_decimals={round_decimals}"
                )
    if initial_point == "paper_best":
        paper_best = {
            "qhalo": round(ZHU_2026_BEST_FIT["qhalo"], round_decimals),
            "phalo": round(ZHU_2026_BEST_FIT["phalo"], round_decimals),
            "rho0": round(ZHU_2026_BEST_FIT["rho0"], round_decimals),
            "rho0_plus_2logrs": round(
                ZHU_2026_BEST_FIT["rho0"]
                + 2 * ZHU_2026_BEST_FIT["log_rs"],
                round_decimals,
            ),
            "gamma": round(ZHU_2026_BEST_FIT["gamma"], round_decimals),
        }
        outside = [
            name
            for name, (lower, upper) in bounds.as_dict().items()
            if not lower <= paper_best[name] <= upper
        ]
        if outside:
            raise ConfigurationError(
                "recipe.search.initial_point='paper_best' lies outside bounds for: "
                + ", ".join(outside)
            )
    search = SearchConfiguration(
        initial_point=initial_point,
        round_decimals=round_decimals,
        bounds=bounds,
    )

    return RecipeConfiguration(
        source_path=source,
        schema_version=_schema_version(
            document["schema_version"], "recipe.schema_version"
        ),
        name=_string(document["name"], "recipe.name"),
        potential=potential,
        density_grid=density_grid,
        density_fit=density_fit,
        velocity_fit=velocity_fit,
        weight_model=weight_model,
        objective=objective,
        orbits=orbits,
        search=search,
    )


def load_run_configuration(path: str | Path) -> RunConfiguration:
    """Load a run TOML and its referenced recipe as one typed configuration."""

    source, document = _read_toml(path, "run configuration")
    _require_exact_fields(
        document,
        {
            "schema_version",
            "recipe",
            "run",
            "data",
            "optimizer",
            "report",
            "coverage",
        },
        "run configuration",
    )
    recipe_path = _resolved_path(document["recipe"], source, "run configuration.recipe")
    recipe = load_recipe_configuration(recipe_path)

    run_table = _table(document, "run", "run configuration")
    _require_exact_fields(run_table, {"id", "output_dir"}, "run configuration.run")
    run = RunIdentityConfiguration(
        id=_string(run_table["id"], "run configuration.run.id"),
        output_dir=_resolved_path(
            run_table["output_dir"], source, "run configuration.run.output_dir"
        ),
    )

    data_table = _table(document, "data", "run configuration")
    _require_exact_fields(
        data_table, {"catalog", "target_density"}, "run configuration.data"
    )
    data = DataConfiguration(
        catalog=_resolved_path(
            data_table["catalog"], source, "run configuration.data.catalog"
        ),
        target_density=_resolved_path(
            data_table["target_density"],
            source,
            "run configuration.data.target_density",
        ),
    )

    optimizer_table = _table(document, "optimizer", "run configuration")
    _require_exact_fields(
        optimizer_table,
        {"iterations", "random_seed"},
        "run configuration.optimizer",
    )
    optimizer = OptimizerConfiguration(
        iterations=_integer(
            optimizer_table["iterations"],
            "run configuration.optimizer.iterations",
            minimum=1,
        ),
        random_seed=_integer(
            optimizer_table["random_seed"],
            "run configuration.optimizer.random_seed",
            minimum=0,
        ),
    )

    report_table = _table(document, "report", "run configuration")
    _require_exact_fields(
        report_table, {"velocity_bin_factor"}, "run configuration.report"
    )
    report = ReportConfiguration(
        velocity_bin_factor=_integer(
            report_table["velocity_bin_factor"],
            "run configuration.report.velocity_bin_factor",
            minimum=1,
        )
    )

    coverage_table = _table(document, "coverage", "run configuration")
    _require_exact_fields(
        coverage_table,
        {"output_dir", "maximum_points", "velocity_limit_km_s", "random_seed"},
        "run configuration.coverage",
    )
    coverage = CoverageConfiguration(
        output_dir=_resolved_path(
            coverage_table["output_dir"],
            source,
            "run configuration.coverage.output_dir",
        ),
        maximum_points=_integer(
            coverage_table["maximum_points"],
            "run configuration.coverage.maximum_points",
            minimum=1,
        ),
        velocity_limit_km_s=_positive_number(
            coverage_table["velocity_limit_km_s"],
            "run configuration.coverage.velocity_limit_km_s",
        ),
        random_seed=_integer(
            coverage_table["random_seed"],
            "run configuration.coverage.random_seed",
            minimum=0,
        ),
    )

    return RunConfiguration(
        source_path=source,
        schema_version=_schema_version(
            document["schema_version"], "run configuration.schema_version"
        ),
        recipe=recipe,
        run=run,
        data=data,
        optimizer=optimizer,
        report=report,
        coverage=coverage,
    )


def load_synthetic_density_configuration(
    path: str | Path,
) -> SyntheticDensityConfiguration:
    """Load a strict analytic-density generation configuration."""

    source, document = _read_toml(path, "synthetic density configuration")
    _require_exact_fields(
        document,
        {
            "schema_version",
            "recipe",
            "model",
            "quadrature",
            "uncertainty",
            "output",
        },
        "synthetic density configuration",
    )
    recipe_path = _resolved_path(
        document["recipe"],
        source,
        "synthetic density configuration.recipe",
    )
    recipe = load_recipe_configuration(recipe_path)

    model_table = _table(document, "model", "synthetic density configuration")
    _require_exact_fields(
        model_table,
        {"name", "source"},
        "synthetic density configuration.model",
    )
    model_name = _string(
        model_table["name"],
        "synthetic density configuration.model.name",
    )
    if model_name not in SYNTHETIC_DENSITY_MODEL_NAMES:
        raise ConfigurationError(
            f"unsupported synthetic density model: {model_name!r}; expected one of "
            + ", ".join(SYNTHETIC_DENSITY_MODEL_NAMES)
        )

    quadrature_table = _table(
        document,
        "quadrature",
        "synthetic density configuration",
    )
    _require_exact_fields(
        quadrature_table,
        {"order", "validation_order"},
        "synthetic density configuration.quadrature",
    )
    quadrature_order = _integer(
        quadrature_table["order"],
        "synthetic density configuration.quadrature.order",
        minimum=1,
    )
    validation_order = _integer(
        quadrature_table["validation_order"],
        "synthetic density configuration.quadrature.validation_order",
        minimum=1,
    )
    if validation_order <= quadrature_order:
        raise ConfigurationError(
            "synthetic density validation_order must exceed quadrature order"
        )

    uncertainty_table = _table(
        document,
        "uncertainty",
        "synthetic density configuration",
    )
    _require_exact_fields(
        uncertainty_table,
        {"fractional"},
        "synthetic density configuration.uncertainty",
    )
    fractional_uncertainty = _positive_number(
        uncertainty_table["fractional"],
        "synthetic density configuration.uncertainty.fractional",
    )

    output_table = _table(document, "output", "synthetic density configuration")
    _require_exact_fields(
        output_table,
        {"path"},
        "synthetic density configuration.output",
    )
    output_path = _resolved_path(
        output_table["path"],
        source,
        "synthetic density configuration.output.path",
    )
    if output_path.suffix.lower() != ".npz":
        raise ConfigurationError("synthetic density output.path must end in .npz")

    return SyntheticDensityConfiguration(
        source_path=source,
        schema_version=_schema_version(
            document["schema_version"],
            "synthetic density configuration.schema_version",
        ),
        recipe=recipe,
        model_name=model_name,
        model_source=_resolved_path(
            model_table["source"],
            source,
            "synthetic density configuration.model.source",
        ),
        quadrature_order=quadrature_order,
        validation_order=validation_order,
        fractional_uncertainty=fractional_uncertainty,
        output_path=output_path,
    )


__all__ = [
    "CONFIGURATION_SCHEMA_VERSION",
    "ConfigurationError",
    "CoverageConfiguration",
    "DataConfiguration",
    "DensityFitConfiguration",
    "DensityGridConfiguration",
    "OptimizerConfiguration",
    "OrbitConfiguration",
    "PotentialConfiguration",
    "RecipeConfiguration",
    "ReportConfiguration",
    "RunConfiguration",
    "RunIdentityConfiguration",
    "SearchBounds",
    "SearchConfiguration",
    "VelocityFitConfiguration",
    "load_recipe_configuration",
    "load_run_configuration",
]
