"""Read-only, stage-aware checks and one-pass data preparation."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Literal, Mapping

from .coverage import DataCoverage, build_data_coverage
from .weights import catalogue_weight_audit
from .catalogue import read_phase_space_catalogue


PreflightStage = Literal["run", "optimize", "evaluate", "coverage"]


@dataclass(frozen=True)
class PreflightCheck:
    """One machine-readable preflight observation."""

    name: str
    status: Literal["pass", "warning", "fail"]
    detail: str

    def document(self) -> dict[str, str]:
        return {"name": self.name, "status": self.status, "detail": self.detail}


@dataclass(frozen=True)
class PreparedExecution:
    """Prepared numerical inputs that may be handed directly to a trial loop."""

    configuration: dict
    model: PreparedModelData
    weight_audit: Mapping[str, object] | None


@dataclass(frozen=True)
class PreparedCoverage:
    """Catalogue-only coverage inputs, with no target-density read."""

    configuration: dict
    coverage: DataCoverage


@dataclass(frozen=True)
class PreflightResult:
    """Checks plus an optional in-memory payload for immediate execution."""

    stage: PreflightStage
    numerical_stage: Literal["optimize", "evaluate"] | None
    checks: tuple[PreflightCheck, ...]
    execution: PreparedExecution | None = None
    coverage: PreparedCoverage | None = None

    @property
    def ok(self) -> bool:
        return not any(check.status == "fail" for check in self.checks)

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(
            check.detail for check in self.checks if check.status == "warning"
        )

    def document(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "numerical_stage": self.numerical_stage,
            "ok": self.ok,
            "checks": [check.document() for check in self.checks],
            "warnings": list(self.warnings),
        }


class PreflightError(ValueError):
    """Raised when a read-only execution prerequisite is not satisfied."""


def _dependency_check(name: str, *, required: bool) -> PreflightCheck:
    available = importlib.util.find_spec(name) is not None
    if available:
        return PreflightCheck(name, "pass", f"Python dependency {name} is available")
    if required:
        return PreflightCheck(name, "fail", f"required Python dependency {name} is missing")
    return PreflightCheck(
        name,
        "warning",
        f"optional Python dependency {name} is missing; NumPy parser fallback will be used",
    )


def preflight_and_prepare(
    configuration: dict,
    *,
    stage: PreflightStage = "run",
) -> PreflightResult:
    """Check one stage without writing files, and read each needed input once."""

    if stage not in {"run", "optimize", "evaluate", "coverage"}:
        raise ValueError(f"unsupported preflight stage: {stage}")
    fixed = configuration["optimizer"]["fixed_points"] is not None
    if stage == "evaluate" and not fixed:
        raise PreflightError("evaluate requires optimizer.fixed_points")
    if stage == "optimize" and fixed:
        raise PreflightError("optimize accepts adaptive configurations only")
    if stage == "coverage":
        numerical_stage = None
    elif stage == "run":
        numerical_stage = "evaluate" if fixed else "optimize"
    else:
        numerical_stage = stage
    checks: list[PreflightCheck] = []

    output = configuration["coverage"]["output_dir"] if stage == "coverage" else configuration["run"]["output_dir"]
    checks.append(PreflightCheck(
        "output_directory", "fail" if output.exists() else "pass",
        f"output directory {'already exists' if output.exists() else 'is available'}: {output}",
    ))
    checks.append(_dependency_check("astropy", required=False))
    checks.append(
        _dependency_check(
            "matplotlib",
            required=stage in {"run", "coverage"},
        )
    )
    if numerical_stage is not None:
        checks.append(_dependency_check("agama", required=True))
        if numerical_stage == "optimize":
            checks.append(_dependency_check("skopt", required=True))
        if configuration["recipe"]["weight_model"]["mode"] == "density_solved":
            checks.append(_dependency_check("scipy", required=True))

    failed_dependencies = any(check.status == "fail" for check in checks)
    if stage == "coverage":
        if not configuration["data"]["catalog"].exists():
            checks.append(
                PreflightCheck(
                    "catalogue",
                    "fail",
                    f"catalogue not found: {configuration['data']['catalog']}",
                )
            )
            return PreflightResult(stage, numerical_stage, tuple(checks))
        try:
            initial = read_phase_space_catalogue(configuration["data"]["catalog"])
            comparison = resolve_model(configuration["recipe"])
            coverage = build_data_coverage(
                initial,
                rzphi_grid=comparison["density_grid"],
                spherical_radius_edges=comparison["velocity_grid"].radius_edges,
                theta_edges=comparison["velocity_grid"].theta_edges,
            )
        except (OSError, TypeError, ValueError) as exc:
            checks.append(PreflightCheck("catalogue", "fail", str(exc)))
            return PreflightResult(stage, numerical_stage, tuple(checks))
        checks.append(
            PreflightCheck(
                "catalogue",
                "pass",
                f"read {coverage.input_rows} catalogue rows",
            )
        )
        payload = PreparedCoverage(configuration, coverage)
        return PreflightResult(
            stage,
            numerical_stage,
            tuple(checks),
            coverage=payload if not failed_dependencies else None,
        )

    missing = [("catalogue", configuration["data"]["catalog"]), ("target_density", configuration["data"]["target_density"])]
    for name, path in missing:
        checks.append(
            PreflightCheck(
                name,
                "pass" if path.exists() else "fail",
                f"{name} {'found' if path.exists() else 'not found'}: {path}",
            )
        )
    if any(not path.exists() for _, path in missing):
        return PreflightResult(stage, numerical_stage, tuple(checks))

    try:
        comparison = resolve_model(configuration["recipe"])
        prepared = prepare_model_data(configuration["data"]["catalog"], configuration["data"]["target_density"], comparison)
        audit = None
        if comparison["weight_model"]["mode"] == "catalogue_fixed":
            audit = catalogue_weight_audit(
                prepared.initial_conditions,
                prepared.seed_weights,
                comparison["density_grid"],
            )
    except (OSError, TypeError, ValueError) as exc:
        checks.append(PreflightCheck("prepared_model", "fail", str(exc)))
        return PreflightResult(stage, numerical_stage, tuple(checks))

    checks.append(
        PreflightCheck(
            "prepared_model",
            "pass",
            (
                f"prepared {prepared.initial_conditions.shape[0]} orbits and "
                f"target grid {list(prepared.target_density.shape)}"
            ),
        )
    )
    payload = PreparedExecution(configuration, prepared, audit)
    return PreflightResult(
        stage,
        numerical_stage,
        tuple(checks),
        execution=payload if not any(c.status == "fail" for c in checks) else None,
    )


def require_preflight(result: PreflightResult) -> PreflightResult:
    """Return a successful result or raise one concise lifecycle error."""

    if result.ok:
        return result
    failures = [check.detail for check in result.checks if check.status == "fail"]
    raise PreflightError("preflight failed: " + "; ".join(failures))


from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np

from .config import resolve_model
from .orbits import (
    SphericalPhaseSpace,
    cartesian_to_spherical_phase_space,
)
from .velocity import (
    VelocityHistogramSummary,
    conditional_velocity_histogram,
    multinomial_histogram_uncertainty,
)
from .catalogue import SeedCatalogue, read_seed_catalogue
from .catalogue import read_target_density


@dataclass(frozen=True)
class PreparedModelData:
    """Core-ready catalogue arrays and fixed observations for all trials."""

    catalogue: SeedCatalogue
    target_density: np.ndarray
    target_error: np.ndarray
    config: dict
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


def prepare_model_data(
    catalog_path: str | Path,
    density_path: str | Path,
    comparison_config: dict,
) -> PreparedModelData:
    """Read inputs once, before any trial potential is evaluated."""

    catalog_source = Path(catalog_path).expanduser().resolve()
    density_source = Path(density_path).expanduser().resolve()
    catalogue = read_seed_catalogue(catalog_source, include_velocity=comparison_config["include_velocity"], require_weights=comparison_config["weight_model"]["mode"] == "catalogue_fixed")
    initial = catalogue.initial_conditions
    phase_space = cartesian_to_spherical_phase_space(initial[:, 0], initial[:, 1], initial[:, 2], initial[:, 3], initial[:, 4], initial[:, 5])

    observed_histograms: dict[str, VelocityHistogramSummary] = {}
    if comparison_config["include_velocity"]:
        velocities = {"vr": phase_space.radial_velocity, "vphi": phase_space.azimuthal_velocity, "vtheta": phase_space.polar_velocity}
        for name, values in velocities.items():
            probability, occupancy = conditional_velocity_histogram(phase_space.radius, phase_space.theta, phase_space.phi, values, comparison_config["velocity_grid"])
            uncertainty = multinomial_histogram_uncertainty(probability, occupancy)
            observed_histograms[name] = VelocityHistogramSummary(probability=probability, uncertainty=uncertainty, occupancy=occupancy)

    target_density, target_error = read_target_density(density_source, comparison_config["density_grid"])
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
    comparison_config: dict,
) -> PreparedModelData:
    if comparison_config["weight_model"]["mode"] != "catalogue_fixed":
        raise ValueError("prepare_fixed_weight_data requires catalogue_fixed mode")
    return prepare_model_data(catalog_path, density_path, comparison_config)
