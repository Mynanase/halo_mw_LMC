"""Generate portable target-density artifacts from analytic tracer models."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from ..configuration import SyntheticDensityConfiguration
from ..core.tracer_density import (
    DESI_YEAR1_KGIANTS_DENSITY,
    cell_average_cylindrical_density,
)


SYNTHETIC_DENSITY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SyntheticDensityResult:
    """Summary of one generated target-density artifact."""

    output_path: Path
    grid_shape: tuple[int, int, int]
    maximum_quadrature_relative_difference: float
    median_quadrature_relative_difference: float


def _selected_model(name: str):
    if name == "desi_year1_kgiants_3d":
        return DESI_YEAR1_KGIANTS_DENSITY
    raise ValueError(f"unsupported synthetic density model: {name!r}")


def generate_synthetic_density(
    configuration: SyntheticDensityConfiguration,
) -> SyntheticDensityResult:
    """Volume-average one analytic model and atomically write a target NPZ."""

    if not configuration.model_source.exists():
        raise FileNotFoundError(
            f"density-model source not found: {configuration.model_source}"
        )
    output_path = configuration.output_path
    if output_path.exists():
        raise FileExistsError(
            f"synthetic density output already exists: {output_path}"
        )

    model = _selected_model(configuration.model_name)
    grid = configuration.grid
    lower_order = cell_average_cylindrical_density(
        model,
        grid,
        quadrature_order=configuration.quadrature_order,
    )
    target_density = cell_average_cylindrical_density(
        model,
        grid,
        quadrature_order=configuration.validation_order,
    )
    quadrature_error = np.abs(target_density - lower_order)
    fractional_error = configuration.fractional_uncertainty * target_density
    target_error = np.hypot(fractional_error, quadrature_error)
    if (
        not np.all(np.isfinite(target_density))
        or np.any(target_density <= 0)
        or not np.all(np.isfinite(target_error))
        or np.any(target_error <= 0)
    ):
        raise ValueError(
            "generated density and uncertainty must be finite and positive"
        )

    relative_difference = quadrature_error / target_density
    source_sha256 = hashlib.sha256(
        configuration.model_source.read_bytes()
    ).hexdigest()
    model_parameters = json.dumps(
        model.parameter_document(),
        sort_keys=True,
        separators=(",", ":"),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".npz",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(
            temporary,
            artifact_schema_version=np.asarray(
                SYNTHETIC_DENSITY_SCHEMA_VERSION,
                dtype=np.int64,
            ),
            target_density=target_density,
            target_error=target_error,
            quadrature_error=quadrature_error,
            r_edges=grid.r_edges,
            z_edges=grid.z_edges,
            phi_edges=grid.phi_edges,
            axis_order=np.asarray("R,z,phi"),
            coordinate_system=np.asarray("Galactocentric Cartesian/cylindrical"),
            length_unit=np.asarray("kpc"),
            density_unit=np.asarray("relative tracer density"),
            normalization=np.asarray("source shape; no absolute normalization"),
            uncertainty_semantics=np.asarray(
                "hypot(configured fractional model error, quadrature difference)"
            ),
            fractional_uncertainty=np.asarray(
                configuration.fractional_uncertainty,
                dtype=float,
            ),
            quadrature_order=np.asarray(
                configuration.quadrature_order,
                dtype=np.int64,
            ),
            validation_order=np.asarray(
                configuration.validation_order,
                dtype=np.int64,
            ),
            model_name=np.asarray(configuration.model_name),
            model_parameters_json=np.asarray(model_parameters),
            model_source=np.asarray(str(configuration.model_source)),
            model_source_sha256=np.asarray(source_sha256),
            generator_config=np.asarray(str(configuration.source_path)),
            recipe_config=np.asarray(str(configuration.recipe.source_path)),
            unused_source_fit_offset=np.asarray(model.unused_fit_offset),
        )
        temporary.replace(output_path)
        output_path.chmod(0o644)
    finally:
        if temporary.exists():
            temporary.unlink()

    return SyntheticDensityResult(
        output_path=output_path,
        grid_shape=grid.shape,
        maximum_quadrature_relative_difference=float(
            np.max(relative_difference)
        ),
        median_quadrature_relative_difference=float(
            np.median(relative_difference)
        ),
    )
