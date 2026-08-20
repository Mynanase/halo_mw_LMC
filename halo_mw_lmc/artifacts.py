"""Versioned, portable artifacts shared by workflows and read-only analysis."""

from __future__ import annotations

import json
import tempfile
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np

from .core.density import DensityComparison, DensityShellDiagnostics
from .core.grids import CylindricalGrid
from .core.orbit_response import OrbitSupportAudit
from .core.velocity import (
    SphericalVelocityGrid,
    VelocityDistributionComparison,
)
from .core.weight_solver import WeightSolution

if TYPE_CHECKING:
    from .core.potentials import ZhuHaloParameters


BEST_EVALUATION_SCHEMA_VERSION = 3
RESOLVED_CONFIG_SCHEMA_VERSION = 5
SUPPORTED_BEST_EVALUATION_SCHEMA_VERSIONS = frozenset({2, 3})
SUPPORTED_RESOLVED_CONFIG_SCHEMA_VERSIONS = frozenset({4, 5})
WEIGHT_AUDIT_SCHEMA_VERSION = 1


class SampleFileError(ValueError):
    """Raised when an optimizer sample file cannot be used safely."""


def _require_columns(data: np.ndarray, columns: Sequence[str]) -> None:
    names = set(data.dtype.names or ())
    missing = [name for name in columns if name not in names]
    if missing:
        raise SampleFileError(
            "sample file is missing required columns: " + ", ".join(missing)
        )


def load_sample_table(
    path: str | Path,
    *,
    required_columns: Sequence[str] = (),
) -> np.ndarray:
    """Read a non-empty ``sample.dat`` as a one-dimensional structured array."""

    sample_path = Path(path)
    if not sample_path.exists():
        raise SampleFileError(f"sample file not found: {sample_path}")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            data = np.genfromtxt(sample_path, names=True, ndmin=1)
    except (OSError, TypeError, ValueError) as exc:
        raise SampleFileError(
            f"could not read sample file {sample_path}: {exc}"
        ) from exc

    if data.dtype.names is None:
        raise SampleFileError(f"sample file has no named header: {sample_path}")
    _require_columns(data, required_columns)
    if data.size == 0:
        raise SampleFileError(f"sample file contains no samples: {sample_path}")
    return data


def best_sample(data: np.ndarray) -> np.void:
    """Return the row with the smallest finite objective."""

    _require_columns(data, ("objective",))
    try:
        objective = np.asarray(data["objective"], dtype=float)
    except (TypeError, ValueError) as exc:
        raise SampleFileError("objective column is not numeric") from exc
    finite = np.flatnonzero(np.isfinite(objective))
    if finite.size == 0:
        raise SampleFileError("sample file contains no finite objective values")
    return data[finite[np.argmin(objective[finite])]]


@dataclass(frozen=True)
class StoredBestEvaluation:
    """Best-so-far numerical snapshot reconstructed from one run directory."""

    metadata: Mapping[str, object]
    density: DensityComparison
    velocity_loglike: Mapping[str, float]
    velocity_loglike_by_phi: Mapping[str, np.ndarray]
    velocity_stars_by_phi: Mapping[str, np.ndarray]
    velocity_distributions: Mapping[str, VelocityDistributionComparison]
    weight_solution: WeightSolution
    density_shells: DensityShellDiagnostics | None = None
    orbit_support_audit: OrbitSupportAudit | None = None


@dataclass(frozen=True)
class RunSummary:
    """Cheap scalar/audit view used by reports and Marimo."""

    run_directory: Path
    config: Mapping[str, object]
    samples: np.ndarray
    weight_audit: Mapping[str, np.ndarray]
    best_metadata: Mapping[str, object] | None


def _atomic_json(path: Path, document: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(document, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def write_resolved_config(path: str | Path, document: Mapping[str, object]) -> None:
    """Atomically persist the complete, path-resolved run configuration."""

    payload = dict(document)
    payload["schema_version"] = RESOLVED_CONFIG_SCHEMA_VERSION
    _atomic_json(Path(path), payload)


def discover_runs(root: str | Path) -> list[Path]:
    """Find immediate child run directories without opening source data."""

    directory = Path(root).expanduser()
    if not directory.exists() or not directory.is_dir():
        return []
    result = []
    for candidate in directory.iterdir():
        if not candidate.is_dir():
            continue
        if (candidate / "resolved_config.json").exists() or (
            candidate / "run_config.json"
        ).exists():
            result.append(candidate.resolve())
    return sorted(result, key=lambda path: path.name)


def _load_json(path: Path) -> dict[str, object]:
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read JSON artifact {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return document


def load_run_summary(run_directory: str | Path) -> RunSummary:
    """Load configuration, scalar samples, and weight audit only."""

    run = Path(run_directory).expanduser().resolve()
    config_path = run / "resolved_config.json"
    if not config_path.exists():
        config_path = run / "run_config.json"
    if not config_path.exists():
        raise ValueError(f"run configuration not found in {run}")
    config = _load_json(config_path)
    if (
        config_path.name == "resolved_config.json"
        and config.get("schema_version")
        not in SUPPORTED_RESOLVED_CONFIG_SCHEMA_VERSIONS
    ):
        raise ValueError(f"unsupported resolved-config schema in {config_path}")
    samples = load_sample_table(
        run / "sample.dat",
        required_columns=("iteration", "objective"),
    )

    audit_path = run / "fixed_seed_weights.npz"
    if not audit_path.exists():
        audit_path = run / "weight_model_inputs.npz"
    weight_audit: dict[str, np.ndarray] = {}
    if audit_path.exists():
        try:
            with np.load(audit_path, allow_pickle=False) as archive:
                if (
                    "artifact_schema_version" in archive
                    and int(archive["artifact_schema_version"])
                    != WEIGHT_AUDIT_SCHEMA_VERSION
                ):
                    raise ValueError("unsupported weight-input audit schema")
                weight_audit = {name: archive[name].copy() for name in archive.files}
        except (OSError, ValueError) as exc:
            raise ValueError(f"could not read weight audit {audit_path}: {exc}") from exc

    metadata_path = run / "best" / "metadata.json"
    metadata = _load_json(metadata_path) if metadata_path.exists() else None
    if metadata is not None:
        if metadata.get("schema_version") not in (
            SUPPORTED_BEST_EVALUATION_SCHEMA_VERSIONS
        ):
            raise ValueError(f"unsupported best metadata schema in {metadata_path}")
        current_best = best_sample(samples)
        try:
            metadata_iteration = int(metadata["iteration"])
            metadata_objective = float(metadata["objective"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"best metadata is missing numeric iteration/objective: {metadata_path}"
            ) from exc
        sample_iteration = int(current_best["iteration"])
        sample_objective = float(current_best["objective"])
        if metadata_iteration != sample_iteration or not np.isclose(
            metadata_objective,
            sample_objective,
            rtol=1e-7,
            atol=1e-10,
        ):
            raise ValueError(
                "best metadata does not match the best row in sample.dat; "
                "the run may be incomplete"
            )
    return RunSummary(
        run_directory=run,
        config=config,
        samples=samples,
        weight_audit=weight_audit,
        best_metadata=metadata,
    )


def _evaluation_arrays(
    evaluation,
    *,
    generation: str,
    iteration: int,
    objective: float,
) -> dict[str, np.ndarray]:
    density = evaluation.density
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray(BEST_EVALUATION_SCHEMA_VERSION),
        "snapshot_generation": np.asarray(generation),
        "snapshot_iteration": np.asarray(iteration, dtype=np.int64),
        "snapshot_objective": np.asarray(objective, dtype=float),
        "density_data": np.asarray(density.data_density),
        "density_error": np.asarray(density.data_error),
        "density_raw_model": np.asarray(density.raw_model_density),
        "density_model": np.asarray(density.model_density),
        "density_residual": np.asarray(density.residual),
        "density_fit_mask": np.asarray(density.fit_mask, dtype=bool),
        "density_normalization_mask": np.asarray(
            density.normalization_mask,
            dtype=bool,
        ),
        "density_scale": np.asarray(density.scale),
        "density_chi2": np.asarray(density.chi2),
        "density_chi2_by_phi": np.asarray(density.chi2_by_phi),
        "density_valid_bins_by_phi": np.asarray(density.valid_bins_by_phi),
        "density_r_edges": np.asarray(density.grid.r_edges),
        "density_z_edges": np.asarray(density.grid.z_edges),
        "density_phi_edges": np.asarray(density.grid.phi_edges),
        "weight_seed_weights": np.asarray(
            evaluation.weight_solution.seed_weights
        ),
        "weight_inner_objective": np.asarray(
            evaluation.weight_solution.inner_objective
        ),
        "weight_regularization_penalty": np.asarray(
            evaluation.weight_solution.regularization_penalty
        ),
        "weight_effective_orbit_count": np.asarray(
            evaluation.weight_solution.effective_orbit_count
        ),
        "weight_maximum_fraction": np.asarray(
            evaluation.weight_solution.maximum_weight_fraction
        ),
        "weight_active_orbit_count": np.asarray(
            evaluation.weight_solution.active_orbit_count,
            dtype=np.int64,
        ),
        "weight_converged": np.asarray(
            evaluation.weight_solution.converged,
            dtype=bool,
        ),
        "weight_status": np.asarray(
            evaluation.weight_solution.status,
            dtype=np.int64,
        ),
        "weight_iterations": np.asarray(
            evaluation.weight_solution.iterations,
            dtype=np.int64,
        ),
        "weight_optimality": np.asarray(
            evaluation.weight_solution.optimality,
            dtype=float,
        ),
        "weight_solver_cost": np.asarray(
            evaluation.weight_solution.solver_cost,
            dtype=float,
        ),
    }
    shells = evaluation.density_shells
    if shells is None:
        arrays.update(
            {
                "density_shell_edges": np.array([], dtype=float),
                "density_shell_chi2": np.array([], dtype=float),
                "density_shell_valid_bins": np.array([], dtype=np.int64),
                "density_shell_phi_chi2": np.empty((0, density.grid.shape[-1])),
                "density_shell_phi_valid_bins": np.empty(
                    (0, density.grid.shape[-1]), dtype=np.int64
                ),
            }
        )
    else:
        arrays.update(
            {
                "density_shell_edges": np.asarray(shells.radius_edges),
                "density_shell_chi2": np.asarray(shells.chi2_by_shell),
                "density_shell_valid_bins": np.asarray(
                    shells.valid_bins_by_shell, dtype=np.int64
                ),
                "density_shell_phi_chi2": np.asarray(shells.chi2_by_shell_phi),
                "density_shell_phi_valid_bins": np.asarray(
                    shells.valid_bins_by_shell_phi, dtype=np.int64
                ),
            }
        )
    support = evaluation.orbit_support_audit
    arrays["orbit_support_available"] = np.asarray(support is not None, dtype=bool)
    arrays["orbit_density_supported_count"] = np.asarray(
        support.density_supported_orbit_count if support is not None else -1,
        dtype=np.int64,
    )
    arrays["orbit_velocity_supported_count"] = np.asarray(
        support.velocity_supported_orbit_count if support is not None else -1,
        dtype=np.int64,
    )
    arrays["orbit_zero_density_response_velocity_count"] = np.asarray(
        support.zero_density_response_velocity_orbit_count
        if support is not None
        else -1,
        dtype=np.int64,
    )
    arrays["orbit_zero_density_response_velocity_sample_count"] = np.asarray(
        support.zero_density_response_velocity_sample_count
        if support is not None
        else -1,
        dtype=np.int64,
    )
    arrays["orbit_zero_density_response_velocity_weight_sum"] = np.asarray(
        support.zero_density_response_velocity_weight_sum
        if support is not None
        else np.nan,
        dtype=float,
    )
    components = tuple(evaluation.velocity_distributions)
    arrays["velocity_components"] = np.asarray(components, dtype="U16")
    if components:
        first = evaluation.velocity_distributions[components[0]].grid
        arrays.update(
            {
                "velocity_radius_edges": np.asarray(first.radius_edges),
                "velocity_theta_edges": np.asarray(first.theta_edges),
                "velocity_phi_edges": np.asarray(first.phi_edges),
                "velocity_edges": np.asarray(first.velocity_edges),
            }
        )
    for component in components:
        distribution = evaluation.velocity_distributions[component]
        prefix = f"velocity_{component}"
        arrays.update(
            {
                f"{prefix}_data_probability": np.asarray(
                    distribution.data_probability
                ),
                f"{prefix}_data_uncertainty": np.asarray(
                    distribution.data_uncertainty
                ),
                f"{prefix}_data_occupancy": np.asarray(
                    distribution.data_occupancy
                ),
                f"{prefix}_model_probability": np.asarray(
                    distribution.model_probability
                ),
                f"{prefix}_model_occupancy": np.asarray(
                    distribution.model_occupancy
                ),
                f"{prefix}_loglike": np.asarray(
                    evaluation.velocity_loglike[component]
                ),
                f"{prefix}_loglike_by_phi": np.asarray(
                    evaluation.velocity_loglike_by_phi[component]
                ),
                f"{prefix}_stars_by_phi": np.asarray(
                    evaluation.velocity_stars_by_phi[component]
                ),
            }
        )
    return arrays


def save_best_evaluation(
    run_directory: str | Path,
    evaluation,
    parameters: ZhuHaloParameters,
    *,
    iteration: int,
    objective: float,
) -> None:
    """Atomically replace the single best numerical snapshot for a run."""

    seed_orbits = int(evaluation.weight_solution.seed_weights.size)
    successful_orbits = int(evaluation.successful_orbits)
    failed_orbits = seed_orbits - successful_orbits
    if failed_orbits < 0:
        raise ValueError("successful orbit count exceeds seed orbit count")
    best_directory = Path(run_directory) / "best"
    best_directory.mkdir(parents=True, exist_ok=True)
    evaluation_path = best_directory / "evaluation.npz"
    generation = uuid.uuid4().hex
    with tempfile.NamedTemporaryFile(
        dir=best_directory,
        prefix=".evaluation.",
        suffix=".npz",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(
            temporary,
            **_evaluation_arrays(
                evaluation,
                generation=generation,
                iteration=iteration,
                objective=objective,
            ),
        )
        temporary.replace(evaluation_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    worst_shell_phi = evaluation.density_worst_shell_phi_chi2_per_bin
    if not np.isfinite(worst_shell_phi):
        worst_shell_phi = None
    metadata = {
        "schema_version": BEST_EVALUATION_SCHEMA_VERSION,
        "generation": generation,
        "iteration": int(iteration),
        "objective": float(objective),
        "parameters": parameters.as_dict(),
        "seed_orbits": seed_orbits,
        "successful_orbits": successful_orbits,
        "failed_orbits": failed_orbits,
        "weight_sum": float(np.sum(evaluation.weight_solution.seed_weights)),
        "include_velocity": bool(evaluation.velocity_distributions),
        "weight_mode": evaluation.weight_mode,
        "weight_solver_message": evaluation.weight_solution.message,
        "weight_solver_iterations": int(
            evaluation.weight_solution.iterations
        ),
        "weight_solver_optimality": float(
            evaluation.weight_solution.optimality
        ),
        "weight_solver_cost": float(evaluation.weight_solution.solver_cost),
        "objective_mode": evaluation.objective_mode,
        "objective_velocity": float(evaluation.objective_velocity),
        "objective_density_velocity": float(
            evaluation.objective_density_velocity
        ),
        "density_chi2_per_bin": float(evaluation.density_chi2_per_bin),
        "density_max_chi2_per_bin": evaluation.density_max_chi2_per_bin,
        "density_shell_phi_max_chi2_per_bin": (
            evaluation.density_shell_phi_max_chi2_per_bin
        ),
        "density_shell_phi_gate_passed": (
            evaluation.density_shell_phi_gate_passed
            if evaluation.density_shells is not None
            else None
        ),
        "density_worst_shell_phi_chi2_per_bin": (
            worst_shell_phi
            if evaluation.density_shells is not None
            else None
        ),
        "density_worst_shell_phi_index": (
            list(evaluation.density_worst_shell_phi_index)
            if evaluation.density_worst_shell_phi_index is not None
            else None
        ),
        "density_gate_passed": (
            evaluation.density_gate_passed
            if evaluation.objective_mode == "velocity_only"
            else None
        ),
        "orbit_support_audit": (
            {
                "density_supported_orbit_count": (
                    evaluation.orbit_support_audit.density_supported_orbit_count
                ),
                "velocity_supported_orbit_count": (
                    evaluation.orbit_support_audit.velocity_supported_orbit_count
                ),
                "zero_density_response_velocity_orbit_count": (
                    evaluation.orbit_support_audit.zero_density_response_velocity_orbit_count
                ),
                "zero_density_response_velocity_orbit_fraction": (
                    evaluation.orbit_support_audit.zero_density_response_velocity_orbit_fraction
                ),
                "zero_density_response_velocity_sample_count": (
                    evaluation.orbit_support_audit.zero_density_response_velocity_sample_count
                ),
                "zero_density_response_velocity_weight_sum": (
                    evaluation.orbit_support_audit.zero_density_response_velocity_weight_sum
                ),
            }
            if evaluation.orbit_support_audit is not None
            else None
        ),
    }
    _atomic_json(best_directory / "metadata.json", metadata)


def _required_arrays(archive, names: Sequence[str], path: Path) -> None:
    missing = [name for name in names if name not in archive]
    if missing:
        raise ValueError(
            f"best-evaluation artifact {path} is missing arrays: "
            + ", ".join(missing)
        )


def load_best_evaluation(run_directory: str | Path) -> StoredBestEvaluation:
    """Load a best snapshot; never reconstruct it from source data or AGAMA."""

    best_directory = Path(run_directory).expanduser().resolve() / "best"
    metadata_path = best_directory / "metadata.json"
    evaluation_path = best_directory / "evaluation.npz"
    if not metadata_path.exists() or not evaluation_path.exists():
        raise ValueError(f"best-evaluation snapshot not found in {best_directory}")
    metadata = _load_json(metadata_path)
    metadata_schema = metadata.get("schema_version")
    if metadata_schema not in SUPPORTED_BEST_EVALUATION_SCHEMA_VERSIONS:
        raise ValueError("unsupported best-evaluation metadata schema")

    required = (
        "schema_version",
        "snapshot_generation",
        "snapshot_iteration",
        "snapshot_objective",
        "density_data",
        "density_error",
        "density_raw_model",
        "density_model",
        "density_residual",
        "density_fit_mask",
        "density_normalization_mask",
        "density_scale",
        "density_chi2",
        "density_chi2_by_phi",
        "density_valid_bins_by_phi",
        "density_r_edges",
        "density_z_edges",
        "density_phi_edges",
        "velocity_components",
        "weight_seed_weights",
        "weight_inner_objective",
        "weight_regularization_penalty",
        "weight_effective_orbit_count",
        "weight_maximum_fraction",
        "weight_active_orbit_count",
        "weight_converged",
        "weight_status",
    )
    try:
        with np.load(evaluation_path, allow_pickle=False) as archive:
            _required_arrays(archive, required, evaluation_path)
            archive_schema = int(archive["schema_version"])
            if archive_schema not in SUPPORTED_BEST_EVALUATION_SCHEMA_VERSIONS:
                raise ValueError("unsupported best-evaluation array schema")
            if archive_schema != metadata_schema:
                raise ValueError("best-evaluation metadata and arrays do not match")
            if str(archive["snapshot_generation"].item()) != metadata.get(
                "generation"
            ):
                raise ValueError("best-evaluation metadata and arrays do not match")
            if int(archive["snapshot_iteration"]) != int(metadata["iteration"]):
                raise ValueError("best-evaluation metadata and arrays do not match")
            if not np.isclose(
                float(archive["snapshot_objective"]),
                float(metadata["objective"]),
                rtol=1e-12,
                atol=1e-12,
            ):
                raise ValueError("best-evaluation metadata and arrays do not match")
            grid = CylindricalGrid(
                archive["density_r_edges"],
                archive["density_z_edges"],
                archive["density_phi_edges"],
            )
            density_shape = grid.shape
            for name in (
                "density_data",
                "density_error",
                "density_raw_model",
                "density_model",
                "density_residual",
                "density_fit_mask",
                "density_normalization_mask",
            ):
                if archive[name].shape != density_shape:
                    raise ValueError(
                        f"array {name} has shape {archive[name].shape}; "
                        f"expected {density_shape}"
                    )
            density = DensityComparison(
                data_density=archive["density_data"].copy(),
                data_error=archive["density_error"].copy(),
                raw_model_density=archive["density_raw_model"].copy(),
                model_density=archive["density_model"].copy(),
                residual=archive["density_residual"].copy(),
                fit_mask=archive["density_fit_mask"].astype(bool),
                normalization_mask=archive["density_normalization_mask"].astype(bool),
                scale=float(archive["density_scale"]),
                chi2=float(archive["density_chi2"]),
                chi2_by_phi=archive["density_chi2_by_phi"].copy(),
                valid_bins_by_phi=archive["density_valid_bins_by_phi"].astype(
                    np.int64
                ),
                grid=grid,
            )
            seed_weights = archive["weight_seed_weights"].copy()
            if seed_weights.ndim != 1 or not np.all(np.isfinite(seed_weights)):
                raise ValueError("array weight_seed_weights must be finite and 1D")
            if np.any(seed_weights < 0):
                raise ValueError("array weight_seed_weights must be non-negative")
            weight_solution = WeightSolution(
                seed_weights=seed_weights,
                model_density=density.raw_model_density.copy(),
                target_density=density.data_density.copy(),
                target_error=density.data_error.copy(),
                inner_objective=float(archive["weight_inner_objective"]),
                regularization_penalty=float(
                    archive["weight_regularization_penalty"]
                ),
                effective_orbit_count=float(
                    archive["weight_effective_orbit_count"]
                ),
                maximum_weight_fraction=float(
                    archive["weight_maximum_fraction"]
                ),
                active_orbit_count=int(archive["weight_active_orbit_count"]),
                converged=bool(archive["weight_converged"]),
                status=int(archive["weight_status"]),
                message=str(metadata.get("weight_solver_message", "")),
                iterations=int(archive.get("weight_iterations", 0)),
                optimality=float(archive.get("weight_optimality", np.inf)),
                solver_cost=float(archive.get("weight_solver_cost", np.inf)),
            )
            n_phi = grid.shape[-1]
            for name in ("density_chi2_by_phi", "density_valid_bins_by_phi"):
                if archive[name].shape != (n_phi,):
                    raise ValueError(
                        f"array {name} has shape {archive[name].shape}; "
                        f"expected {(n_phi,)}"
                    )

            density_shells = None
            orbit_support_audit = None
            if archive_schema >= 3:
                shell_names = (
                    "density_shell_edges",
                    "density_shell_chi2",
                    "density_shell_valid_bins",
                    "density_shell_phi_chi2",
                    "density_shell_phi_valid_bins",
                    "orbit_support_available",
                    "orbit_density_supported_count",
                    "orbit_velocity_supported_count",
                    "orbit_zero_density_response_velocity_count",
                    "orbit_zero_density_response_velocity_sample_count",
                    "orbit_zero_density_response_velocity_weight_sum",
                )
                _required_arrays(archive, shell_names, evaluation_path)
                shell_edges = archive["density_shell_edges"].copy()
                if shell_edges.size:
                    n_shell = shell_edges.size - 1
                    if archive["density_shell_chi2"].shape != (n_shell,):
                        raise ValueError("array density_shell_chi2 has invalid shape")
                    if archive["density_shell_valid_bins"].shape != (n_shell,):
                        raise ValueError(
                            "array density_shell_valid_bins has invalid shape"
                        )
                    expected_shell_phi = (n_shell, n_phi)
                    for name in (
                        "density_shell_phi_chi2",
                        "density_shell_phi_valid_bins",
                    ):
                        if archive[name].shape != expected_shell_phi:
                            raise ValueError(f"array {name} has invalid shape")
                    density_shells = DensityShellDiagnostics(
                        radius_edges=shell_edges,
                        chi2_by_shell=archive["density_shell_chi2"].copy(),
                        valid_bins_by_shell=archive[
                            "density_shell_valid_bins"
                        ].astype(np.int64),
                        chi2_by_shell_phi=archive[
                            "density_shell_phi_chi2"
                        ].copy(),
                        valid_bins_by_shell_phi=archive[
                            "density_shell_phi_valid_bins"
                        ].astype(np.int64),
                    )
                if bool(archive["orbit_support_available"]):
                    orbit_support_audit = OrbitSupportAudit(
                        density_supported_orbit_count=int(
                            archive["orbit_density_supported_count"]
                        ),
                        velocity_supported_orbit_count=int(
                            archive["orbit_velocity_supported_count"]
                        ),
                        zero_density_response_velocity_orbit_count=int(
                            archive["orbit_zero_density_response_velocity_count"]
                        ),
                        zero_density_response_velocity_sample_count=int(
                            archive[
                                "orbit_zero_density_response_velocity_sample_count"
                            ]
                        ),
                        zero_density_response_velocity_weight_sum=float(
                            archive[
                                "orbit_zero_density_response_velocity_weight_sum"
                            ]
                        ),
                    )

            components = tuple(str(value) for value in archive["velocity_components"])
            distributions: dict[str, VelocityDistributionComparison] = {}
            loglike: dict[str, float] = {}
            by_phi: dict[str, np.ndarray] = {}
            stars: dict[str, np.ndarray] = {}
            if components:
                edge_names = (
                    "velocity_radius_edges",
                    "velocity_theta_edges",
                    "velocity_phi_edges",
                    "velocity_edges",
                )
                _required_arrays(archive, edge_names, evaluation_path)
                velocity_grid = SphericalVelocityGrid(
                    archive["velocity_radius_edges"],
                    archive["velocity_theta_edges"],
                    archive["velocity_phi_edges"],
                    archive["velocity_edges"],
                )
                for component in components:
                    prefix = f"velocity_{component}"
                    names = tuple(
                        f"{prefix}_{suffix}"
                        for suffix in (
                            "data_probability",
                            "data_uncertainty",
                            "data_occupancy",
                            "model_probability",
                            "model_occupancy",
                            "loglike",
                            "loglike_by_phi",
                            "stars_by_phi",
                        )
                    )
                    _required_arrays(archive, names, evaluation_path)
                    probability_shape = velocity_grid.shape
                    occupancy_shape = velocity_grid.shape[:-1]
                    for suffix in (
                        "data_probability",
                        "data_uncertainty",
                        "model_probability",
                    ):
                        if archive[f"{prefix}_{suffix}"].shape != probability_shape:
                            raise ValueError(
                                f"velocity component {component} has an invalid "
                                f"{suffix} shape"
                            )
                    for suffix in ("data_occupancy", "model_occupancy"):
                        if archive[f"{prefix}_{suffix}"].shape != occupancy_shape:
                            raise ValueError(
                                f"velocity component {component} has an invalid "
                                f"{suffix} shape"
                            )
                    for suffix in ("loglike_by_phi", "stars_by_phi"):
                        if archive[f"{prefix}_{suffix}"].shape != (
                            velocity_grid.shape[2],
                        ):
                            raise ValueError(
                                f"velocity component {component} has an invalid "
                                f"{suffix} shape"
                            )
                    distributions[component] = VelocityDistributionComparison(
                        component=component,
                        grid=velocity_grid,
                        data_probability=archive[
                            f"{prefix}_data_probability"
                        ].copy(),
                        data_uncertainty=archive[
                            f"{prefix}_data_uncertainty"
                        ].copy(),
                        data_occupancy=archive[f"{prefix}_data_occupancy"].copy(),
                        model_probability=archive[
                            f"{prefix}_model_probability"
                        ].copy(),
                        model_occupancy=archive[
                            f"{prefix}_model_occupancy"
                        ].copy(),
                    )
                    loglike[component] = float(archive[f"{prefix}_loglike"])
                    by_phi[component] = archive[
                        f"{prefix}_loglike_by_phi"
                    ].copy()
                    stars[component] = archive[f"{prefix}_stars_by_phi"].astype(
                        np.int64
                    )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(
            ("unsupported", "array ", "velocity component", "best-evaluation")
        ):
            raise
        raise ValueError(
            f"could not read best-evaluation artifact {evaluation_path}: {exc}"
        ) from exc

    return StoredBestEvaluation(
        metadata=metadata,
        density=density,
        velocity_loglike=loglike,
        velocity_loglike_by_phi=by_phi,
        velocity_stars_by_phi=stars,
        velocity_distributions=distributions,
        weight_solution=weight_solution,
        density_shells=density_shells,
        orbit_support_audit=orbit_support_audit,
    )
