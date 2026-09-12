"""Sequential execution and persistence for the density-solved solver budget.

Step 1 of ``docs/solver_budget_experiment.md`` owns the strict benchmark
configuration, the output-directory contract, and the read-only preflight.
The later phases (``prepare``, ``parity``, ``pilot``, ``budget``, ``repeat``,
``holdout-prepare``, ``holdout``, ``confirm``) are declared here so the entry
point accepts them, but each refuses to run until its contract step is
implemented.  No orbit integration, weight solve, or GPU/GP work happens in
this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import subprocess
import tomllib
from typing import Any, Mapping

import numpy as np

from .config import (
    CONFIGURATION_SCHEMA_VERSION,
    ConfigurationError,
    RecipeConfiguration,
    SEARCH_PARAMETER_NAMES,
    load_recipe_configuration,
)
from .config import WeightModelSettings


SOLVER_BUDGET_SCHEMA_VERSION = CONFIGURATION_SCHEMA_VERSION
SOLVER_BUDGET_PHASES = (
    "preflight",
    "prepare",
    "parity",
    "pilot",
    "budget",
    "repeat",
    "holdout-prepare",
    "holdout",
    "confirm",
)
SOLVER_BUDGET_BACKENDS = ("lsq_linear", "dense_nnls", "dual_ridge")
GNU_TIME_PROGRAM = "/usr/bin/time"
THREAD_ENVIRONMENT_VARIABLES = (
    ("OPENBLAS_NUM_THREADS", "openblas"),
    ("MKL_NUM_THREADS", "mkl"),
    ("BLIS_NUM_THREADS", "blis"),
    ("OMP_NUM_THREADS", "omp"),
)


@dataclass(frozen=True)
class SolverBudgetPoint:
    """One fixed five-dimensional potential point of the experiment."""

    name: str
    coordinates: tuple[float, float, float, float, float]
    source: str


@dataclass(frozen=True)
class SolverBudgetMethod:
    """One solver method: scientific settings come from the recipe."""

    name: str
    solver: str
    max_iter: int
    lsmr_tol: float | None
    solver_tolerance: float | None


@dataclass(frozen=True)
class SolverBudgetThresholds:
    """Pre-fixed engineering gates; never relaxed after seeing results."""

    reference_kkt_maximum: float
    inner_objective_relative: float
    joint_objective_absolute: float
    component_absolute: float
    pairwise_delta_absolute: float
    ranking_flip_reference_difference: float
    density_prediction_rms: float
    velocity_tv_weighted_mean: float
    velocity_tv_cell_maximum: float
    velocity_tv_minimum_observations: int
    repeat_rtol: float
    repeat_atol: float
    environment_instability_ratio: float
    target_full_evaluation_seconds: float


@dataclass(frozen=True)
class SolverBudgetPlan:
    """Resolved benchmark configuration plus its experiment provenance."""

    source_path: Path
    recipe_source_path: Path
    recipe: RecipeConfiguration
    benchmark_id: str
    output_root: Path
    timeout_seconds: float
    repeats: int
    threads: tuple[tuple[str, int], ...]
    points: tuple[SolverBudgetPoint, ...]
    methods: tuple[SolverBudgetMethod, ...]
    budget_max_iter: tuple[int, ...]
    phases: tuple[str, ...]
    thresholds: SolverBudgetThresholds
    catalog_path: Path
    density_path: Path
    baseline_ref: str | None = None


def _read_document(path: str | Path, context: str) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    try:
        with source.open("rb") as stream:
            document = tomllib.load(stream)
    except FileNotFoundError as exc:
        raise ConfigurationError(f"{context} file not found: {source}") from exc
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigurationError(
            f"could not read {context} file {source}: {exc}"
        ) from exc
    if not isinstance(document, dict):
        raise ConfigurationError(f"{context} document must be a TOML table")
    return source, document


def _require_exact_fields(
    table: Mapping[str, Any],
    expected: set[str],
    context: str,
    *,
    optional: set[str] | None = None,
) -> None:
    optional = set() if optional is None else optional
    unknown = sorted(set(table) - expected - optional)
    missing = sorted(expected - set(table))
    if unknown:
        raise ConfigurationError(f"unknown field(s) in {context}: {', '.join(unknown)}")
    if missing:
        raise ConfigurationError(
            f"missing required field(s) in {context}: {', '.join(missing)}"
        )


def _table(document: Mapping[str, Any], name: str, context: str) -> Mapping[str, Any]:
    value = document.get(name)
    if not isinstance(value, dict):
        raise ConfigurationError(f"{context}.{name} must be a TOML table")
    return value


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{context} must be a non-empty string")
    return value


def _integer(value: Any, context: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{context} must be an integer")
    if minimum is not None and value < minimum:
        raise ConfigurationError(f"{context} must be at least {minimum}")
    return value


def _finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{context} must be a number")
    result = float(value)
    if not np.isfinite(result):
        raise ConfigurationError(f"{context} must be finite")
    return result


def _positive_number(value: Any, context: str) -> float:
    result = _finite_number(value, context)
    if result <= 0:
        raise ConfigurationError(f"{context} must be a positive finite number")
    return result


def _resolved_path(value: Any, source: Path, context: str) -> Path:
    raw = Path(_string(value, context)).expanduser()
    return raw.resolve() if raw.is_absolute() else (source.parent / raw).resolve()


def _point(
    name: str,
    table: Mapping[str, Any],
    context: str,
    *,
    recipe: RecipeConfiguration,
) -> SolverBudgetPoint:
    _require_exact_fields(table, {"coordinates", "source"}, context)
    raw = table["coordinates"]
    if not isinstance(raw, list) or len(raw) != len(SEARCH_PARAMETER_NAMES):
        raise ConfigurationError(
            f"{context}.coordinates must contain exactly "
            f"{len(SEARCH_PARAMETER_NAMES)} values in "
            f"{', '.join(SEARCH_PARAMETER_NAMES)} order"
        )
    coordinates = tuple(_finite_number(value, f"{context}.coordinates[{index}]") for index, value in enumerate(raw))
    bounds = recipe.search.bounds.as_dict()
    decimals = recipe.search.round_decimals
    for parameter, coordinate in zip(SEARCH_PARAMETER_NAMES, coordinates):
        lower, upper = bounds[parameter]
        if not lower <= coordinate <= upper:
            raise ConfigurationError(
                f"{context}.coordinates {parameter}={coordinate} lies outside "
                f"the recipe bounds [{lower}, {upper}]"
            )
        if not np.isclose(coordinate, round(coordinate, decimals), rtol=0.0, atol=10 ** (-(decimals + 10))):
            raise ConfigurationError(
                f"{context}.coordinates {parameter} is not representable with "
                f"round_decimals={decimals}"
            )
    return SolverBudgetPoint(name=name, coordinates=coordinates, source=_string(table["source"], f"{context}.source"))


def _method(
    name: str,
    table: Mapping[str, Any],
    context: str,
    *,
    recipe: RecipeConfiguration,
) -> SolverBudgetMethod:
    _require_exact_fields(
        table,
        {"solver", "max_iter"},
        context,
        optional={"lsmr_tol", "solver_tolerance"},
    )
    solver = _string(table["solver"], f"{context}.solver")
    if solver not in SOLVER_BUDGET_BACKENDS:
        raise ConfigurationError(
            f"{context}.solver must be one of " + ", ".join(SOLVER_BUDGET_BACKENDS)
        )
    if solver == "lsq_linear":
        if "lsmr_tol" not in table:
            raise ConfigurationError(
                f"{context} must declare lsmr_tol for solver='lsq_linear'"
            )
        lsmr_tol = _positive_number(table["lsmr_tol"], f"{context}.lsmr_tol")
        solver_tolerance = None
        if "solver_tolerance" in table:
            solver_tolerance = _positive_number(
                table["solver_tolerance"], f"{context}.solver_tolerance"
            )
    else:
        if "lsmr_tol" in table:
            raise ConfigurationError(
                f"{context} cannot declare lsmr_tol for solver={solver!r}"
            )
        lsmr_tol = None
        if "solver_tolerance" not in table:
            raise ConfigurationError(
                f"{context}.solver_tolerance is required for solver={solver!r}"
            )
        solver_tolerance = _positive_number(
            table["solver_tolerance"], f"{context}.solver_tolerance"
        )
    method = SolverBudgetMethod(
        name=name,
        solver=solver,
        max_iter=_integer(table["max_iter"], f"{context}.max_iter", minimum=1),
        lsmr_tol=lsmr_tol,
        solver_tolerance=solver_tolerance,
    )
    resolved_weight_settings(recipe, method)
    return method


def resolved_weight_settings(
    recipe: RecipeConfiguration,
    method: SolverBudgetMethod,
) -> WeightModelSettings:
    """Return the method's weight settings: solver knobs override the recipe.

    Only ``solver``, ``max_iter``, and the backend tolerance change between
    methods; normalization, regularization, and everything else stays exactly
    as the shared recipe declares.  Constructing the dataclass also applies the
    core validation contract for the chosen backend.
    """

    base = recipe.weight_model
    if base.mode != "density_solved":
        raise ConfigurationError(
            "the solver-budget experiment requires a density_solved recipe"
        )
    return WeightModelSettings(
        mode=base.mode,
        solver=method.solver,
        target_normalization=base.target_normalization,
        regularization=base.regularization,
        regularization_strength=base.regularization_strength,
        max_iter=method.max_iter,
        solver_tolerance=method.solver_tolerance,
        lsmr_tol=method.lsmr_tol,
    )


def _thresholds(table: Mapping[str, Any]) -> SolverBudgetThresholds:
    expected = {
        "reference_kkt_maximum",
        "inner_objective_relative",
        "joint_objective_absolute",
        "component_absolute",
        "pairwise_delta_absolute",
        "ranking_flip_reference_difference",
        "density_prediction_rms",
        "velocity_tv_weighted_mean",
        "velocity_tv_cell_maximum",
        "velocity_tv_minimum_observations",
        "repeat_rtol",
        "repeat_atol",
        "environment_instability_ratio",
        "target_full_evaluation_seconds",
    }
    _require_exact_fields(table, expected, "thresholds")
    thresholds = SolverBudgetThresholds(
        reference_kkt_maximum=_positive_number(table["reference_kkt_maximum"], "thresholds.reference_kkt_maximum"),
        inner_objective_relative=_positive_number(table["inner_objective_relative"], "thresholds.inner_objective_relative"),
        joint_objective_absolute=_positive_number(table["joint_objective_absolute"], "thresholds.joint_objective_absolute"),
        component_absolute=_positive_number(table["component_absolute"], "thresholds.component_absolute"),
        pairwise_delta_absolute=_positive_number(table["pairwise_delta_absolute"], "thresholds.pairwise_delta_absolute"),
        ranking_flip_reference_difference=_positive_number(table["ranking_flip_reference_difference"], "thresholds.ranking_flip_reference_difference"),
        density_prediction_rms=_positive_number(table["density_prediction_rms"], "thresholds.density_prediction_rms"),
        velocity_tv_weighted_mean=_positive_number(table["velocity_tv_weighted_mean"], "thresholds.velocity_tv_weighted_mean"),
        velocity_tv_cell_maximum=_positive_number(table["velocity_tv_cell_maximum"], "thresholds.velocity_tv_cell_maximum"),
        velocity_tv_minimum_observations=_integer(table["velocity_tv_minimum_observations"], "thresholds.velocity_tv_minimum_observations", minimum=1),
        repeat_rtol=_positive_number(table["repeat_rtol"], "thresholds.repeat_rtol"),
        repeat_atol=_positive_number(table["repeat_atol"], "thresholds.repeat_atol"),
        environment_instability_ratio=_positive_number(table["environment_instability_ratio"], "thresholds.environment_instability_ratio"),
        target_full_evaluation_seconds=_positive_number(table["target_full_evaluation_seconds"], "thresholds.target_full_evaluation_seconds"),
    )
    # The repeat tolerance is fixed by the contract; only make it stricter.
    if thresholds.repeat_rtol > 1e-12 or thresholds.repeat_atol > 1e-12:
        raise ConfigurationError(
            "thresholds.repeat_rtol and thresholds.repeat_atol cannot be loosened "
            "beyond 1e-12"
        )
    if thresholds.environment_instability_ratio < 1.0:
        raise ConfigurationError(
            "thresholds.environment_instability_ratio must be at least one"
        )
    return thresholds


def load_solver_budget_plan(
    path: str | Path,
    *,
    baseline_ref: str | None = None,
) -> SolverBudgetPlan:
    """Load and strictly validate one solver-budget benchmark TOML."""

    source, document = _read_document(path, "solver-budget configuration")
    _require_exact_fields(
        document,
        {
            "schema_version",
            "recipe",
            "benchmark",
            "data",
            "threads",
            "points",
            "methods",
            "budget",
            "phases",
            "thresholds",
        },
        "solver-budget configuration",
    )
    schema_version = _integer(document["schema_version"], "schema_version", minimum=1)
    if schema_version != SOLVER_BUDGET_SCHEMA_VERSION:
        raise ConfigurationError(
            f"unsupported schema_version: {schema_version}; "
            f"expected {SOLVER_BUDGET_SCHEMA_VERSION}"
        )
    recipe_path = _resolved_path(document["recipe"], source, "recipe")
    recipe = load_recipe_configuration(recipe_path)

    benchmark_table = _table(document, "benchmark", "solver-budget configuration")
    _require_exact_fields(
        benchmark_table,
        {"id", "output_root", "timeout_seconds", "repeats"},
        "benchmark",
    )
    output_root = _resolved_path(benchmark_table["output_root"], source, "benchmark.output_root")

    data_table = _table(document, "data", "solver-budget configuration")
    _require_exact_fields(data_table, {"catalog", "target_density"}, "data")
    catalog_path = _resolved_path(data_table["catalog"], source, "data.catalog")
    density_path = _resolved_path(data_table["target_density"], source, "data.target_density")

    threads_table = _table(document, "threads", "solver-budget configuration")
    thread_names = {name for _, name in THREAD_ENVIRONMENT_VARIABLES}
    _require_exact_fields(threads_table, thread_names, "threads")
    threads = tuple((name, _integer(threads_table[name], f"threads.{name}", minimum=1)) for _, name in THREAD_ENVIRONMENT_VARIABLES)

    points_table = _table(document, "points", "solver-budget configuration")
    if len(points_table) < 3:
        raise ConfigurationError("points must declare at least three pilot points")
    points = tuple(
        _point(name, table, f"points.{name}", recipe=recipe)
        for name, table in points_table.items()
        if isinstance(table, dict)
    )
    if len(points) != len(points_table):
        raise ConfigurationError("every entry of points must be a TOML table")
    coordinates = [point.coordinates for point in points]
    if len(set(coordinates)) != len(coordinates):
        raise ConfigurationError("points must not duplicate coordinates")

    methods_table = _table(document, "methods", "solver-budget configuration")
    if not methods_table:
        raise ConfigurationError("methods must declare at least one method")
    methods = tuple(
        _method(name, table, f"methods.{name}", recipe=recipe)
        for name, table in methods_table.items()
        if isinstance(table, dict)
    )
    if len(methods) != len(methods_table):
        raise ConfigurationError("every entry of methods must be a TOML table")

    budget_table = _table(document, "budget", "solver-budget configuration")
    _require_exact_fields(budget_table, {"solver", "lsmr_tol", "max_iter"}, "budget")
    if _string(budget_table["solver"], "budget.solver") != "lsq_linear":
        raise ConfigurationError("the budget curve is fixed to solver='lsq_linear'")
    budget_lsmr_tol = _positive_number(budget_table["lsmr_tol"], "budget.lsmr_tol")
    raw_levels = budget_table["max_iter"]
    if not isinstance(raw_levels, list) or not raw_levels:
        raise ConfigurationError("budget.max_iter must be a non-empty list")
    budget_max_iter = tuple(
        _integer(value, f"budget.max_iter[{index}]", minimum=1)
        for index, value in enumerate(raw_levels)
    )
    if any(left >= right for left, right in zip(budget_max_iter, budget_max_iter[1:])):
        raise ConfigurationError("budget.max_iter must be strictly increasing")
    for index, level in enumerate(budget_max_iter):
        resolved_weight_settings(recipe, SolverBudgetMethod(name=f"budget_{level}", solver="lsq_linear", max_iter=level, lsmr_tol=budget_lsmr_tol, solver_tolerance=None))

    phase_list = document["phases"]
    if not isinstance(phase_list, list) or not all(
        isinstance(entry, str) for entry in phase_list
    ):
        raise ConfigurationError("phases must be a list of phase names")
    phases = tuple(phase_list)
    if phases != SOLVER_BUDGET_PHASES:
        raise ConfigurationError(
            "phases must be exactly " + ", ".join(SOLVER_BUDGET_PHASES)
        )

    thresholds = _thresholds(_table(document, "thresholds", "solver-budget configuration"))

    return SolverBudgetPlan(
        source_path=source,
        recipe_source_path=recipe_path,
        recipe=recipe,
        benchmark_id=_string(benchmark_table["id"], "benchmark.id"),
        output_root=output_root,
        timeout_seconds=_positive_number(
            benchmark_table["timeout_seconds"], "benchmark.timeout_seconds"
        ),
        repeats=_integer(benchmark_table["repeats"], "benchmark.repeats", minimum=1),
        threads=threads,
        points=points,
        methods=methods,
        budget_max_iter=budget_max_iter,
        phases=phases,
        thresholds=thresholds,
        catalog_path=catalog_path,
        density_path=density_path,
        baseline_ref=baseline_ref,
    )


def case_directory(
    plan: SolverBudgetPlan,
    phase: str,
    case_name: str,
    *,
    attempt: int = 1,
) -> Path:
    """Return the output directory for one case attempt (never overwritten)."""

    if phase not in SOLVER_BUDGET_PHASES or phase == "preflight":
        raise ValueError(f"phase {phase!r} does not produce case directories")
    if attempt < 1:
        raise ValueError("attempt must be a positive integer")
    if not case_name or "/" in case_name:
        raise ValueError("case_name must be a non-empty single path component")
    return plan.output_root / phase / case_name / f"attempt_{attempt}"


def require_absent_case(path: Path) -> None:
    """Reject a case directory that already holds evidence."""

    if path.exists():
        raise RuntimeError(
            f"case directory already exists: {path}; use a new attempt id or a "
            "new output root"
        )


def existing_case_directories(output_root: Path) -> list[Path]:
    """List case directories already present under a solver-budget output root.

    Only directories named after an experiment phase are inspected, so receipt
    folders such as ``step0/`` can live in the same ignored output root.
    """

    root = Path(output_root)
    if not root.is_dir():
        return []
    found: list[Path] = []
    for phase_root in sorted(root.iterdir()):
        if not phase_root.is_dir() or phase_root.name not in SOLVER_BUDGET_PHASES:
            continue
        found.extend(path for path in sorted(phase_root.iterdir()) if path.is_dir())
    return found


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_provenance() -> dict[str, object]:
    root = _repository_root()
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        status = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"head": "unknown", "dirty": None}
    return {"head": head, "dirty": bool(status.strip())}


def _require_gnu_time() -> str:
    program = Path(GNU_TIME_PROGRAM)
    if not program.is_file() or not os.access(program, os.X_OK):
        raise RuntimeError(f"GNU time executable not found: {program}")
    try:
        completed = subprocess.run([str(program), "--version"], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"could not execute GNU time: {program}") from exc
    version_text = (completed.stdout + completed.stderr).lower()
    if "gnu time" not in version_text:
        raise RuntimeError(f"the experiment requires GNU time, not {program}")
    return str(program)


def _thread_environment_report(plan: SolverBudgetPlan) -> list[dict[str, object]]:
    report: list[dict[str, object]] = []
    for variable, name in THREAD_ENVIRONMENT_VARIABLES:
        configured = str(dict(plan.threads)[name])
        actual = os.environ.get(variable)
        report.append(
            {
                "variable": variable,
                "configured": configured,
                "actual": actual,
                "matches": actual == configured,
            }
        )
    return report


def run_preflight(plan: SolverBudgetPlan) -> dict[str, object]:
    """Validate configuration, inputs, tools, and the output directory.

    Read-only with respect to science: no integration, solve, or GP.  The only
    filesystem effect is creating the empty output root.
    """

    output_root = plan.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    if not output_root.is_dir():
        raise RuntimeError(f"solver-budget output root is not a directory: {output_root}")
    conflicts = existing_case_directories(output_root)
    if conflicts:
        raise RuntimeError(
            "solver-budget output root already contains case directories; use a "
            "new output root or a new attempt id: "
            + ", ".join(str(path) for path in conflicts[:5])
        )
    inputs = []
    for label, path in (
        ("catalogue", plan.catalog_path),
        ("target density", plan.density_path),
    ):
        if not path.is_file():
            raise RuntimeError(f"{label} not found: {path}")
        if not os.access(path, os.R_OK):
            raise RuntimeError(f"{label} is not readable: {path}")
        inputs.append(
            {
                "label": label,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    gnu_time = _require_gnu_time()
    environment = _thread_environment_report(plan)
    warnings = [
        f"{entry['variable']} is {entry['actual']!r}, configured {entry['configured']}"
        for entry in environment
        if not entry["matches"]
    ]
    agama_path = any(
        "agama" in part.lower() for part in os.environ.get("PYTHONPATH", "").split(os.pathsep) if part
    )
    if not agama_path:
        warnings.append(
            "PYTHONPATH does not list Agama-master; numerical phases need it set "
            "before Python starts"
        )
    return {
        "phase": "preflight",
        "benchmark_id": plan.benchmark_id,
        "config_path": str(plan.source_path),
        "recipe_path": str(plan.recipe_source_path),
        "output_root": str(output_root),
        "timeout_seconds": plan.timeout_seconds,
        "repeats": plan.repeats,
        "points": [
            {
                "name": point.name,
                "coordinates": list(point.coordinates),
                "source": point.source,
            }
            for point in plan.points
        ],
        "methods": [
            {
                "name": method.name,
                "solver": method.solver,
                "max_iter": method.max_iter,
                "lsmr_tol": method.lsmr_tol,
                "solver_tolerance": method.solver_tolerance,
            }
            for method in plan.methods
        ],
        "budget_max_iter": list(plan.budget_max_iter),
        "phases": list(plan.phases),
        "threads": [{"variable": variable, "value": value} for variable, value in plan.threads],
        "thresholds": {
            "reference_kkt_maximum": plan.thresholds.reference_kkt_maximum,
            "inner_objective_relative": plan.thresholds.inner_objective_relative,
            "joint_objective_absolute": plan.thresholds.joint_objective_absolute,
            "component_absolute": plan.thresholds.component_absolute,
            "pairwise_delta_absolute": plan.thresholds.pairwise_delta_absolute,
            "ranking_flip_reference_difference": plan.thresholds.ranking_flip_reference_difference,
            "density_prediction_rms": plan.thresholds.density_prediction_rms,
            "velocity_tv_weighted_mean": plan.thresholds.velocity_tv_weighted_mean,
            "velocity_tv_cell_maximum": plan.thresholds.velocity_tv_cell_maximum,
            "velocity_tv_minimum_observations": plan.thresholds.velocity_tv_minimum_observations,
            "repeat_rtol": plan.thresholds.repeat_rtol,
            "repeat_atol": plan.thresholds.repeat_atol,
            "environment_instability_ratio": plan.thresholds.environment_instability_ratio,
            "target_full_evaluation_seconds": plan.thresholds.target_full_evaluation_seconds,
        },
        "inputs": inputs,
        "gnu_time": gnu_time,
        "environment": environment,
        "warnings": warnings,
        "existing_cases": [],
        "provenance": _git_provenance(),
        "baseline_ref": plan.baseline_ref,
    }


def run_solver_budget_phase(
    config_path: str | Path,
    phase: str,
    *,
    baseline_ref: str | None = None,
) -> dict[str, object]:
    """Dispatch one experiment phase; only ``preflight`` runs today."""

    if phase not in SOLVER_BUDGET_PHASES:
        raise ValueError(
            f"unknown solver-budget phase {phase!r}; expected one of "
            + ", ".join(SOLVER_BUDGET_PHASES)
        )
    if phase == "parity" and not baseline_ref:
        raise ValueError("phase 'parity' requires --baseline-ref GIT_REF")
    plan = load_solver_budget_plan(config_path, baseline_ref=baseline_ref)
    if phase == "preflight":
        return run_preflight(plan)
    raise NotImplementedError(
        f"phase {phase!r} is implemented in a later contract step"
    )
