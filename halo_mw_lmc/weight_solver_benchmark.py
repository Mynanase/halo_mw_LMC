"""Artifact-only comparison of density-weight solver benchmark runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .artifacts import load_run_summary


SOLVER_BACKENDS = ("lsq_linear", "dense_nnls", "dual_ridge")
FORMULATION_MEMORY_RANK = {
    "lsq_linear": 0,
    "dual_ridge": 1,
    "dense_nnls": 2,
}
REQUIRED_SAMPLE_COLUMNS = (
    "objective",
    "objective_velocity",
    "inner_weight_objective",
    "density_chi2_per_bin",
    "effective_orbit_count",
    "max_weight_fraction",
    "active_orbit_count",
    "zero_weight_fraction",
    "weight_solver_converged",
    "weight_solver_kkt_residual",
    "weight_solver_wall_seconds",
)


def _maximum_rss_kib(run: Path) -> int | None:
    path = run / "benchmark_metadata" / "time-v.txt"
    if not path.exists():
        return None
    for line in path.read_text(errors="replace").splitlines():
        if "Maximum resident set size (kbytes)" not in line:
            continue
        try:
            return int(line.rsplit(":", 1)[1].strip())
        except (IndexError, ValueError):
            return None
    return None


def _numeric_column(samples: np.ndarray, name: str) -> np.ndarray:
    names = set(samples.dtype.names or ())
    if name not in names:
        raise ValueError(f"sample.dat is missing required column: {name}")
    return np.asarray(samples[name], dtype=float)


def _run_metrics(run_directory: str | Path) -> dict[str, object]:
    run = Path(run_directory).expanduser().resolve()
    summary = load_run_summary(run)
    weight_model = summary.config.get("weight_model")
    if not isinstance(weight_model, dict):
        raise ValueError(f"resolved config has no weight_model table: {run}")
    backend = str(weight_model.get("solver"))
    if backend not in SOLVER_BACKENDS:
        raise ValueError(f"unsupported solver backend {backend!r} in {run}")
    tolerance = float(weight_model.get("solver_tolerance", 1e-8))
    samples = summary.samples
    if samples.size != 1:
        raise ValueError(
            "each solver benchmark directory must contain exactly one "
            f"cold-start paper-best evaluation: {run}"
        )
    for name in REQUIRED_SAMPLE_COLUMNS:
        _numeric_column(samples, name)
    objective = _numeric_column(samples, "objective")
    objective_velocity = _numeric_column(samples, "objective_velocity")
    inner = _numeric_column(samples, "inner_weight_objective")
    density = _numeric_column(samples, "density_chi2_per_bin")
    effective = _numeric_column(samples, "effective_orbit_count")
    maximum_weight = _numeric_column(samples, "max_weight_fraction")
    active = _numeric_column(samples, "active_orbit_count")
    zero_weight = _numeric_column(samples, "zero_weight_fraction")
    converged = _numeric_column(samples, "weight_solver_converged") > 0.5
    kkt = _numeric_column(samples, "weight_solver_kkt_residual")
    wall = _numeric_column(samples, "weight_solver_wall_seconds")
    density_limit = summary.config.get("objective", {}).get(
        "density_max_chi2_per_bin"
    )
    density_gate = np.ones(samples.size, dtype=bool)
    if density_limit is not None:
        density_gate &= density <= float(density_limit)
    names = set(samples.dtype.names or ())
    if "density_shell_phi_gate_passed" in names:
        density_gate &= (
            _numeric_column(samples, "density_shell_phi_gate_passed") > 0.5
        )
    finite = np.ones(samples.size, dtype=bool)
    for values in (
        objective,
        objective_velocity,
        inner,
        density,
        effective,
        maximum_weight,
        active,
        zero_weight,
        kkt,
        wall,
    ):
        finite &= np.isfinite(values)
    finite &= objective < 1e30
    run_qualified = bool(
        np.all(finite)
        and np.all(converged)
        and np.all(density_gate)
        and np.all(kkt <= tolerance)
    )
    metadata = summary.best_metadata or {}
    return {
        "run_directory": str(run),
        "backend": backend,
        "repeats": int(samples.size),
        "solver_tolerance": tolerance,
        "problem_fingerprint": str(
            metadata.get("weight_problem_fingerprint", "")
        ),
        "median_wall_seconds": float(np.median(wall)),
        "wall_seconds": wall.tolist(),
        "maximum_rss_kib": _maximum_rss_kib(run),
        "median_inner_objective": float(np.median(inner)),
        "median_selected_objective": float(np.median(objective)),
        "median_velocity_objective": float(np.median(objective_velocity)),
        "maximum_kkt_residual": float(np.max(kkt)),
        "median_density_chi2_per_bin": float(np.median(density)),
        "median_effective_orbit_count": float(np.median(effective)),
        "median_active_orbit_count": float(np.median(active)),
        "median_maximum_weight_fraction": float(np.median(maximum_weight)),
        "median_zero_weight_fraction": float(np.median(zero_weight)),
        "finite": bool(np.all(finite)),
        "solver_converged": bool(np.all(converged)),
        "density_gates_passed": bool(np.all(density_gate)),
        "individual_qualified": run_qualified,
        "_objective": objective.tolist(),
        "_objective_velocity": objective_velocity.tolist(),
        "_inner": inner.tolist(),
        "_density": density.tolist(),
        "_effective": effective.tolist(),
        "_maximum_weight": maximum_weight.tolist(),
        "_active": active.tolist(),
        "_zero_weight": zero_weight.tolist(),
        "_kkt": kkt.tolist(),
    }


def _aggregate_backend_runs(items: list[dict[str, object]]) -> dict[str, object]:
    backend = str(items[0]["backend"])
    tolerance = float(items[0]["solver_tolerance"])
    directories = {str(item["run_directory"]) for item in items}
    if len(items) != 3 or len(directories) != 3:
        raise ValueError(
            f"solver {backend!r} requires exactly three distinct one-point runs"
        )

    def combined(name: str) -> np.ndarray:
        return np.concatenate(
            [np.asarray(item[name], dtype=float) for item in items]
        )

    wall = combined("wall_seconds")
    objective = combined("_objective")
    objective_velocity = combined("_objective_velocity")
    inner = combined("_inner")
    density = combined("_density")
    effective = combined("_effective")
    maximum_weight = combined("_maximum_weight")
    active = combined("_active")
    zero_weight = combined("_zero_weight")
    kkt = combined("_kkt")
    fingerprints = {str(item["problem_fingerprint"]) for item in items}
    rss_values = [
        int(item["maximum_rss_kib"])
        for item in items
        if item["maximum_rss_kib"] is not None
    ]
    repeats = int(wall.size)
    return {
        "run_directories": [str(item["run_directory"]) for item in items],
        "backend": backend,
        "repeats": repeats,
        "solver_tolerance": tolerance,
        "problem_fingerprint": (
            next(iter(fingerprints)) if len(fingerprints) == 1 else ""
        ),
        "median_wall_seconds": float(np.median(wall)),
        "wall_seconds": wall.tolist(),
        "maximum_rss_kib": max(rss_values) if rss_values else None,
        "median_inner_objective": float(np.median(inner)),
        "median_selected_objective": float(np.median(objective)),
        "median_velocity_objective": float(np.median(objective_velocity)),
        "maximum_kkt_residual": float(np.max(kkt)),
        "median_density_chi2_per_bin": float(np.median(density)),
        "median_effective_orbit_count": float(np.median(effective)),
        "median_active_orbit_count": float(np.median(active)),
        "median_maximum_weight_fraction": float(np.median(maximum_weight)),
        "median_zero_weight_fraction": float(np.median(zero_weight)),
        "finite": bool(all(item["finite"] for item in items)),
        "solver_converged": bool(
            all(item["solver_converged"] for item in items)
        ),
        "density_gates_passed": bool(
            all(item["density_gates_passed"] for item in items)
        ),
        "individual_qualified": bool(
            repeats == 3 and all(item["individual_qualified"] for item in items)
        ),
    }


def compare_solver_runs(
    run_directories: Iterable[str | Path],
) -> dict[str, object]:
    """Compare repeated paper-best runs and apply the promotion rules."""

    run_metrics = [_run_metrics(path) for path in run_directories]
    grouped = {
        backend: [
            item for item in run_metrics if str(item["backend"]) == backend
        ]
        for backend in SOLVER_BACKENDS
    }
    missing = sorted(backend for backend, items in grouped.items() if not items)
    if missing:
        raise ValueError("missing solver benchmark run(s): " + ", ".join(missing))
    metrics = [
        _aggregate_backend_runs(grouped[backend]) for backend in SOLVER_BACKENDS
    ]
    by_backend = {str(item["backend"]): item for item in metrics}

    fingerprints = {
        str(item["problem_fingerprint"])
        for item in metrics
        if item["problem_fingerprint"]
    }
    same_problem = len(fingerprints) == 1 and all(
        item["problem_fingerprint"] for item in metrics
    )
    numerically_qualified = [
        by_backend[name]
        for name in SOLVER_BACKENDS
        if by_backend[name]["individual_qualified"]
    ]
    objective_agreement = None
    if len(numerically_qualified) >= 2:
        objectives = np.asarray(
            [
                float(item["median_inner_objective"])
                for item in numerically_qualified
            ],
            dtype=float,
        )
        objective_agreement = float(
            (np.max(objectives) - np.min(objectives))
            / max(1.0, float(np.max(np.abs(objectives))))
        )
    agreement_passed = (
        objective_agreement is None or objective_agreement <= 1e-8
    )
    qualified = (
        numerically_qualified
        if same_problem and agreement_passed
        else []
    )

    winner = None
    selection_reason = "no backend passed every numerical gate"
    if qualified:
        qualified.sort(key=lambda item: float(item["median_wall_seconds"]))
        fastest = qualified[0]
        limit = 1.2 * float(fastest["median_wall_seconds"])
        contenders = [
            item
            for item in qualified
            if float(item["median_wall_seconds"]) <= limit
        ]
        with_memory = all(
            item["maximum_rss_kib"] is not None for item in contenders
        )
        if len(contenders) == 1:
            selected = fastest
            selection_reason = "selected fastest qualified backend"
        elif with_memory:
            selected = min(
                contenders,
                key=lambda item: (
                    int(item["maximum_rss_kib"]),
                    str(item["backend"]),
                ),
            )
            selection_reason = (
                "within the 20% wall-time band; selected lower peak RSS"
            )
        else:
            selected = min(
                contenders,
                key=lambda item: (
                    FORMULATION_MEMORY_RANK[str(item["backend"])],
                    str(item["backend"]),
                ),
            )
            selection_reason = (
                "within the 20% wall-time band; peak RSS unavailable, selected "
                "the lower-memory formulation"
            )
        winner = str(selected["backend"])

    for item in metrics:
        item["qualified_for_speed_selection"] = bool(item in qualified)
    return {
        "same_problem_fingerprint": same_problem,
        "inner_objective_max_relative_difference": objective_agreement,
        "inner_objective_agreement_passed": agreement_passed,
        "winner": winner,
        "selection_reason": selection_reason,
        "production_ready": False,
        "production_blocker": (
            "paper-best timing must be followed by the five fixed-potential "
            "ranking validation before changing the active recipe"
        ),
        "runs": {str(item["backend"]): item for item in metrics},
    }


def write_solver_comparison(
    run_directories: Iterable[str | Path],
    output: str | Path,
) -> dict[str, object]:
    result = compare_solver_runs(run_directories)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
