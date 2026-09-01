"""Reconstruct compact run status from authoritative numerical artifacts."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np

from .artifacts import (
    SUPPORTED_RESOLVED_CONFIG_SCHEMA_VERSIONS,
    best_sample,
    load_best_evaluation,
    load_sample_table,
)


INSPECTION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RunInspection:
    """JSON-safe, derived view of one run directory."""

    run_directory: Path
    document: Mapping[str, object]

    @property
    def numerical_status(self) -> str:
        return str(self.document["numerical_status"])

    @property
    def report_status(self) -> str:
        return str(self.document["report_status"])

    @property
    def valid(self) -> bool:
        return self.numerical_status != "invalid"


def _read_json(path: Path) -> dict[str, object]:
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read JSON artifact {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return document


def _safe_int(value, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _report_state(
    run: Path,
    *,
    best_generation: str | None,
    transient_failure: str | None,
) -> tuple[str, dict[str, object], list[str]]:
    if transient_failure is not None:
        return (
            "failed",
            {"directory": str(run / "report"), "error": transient_failure},
            [],
        )
    report = run / "report"
    if not report.exists():
        return "missing", {"directory": str(report)}, []
    manifest_path = report / "manifest.json"
    if not manifest_path.exists():
        return (
            "invalid",
            {"directory": str(report), "manifest": str(manifest_path)},
            ["report directory has no manifest.json"],
        )
    try:
        manifest = _read_json(manifest_path)
    except ValueError as exc:
        return "invalid", {"directory": str(report)}, [str(exc)]
    if manifest.get("schema_version") != 1:
        return "invalid", {"manifest": manifest}, ["unsupported report manifest schema"]
    manifest_generation = manifest.get("best_generation")
    if best_generation is None or manifest_generation != best_generation:
        return "stale", {"manifest": manifest}, []
    files = manifest.get("files")
    if not isinstance(files, list) or not all(isinstance(item, str) for item in files):
        return "invalid", {"manifest": manifest}, ["report manifest files must be a list"]
    invalid_files = []
    for relative in files:
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            invalid_files.append(relative)
            continue
        candidate = report / relative
        if not candidate.is_file() or candidate.stat().st_size == 0:
            invalid_files.append(relative)
    if invalid_files:
        return (
            "invalid",
            {"manifest": manifest, "invalid_files": invalid_files},
            ["report manifest names missing or empty files"],
        )
    return "current", {"manifest": manifest}, []


def inspect_run(
    run_directory: str | Path,
    *,
    report_failure: str | None = None,
) -> RunInspection:
    """Recompute status without trusting an existing ``inspection.json``."""

    run = Path(run_directory).expanduser().resolve()
    warnings: list[str] = []
    errors: list[str] = []
    artifacts: dict[str, object] = {
        "resolved_config": False,
        "sample": False,
        "best_metadata": False,
        "best_evaluation": False,
        "best_matches_sample": False,
    }
    schedule = None
    planned = None
    completed = 0
    config: dict[str, object] | None = None
    config_schema = None
    provenance: dict[str, object] = {"git_commit": None, "git_dirty": None}

    config_path = run / "resolved_config.json"
    if not config_path.exists():
        errors.append(f"resolved config not found: {config_path}")
    else:
        try:
            config = _read_json(config_path)
            config_schema = config.get("schema_version")
            if config_schema not in SUPPORTED_RESOLVED_CONFIG_SCHEMA_VERSIONS:
                raise ValueError(f"unsupported resolved-config schema: {config_schema}")
            artifacts["resolved_config"] = True
            optimizer = config.get("optimizer", {})
            if not isinstance(optimizer, dict):
                raise ValueError("resolved config optimizer must be an object")
            schedule = optimizer.get("schedule")
            if schedule not in {"fixed_points", "adaptive"}:
                fixed = optimizer.get("fixed_points")
                schedule = "fixed_points" if fixed is not None else "adaptive"
            planned = _safe_int(optimizer.get("iterations"))
            if planned is None or planned < 1:
                raise ValueError("resolved config has no positive optimizer iteration count")
            provenance = {
                "git_commit": config.get("git_commit"),
                "git_dirty": config.get("git_dirty"),
            }
        except ValueError as exc:
            errors.append(str(exc))

    samples = None
    current_best = None
    sample_path = run / "sample.dat"
    if sample_path.exists():
        try:
            samples = load_sample_table(
                sample_path,
                required_columns=("iteration", "objective"),
            )
            artifacts["sample"] = True
            completed = int(samples.size)
            current_best = best_sample(samples)
            iterations = np.asarray(samples["iteration"], dtype=int)
            if np.unique(iterations).size != iterations.size:
                errors.append("sample.dat contains duplicate iteration values")
            if planned is not None and completed > planned:
                errors.append("sample.dat contains more trials than planned")
        except ValueError as exc:
            errors.append(str(exc))
    elif artifacts["resolved_config"]:
        warnings.append("sample.dat has not been created")

    metadata_path = run / "best" / "metadata.json"
    evaluation_path = run / "best" / "evaluation.npz"
    metadata = None
    best = None
    if metadata_path.exists() != evaluation_path.exists():
        errors.append("best snapshot is missing either metadata.json or evaluation.npz")
    elif metadata_path.exists():
        try:
            metadata = _read_json(metadata_path)
            artifacts["best_metadata"] = True
            best = load_best_evaluation(run)
            artifacts["best_evaluation"] = True
            if current_best is not None:
                matches = (
                    _safe_int(metadata.get("iteration"))
                    == int(current_best["iteration"])
                    and np.isclose(
                        float(metadata.get("objective")),
                        float(current_best["objective"]),
                        rtol=1e-7,
                        atol=1e-10,
                    )
                )
                artifacts["best_matches_sample"] = bool(matches)
                if not matches:
                    warnings.append("best snapshot is stale relative to sample.dat")
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
    elif artifacts["sample"]:
        warnings.append("best snapshot has not been published")

    incomplete_reasons = []
    if artifacts["resolved_config"] and not artifacts["sample"]:
        incomplete_reasons.append("sample missing")
    if artifacts["sample"] and not artifacts["best_evaluation"]:
        incomplete_reasons.append("best snapshot missing")
    if planned is not None and completed < planned:
        incomplete_reasons.append("planned trials not complete")
    if artifacts["best_evaluation"] and not artifacts["best_matches_sample"]:
        incomplete_reasons.append("best snapshot stale")
    numerical_status = (
        "invalid"
        if errors
        else "incomplete"
        if incomplete_reasons
        else "complete"
    )

    best_generation = str(metadata.get("generation")) if metadata else None
    report_status, report, report_warnings = _report_state(
        run,
        best_generation=best_generation,
        transient_failure=report_failure,
    )
    warnings.extend(report_warnings)

    best_document = None
    density_document = None
    weight_document = None
    orbit_document = None
    if metadata is not None:
        density_chi2_per_bin = _safe_float(
            metadata.get("density_chi2_per_bin")
        )
        density_global_limit = _safe_float(
            metadata.get("density_max_chi2_per_bin")
        )
        density_global_gate = metadata.get("density_gate_passed")
        if (
            density_global_gate is None
            and density_chi2_per_bin is not None
            and density_global_limit is not None
        ):
            density_global_gate = density_chi2_per_bin <= density_global_limit
        best_document = {
            "generation": metadata.get("generation"),
            "iteration": _safe_int(metadata.get("iteration")),
            "parameters": metadata.get("parameters"),
            "objective": _safe_float(metadata.get("objective")),
            "objective_velocity": _safe_float(metadata.get("objective_velocity")),
            "objective_density_velocity": _safe_float(
                metadata.get("objective_density_velocity")
            ),
        }
        density_document = {
            "chi2_per_bin": density_chi2_per_bin,
            "global_limit": density_global_limit,
            "global_gate_passed": density_global_gate,
            "shell_phi_limit": _safe_float(
                metadata.get("density_shell_phi_max_chi2_per_bin")
            ),
            "shell_phi_gate_passed": metadata.get(
                "density_shell_phi_gate_passed"
            ),
            "worst_shell_phi_chi2_per_bin": _safe_float(
                metadata.get("density_worst_shell_phi_chi2_per_bin")
            ),
            "worst_shell_phi_index": metadata.get("density_worst_shell_phi_index"),
        }
        weight_document = {
            "mode": metadata.get("weight_mode"),
            "solver_backend": metadata.get("weight_solver_backend"),
            "solver_converged": bool(best.weight_solution.converged) if best else None,
            "solver_status": int(best.weight_solution.status) if best else None,
            "solver_iterations": _safe_int(metadata.get("weight_solver_iterations")),
            "solver_optimality": _safe_float(metadata.get("weight_solver_optimality")),
            "solver_cost": _safe_float(metadata.get("weight_solver_cost")),
            "solver_kkt_residual": _safe_float(
                metadata.get("weight_solver_kkt_residual")
            ),
            "solver_wall_seconds": _safe_float(
                metadata.get("weight_solver_wall_seconds")
            ),
            "problem_fingerprint": metadata.get("weight_problem_fingerprint"),
            "message": metadata.get("weight_solver_message"),
        }
        orbit_document = {
            "seed": _safe_int(metadata.get("seed_orbits")),
            "successful": _safe_int(metadata.get("successful_orbits")),
            "failed": _safe_int(metadata.get("failed_orbits")),
            "support": metadata.get("orbit_support_audit"),
        }

    document = {
        "schema_version": INSPECTION_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_directory": str(run),
        "numerical_status": numerical_status,
        "report_status": report_status,
        "schedule": schedule,
        "trials": {"planned": planned, "completed": completed},
        "best": best_document,
        "density": density_document,
        "weights": weight_document,
        "orbits": orbit_document,
        "artifacts": {
            **artifacts,
            "resolved_config_schema": config_schema,
        },
        "provenance": provenance,
        "report": report,
        "warnings": warnings,
        "errors": errors,
    }
    return RunInspection(run, document)


def save_inspection(inspection: RunInspection) -> Path:
    """Atomically refresh the derived ``inspection.json`` cache."""

    destination = inspection.run_directory / "inspection.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=".inspection.",
        suffix=".json.tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(inspection.document, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(destination)
    return destination
