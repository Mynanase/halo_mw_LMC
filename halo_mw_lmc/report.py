"""Managed static reports generated exclusively from saved run artifacts."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .artifacts import (
    load_best_evaluation,
    load_resolved_config,
    load_sample_table,
)
from .config import RunConfiguration
from .inspection import inspect_run, save_inspection
from .plot_convergence import build_convergence_figure
from .plot_model import (
    plot_density_comparison,
    plot_density_phi_pages,
    plot_density_shape,
    plot_density_shell_gate,
    plot_velocity_distributions,
)
from .plot_constraints import (
    build_parameter_constraints_corner_figure,
    build_parameter_constraints_figure,
    persist_corner_surfaces,
    search_bounds_from_resolved_config,
)
from .plot_weights import plot_orbit_weight_histograms

from .plot_coverage import plot_all_data_coverage
from .prepare import PreparedCoverage, preflight_and_prepare, require_preflight


REPORT_MANIFEST_SCHEMA_VERSION = 1


def _git_provenance(repository: Path) -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": bool(status.strip())}


def _velocity_bin_factor(config: dict[str, object]) -> int:
    report = config.get("report")
    if not isinstance(report, dict):
        raise ValueError("resolved config has no report settings")
    value = report.get("velocity_bin_factor")
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("resolved report velocity_bin_factor must be positive")
    return value


def _rename_generated(paths: list[Path], prefix: str) -> list[Path]:
    renamed: list[Path] = []
    for path in paths:
        name = path.name
        if prefix == "density" and name.startswith("density_phi"):
            destination = path.with_name(name.replace("density_phi", "phi_", 1))
        elif prefix == "velocity" and name == "velocity_phi_average.pdf":
            destination = path.with_name("phi_average.pdf")
        elif prefix == "velocity" and name.startswith("velocity_phi"):
            destination = path.with_name(name.replace("velocity_phi", "phi_", 1))
        else:
            destination = path
        if destination != path:
            path.replace(destination)
        renamed.append(destination)
    return renamed


def _summary_markdown(run: Path, config: dict[str, object], best) -> str:
    metadata = best.metadata
    lines = [
        "# Run report",
        "",
        f"- Run directory: `{run}`",
        f"- Best generation: `{metadata.get('generation')}`",
        f"- Best iteration: {metadata.get('iteration')}",
        f"- Selected objective: {metadata.get('objective')}",
        f"- Objective mode: {metadata.get('objective_mode')}",
        f"- Density chi2 per bin: {metadata.get('density_chi2_per_bin')}",
        f"- Density global gate: {metadata.get('density_gate_passed')}",
        f"- Density shell-phi gate: {metadata.get('density_shell_phi_gate_passed')}",
        f"- Weight mode: {metadata.get('weight_mode')}",
        f"- Weight solver converged: {best.weight_solution.converged}",
        f"- Successful/failed orbits: {metadata.get('successful_orbits')} / {metadata.get('failed_orbits')}",
        "",
        "## Best parameters",
        "",
        "```json",
        json.dumps(metadata.get("parameters"), indent=2, sort_keys=True),
        "```",
        "",
        "## Orbit support",
        "",
        "```json",
        json.dumps(metadata.get("orbit_support_audit"), indent=2, sort_keys=True),
        "```",
        "",
        "## Numerical provenance",
        "",
        f"- Git commit: `{config.get('git_commit')}`",
        f"- Dirty checkout: {config.get('git_dirty')}",
        "",
        "This report is derived from resolved configuration, scalar samples, and the saved best snapshot only.",
    ]
    return "\n".join(lines) + "\n"


def _render_report(run: Path, staging: Path) -> dict[str, object]:
    config = load_resolved_config(run)
    samples = load_sample_table(
        run / "sample.dat",
        required_columns=("iteration", "objective"),
    )
    best = load_best_evaluation(run)
    factor = _velocity_bin_factor(config)
    density_directory = staging / "density"
    velocity_directory = staging / "velocity"
    weights_directory = staging / "weights"
    density_directory.mkdir(parents=True)

    plot_density_comparison(best.density, density_directory / "overview.pdf")
    plot_density_shape(best.density, density_directory / "flattening.pdf")
    omitted: list[str] = []
    if best.density_shells is not None:
        plot_density_shell_gate(
            best.density_shells,
            best.density.grid.phi_edges,
            density_directory / "shell_phi_gate.pdf",
            limit=best.metadata.get("density_shell_phi_max_chi2_per_bin"),
        )
    else:
        omitted.append("density/shell_phi_gate.pdf: no saved shell diagnostics")
    _rename_generated(
        plot_density_phi_pages(best.density, density_directory),
        "density",
    )
    if best.velocity_distributions:
        _rename_generated(
            plot_velocity_distributions(
                best.velocity_distributions,
                velocity_directory,
                velocity_bin_factor=factor,
            ),
            "velocity",
        )
    else:
        omitted.append("velocity/: no saved velocity distributions")

    plot_orbit_weight_histograms(
        {"best": best.weight_solution.seed_weights},
        weights_directory / "distribution.pdf",
        title="Best-evaluation orbit-weight distribution",
    )
    convergence = build_convergence_figure(samples)
    convergence.savefig(staging / "convergence.pdf", bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(convergence)
    constraints = build_parameter_constraints_figure(samples, search_bounds_from_resolved_config(config))
    constraints.savefig(staging / "parameter_constraints.pdf", bbox_inches="tight")
    plt.close(constraints)
    try:
        corner, corner_surfaces = build_parameter_constraints_corner_figure(
            samples,
            search_bounds_from_resolved_config(config),
            return_artifacts=True,
        )
        corner.savefig(
            staging / "parameter_constraints_corner.pdf",
            bbox_inches="tight",
        )
        plt.close(corner)
        try:
            if corner_surfaces:
                persist_corner_surfaces(
                    corner_surfaces,
                    staging / "parameter_constraints_corner_surfaces.npz",
                )
        except Exception as exc:
            omitted.append(
                "parameter_constraints_corner_surfaces.npz: "
                f"{type(exc).__name__}: {exc}"
            )
    except Exception as exc:
        omitted.append(f"parameter_constraints_corner.pdf: {type(exc).__name__}: {exc}")
    (staging / "summary.md").write_text(_summary_markdown(run, config, best), encoding="utf-8")
    files = sorted(
        str(path.relative_to(staging))
        for path in staging.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    empty = [relative for relative in files if (staging / relative).stat().st_size == 0]
    if not files or empty:
        raise RuntimeError(f"report staging validation failed; empty files: {empty}")
    repository = Path(__file__).resolve().parents[2]
    manifest = {
        "schema_version": REPORT_MANIFEST_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "best_generation": best.metadata.get("generation"),
        "resolved_config_schema": config.get("schema_version"),
        "velocity_display_factor": factor,
        "package_version": __version__,
        "report_git": _git_provenance(repository),
        "files": files,
        "omitted": omitted,
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def generate_report_from_run(
    run_directory: str | Path,
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Render and atomically publish the managed ``report/`` directory."""

    if importlib.util.find_spec("matplotlib") is None:
        raise RuntimeError("Matplotlib is required to generate a static report")
    run = Path(run_directory).expanduser().resolve()
    destination = run / "report"
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"managed report already exists: {destination}; use --overwrite"
        )
    staging = Path(tempfile.mkdtemp(prefix=".report-staging-", dir=run))
    backup = run / f".report-backup-{uuid.uuid4().hex}"
    try:
        _render_report(run, staging)
        if destination.exists():
            destination.replace(backup)
        try:
            staging.replace(destination)
        except Exception:
            if backup.exists() and not destination.exists():
                backup.replace(destination)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    save_inspection(inspect_run(run))
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError(f"invalid report manifest: {manifest_path}")
    return [destination / relative for relative in manifest["files"]]


def generate_report(configuration: RunConfiguration) -> list[Path]:
    """Compatibility wrapper for callers that still hold a run configuration."""

    return generate_report_from_run(configuration.output_dir)


def generate_coverage_report(
    configuration: RunConfiguration,
    prepared: PreparedCoverage | None = None,
) -> list[Path]:
    """Measure and render raw catalogue coverage from one run configuration."""

    catalog_path = configuration.data.catalog
    output_directory = configuration.coverage.output_dir
    if prepared is None:
        result = require_preflight(preflight_and_prepare(configuration, stage="coverage"))
        prepared = result.coverage
    if prepared is None:
        raise RuntimeError("coverage preflight did not return prepared data")
    if prepared.configuration != configuration:
        raise ValueError("prepared coverage belongs to a different configuration")
    comparison = configuration.to_comparison_config()
    grid = comparison.density_grid
    coverage = prepared.coverage
    output_directory.mkdir(parents=True, exist_ok=False)
    written = plot_all_data_coverage(
        coverage,
        output_directory,
        spatial_limit=float(max(grid.r_edges[-1], np.max(np.abs(grid.z_edges)))),
        velocity_limit=configuration.coverage.velocity_limit_km_s,
        maximum_points=configuration.coverage.maximum_points,
        random_state=configuration.coverage.random_seed,
    )

    summary = {
        "catalog_path": str(catalog_path),
        "density_interpretation": "raw catalogue sampling density; no selection-function correction",
        "configuration": {
            "r_edges_kpc": grid.r_edges.tolist(),
            "z_edges_kpc": grid.z_edges.tolist(),
            "phi_edges_rad": grid.phi_edges.tolist(),
            "velocity_display_limit_km_s": configuration.coverage.velocity_limit_km_s,
            "random_seed": configuration.coverage.random_seed,
        },
        **coverage.summary(),
    }
    summary_path = output_directory / "coverage_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    counts_path = output_directory / "coverage_counts.npz"
    np.savez_compressed(
        counts_path,
        rzphi_counts=coverage.rzphi_counts,
        rzphi_sampling_density=coverage.rzphi_sampling_density,
        r_edges=coverage.rzphi_grid.r_edges,
        z_edges=coverage.rzphi_grid.z_edges,
        phi_edges=coverage.phi_edges,
        rtheta_phi_counts=coverage.rtheta_phi_counts,
        rtheta_phi_sampling_density=coverage.rtheta_phi_sampling_density,
        spherical_radius_edges=coverage.spherical_radius_edges,
        theta_edges=coverage.theta_edges,
    )
    return [*written, summary_path, counts_path]
