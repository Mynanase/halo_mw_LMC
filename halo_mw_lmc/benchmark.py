"""Preflight for the 8--40 kpc one-trial benchmark cases."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys

import numpy as np

from .configuration import RunConfiguration, load_run_configuration


R8_40_RUN_CONFIG_NAMES = frozenset(
    {
        "density_solved_r8_40_benchmark.toml",
        "density_solved_r8_40_tol1e7_benchmark.toml",
        "density_solved_r8_40_tol1e8_benchmark.toml",
        "density_solved_r8_40_reg1e5_benchmark.toml",
        "density_solved_r8_40_reg1e4_benchmark.toml",
        "density_solved_r8_40_potential_ranking_tol1e7.toml",
        "density_solved_r8_40_potential_ranking_tol1e8.toml",
    }
)
R8_40_CASE_PARAMETERS = {
    "density_solved_r8_40_benchmark.toml": (1e-6, 1e-6),
    "density_solved_r8_40_tol1e7_benchmark.toml": (1e-7, 1e-6),
    "density_solved_r8_40_tol1e8_benchmark.toml": (1e-8, 1e-6),
    "density_solved_r8_40_reg1e5_benchmark.toml": (1e-6, 1e-5),
    "density_solved_r8_40_reg1e4_benchmark.toml": (1e-6, 1e-4),
}
R8_40_POTENTIAL_RANKING_CASE_PARAMETERS = {
    "density_solved_r8_40_potential_ranking_tol1e7.toml": (1e-7, 1e-6),
    "density_solved_r8_40_potential_ranking_tol1e8.toml": (1e-8, 1e-6),
}
R8_40_POTENTIAL_RANKING_FIXED_POINTS = (
    (0.920, 0.800, 6.200, 9.890, 1.000),
    (0.820, 0.700, 6.200, 9.890, 1.000),
    (1.020, 0.950, 6.200, 9.890, 1.000),
    (0.920, 0.800, 6.500, 9.800, 1.200),
    (0.920, 0.800, 5.900, 10.050, 0.800),
)


@dataclass(frozen=True)
class BenchmarkPreflight:
    configuration: RunConfiguration


def validate_benchmark_preflight(
    repository: str | Path,
    config_path: str | Path,
    *,
    time_program: str | Path = "/usr/bin/time",
) -> BenchmarkPreflight:
    """Reject an invalid, non-GNU-time, or non-cold-start run."""

    root = Path(repository).resolve()
    config = Path(config_path)
    if not config.is_absolute():
        config = root / config
    config = config.resolve()
    allowed_directory = (root / "configs" / "runs").resolve()
    if config.parent != allowed_directory or config.name not in R8_40_RUN_CONFIG_NAMES:
        raise RuntimeError("the launcher only accepts the named 8--40 benchmark configs")

    time_path = Path(time_program)
    if not time_path.is_file() or not os.access(time_path, os.X_OK):
        raise RuntimeError(f"GNU time executable not found: {time_path}")
    try:
        version = subprocess.run(
            [str(time_path), "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"could not execute GNU time: {time_path}") from exc
    version_text = version.stdout + version.stderr
    if "gnu time" not in version_text.lower():
        raise RuntimeError(f"benchmark requires GNU time, not {time_path}")

    configuration = load_run_configuration(config)
    comparison = configuration.to_comparison_config()
    expected_shells = np.array([8, 10, 12, 15, 20, 30, 40], dtype=float)
    expected_velocity_edges = np.array(
        [4, 6, 8, 10, 12, 15, 20, 30, 40],
        dtype=float,
    )
    actual_shells = np.asarray(comparison.objective.density_shell_edges, dtype=float)
    ranking_case = config.name in R8_40_POTENTIAL_RANKING_CASE_PARAMETERS
    case_parameters = (
        R8_40_POTENTIAL_RANKING_CASE_PARAMETERS
        if ranking_case
        else R8_40_CASE_PARAMETERS
    )
    expected_tol, expected_regularization = case_parameters[config.name]
    expected_iterations = (
        len(R8_40_POTENTIAL_RANKING_FIXED_POINTS) if ranking_case else 1
    )
    expected_fixed_points = (
        R8_40_POTENTIAL_RANKING_FIXED_POINTS if ranking_case else None
    )
    if (
        configuration.iterations != expected_iterations
        or configuration.random_seed != 0
        or configuration.fixed_optimizer_points != expected_fixed_points
        or configuration.recipe.search.initial_point != "paper_best"
        or comparison.density_fit.min_spherical_radius != 8.0
        or comparison.density_fit.max_spherical_radius != 40.0
        or comparison.density_fit.min_abs_z != 2.0
        or comparison.velocity_fit_min_radius != 8.0
        or comparison.velocity_grid.radius_edges.shape
        != expected_velocity_edges.shape
        or not np.allclose(
            comparison.velocity_grid.radius_edges,
            expected_velocity_edges,
        )
        or actual_shells.shape != expected_shells.shape
        or not np.allclose(actual_shells, expected_shells)
        or comparison.objective.density_max_chi2_per_bin != 2.0
        or comparison.objective.density_shell_phi_max_chi2_per_bin != 2.0
        or comparison.weight_model.mode != "density_solved"
        or comparison.weight_model.lsmr_tol != expected_tol
        or comparison.weight_model.regularization_strength
        != expected_regularization
        or comparison.orbit_periods != 10.0
        or comparison.orbit_samples_per_orbit != 1000
    ):
        raise RuntimeError("run config does not match its named 8--40 benchmark")
    if configuration.output_dir.exists():
        raise RuntimeError(
            f"cold-start output directory already exists: {configuration.output_dir}"
        )
    for label, path in (
        ("catalogue", configuration.data.catalog),
        ("target density", configuration.data.target_density),
    ):
        if not path.is_file():
            raise RuntimeError(f"{label} not found: {path}")
    return BenchmarkPreflight(configuration=configuration)


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        print(
            "usage: python -m halo_mw_lmc.benchmark RUN_CONFIG",
            file=sys.stderr,
        )
        return 2
    repository = Path(__file__).resolve().parents[1]
    try:
        result = validate_benchmark_preflight(
            repository,
            arguments[0],
        )
    except (RuntimeError, ValueError) as exc:
        print(f"benchmark preflight failed: {exc}", file=sys.stderr)
        return 1
    configuration = result.configuration
    for value in (
        configuration.run_id,
        configuration.output_dir,
        configuration.data.catalog,
        configuration.data.target_density,
    ):
        print(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
