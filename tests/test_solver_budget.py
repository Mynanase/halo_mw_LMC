"""Small-array contract tests for the solver-budget experiment (step 1)."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from halo_mw_lmc.config import (
    DensityFitSettings,
    ObjectiveSettings,
    WeightModelSettings,
    ZhuComparisonConfig,
)
from halo_mw_lmc.grids import CylindricalGrid
from halo_mw_lmc.density import build_orbit_density_response
from halo_mw_lmc.orbits import OrbitLibrary
from halo_mw_lmc.orbits import cartesian_to_spherical_phase_space
from halo_mw_lmc.potential import ZhuHaloParameters
from halo_mw_lmc.catalogue import SeedCatalogue
from halo_mw_lmc.evaluate import (
    INVALID_TRIAL_PENALTY,
    evaluate_orbit_library,
    evaluate_prepared_model,
)
from halo_mw_lmc.prepare import PreparedFixedWeightData
from halo_mw_lmc.solver_budget import (
    SOLVER_BUDGET_PHASES,
    case_directory,
    existing_case_directories,
    load_solver_budget_plan,
    require_absent_case,
    resolved_weight_settings,
    run_solver_budget_phase,
)


REPOSITORY = Path(__file__).resolve().parents[1]
BENCHMARK_CONFIG = REPOSITORY / "configs" / "benchmarks" / "r8_40_solver_budget.toml"
BENCHMARK_SCRIPT = REPOSITORY / "scripts" / "benchmark_density_solved_r8_40_solver_budget.py"
COMPARATOR_SCRIPT = REPOSITORY / "scripts" / "compare_density_solved_r8_40_solver_budget.py"
OUTPUT_ROOT_LINE = 'output_root = "../../.agent-local/benchmarks/r8_40_solver_budget_v1"'

VELOCITY_STUB = (
    {"vr": -2.0, "vphi": -1.0, "vtheta": -3.0},
    {name: np.array([-value / 2, -value / 2]) for name, value in (("vr", 2), ("vphi", 1), ("vtheta", 3))},
    {name: np.array([1, 1]) for name in ("vr", "vphi", "vtheta")},
    {},
)


def _initial_conditions() -> np.ndarray:
    return np.array(
        [
            [0.5, -0.5, 0.5, 0, 0, 0],
            [0.2, 0.0, 0.5, 0, 0, 0],
            [0.5, 0.5, 0.5, 0, 0, 0],
        ],
        dtype=float,
    )


def _toy_library() -> OrbitLibrary:
    initial = _initial_conditions()
    return OrbitLibrary(
        seed_index=np.array([0, 0, 2, 2, 2], dtype=np.int64),
        time=np.arange(5, dtype=float),
        phase_space=np.array([initial[0], initial[0], initial[2], initial[2], initial[2]]),
    )


def _toy_prepared(mode: str = "density_solved") -> PreparedFixedWeightData:
    initial = _initial_conditions()
    grid = CylindricalGrid.uniform(n_r=1, r_range=(0.0, 1.0), n_z=1, z_range=(0.0, 1.0), n_phi=2)
    if mode == "density_solved":
        config = ZhuComparisonConfig(
            density_grid=grid,
            density_fit=DensityFitSettings(
                min_abs_z=0,
                min_spherical_radius=0,
                max_spherical_radius=10,
                normalization_min_radius=0,
                normalization="none",
            ),
            include_velocity=True,
            orbit_samples_per_orbit=3,
            weight_model=WeightModelSettings(
                mode="density_solved",
                solver="lsq_linear",
                target_normalization="absolute",
                regularization="l2",
                regularization_strength=0.0,
            ),
            objective=ObjectiveSettings(mode="velocity_only", density_max_chi2_per_bin=1.0),
        )
        seed_weights = None
        target_density = np.array([[[2.0, 3.0]]]) / grid.volumes
        target_error = np.full(grid.shape, 0.01)
    else:
        config = ZhuComparisonConfig(
            density_grid=grid,
            density_fit=DensityFitSettings(
                min_abs_z=0,
                min_spherical_radius=0,
                max_spherical_radius=10,
                normalization_min_radius=0,
            ),
            orbit_samples_per_orbit=1,
            orbit_sample_divisor=1,
        )
        seed_weights = np.array([1.0, 2.0, 4.0])
        target_density = np.ones(grid.shape)
        target_error = np.ones(grid.shape)
    catalogue = SeedCatalogue(
        initial_conditions=initial,
        seed_weights=seed_weights,
        velocity_errors={},
    )
    return PreparedFixedWeightData(
        catalogue=catalogue,
        target_density=target_density,
        target_error=target_error,
        config=config,
        catalog_path=Path("synthetic-catalog"),
        density_path=Path("synthetic-target"),
        catalog_phase_space=cartesian_to_spherical_phase_space(
            *[initial[:, index] for index in range(6)]
        ),
    )


def _temp_benchmark_config(
    directory: Path,
    *,
    output_root: Path | None = None,
    replacements: tuple[tuple[str, str], ...] = (),
) -> Path:
    text = BENCHMARK_CONFIG.read_text()
    text = text.replace('recipe = "../recipes/', f'recipe = "{REPOSITORY}/configs/recipes/')
    text = text.replace('catalog = "../../', f'catalog = "{REPOSITORY}/')
    text = text.replace('target_density = "../../', f'target_density = "{REPOSITORY}/')
    if output_root is not None:
        text = text.replace(OUTPUT_ROOT_LINE, f'output_root = "{output_root}"')
    for old, new in replacements:
        text = text.replace(old, new)
    path = directory / "bench.toml"
    path.write_text(text)
    return path


def _load_script_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SharedEvaluationBoundaryTests(unittest.TestCase):
    def test_prepared_and_library_paths_agree(self):
        library = _toy_library()
        prepared = _toy_prepared("density_solved")
        parameters = ZhuHaloParameters(rho0=6.0, log_rs=1.0, phalo=1.0, qhalo=1.0, gamma=1.0)
        with (
            patch("halo_mw_lmc.evaluate.build_potential_from_parameters", return_value=object()),
            patch("halo_mw_lmc.evaluate.integrate_agama_orbits", return_value=library),
            patch("halo_mw_lmc.evaluate._score_velocities", return_value=VELOCITY_STUB),
        ):
            from_prepared = evaluate_prepared_model(parameters, prepared)
            from_library = evaluate_orbit_library(library, prepared)

        self.assertAlmostEqual(from_prepared.density.chi2, from_library.density.chi2)
        self.assertAlmostEqual(from_prepared.log_likelihood, from_library.log_likelihood)
        self.assertAlmostEqual(from_prepared.objective_velocity, from_library.objective_velocity)
        self.assertAlmostEqual(
            from_prepared.objective_density_velocity,
            from_library.objective_density_velocity,
        )
        self.assertEqual(from_prepared.successful_orbits, from_library.successful_orbits)
        self.assertAlmostEqual(from_prepared.weight_sum, from_library.weight_sum)
        np.testing.assert_allclose(
            from_prepared.weight_solution.seed_weights,
            from_library.weight_solution.seed_weights,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(from_prepared.density.model_density, from_library.density.model_density)

    def test_external_response_is_reused_without_rebuilding(self):
        library = _toy_library()
        prepared = _toy_prepared("density_solved")
        response = build_orbit_density_response(
            library,
            prepared.config.density_grid,
            seed_count=prepared.initial_conditions.shape[0],
        )
        with patch("halo_mw_lmc.evaluate._score_velocities", return_value=VELOCITY_STUB):
            with patch(
                "halo_mw_lmc.evaluate.build_orbit_density_response",
                side_effect=AssertionError("frozen response was rebuilt"),
            ):
                reused = evaluate_orbit_library(library, prepared, response=response)
            rebuilt = evaluate_orbit_library(library, prepared)

        self.assertAlmostEqual(reused.objective_density_velocity, rebuilt.objective_density_velocity)
        np.testing.assert_allclose(reused.density.model_density, rebuilt.density.model_density)

    def test_response_validation_rejects_mismatched_provenance(self):
        library = _toy_library()
        prepared = _toy_prepared("density_solved")
        good = build_orbit_density_response(
            library,
            prepared.config.density_grid,
            seed_count=prepared.initial_conditions.shape[0],
        )
        other_grid = CylindricalGrid.uniform(n_r=2, r_range=(0.0, 1.0), n_z=1, z_range=(0.0, 1.0), n_phi=2)
        wrong_grid = build_orbit_density_response(
            library,
            other_grid,
            seed_count=prepared.initial_conditions.shape[0],
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            evaluate_orbit_library(library, prepared, response=wrong_grid)
        wrong_columns = replace(good, successful_seed_index=np.array([0, 1], dtype=np.int64))
        with self.assertRaisesRegex(ValueError, "column mapping"):
            evaluate_orbit_library(library, prepared, response=wrong_columns)
        wrong_seed_count = replace(good, seed_count=prepared.initial_conditions.shape[0] + 1)
        with self.assertRaisesRegex(ValueError, "seed_count"):
            evaluate_orbit_library(library, prepared, response=wrong_seed_count)
        non_finite_counts = replace(good, sample_count=np.array([2.0, np.nan]))
        with self.assertRaisesRegex(ValueError, "sample counts must be finite"):
            evaluate_orbit_library(library, prepared, response=non_finite_counts)
        fractional_counts = replace(good, sample_count=np.array([2.5, 3.0]))
        with self.assertRaisesRegex(ValueError, "non-negative integers"):
            evaluate_orbit_library(library, prepared, response=fractional_counts)
        truncated_counts = replace(good, sample_count=np.array([2.0, 2.9999999]))
        with self.assertRaisesRegex(ValueError, "non-negative integers"):
            evaluate_orbit_library(library, prepared, response=truncated_counts)
        wrong_counts = replace(good, sample_count=np.array([2, 1]))
        with self.assertRaisesRegex(ValueError, "provenance"):
            evaluate_orbit_library(library, prepared, response=wrong_counts)
        from scipy.sparse import csr_matrix

        non_finite_matrix = replace(
            good,
            matrix=csr_matrix(np.full(good.matrix.shape, np.nan)),
        )
        with self.assertRaisesRegex(ValueError, "non-finite entries"):
            evaluate_orbit_library(library, prepared, response=non_finite_matrix)

    def test_fixed_weight_scoring_rejects_orbit_response(self):
        library = _toy_library()
        prepared = _toy_prepared("catalogue_fixed")
        response = build_orbit_density_response(
            library,
            prepared.config.density_grid,
            seed_count=prepared.initial_conditions.shape[0],
        )
        with self.assertRaisesRegex(ValueError, "catalogue_fixed"):
            evaluate_orbit_library(library, prepared, response=response)

    def test_selected_objective_stays_separate_from_raw_j(self):
        library = _toy_library()
        prepared = _toy_prepared("density_solved")
        with patch("halo_mw_lmc.evaluate._score_velocities", return_value=VELOCITY_STUB):
            accepted = evaluate_orbit_library(library, prepared)
        rejected = replace(
            accepted,
            weight_solution=replace(accepted.weight_solution, converged=False),
        )
        self.assertGreaterEqual(rejected.selected_objective, INVALID_TRIAL_PENALTY)
        self.assertAlmostEqual(
            rejected.objective_density_velocity,
            accepted.objective_density_velocity,
        )
        self.assertLess(accepted.selected_objective, INVALID_TRIAL_PENALTY)


class SolverBudgetConfigurationTests(unittest.TestCase):
    def test_plan_records_contract_points_methods_and_gates(self):
        plan = load_solver_budget_plan(BENCHMARK_CONFIG)
        self.assertEqual(
            [point.name for point in plan.points],
            ["paper_best", "accepted_best", "capped_candidate"],
        )
        by_name = {point.name: point.coordinates for point in plan.points}
        self.assertEqual(by_name["paper_best"], (0.920, 0.800, 6.200, 9.890, 1.000))
        self.assertEqual(by_name["accepted_best"], (0.985, 0.714, 6.099, 9.829, 0.751))
        self.assertEqual(by_name["capped_candidate"], (1.071, 0.738, 5.687, 9.541, 1.214))
        self.assertEqual([method.name for method in plan.methods], ["trf_default", "dense_reference"])
        self.assertEqual(plan.methods[0].solver, "lsq_linear")
        self.assertEqual(plan.methods[1].solver, "dense_nnls")
        self.assertEqual(plan.budget_max_iter, (300, 1000, 5000))
        self.assertEqual(plan.phases, SOLVER_BUDGET_PHASES)
        self.assertEqual(plan.repeats, 3)
        self.assertAlmostEqual(plan.timeout_seconds, 3600.0)
        self.assertEqual(dict(plan.threads), {"openblas": 1, "mkl": 1, "blis": 1, "omp": 16})
        self.assertEqual(plan.benchmark_id, "r8_40_solver_budget_v1")
        self.assertEqual(
            plan.output_root,
            (REPOSITORY / ".agent-local" / "benchmarks" / "r8_40_solver_budget_v1").resolve(),
        )

    def test_method_overrides_change_only_solver_settings(self):
        plan = load_solver_budget_plan(BENCHMARK_CONFIG)
        base = plan.recipe.weight_model
        for method in plan.methods:
            settings = resolved_weight_settings(plan.recipe, method)
            self.assertEqual(settings.mode, base.mode)
            self.assertEqual(settings.target_normalization, base.target_normalization)
            self.assertEqual(settings.regularization, base.regularization)
            self.assertEqual(settings.regularization_strength, base.regularization_strength)
            self.assertEqual(settings.solver, method.solver)
            self.assertEqual(settings.max_iter, method.max_iter)
            self.assertEqual(
                settings,
                replace(
                    base,
                    solver=method.solver,
                    max_iter=method.max_iter,
                    lsmr_tol=method.lsmr_tol,
                    solver_tolerance=method.solver_tolerance,
                ),
            )

    def test_configuration_rejects_out_of_bounds_point(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _temp_benchmark_config(
                Path(tmp),
                replacements=(
                    (
                        "coordinates = [0.920, 0.800, 6.200, 9.890, 1.000]",
                        "coordinates = [0.920, 0.800, 6.200, 9.890, 3.500]",
                    ),
                ),
            )
            with self.assertRaisesRegex(ValueError, "outside the recipe bounds"):
                load_solver_budget_plan(config)

    def test_configuration_rejects_loosened_repeat_tolerance(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _temp_benchmark_config(
                Path(tmp),
                replacements=(("repeat_rtol = 1e-12", "repeat_rtol = 1e-6"),),
            )
            with self.assertRaisesRegex(ValueError, "cannot be loosened"):
                load_solver_budget_plan(config)


class SolverBudgetPreflightTests(unittest.TestCase):
    def test_preflight_accepts_committed_benchmark(self):
        # Run against a temporary output root: once later contract steps write
        # real case directories into the committed output root, the no-overwrite
        # guard must keep this test independent of experiment state.
        with tempfile.TemporaryDirectory() as tmp:
            config = _temp_benchmark_config(Path(tmp), output_root=Path(tmp) / "root")
            summary = run_solver_budget_phase(config, "preflight")
        self.assertEqual(summary["phase"], "preflight")
        self.assertEqual(summary["benchmark_id"], "r8_40_solver_budget_v1")
        self.assertEqual(len(summary["inputs"]), 2)
        for entry in summary["inputs"]:
            self.assertEqual(len(entry["sha256"]), 64)
            self.assertGreater(entry["bytes"], 0)
        self.assertTrue(str(summary["gnu_time"]).endswith("time"))
        self.assertEqual(summary["existing_cases"], [])
        self.assertEqual(
            summary["thresholds"]["target_full_evaluation_seconds"],
            600.0,
        )

    def test_preflight_rejects_existing_case_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "out"
            config = _temp_benchmark_config(Path(tmp), output_root=root)
            plan = load_solver_budget_plan(config)
            case = case_directory(plan, "pilot", "paper_best__trf_default")
            case.mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError, "already contains case directories"):
                run_solver_budget_phase(config, "preflight")
            self.assertEqual(
                existing_case_directories(root),
                [root / "pilot" / "paper_best__trf_default"],
            )
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                require_absent_case(case)
            require_absent_case(case_directory(plan, "pilot", "paper_best__trf_default", attempt=2))

    def test_later_phases_are_deferred(self):
        with self.assertRaisesRegex(ValueError, "requires --baseline-ref"):
            run_solver_budget_phase(BENCHMARK_CONFIG, "parity")
        with self.assertRaisesRegex(NotImplementedError, "later contract step"):
            run_solver_budget_phase(BENCHMARK_CONFIG, "parity", baseline_ref="ec380dd")
        with self.assertRaisesRegex(NotImplementedError, "later contract step"):
            run_solver_budget_phase(BENCHMARK_CONFIG, "prepare")
        with self.assertRaisesRegex(ValueError, "unknown solver-budget phase"):
            run_solver_budget_phase(BENCHMARK_CONFIG, "sparkle")

    def test_cli_runs_preflight_and_defers_later_phases(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _temp_benchmark_config(Path(tmp), output_root=Path(tmp) / "root")
            preflight = subprocess.run(
                [sys.executable, str(BENCHMARK_SCRIPT), str(config), "--phase", "preflight"],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("preflight: PASS", preflight.stdout)
            deferred = subprocess.run(
                [sys.executable, str(BENCHMARK_SCRIPT), str(config), "--phase", "prepare"],
                capture_output=True,
                text=True,
            )
        self.assertEqual(deferred.returncode, 1)
        self.assertIn("later contract step", deferred.stderr)


class SolverBudgetComparatorTests(unittest.TestCase):
    def test_comparator_reports_missing_and_present_evidence(self):
        module = _load_script_module(COMPARATOR_SCRIPT)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "out"
            config = _temp_benchmark_config(Path(tmp), output_root=root)
            with self.assertRaisesRegex(RuntimeError, "no saved artifacts"):
                module.compare_solver_budget_phase(config, "pilot")
            case = root / "pilot" / "paper_best__trf_default"
            case.mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError, "missing case evidence"):
                module.compare_solver_budget_phase(config, "pilot")
            (case / "case.json").write_text("{}")
            summary = module.compare_solver_budget_phase(config, "pilot")
            self.assertEqual(summary["cases"][0]["artifacts"], ["case.json"])
            self.assertFalse(summary["gates_evaluated"])

    def test_comparator_does_not_import_agama(self):
        code = (
            "import importlib.util, sys\n"
            f"spec = importlib.util.spec_from_file_location('cmp', r'{COMPARATOR_SCRIPT}')\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(module)\n"
            "print('agama' in sys.modules)\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()
