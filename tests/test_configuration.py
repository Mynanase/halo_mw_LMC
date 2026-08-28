import tempfile
import unittest
from pathlib import Path

import numpy as np

from halo_mw_lmc.configuration import (
    ConfigurationError,
    RunConfiguration,
    load_recipe_configuration,
    load_run_configuration,
)
from halo_mw_lmc.core.config import ZhuComparisonConfig


REPOSITORY = Path(__file__).resolve().parents[1]
RUN_FILE = REPOSITORY / "configs" / "runs" / "fix_weight.toml"
RECIPE_FILE = REPOSITORY / "configs" / "recipes" / "zhu_2026_fixed_weight.toml"
DENSITY_SOLVED_RUN_FILE = REPOSITORY / "configs" / "runs" / "density_solved.toml"
BENCHMARK_RUN_FILE = (
    REPOSITORY / "configs" / "runs" / "density_solved_benchmark.toml"
)
R8_50_BENCHMARK_RUN_FILE = (
    REPOSITORY
    / "configs"
    / "runs"
    / "density_solved_r8_50_benchmark.toml"
)
R8_40_BENCHMARK_RUN_FILE = (
    REPOSITORY
    / "configs"
    / "runs"
    / "density_solved_r8_40_benchmark.toml"
)
R8_40_CASES = {
    "density_solved_r8_40_benchmark.toml": (1e-6, 1e-6),
    "density_solved_r8_40_tol1e7_benchmark.toml": (1e-7, 1e-6),
    "density_solved_r8_40_tol1e8_benchmark.toml": (1e-8, 1e-6),
    "density_solved_r8_40_reg1e5_benchmark.toml": (1e-6, 1e-5),
    "density_solved_r8_40_reg1e4_benchmark.toml": (1e-6, 1e-4),
}
R8_40_RANKING_CASES = {
    "density_solved_r8_40_potential_ranking_tol1e7.toml": 1e-7,
    "density_solved_r8_40_potential_ranking_tol1e8.toml": 1e-8,
}
R8_40_RANKING_POINTS = (
    (0.920, 0.800, 6.200, 9.890, 1.000),
    (0.820, 0.700, 6.200, 9.890, 1.000),
    (1.020, 0.950, 6.200, 9.890, 1.000),
    (0.920, 0.800, 6.500, 9.800, 1.200),
    (0.920, 0.800, 5.900, 10.050, 0.800),
)


class ConfigurationTests(unittest.TestCase):
    def test_repository_run_loads_as_typed_configuration(self):
        configuration = load_run_configuration(RUN_FILE)

        self.assertIsInstance(configuration, RunConfiguration)
        self.assertEqual(configuration.run_id, "fix-weight")
        self.assertEqual(configuration.output_dir, REPOSITORY / "runs/fix-weight")
        self.assertEqual(configuration.data.catalog, REPOSITORY / (
            "data_for_model/lamost_dr8_SFlast_cut4_4phi/halo_clean_N.txt"
        ))
        self.assertEqual(configuration.orbit_periods, 10.0)
        self.assertEqual(configuration.iterations, 1000)
        self.assertEqual(configuration.random_seed, 0)
        self.assertEqual(configuration.round_decimals, 3)
        self.assertEqual(configuration.report.velocity_bin_factor, 3)
        self.assertEqual(configuration.coverage.maximum_points, 20_000)
        self.assertEqual(
            configuration.search_bounds["rho0_plus_2logrs"],
            (9.5, 10.3),
        )

    def test_recipe_constructs_the_core_comparison_config(self):
        configuration = load_run_configuration(RUN_FILE)
        comparison = configuration.to_comparison_config()

        self.assertIsInstance(comparison, ZhuComparisonConfig)
        self.assertEqual(comparison.density_grid.shape, (25, 25, 4))
        self.assertEqual(comparison.velocity_grid.shape, (8, 5, 4, 201))
        np.testing.assert_allclose(
            comparison.density_grid.phi_edges,
            comparison.velocity_grid.phi_edges,
        )
        self.assertFalse(comparison.include_velocity)
        self.assertEqual(comparison.velocity_fit_min_radius, 8.0)
        self.assertEqual(comparison.velocity_probability_floor, 1e-300)
        self.assertEqual(comparison.orbit_periods, 10.0)
        self.assertEqual(comparison.orbit_samples_per_orbit, 1000)
        self.assertEqual(comparison.orbit_sample_divisor, 500.0)
        self.assertEqual(comparison.weight_model.mode, "catalogue_fixed")
        self.assertEqual(comparison.objective.mode, "density_velocity")

    def test_density_solved_recipe_has_no_free_density_scale(self):
        configuration = load_run_configuration(DENSITY_SOLVED_RUN_FILE)
        comparison = configuration.to_comparison_config()

        self.assertEqual(comparison.weight_model.mode, "density_solved")
        self.assertEqual(comparison.weight_model.solver, "lsq_linear")
        self.assertEqual(comparison.weight_model.target_normalization, "unit_mass")
        self.assertEqual(comparison.density_fit.normalization, "none")
        self.assertTrue(comparison.include_velocity)
        self.assertEqual(comparison.objective.mode, "velocity_only")
        self.assertEqual(comparison.objective.density_max_chi2_per_bin, 2.0)

    def test_density_solved_recipe_rejects_a_second_density_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipe.toml"
            source = (REPOSITORY / "configs/recipes/zhu_2026_density_solved.toml")
            path.write_text(
                source.read_text().replace(
                    'normalization = "none"',
                    'normalization = "volume"',
                )
            )
            with self.assertRaisesRegex(ConfigurationError, "normalization='none'"):
                load_recipe_configuration(path)

    def test_density_solved_benchmark_is_one_paper_best_evaluation(self):
        configuration = load_run_configuration(BENCHMARK_RUN_FILE)

        self.assertEqual(
            configuration.run_id,
            "density-solved-paper-best-benchmark",
        )
        self.assertEqual(configuration.iterations, 1)
        self.assertEqual(configuration.recipe.search.initial_point, "paper_best")
        self.assertEqual(
            configuration.output_dir,
            REPOSITORY / "runs/density-solved-paper-best-benchmark",
        )
        self.assertEqual(
            configuration.coverage.output_dir,
            REPOSITORY / "data_coverage-density-solved-paper-best-benchmark",
        )

    def test_r8_50_benchmark_aligns_density_and_velocity_radial_support(self):
        configuration = load_run_configuration(R8_50_BENCHMARK_RUN_FILE)
        comparison = configuration.to_comparison_config()

        self.assertEqual(
            configuration.run_id,
            "density-solved-r8-50-paper-best-benchmark",
        )
        self.assertEqual(configuration.iterations, 1)
        self.assertEqual(configuration.recipe.search.initial_point, "paper_best")
        self.assertEqual(comparison.density_fit.min_spherical_radius, 8.0)
        self.assertEqual(comparison.density_fit.max_spherical_radius, 50.0)
        self.assertEqual(comparison.velocity_fit_min_radius, 8.0)
        self.assertEqual(comparison.velocity_grid.radius_edges[-1], 50.0)
        self.assertEqual(comparison.density_fit.min_abs_z, 2.0)
        self.assertEqual(comparison.weight_model.mode, "density_solved")
        self.assertEqual(comparison.density_fit.normalization, "none")
        self.assertEqual(
            configuration.output_dir,
            REPOSITORY / "runs/density-solved-r8-50-paper-best-benchmark",
        )
        self.assertEqual(
            configuration.coverage.output_dir,
            REPOSITORY
            / "data_coverage-density-solved-r8-50-paper-best-benchmark",
        )

    def test_r8_40_cases_are_one_factor_paper_best_benchmarks(self):
        for filename, (expected_tol, expected_regularization) in R8_40_CASES.items():
            with self.subTest(filename=filename):
                configuration = load_run_configuration(
                    REPOSITORY / "configs" / "runs" / filename
                )
                comparison = configuration.to_comparison_config()
                self.assertEqual(configuration.iterations, 1)
                self.assertEqual(configuration.random_seed, 0)
                self.assertEqual(
                    configuration.recipe.search.initial_point,
                    "paper_best",
                )
                self.assertEqual(comparison.density_fit.min_spherical_radius, 8.0)
                self.assertEqual(comparison.density_fit.max_spherical_radius, 40.0)
                self.assertEqual(comparison.velocity_fit_min_radius, 8.0)
                np.testing.assert_allclose(
                    comparison.velocity_grid.radius_edges,
                    [4, 6, 8, 10, 12, 15, 20, 30, 40],
                )
                np.testing.assert_allclose(
                    comparison.objective.density_shell_edges,
                    [8, 10, 12, 15, 20, 30, 40],
                )
                self.assertEqual(
                    comparison.objective.density_shell_phi_max_chi2_per_bin,
                    2.0,
                )
                self.assertEqual(comparison.weight_model.lsmr_tol, expected_tol)
                self.assertEqual(
                    comparison.weight_model.regularization_strength,
                    expected_regularization,
                )

    def test_r8_40_ranking_cases_use_identical_fixed_points(self):
        for filename, expected_tol in R8_40_RANKING_CASES.items():
            with self.subTest(filename=filename):
                configuration = load_run_configuration(
                    REPOSITORY / "configs" / "runs" / filename
                )
                comparison = configuration.to_comparison_config()
                self.assertEqual(configuration.iterations, 5)
                self.assertEqual(
                    configuration.fixed_optimizer_points,
                    R8_40_RANKING_POINTS,
                )
                self.assertEqual(comparison.weight_model.lsmr_tol, expected_tol)
                self.assertEqual(
                    comparison.weight_model.regularization_strength,
                    1e-6,
                )

    def test_fixed_point_count_must_match_iterations(self):
        source = (
            REPOSITORY
            / "configs/runs/density_solved_r8_40_potential_ranking_tol1e7.toml"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.toml"
            text = source.read_text().replace(
                'recipe = "../recipes/zhu_2026_density_solved_r8_40_tol1e7.toml"',
                f'recipe = "{REPOSITORY / "configs/recipes/zhu_2026_density_solved_r8_40_tol1e7.toml"}"',
            )
            path.write_text(text.replace("iterations = 5", "iterations = 4"))
            with self.assertRaisesRegex(ConfigurationError, "must equal"):
                load_run_configuration(path)

    def test_density_shell_gate_requires_matching_velocity_boundaries(self):
        source = REPOSITORY / "configs/recipes/zhu_2026_density_solved_r8_40.toml"
        cases = (
            (
                "density_shell_edges_kpc = [8.0, 10.0, 12.0, 15.0, 20.0, 30.0, 40.0]",
                "density_shell_edges_kpc = [8.0, 11.0, 12.0, 15.0, 20.0, 30.0, 40.0]",
                "velocity radius edges",
            ),
            (
                "density_shell_phi_max_chi2_per_bin = 2.0",
                "",
                "configured together",
            ),
        )
        for original, replacement, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "recipe.toml"
                path.write_text(source.read_text().replace(original, replacement))
                with self.assertRaisesRegex(ConfigurationError, message):
                    load_recipe_configuration(path)

    def test_every_relative_path_is_resolved_from_its_declaring_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipes = root / "recipe-files"
            runs = root / "run-files"
            recipes.mkdir()
            runs.mkdir()
            recipe_path = recipes / "recipe.toml"
            recipe_path.write_text(RECIPE_FILE.read_text())
            run_path = runs / "run.toml"
            run_path.write_text(
                """\
schema_version = 1
recipe = "../recipe-files/recipe.toml"

[run]
id = "relative-path-test"
output_dir = "../outputs/run"

[data]
catalog = "inputs/catalog.txt"
target_density = "inputs/density.txt"

[optimizer]
iterations = 2
random_seed = 7

[report]
velocity_bin_factor = 3

[coverage]
output_dir = "../outputs/coverage"
maximum_points = 10
velocity_limit_km_s = 500.0
random_seed = 11
"""
            )

            configuration = load_run_configuration(run_path)

        self.assertEqual(configuration.recipe.source_path, recipe_path.resolve())
        self.assertEqual(configuration.output_dir, (root / "outputs/run").resolve())
        self.assertEqual(
            configuration.data.catalog,
            (runs / "inputs/catalog.txt").resolve(),
        )
        self.assertEqual(
            configuration.data.target_density,
            (runs / "inputs/density.txt").resolve(),
        )

    def test_unknown_run_field_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.toml"
            text = RUN_FILE.read_text().replace(
                "velocity_bin_factor = 3",
                "velocity_bin_factor = 3\nunexpected = true",
            ).replace(
                "../recipes/zhu_2026_fixed_weight.toml",
                str(RECIPE_FILE),
            )
            path.write_text(text)

            with self.assertRaisesRegex(
                ConfigurationError,
                r"unknown field\(s\) in run configuration.report: unexpected",
            ):
                load_run_configuration(path)

    def test_unknown_recipe_field_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipe.toml"
            path.write_text(
                RECIPE_FILE.read_text().replace(
                    "n_phi = 4",
                    "n_phi = 4\nunexpected = 1",
                )
            )

            with self.assertRaisesRegex(
                ConfigurationError,
                r"unknown field\(s\) in recipe.density_grid: unexpected",
            ):
                load_recipe_configuration(path)

    def test_invalid_search_interval_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipe.toml"
            path.write_text(
                RECIPE_FILE.read_text().replace(
                    "qhalo = [0.70, 1.15]",
                    "qhalo = [1.15, 0.70]",
                )
            )

            with self.assertRaisesRegex(
                ConfigurationError,
                "recipe.search.bounds.qhalo must be strictly increasing",
            ):
                load_recipe_configuration(path)

    def test_paper_initial_point_must_lie_inside_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipe.toml"
            path.write_text(
                RECIPE_FILE.read_text().replace(
                    "qhalo = [0.70, 1.15]",
                    "qhalo = [0.70, 0.80]",
                )
            )
            with self.assertRaisesRegex(ConfigurationError, "outside bounds.*qhalo"):
                load_recipe_configuration(path)

    def test_search_bounds_respect_the_potential_domain(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipe.toml"
            path.write_text(
                RECIPE_FILE.read_text().replace(
                    "gamma = [0.50, 1.80]",
                    "gamma = [0.50, 3.00]",
                )
            )
            with self.assertRaisesRegex(ConfigurationError, "0 <= gamma < 3"):
                load_recipe_configuration(path)

    def test_bounds_must_align_with_optimizer_rounding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipe.toml"
            path.write_text(
                RECIPE_FILE.read_text().replace(
                    "qhalo = [0.70, 1.15]",
                    "qhalo = [0.7005, 1.15]",
                )
            )
            with self.assertRaisesRegex(ConfigurationError, "representable"):
                load_recipe_configuration(path)


if __name__ == "__main__":
    unittest.main()
