import dataclasses
import tempfile
import unittest
from pathlib import Path

import numpy as np

from halo_mw_lmc.config import (
    load_recipe_configuration,
    load_run_configuration,
    resolve_model,
)


def comparable_recipe(recipe: dict) -> dict:
    """Recipe dict with grid objects replaced by plain edge lists for equality."""

    resolved = dict(recipe)
    for key in ("density_grid", "velocity_grid"):
        resolved[key] = {
            name: value.tolist()
            for name, value in dataclasses.asdict(recipe[key]).items()
        }
    return resolved


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
R8_40_SOLVER_CASES = {
    "density_solved_r8_40_solver_lsq_linear_benchmark.toml": (
        "lsq_linear",
        1e-6,
    ),
    "density_solved_r8_40_solver_lsq_linear_repeat2.toml": (
        "lsq_linear",
        1e-6,
    ),
    "density_solved_r8_40_solver_lsq_linear_repeat3.toml": (
        "lsq_linear",
        1e-6,
    ),
    "density_solved_r8_40_solver_dense_nnls_benchmark.toml": (
        "dense_nnls",
        None,
    ),
    "density_solved_r8_40_solver_dense_nnls_repeat2.toml": (
        "dense_nnls",
        None,
    ),
    "density_solved_r8_40_solver_dense_nnls_repeat3.toml": (
        "dense_nnls",
        None,
    ),
    "density_solved_r8_40_solver_dual_ridge_benchmark.toml": (
        "dual_ridge",
        None,
    ),
    "density_solved_r8_40_solver_dual_ridge_repeat2.toml": (
        "dual_ridge",
        None,
    ),
    "density_solved_r8_40_solver_dual_ridge_repeat3.toml": (
        "dual_ridge",
        None,
    ),
}


class ConfigurationTests(unittest.TestCase):
    def test_repository_run_loads_as_resolved_configuration(self):
        configuration = load_run_configuration(RUN_FILE)

        self.assertEqual(configuration["run"]["id"], "fix-weight")
        self.assertEqual(
            configuration["run"]["output_dir"], REPOSITORY / "runs/fix-weight"
        )
        self.assertEqual(configuration["data"]["catalog"], REPOSITORY / (
            "data_for_model/lamost_dr8_SFlast_cut4_4phi/halo_clean_N.txt"
        ))
        self.assertEqual(configuration["recipe"]["orbit_periods"], 10.0)
        self.assertEqual(configuration["optimizer"]["iterations"], 1000)
        self.assertEqual(configuration["optimizer"]["random_seed"], 0)
        self.assertEqual(
            configuration["recipe"]["search"]["round_decimals"], 3
        )
        self.assertEqual(
            configuration["report"]["velocity_bin_factor"], 3
        )
        self.assertEqual(configuration["coverage"]["maximum_points"], 20_000)
        self.assertEqual(
            configuration["recipe"]["search"]["bounds"]["rho0_plus_2logrs"],
            (9.5, 10.3),
        )

    def test_recipe_resolves_the_core_comparison_model(self):
        configuration = load_run_configuration(RUN_FILE)
        comparison = resolve_model(configuration["recipe"])

        self.assertEqual(comparison["density_grid"].shape, (25, 25, 4))
        self.assertEqual(comparison["velocity_grid"].shape, (8, 5, 4, 201))
        np.testing.assert_allclose(
            comparison["density_grid"].phi_edges,
            comparison["velocity_grid"].phi_edges,
        )
        self.assertFalse(comparison["include_velocity"])
        self.assertEqual(comparison["velocity_fit_min_radius"], 8.0)
        self.assertEqual(comparison["velocity_probability_floor"], 1e-300)
        self.assertEqual(comparison["orbit_periods"], 10.0)
        self.assertEqual(comparison["orbit_samples_per_orbit"], 1000)
        self.assertEqual(comparison["orbit_sample_divisor"], 500.0)
        self.assertEqual(comparison["weight_model"]["mode"], "catalogue_fixed")
        self.assertEqual(comparison["objective"]["mode"], "density_velocity")

    def test_density_solved_recipe_has_no_free_density_scale(self):
        configuration = load_run_configuration(DENSITY_SOLVED_RUN_FILE)
        comparison = resolve_model(configuration["recipe"])

        self.assertEqual(comparison["weight_model"]["mode"], "density_solved")
        self.assertEqual(comparison["weight_model"]["solver"], "lsq_linear")
        self.assertEqual(
            comparison["weight_model"]["target_normalization"], "unit_mass"
        )
        self.assertEqual(comparison["density_fit"]["normalization"], "none")
        self.assertTrue(comparison["include_velocity"])
        self.assertEqual(comparison["objective"]["mode"], "velocity_only")
        self.assertEqual(
            comparison["objective"]["density_max_chi2_per_bin"], 2.0
        )

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
            with self.assertRaisesRegex(ValueError, "normalization='none'"):
                load_recipe_configuration(path)

    def test_density_solved_benchmark_is_one_paper_best_evaluation(self):
        configuration = load_run_configuration(BENCHMARK_RUN_FILE)

        self.assertEqual(
            configuration["run"]["id"],
            "density-solved-paper-best-benchmark",
        )
        self.assertEqual(configuration["optimizer"]["iterations"], 1)
        self.assertEqual(
            configuration["recipe"]["search"]["initial_point"], "paper_best"
        )
        self.assertEqual(
            configuration["run"]["output_dir"],
            REPOSITORY / "runs/density-solved-paper-best-benchmark",
        )
        self.assertEqual(
            configuration["coverage"]["output_dir"],
            REPOSITORY / "data_coverage-density-solved-paper-best-benchmark",
        )

    def test_r8_50_benchmark_aligns_density_and_velocity_radial_support(self):
        configuration = load_run_configuration(R8_50_BENCHMARK_RUN_FILE)
        comparison = resolve_model(configuration["recipe"])

        self.assertEqual(
            configuration["run"]["id"],
            "density-solved-r8-50-paper-best-benchmark",
        )
        self.assertEqual(configuration["optimizer"]["iterations"], 1)
        self.assertEqual(
            configuration["recipe"]["search"]["initial_point"], "paper_best"
        )
        self.assertEqual(
            comparison["density_fit"]["min_spherical_radius"], 8.0
        )
        self.assertEqual(
            comparison["density_fit"]["max_spherical_radius"], 50.0
        )
        self.assertEqual(comparison["velocity_fit_min_radius"], 8.0)
        self.assertEqual(comparison["velocity_grid"].radius_edges[-1], 50.0)
        self.assertEqual(comparison["density_fit"]["min_abs_z"], 2.0)
        self.assertEqual(comparison["weight_model"]["mode"], "density_solved")
        self.assertEqual(comparison["density_fit"]["normalization"], "none")
        self.assertEqual(
            configuration["run"]["output_dir"],
            REPOSITORY / "runs/density-solved-r8-50-paper-best-benchmark",
        )
        self.assertEqual(
            configuration["coverage"]["output_dir"],
            REPOSITORY
            / "data_coverage-density-solved-r8-50-paper-best-benchmark",
        )

    def test_r8_40_cases_are_one_factor_paper_best_benchmarks(self):
        for filename, (expected_tol, expected_regularization) in R8_40_CASES.items():
            with self.subTest(filename=filename):
                configuration = load_run_configuration(
                    REPOSITORY / "configs" / "runs" / filename
                )
                comparison = resolve_model(configuration["recipe"])
                self.assertEqual(configuration["optimizer"]["iterations"], 1)
                self.assertEqual(configuration["optimizer"]["random_seed"], 0)
                self.assertEqual(
                    configuration["recipe"]["search"]["initial_point"],
                    "paper_best",
                )
                self.assertEqual(
                    comparison["density_fit"]["min_spherical_radius"], 8.0
                )
                self.assertEqual(
                    comparison["density_fit"]["max_spherical_radius"], 40.0
                )
                self.assertEqual(comparison["velocity_fit_min_radius"], 8.0)
                np.testing.assert_allclose(
                    comparison["velocity_grid"].radius_edges,
                    [4, 6, 8, 10, 12, 15, 20, 30, 40],
                )
                np.testing.assert_allclose(
                    comparison["objective"]["density_shell_edges"],
                    [8, 10, 12, 15, 20, 30, 40],
                )
                self.assertEqual(
                    comparison["objective"][
                        "density_shell_phi_max_chi2_per_bin"
                    ],
                    2.0,
                )
                self.assertEqual(
                    comparison["weight_model"]["lsmr_tol"], expected_tol
                )
                self.assertEqual(
                    comparison["weight_model"]["regularization_strength"],
                    expected_regularization,
                )

    def test_joint_r8_40_runs_change_only_outer_objective_and_run_identity(self):
        for suffix in ("benchmark", "wide_scan"):
            with self.subTest(suffix=suffix):
                baseline = load_run_configuration(
                    REPOSITORY / "configs/runs"
                    / f"density_solved_r8_40_{suffix}.toml"
                )
                joint = load_run_configuration(
                    REPOSITORY / "configs/runs"
                    / f"density_solved_r8_40_joint_{suffix}.toml"
                )
                self.assertEqual(
                    baseline["recipe"]["objective"]["mode"], "velocity_only"
                )
                self.assertEqual(
                    joint["recipe"]["objective"]["mode"], "density_velocity"
                )
                self.assertIsNone(
                    joint["recipe"]["objective"]["density_max_chi2_per_bin"]
                )
                self.assertIsNone(
                    joint["recipe"]["objective"]["density_shell_edges"]
                )
                self.assertIsNone(
                    joint["recipe"]["objective"][
                        "density_shell_phi_max_chi2_per_bin"
                    ]
                )
                self.assertTrue(
                    resolve_model(joint["recipe"])["include_velocity"]
                )
                joint_recipe = dict(joint["recipe"])
                joint_recipe["source_path"] = baseline["recipe"]["source_path"]
                joint_recipe["name"] = baseline["recipe"]["name"]
                joint_recipe["objective"] = baseline["recipe"]["objective"]
                self.assertEqual(
                    comparable_recipe(joint_recipe),
                    comparable_recipe(baseline["recipe"]),
                )
                self.assertEqual(joint["data"], baseline["data"])
                self.assertEqual(
                    joint["optimizer"]["iterations"],
                    baseline["optimizer"]["iterations"],
                )
                self.assertEqual(
                    joint["optimizer"]["random_seed"],
                    baseline["optimizer"]["random_seed"],
                )
                self.assertIsNone(joint["optimizer"]["fixed_points"])
                self.assertNotEqual(
                    joint["run"]["id"], baseline["run"]["id"]
                )
                self.assertNotEqual(
                    joint["run"]["output_dir"], baseline["run"]["output_dir"]
                )
                self.assertNotEqual(
                    joint["coverage"]["output_dir"],
                    baseline["coverage"]["output_dir"],
                )

    def test_stage1_screen_preserves_wide_bounds_and_fixed_anchors(self):
        benchmark = load_run_configuration(REPOSITORY / "configs/runs/density_solved_r8_40_joint_benchmark.toml")
        screen = load_recipe_configuration(REPOSITORY / "configs/recipes/zhu_2026_density_solved_r8_40_joint_screen.toml")
        screen_copy = dict(screen)
        screen_copy["source_path"] = benchmark["recipe"]["source_path"]
        screen_copy["name"] = benchmark["recipe"]["name"]
        screen_copy["search"] = benchmark["recipe"]["search"]
        self.assertEqual(
            comparable_recipe(screen_copy),
            comparable_recipe(benchmark["recipe"]),
        )
        self.assertEqual(screen["search"]["bounds"]["qhalo"], (0.70, 1.30))
        self.assertEqual(
            screen["search"]["bounds"]["rho0_plus_2logrs"], (9.20, 10.30)
        )
        self.assertEqual(screen["search"]["bounds"]["gamma"], (0.50, 2.00))
        points = []
        for number in range(1, 13):
            run = load_run_configuration(REPOSITORY / f"configs/runs/density_solved_r8_40_stage1_screen_shard{number:02d}.toml")
            self.assertEqual(
                comparable_recipe(run["recipe"]), comparable_recipe(screen)
            )
            self.assertEqual(run["optimizer"]["iterations"], 4)
            self.assertEqual(run["optimizer"]["random_seed"], 0)
            points.extend(run["optimizer"]["fixed_points"])
        self.assertEqual(len(points), 48)
        self.assertEqual(len(set(points)), 48)
        self.assertIn((1.222, 0.895, 5.616, 9.354, 1.330), points)

    def test_stage2_side_runs_preserve_screen_settings(self):
        anchor = load_run_configuration(REPOSITORY / "configs/runs/density_solved_r8_40_stage2_anchor_9353.toml")
        converged = load_run_configuration(REPOSITORY / "configs/runs/density_solved_r8_40_stage2_rank1_converged.toml")
        self.assertEqual(
            anchor["recipe"]["source_path"].name,
            "zhu_2026_density_solved_r8_40_joint_screen.toml",
        )
        self.assertEqual(
            anchor["optimizer"]["fixed_points"][0][3], 9.353
        )
        self.assertEqual(
            converged["recipe"]["weight_model"]["max_iter"], 60000
        )
        converged_recipe = dict(converged["recipe"])
        converged_recipe["source_path"] = anchor["recipe"]["source_path"]
        converged_recipe["name"] = anchor["recipe"]["name"]
        converged_recipe["weight_model"] = anchor["recipe"]["weight_model"]
        self.assertEqual(
            comparable_recipe(converged_recipe),
            comparable_recipe(anchor["recipe"]),
        )

    def test_r8_40_ranking_cases_use_identical_fixed_points(self):
        for filename, expected_tol in R8_40_RANKING_CASES.items():
            with self.subTest(filename=filename):
                configuration = load_run_configuration(
                    REPOSITORY / "configs" / "runs" / filename
                )
                comparison = resolve_model(configuration["recipe"])
                self.assertEqual(configuration["optimizer"]["iterations"], 5)
                self.assertEqual(
                    configuration["optimizer"]["fixed_points"],
                    R8_40_RANKING_POINTS,
                )
                self.assertEqual(
                    comparison["weight_model"]["lsmr_tol"], expected_tol
                )
                self.assertEqual(
                    comparison["weight_model"]["regularization_strength"],
                    1e-6,
                )

    def test_solver_backends_use_independent_paper_best_cold_starts(self):
        for filename, (solver, lsmr_tol) in R8_40_SOLVER_CASES.items():
            with self.subTest(filename=filename):
                configuration = load_run_configuration(
                    REPOSITORY / "configs" / "runs" / filename
                )
                model = resolve_model(configuration["recipe"])["weight_model"]
                self.assertEqual(configuration["optimizer"]["iterations"], 1)
                self.assertIsNone(configuration["optimizer"]["fixed_points"])
                self.assertEqual(
                    configuration["recipe"]["search"]["initial_point"],
                    "paper_best",
                )
                self.assertEqual(model["solver"], solver)
                self.assertEqual(model["lsmr_tol"], lsmr_tol)
                self.assertEqual(model["solver_tolerance"], 1e-8)
                self.assertEqual(model["regularization_strength"], 1e-6)

    def test_alternative_solver_rejects_lsmr_tolerance(self):
        source = (
            REPOSITORY
            / "configs/recipes/zhu_2026_density_solved_r8_40_dual_ridge.toml"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipe.toml"
            path.write_text(
                source.read_text().replace(
                    "solver_tolerance = 1e-8",
                    "solver_tolerance = 1e-8\nlsmr_tol = 1e-6",
                )
            )
            with self.assertRaisesRegex(ValueError, "lsmr_tol is only valid"):
                load_recipe_configuration(path)

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
            with self.assertRaisesRegex(ValueError, "must equal"):
                load_run_configuration(path)

    def test_shell_phi_limit_requires_matching_shell_edges(self):
        source = REPOSITORY / "configs/recipes/zhu_2026_density_solved_r8_40.toml"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipe.toml"
            path.write_text(
                source.read_text().replace(
                    "density_shell_phi_max_chi2_per_bin = 2.0",
                    "",
                )
            )
            with self.assertRaisesRegex(ValueError, "configured together"):
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

        self.assertEqual(
            configuration["recipe"]["source_path"], recipe_path.resolve()
        )
        self.assertEqual(
            configuration["run"]["output_dir"],
            (root / "outputs/run").resolve(),
        )
        self.assertEqual(
            configuration["data"]["catalog"],
            (runs / "inputs/catalog.txt").resolve(),
        )
        self.assertEqual(
            configuration["data"]["target_density"],
            (runs / "inputs/density.txt").resolve(),
        )

    def test_catalogue_fixed_recipe_rejects_solver_options(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipe.toml"
            path.write_text(
                RECIPE_FILE.read_text().replace(
                    '[weight_model]\nmode = "catalogue_fixed"',
                    '[weight_model]\nmode = "catalogue_fixed"\nmax_iter = 20000',
                )
            )

            with self.assertRaisesRegex(
                ValueError,
                "cannot define solver options: max_iter",
            ):
                load_recipe_configuration(path)

    def test_schema_version_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipe.toml"
            path.write_text(
                RECIPE_FILE.read_text().replace(
                    "schema_version = 1",
                    "schema_version = 2",
                )
            )

            with self.assertRaisesRegex(ValueError, "schema_version"):
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
                ValueError,
                "bounds.qhalo must be strictly increasing",
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
            with self.assertRaisesRegex(ValueError, "outside bounds.*qhalo"):
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
            with self.assertRaisesRegex(ValueError, "0 <= gamma < 3"):
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
            with self.assertRaisesRegex(ValueError, "representable"):
                load_recipe_configuration(path)


if __name__ == "__main__":
    unittest.main()
