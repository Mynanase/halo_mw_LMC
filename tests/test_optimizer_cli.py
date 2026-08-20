import unittest
from pathlib import Path

import numpy as np

from halo_mw_lmc.cli import build_parser
from halo_mw_lmc.configuration import load_run_configuration
from halo_mw_lmc.core.grids import CylindricalGrid
from halo_mw_lmc.core.potentials import (
    ZHU_2026_BEST_FIT,
    ZHU_2026_LOCAL_SEARCH_BOUNDS,
)
from halo_mw_lmc.core.weights import catalogue_weight_audit
from halo_mw_lmc.workflows.optimization import (
    paper_best_optimizer_point,
    resolved_configuration_document,
    rounded_trial,
    sample_header,
)


RUN_CONFIG = Path(__file__).resolve().parents[1] / "configs/runs/fix_weight.toml"
R8_40_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs/runs/density_solved_r8_40_benchmark.toml"
)


class OptimizerCliTests(unittest.TestCase):
    def test_cli_defaults_to_full_run_with_only_a_config(self):
        args = build_parser().parse_args([str(RUN_CONFIG)])
        self.assertEqual(args.mode, "run")
        self.assertEqual(args.config, RUN_CONFIG)

    def test_cli_short_flags_select_isolated_modes(self):
        cases = (("-v", "validate"), ("-c", "coverage"), ("-o", "optimize"))
        for flag, expected in cases:
            with self.subTest(flag=flag):
                args = build_parser().parse_args([flag, str(RUN_CONFIG)])
                self.assertEqual(args.mode, expected)
                self.assertEqual(args.config, RUN_CONFIG)

    def test_cli_modes_are_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["-v", "-c", str(RUN_CONFIG)])

    def test_cold_start_has_reproducible_config_seed(self):
        configuration = load_run_configuration(RUN_CONFIG)
        self.assertEqual(configuration.random_seed, 0)
        self.assertEqual(configuration.report.velocity_bin_factor, 3)

    def test_default_bounds_are_paper_centered_and_contain_best_fit(self):
        actual = load_run_configuration(RUN_CONFIG).search_bounds
        self.assertEqual(actual, ZHU_2026_LOCAL_SEARCH_BOUNDS)

        best = dict(ZHU_2026_BEST_FIT)
        best["rho0_plus_2logrs"] = best["rho0"] + 2 * best["log_rs"]
        for name, (lower, upper) in actual.items():
            self.assertLess(lower, best[name])
            self.assertLess(best[name], upper)

    def test_first_optimizer_point_is_the_paper_best_fit(self):
        best = ZHU_2026_BEST_FIT
        self.assertEqual(
            paper_best_optimizer_point(),
            [
                best["qhalo"],
                best["phalo"],
                best["rho0"],
                best["rho0"] + 2 * best["log_rs"],
                best["gamma"],
            ],
        )

    def test_catalogue_weight_audit_reports_orbit_dominance(self):
        grid = CylindricalGrid.uniform(
            n_r=1,
            r_range=(0.0, 2.0),
            n_z=1,
            z_range=(0.0, 1.0),
            n_phi=1,
        )
        initial = np.array(
            [[0.5, 0.0, 0.5, 0, 0, 0], [1.5, 0.0, 0.5, 0, 0, 0]],
            dtype=float,
        )
        audit = catalogue_weight_audit(initial, np.array([1.0, 3.0]), grid)
        self.assertAlmostEqual(float(audit["effective_seed_count"]), 1.6)
        self.assertAlmostEqual(float(audit["max_weight_fraction"]), 0.75)
        self.assertAlmostEqual(float(audit["cell_max_weight_fraction"][0, 0, 0]), 0.75)

    def test_rounded_coordinates_define_evaluation_and_optimizer_point(self):
        point, parameters = rounded_trial(
            [0.9234, 0.8126, 6.2341, 9.9231, 1.0129],
            decimals=3,
        )
        self.assertEqual(point, [0.923, 0.813, 6.234, 9.923, 1.013])
        self.assertEqual(parameters.log_rs, 1.8445)
        self.assertEqual(
            parameters.rho0 + 2 * parameters.log_rs,
            point[3],
        )

    def test_sample_schema_records_both_objectives_and_weight_diagnostics(self):
        header = sample_header(4, include_velocity=True, n_density_shells=2)

        for column in (
            "objective_velocity",
            "objective_density_velocity",
            "density_chi2_per_bin",
            "regularization_penalty",
            "effective_orbit_count",
            "max_weight_fraction",
            "active_orbit_count",
            "weight_solver_converged",
            "weight_solver_status",
            "successful_orbits",
            "failed_orbits",
            "weight_sum",
            "density_shell_phi_gate_passed",
            "density_worst_shell_phi_chi2_per_bin",
            "density_chi2_per_bin_shell0",
            "density_chi2_per_bin_shell1_phi3",
        ):
            self.assertIn(column, header.split())

    def test_resolved_config_records_inner_solver_and_shell_gate(self):
        document = resolved_configuration_document(
            load_run_configuration(R8_40_CONFIG)
        )

        self.assertEqual(document["weight_model"]["max_iter"], 20000)
        self.assertEqual(document["weight_model"]["lsmr_tol"], 1e-6)
        self.assertEqual(
            document["objective"]["density_shell_edges_kpc"],
            [8.0, 10.0, 12.0, 15.0, 20.0, 30.0, 40.0],
        )
        self.assertEqual(
            document["objective"]["density_shell_phi_max_chi2_per_bin"],
            2.0,
        )


if __name__ == "__main__":
    unittest.main()
