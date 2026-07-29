import unittest

from halo_mw_lmc.potentials import (
    ZHU_2026_BEST_FIT,
    ZHU_2026_LOCAL_SEARCH_BOUNDS,
)
from run_skopt_lamost_4phi import _paper_best_optimizer_point, build_parser


class OptimizerCliTests(unittest.TestCase):
    def test_cold_start_has_reproducible_default_seed(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.random_state, 0)

    def test_default_bounds_are_paper_centered_and_contain_best_fit(self):
        args = build_parser().parse_args([])
        actual = {
            "qhalo": (args.qhalo_min, args.qhalo_max),
            "phalo": (args.phalo_min, args.phalo_max),
            "rho0": (args.rho0_min, args.rho0_max),
            "rho0_plus_2logrs": (
                args.rho0_plus_2logrs_min,
                args.rho0_plus_2logrs_max,
            ),
            "gamma": (args.gamma_min, args.gamma_max),
        }
        self.assertEqual(actual, ZHU_2026_LOCAL_SEARCH_BOUNDS)

        best = dict(ZHU_2026_BEST_FIT)
        best["rho0_plus_2logrs"] = best["rho0"] + 2 * best["log_rs"]
        for name, (lower, upper) in actual.items():
            self.assertLess(lower, best[name])
            self.assertLess(best[name], upper)

    def test_first_optimizer_point_is_the_paper_best_fit(self):
        best = ZHU_2026_BEST_FIT
        self.assertEqual(
            _paper_best_optimizer_point(),
            [
                best["qhalo"],
                best["phalo"],
                best["rho0"],
                best["rho0"] + 2 * best["log_rs"],
                best["gamma"],
            ],
        )


if __name__ == "__main__":
    unittest.main()
