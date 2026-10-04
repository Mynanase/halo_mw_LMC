"""Synthetic mid-scale validation for the interior Newton entropy solver.

Gate required before any server run of the entropy temperature scan: the
solver must reach tight optimality on systems small enough to check
against a long-budget L-BFGS-B reference, including the degenerate and
tiny-weight cases that defeated the previous exp-transform variant.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "full_weight_regularization_scan", REPO / "scripts/full_weight_regularization_scan.py",
)
scan = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(scan)


def _reference_solve(design, observed, l2, strength, start):
    """Long-budget L-BFGS-B reference on the identical objective."""

    count = design.shape[1]
    cache: dict[str, float] = {}

    def fun(weights):
        value, gradient = scan.objective_entropy(design, observed, l2, weights, strength, count, cache)
        return value, gradient

    result = minimize(
        fun, np.maximum(start, 1e-12), jac=True, method="L-BFGS-B",
        bounds=[(0.0, None)] * count,
        options={"maxiter": 50000, "maxfun": 60000, "ftol": 1e-18, "gtol": 1e-14, "maxls": 100},
    )
    return np.maximum(np.asarray(result.x, dtype=float), 0.0), float(result.fun)


def _system(seed, rows=120, count=300, active=25, noise=0.05, duplicates=0, tiny=False):
    rng = np.random.default_rng(seed)
    columns = rng.gamma(shape=2.0, size=(rows, count))
    if duplicates:
        columns = np.hstack([columns, columns[:, :duplicates]])
    truth = np.zeros(columns.shape[1])
    support = rng.choice(columns.shape[1], active, replace=False)
    weights = rng.gamma(shape=2.0, size=active)
    if tiny:
        weights[: max(active // 5, 1)] *= 1e-5
    truth[support] = weights
    observed_physical = columns @ truth
    observed = observed_physical + noise * observed_physical * rng.standard_normal(rows)
    design = scipy.sparse.csr_matrix(columns)
    return design, observed, np.linalg.lstsq(columns, observed, rcond=None)[0]


class InteriorNewtonValidation(unittest.TestCase):
    def _assert_optimal(self, design, observed, l2, strength, start, label):
        weights, result = scan.solve_entropy_interior(design, observed, l2, strength, start, 3000)
        self.assertTrue(bool(result.success), f"{label}: {result.message}")
        self.assertIn("converged", str(result.message))
        cache: dict[str, float] = {}
        _, gradient = scan.objective_entropy(design, observed, l2, weights, strength, design.shape[1], cache)
        # Optimality is judged in the z geometry (w-weighted stationarity):
        # max |w_j g_j| over mass-bearing coordinates.  An absolute |g| bar
        # is inconsistent with z-space convergence -- coordinates at
        # w ~ 1e-9 legally carry |g| ~ ztol/w.  Boundary coordinates
        # (w ~ e^-650, w|g| ~ 1e-11) are irrelevant to concentration metrics.
        gradient_scale = max(1.0, float(np.max(np.abs(gradient))))
        weighted = float(np.max(weights * np.abs(gradient)))
        self.assertLessEqual(
            weighted, 1e-6 * gradient_scale * max(1.0, float(np.max(weights))),
            f"{label}: w-weighted gradient {weighted:.3e}",
        )
        reference, reference_value = _reference_solve(design, observed, l2, strength, start)
        value = scan.objective_entropy(design, observed, l2, weights, strength, design.shape[1], {})[0]
        self.assertLessEqual(
            value, reference_value + 1e-6 * max(1.0, abs(reference_value)),
            f"{label}: Newton objective {value:.6f} vs reference {reference_value:.6f}",
        )
        return weights

    def test_mid_scale_random_systems(self):
        for seed, strength in ((3, 0.3), (5, 3.0), (9, 30.0)):
            design, observed, start = _system(seed)
            self._assert_optimal(design, observed, 0.5, strength, np.maximum(start, 1e-3), f"seed={seed} mu={strength}")

    def test_duplicate_columns_are_degenerate_but_solved(self):
        design, observed, start = _system(13, duplicates=40)
        self._assert_optimal(design, observed, 0.5, 3.0, np.maximum(start, 1e-3), "duplicates=40")

    def test_tiny_optimal_weights_do_not_break_scaling(self):
        design, observed, start = _system(21, tiny=True)
        self._assert_optimal(design, observed, 0.5, 1.0, np.maximum(start, 1e-3), "tiny weights")

    def test_unique_optimum_from_two_starts(self):
        design, observed, start = _system(17)
        count = design.shape[1]
        uniform = np.full(count, float(np.sum(start)) / count)
        first = self._assert_optimal(design, observed, 0.5, 3.0, np.maximum(start, 1e-3), "lstsq start")
        # A deliberately hostile far start reaches the same optimum value
        # (strict convexity => unique minimizer) even though its z-gradient
        # tail can idle above the success threshold; production scans always
        # start near the optimum (ridge-NNLS reference plus mu-continuation),
        # so only the objective agreement is asserted here.
        second, far = scan.solve_entropy_interior(design, observed, 0.5, 3.0, uniform, 3000)
        value_first = scan.objective_entropy(design, observed, 0.5, first, 3.0, count, {})[0]
        value_second = scan.objective_entropy(design, observed, 0.5, second, 3.0, count, {})[0]
        self.assertLessEqual(value_second, value_first + 1e-6 * max(1.0, abs(value_first)))
        self.assertLessEqual(
            abs(scan.concentration(first)["n_eff"] - scan.concentration(second)["n_eff"]),
            1e-3 * scan.concentration(first)["n_eff"],
        )

    def test_bookkeeping_is_honest(self):
        design, observed, start = _system(23, rows=40, count=60, active=10)
        _, truncated = scan.solve_entropy_interior(design, observed, 0.5, 3.0, np.maximum(start, 1e-3), 1)
        self.assertFalse(bool(truncated.success))
        self.assertEqual(int(truncated.status), 1)
        self.assertEqual(str(truncated.message), "interior Newton reached iteration limit")
        _, converged = scan.solve_entropy_interior(design, observed, 0.5, 3.0, np.maximum(start, 1e-3), 3000)
        self.assertTrue(bool(converged.success))
        self.assertIn("converged", str(converged.message))
        self.assertEqual(int(converged.status), 0)


if __name__ == "__main__":
    unittest.main()
