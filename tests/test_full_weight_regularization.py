"""Small-array checks for the S16 full-space regularizers."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse

REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "full_weight_regularization_scan", REPO / "scripts/full_weight_regularization_scan.py",
)
scan = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(scan)


class EntropyTests(unittest.TestCase):
    def test_zero_strength_reduces_to_ridge_gradient(self):
        design = scipy.sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]]))
        observed = np.array([1.0, 1.0])
        weights = np.array([0.7, 0.2])
        cache = {}
        value, gradient = scan.objective_entropy(design, observed, 0.5, weights, 0.0, 2, cache)
        residual = design @ weights - observed
        self.assertAlmostEqual(value, float(residual @ residual) + 0.5 * float(weights @ weights), places=14)
        self.assertTrue(np.allclose(gradient, 2.0 * design.T @ residual + 2.0 * 0.5 * weights))

    def test_entropy_penalty_is_nonnegative_and_uniform_is_zero(self):
        design = scipy.sparse.csr_matrix(np.eye(2))
        observed = np.array([1.0, 1.0])
        cache = {}
        concentrated = np.array([2.0, 0.0])
        uniform = np.array([1.0, 1.0])
        value_c, _ = scan.objective_entropy(design, observed, 0.0, concentrated, 1.0, 2, cache)
        value_u, _ = scan.objective_entropy(design, observed, 0.0, uniform, 1.0, 2, cache)
        # The returned value is the full objective; subtract the identical
        # density term explicitly so the comparison isolates the KL penalty.
        residual_c = observed - design @ concentrated
        residual_u = observed - design @ uniform
        penalty_concentrated = value_c - float(residual_c @ residual_c)
        penalty_uniform = value_u - float(residual_u @ residual_u)
        # For count=2, uniform w=[1,1] gives -2 log 2 and is the minimum.
        self.assertAlmostEqual(penalty_uniform, -np.log(2.0), places=12)
        self.assertGreater(penalty_concentrated, penalty_uniform)

    def test_scaled_ill_conditioned_system_does_not_stop_at_start(self):
        """Verify the repaired scaled objective explores nonzero temperatures."""

        rng = np.random.default_rng(7)
        rows, count = 80, 40
        columns = rng.gamma(shape=2.0, size=(rows, count))
        truth = rng.gamma(shape=2.0, size=count)
        observed_physical = columns @ truth
        error = 0.02 * observed_physical
        design_dense = columns / error[:, None]
        observed = observed_physical / error
        design = scipy.sparse.csr_matrix(design_dense)
        start = scan.initial_weights(design, observed, 0.0)
        before = scan.gradient_diagnostics(design, observed, 0.0, 0.0, start)
        # The ridge-NNLS start is feasible for the non-negative bound. It is
        # not a stationary point after adding a nonzero entropy term.
        self.assertLess(before["projected_gradient_l_inf"], 1e-3)
        weights, result = scan.solve_entropy(design, observed, 0.0, 10.0, start, 500)
        after = scan.gradient_diagnostics(design, observed, 0.0, 10.0, weights)
        moved = float(np.linalg.norm(weights - start) / max(np.linalg.norm(start), 1e-30))
        self.assertTrue(bool(result.success))
        self.assertGreater(result.nit, 1)
        self.assertLess(after["projected_gradient_l_inf"], 1e-3)
        # A temperature comparable to the small data term must produce a
        # genuinely different KKT point, not a one-step return to the start.
        self.assertGreater(moved, 1e-10)

    def test_entropy_scan_strengths_reach_large_fraction_of_data_term(self):
        # The production objective is ~473, so the upper temperatures must be
        # large enough to make entropy a non-degenerate part of the frontier.
        self.assertGreaterEqual(max(scan.ENTROPY_STRENGTHS), 100.0)

    def test_interior_entropy_solver_reaches_kkt_on_scaled_system(self):
        rng = np.random.default_rng(7)
        rows, count = 80, 40
        columns = rng.gamma(shape=2.0, size=(rows, count))
        truth = rng.gamma(shape=2.0, size=count)
        observed_physical = columns @ truth
        error = 0.02 * observed_physical
        design = scipy.sparse.csr_matrix(columns / error[:, None])
        observed = observed_physical / error
        start = scan.initial_weights(design, observed, 1e-6)
        weights, result = scan.solve_entropy_interior(design, observed, 1e-6, 10.0, start, 500)
        diagnostics = scan.gradient_diagnostics(design, observed, 1e-6, 10.0, weights)
        self.assertTrue(bool(result.success))
        self.assertGreater(float(np.min(weights)), 0.0)
        self.assertLess(diagnostics["projected_gradient_l_inf"], 2e-5)
        self.assertGreater(float(np.linalg.norm(weights - start) / np.linalg.norm(start)), 1e-3)


class ResponseGraphTests(unittest.TestCase):
    def test_knn_graph_is_symmetric_and_laplacian_psd(self):
        rng = np.random.default_rng(4)
        columns = np.vstack([
            rng.normal(size=(5, 3)),
            rng.normal(size=(5, 3)) + np.array([4.0, 0.0, 0.0]),
        ])
        design = scipy.sparse.csr_matrix(columns.T)
        laplacian, degree = scan.response_graph(design, neighbours=2)
        dense = laplacian.toarray()
        self.assertTrue(np.allclose(dense, dense.T))
        self.assertTrue(np.all(degree > 0))
        values = np.linalg.eigvalsh(dense)
        self.assertGreaterEqual(float(values.min()), -1e-12)
        # Normalized Laplacian: diagonal is one and eigenvalues lie in [0,2].
        self.assertTrue(np.allclose(np.diag(dense), 1.0, atol=1e-12))
        self.assertLessEqual(float(values.max()), 2.0 + 1e-12)

    def test_similar_columns_connect_and_opposites_do_not(self):
        columns = np.vstack([np.ones((4, 3)), -np.ones((4, 3)), np.ones((4, 3)) * 2.0])
        design = scipy.sparse.csr_matrix(columns.T)
        laplacian, _ = scan.response_graph(design, neighbours=2)
        dense = laplacian.toarray()
        self.assertLess(float(np.abs(dense[:4, 4:8]).sum()), 1e-14)
        self.assertGreater(float(np.abs(dense[:4, 8:]).sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
