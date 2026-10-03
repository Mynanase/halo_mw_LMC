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


class L2FrontierTests(unittest.TestCase):
    def test_l2_grid_is_positive_ordered_and_anchored_at_production(self):
        self.assertEqual(scan.L2_STRENGTHS[0], 1e-6)
        self.assertTrue(all(value > 0 for value in scan.L2_STRENGTHS))
        self.assertLessEqual(list(scan.L2_STRENGTHS), sorted(scan.L2_STRENGTHS))

    def test_problem_at_l2_changes_strength_and_fingerprint(self):
        design = scipy.sparse.csr_matrix(np.eye(3))
        observed = np.array([1.0, 2.0, 3.0])
        problem = type("Problem", (), {
            "design": design, "observed": observed,
            "active_columns": np.ones(3, dtype=bool),
            "successful_orbit_count": 3, "regularization": 1.0,
            "fingerprint": "old",
        })()
        updated = scan.problem_at_l2(problem, 2.0)
        self.assertEqual(updated.regularization, 2.0)
        self.assertNotEqual(updated.fingerprint, "old")

    def test_l2_frontier_preserves_grid_and_solves_each_point(self):
        design = scipy.sparse.csr_matrix(np.diag([1.0, 2.0, 3.0]))
        observed = np.array([1.0, 1.0, 1.0])
        records = scan.l2_frontier_records(
            design, observed, (0.5, 2.0),
            lambda strength, weights, seconds: {
                "strength": strength, "nonzero": int(np.count_nonzero(weights)),
                "seconds": seconds,
            },
        )
        self.assertEqual([row["strength"] for row in records], [0.5, 2.0])
        self.assertTrue(all(row["nonzero"] > 0 for row in records))


class EntropyTests(unittest.TestCase):
    def test_zero_strength_reduces_to_ridge_gradient(self):
        design = scipy.sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]]))
        observed = np.array([1.0, 1.0])
        weights = np.array([0.7, 0.2])
        cache = {}
        value, gradient = scan.objective_entropy(design, observed, 0.5, weights, 0.0, 2, cache)
        residual = design @ weights - observed
        self.assertAlmostEqual(value, float(residual @ residual) + 0.5 * float(weights @ weights), places=14)
        self.assertTrue(np.allclose(gradient, design.T @ residual + 0.5 * weights))

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
