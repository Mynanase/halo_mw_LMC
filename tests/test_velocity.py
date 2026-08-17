import unittest

import numpy as np

from halo_mw_lmc.core.phase_space import cartesian_to_spherical_phase_space
from halo_mw_lmc.core.velocity import (
    SphericalVelocityGrid,
    conditional_velocity_histogram,
    multinomial_histogram_uncertainty,
    velocity_log_likelihood,
)


class VelocityTests(unittest.TestCase):
    def setUp(self):
        self.grid = SphericalVelocityGrid(
            radius_edges=np.array([0.0, 2.0]),
            theta_edges=np.array([-1.0, 1.0]),
            phi_edges=np.array([-np.pi, 0.0, np.pi]),
            velocity_edges=np.array([-2.0, 0.0, 2.0]),
        )

    def test_empty_cells_are_zero_not_nan(self):
        probability, occupancy = conditional_velocity_histogram(
            [1.0], [0.0], [-np.pi / 2], [-1.0], self.grid
        )
        self.assertTrue(np.all(np.isfinite(probability)))
        self.assertEqual(occupancy[0, 0, 1], 0)
        self.assertTrue(np.all(probability[0, 0, 1] == 0))

    def test_histogram_uncertainty_uses_cell_occupancy(self):
        probability = np.array([[[[0.25, 0.75], [0.0, 0.0]]]])
        occupancy = np.array([[[4.0, 0.0]]])
        uncertainty = multinomial_histogram_uncertainty(
            probability,
            occupancy,
        )
        expected = np.sqrt(0.25 * 0.75 / 4.0)
        np.testing.assert_allclose(uncertainty[0, 0, 0], [expected, expected])
        np.testing.assert_array_equal(uncertainty[0, 0, 1], [0.0, 0.0])

    def test_histogram_uncertainty_rejects_mismatched_shapes(self):
        with self.assertRaises(ValueError):
            multinomial_histogram_uncertainty(
                np.zeros((1, 2)),
                np.zeros((2,)),
            )

    def test_velocity_likelihood_keeps_phi_bins_separate(self):
        probability, _ = conditional_velocity_histogram(
            [1.0, 1.0],
            [0.0, 0.0],
            [-np.pi / 2, np.pi / 2],
            [-1.0, 1.0],
            self.grid,
        )
        correct, by_phi, used = velocity_log_likelihood(
            [1.0, 1.0],
            [0.0, 0.0],
            [-np.pi / 2, np.pi / 2],
            [-1.0, 1.0],
            [0.2, 0.2],
            probability,
            self.grid,
        )
        swapped, _, _ = velocity_log_likelihood(
            [1.0, 1.0],
            [0.0, 0.0],
            [-np.pi / 2, np.pi / 2],
            [1.0, -1.0],
            [0.2, 0.2],
            probability,
            self.grid,
        )
        self.assertGreater(correct, swapped)
        np.testing.assert_array_equal(used, [1, 1])
        self.assertAlmostEqual(correct, by_phi.sum())

    def test_empty_model_cell_is_penalized(self):
        model = np.zeros(self.grid.shape)
        total, by_phi, used = velocity_log_likelihood(
            [1.0],
            [0.0],
            [np.pi / 2],
            [0.0],
            [1.0],
            model,
            self.grid,
            probability_floor=1e-12,
        )
        self.assertAlmostEqual(total, np.log(1e-12))
        self.assertAlmostEqual(by_phi[1], np.log(1e-12))
        np.testing.assert_array_equal(used, [0, 1])

    def test_minimum_radius_excludes_incomplete_inner_orbit_library(self):
        probability, _ = conditional_velocity_histogram(
            [0.5, 1.5],
            [0.0, 0.0],
            [-np.pi / 2, -np.pi / 2],
            [-1.0, 1.0],
            self.grid,
        )
        total, by_phi, used = velocity_log_likelihood(
            [0.5, 1.5],
            [0.0, 0.0],
            [-np.pi / 2, -np.pi / 2],
            [-1.0, 1.0],
            [0.2, 0.2],
            probability,
            self.grid,
            minimum_radius=1.0,
        )
        self.assertTrue(np.isfinite(total))
        self.assertAlmostEqual(total, by_phi.sum())
        np.testing.assert_array_equal(used, [1, 0])

    def test_phase_space_transform(self):
        phase = cartesian_to_spherical_phase_space(
            [1.0], [0.0], [0.0], [2.0], [3.0], [4.0]
        )
        np.testing.assert_allclose(phase.radial_velocity, [2.0])
        np.testing.assert_allclose(phase.azimuthal_velocity, [3.0])
        np.testing.assert_allclose(phase.polar_velocity, [4.0])


if __name__ == "__main__":
    unittest.main()
