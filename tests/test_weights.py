import unittest

import numpy as np

from halo_mw_lmc.grids import CylindricalGrid
from halo_mw_lmc.weights import (
    catalogue_seed_weights,
    representative_weights_from_target,
)


class RepresentativeWeightTests(unittest.TestCase):
    def setUp(self):
        self.grid = CylindricalGrid.uniform(
            n_r=1,
            r_range=(0.0, 1.0),
            n_z=1,
            z_range=(0.0, 1.0),
            n_phi=2,
        )
        self.x = np.array([0.5, 0.5, 0.5])
        self.y = np.array([-0.5, -0.5, 0.5])
        self.z = np.array([0.5, 0.5, 0.5])
        self.target = np.array([[[2.0, 4.0]]])

    def test_seed_weights_reconstruct_target_mass_per_phi_cell(self):
        result = representative_weights_from_target(
            self.x,
            self.y,
            self.z,
            self.target,
            self.grid,
        )
        reconstructed = self.grid.histogram(
            np.hypot(self.x, self.y),
            self.z,
            np.arctan2(self.y, self.x),
            weights=result.weights,
        )
        np.testing.assert_allclose(reconstructed, result.target_mass)
        self.assertEqual(result.weighted_seed_count, 3)
        self.assertAlmostEqual(result.supported_mass_fraction, 1.0)

    def test_minimum_seed_count_marks_unsupported_target_mass(self):
        result = representative_weights_from_target(
            self.x,
            self.y,
            self.z,
            self.target,
            self.grid,
            minimum_seed_count=2,
        )
        self.assertTrue(result.supported_cells[0, 0, 0])
        self.assertFalse(result.supported_cells[0, 0, 1])
        self.assertEqual(result.weighted_seed_count, 2)
        self.assertGreater(result.unsupported_positive_mass, 0)
        self.assertLess(result.supported_mass_fraction, 1.0)

    def test_target_density_must_match_r_z_phi_shape(self):
        with self.assertRaisesRegex(ValueError, "expected"):
            representative_weights_from_target(
                self.x,
                self.y,
                self.z,
                np.ones((1, 1)),
                self.grid,
            )

    def test_catalogue_weights_are_validated_and_copied(self):
        source = np.array([0.0, 1.0, 3.0])
        weights = catalogue_seed_weights(source)
        np.testing.assert_array_equal(weights, source)
        self.assertIsNot(weights, source)

        for invalid in (
            np.array([]),
            np.array([0.0, 0.0]),
            np.array([1.0, -1.0]),
            np.array([1.0, np.nan]),
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    catalogue_seed_weights(invalid)


if __name__ == "__main__":
    unittest.main()
