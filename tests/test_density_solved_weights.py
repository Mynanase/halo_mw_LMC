import unittest
from unittest.mock import patch

import numpy as np

from halo_mw_lmc.core.config import DensityFitSettings, WeightModelSettings
from halo_mw_lmc.core.grids import CylindricalGrid
from halo_mw_lmc.core.orbit_response import build_orbit_density_response
from halo_mw_lmc.core.orbits import OrbitLibrary
from halo_mw_lmc.core.weight_solver import solve_density_weights


class DensitySolvedWeightTests(unittest.TestCase):
    def setUp(self):
        self.grid = CylindricalGrid.uniform(
            n_r=1,
            r_range=(0.0, 1.0),
            n_z=1,
            z_range=(0.0, 1.0),
            n_phi=2,
        )
        phase_space = np.array(
            [
                [0.5, -0.5, 0.5, 0, 0, 0],
                [0.5, -0.5, 0.5, 0, 0, 0],
                [0.5, 0.5, 0.5, 0, 0, 0],
                [0.5, 0.5, 0.5, 0, 0, 0],
                [0.5, 0.5, 0.5, 0, 0, 0],
            ],
            dtype=float,
        )
        self.library = OrbitLibrary(
            seed_index=np.array([0, 0, 2, 2, 2], dtype=np.int64),
            time=np.arange(5, dtype=float),
            phase_space=phase_space,
        )
        self.response = build_orbit_density_response(
            self.library,
            self.grid,
            seed_count=3,
        )
        self.fit = DensityFitSettings(
            min_abs_z=0,
            min_spherical_radius=0,
            max_spherical_radius=10,
            normalization_min_radius=0,
            normalization="none",
        )

    def test_response_uses_time_fraction_and_preserves_failed_seed_slot(self):
        expected = np.array([1.0, 0.0, 1.0])
        density = self.response.model_density(expected)

        np.testing.assert_allclose(
            density * self.grid.volumes,
            [[[1.0, 1.0]]],
        )
        np.testing.assert_array_equal(
            self.response.successful_seed_index,
            [0, 2],
        )
        np.testing.assert_array_equal(self.response.sample_count, [2, 3])
        np.testing.assert_allclose(
            self.response.sample_weights(expected, self.library),
            [0.5, 0.5, 1 / 3, 1 / 3, 1 / 3],
        )

    def test_chunked_response_accumulates_orbits_across_chunk_boundaries(self):
        with patch(
            "halo_mw_lmc.core.orbit_response.RESPONSE_BUILD_CHUNK_SIZE",
            2,
        ):
            chunked = build_orbit_density_response(
                self.library,
                self.grid,
                seed_count=3,
            )

        np.testing.assert_allclose(
            chunked.matrix.toarray(),
            self.response.matrix.toarray(),
        )

    def test_absolute_density_recovers_nonnegative_orbit_weights(self):
        expected = np.array([2.0, 0.0, 3.0])
        target = self.response.model_density(expected)
        result = solve_density_weights(
            self.response,
            target,
            np.full(self.grid.shape, 0.01),
            self.fit,
            WeightModelSettings(
                mode="density_solved",
                solver="lsq_linear",
                target_normalization="absolute",
                regularization="l2",
                regularization_strength=0.0,
            ),
        )

        self.assertTrue(result.converged)
        np.testing.assert_allclose(result.seed_weights, expected, atol=1e-8)
        np.testing.assert_allclose(result.model_density, target, atol=1e-8)
        self.assertEqual(result.active_orbit_count, 2)
        self.assertAlmostEqual(result.effective_orbit_count, 25 / 13)
        self.assertAlmostEqual(result.maximum_weight_fraction, 3 / 5)

    def test_unit_mass_normalizes_target_and_uncertainty_together(self):
        original_weights = np.array([2.0, 0.0, 3.0])
        target = self.response.model_density(original_weights)
        error = np.full(self.grid.shape, 0.02)
        result = solve_density_weights(
            self.response,
            target,
            error,
            self.fit,
            WeightModelSettings(
                mode="density_solved",
                solver="lsq_linear",
                target_normalization="unit_mass",
                regularization="l2",
                regularization_strength=0.0,
            ),
        )

        self.assertAlmostEqual(
            float(np.sum(result.target_density * self.grid.volumes)),
            1.0,
        )
        np.testing.assert_allclose(
            result.seed_weights,
            original_weights / np.sum(original_weights),
            atol=1e-8,
        )
        np.testing.assert_allclose(
            result.target_error,
            error / np.sum(original_weights),
        )

    def test_unit_mass_does_not_constrain_the_total_weight(self):
        library = OrbitLibrary(
            seed_index=np.array([0, 0], dtype=np.int64),
            time=np.arange(2, dtype=float),
            phase_space=np.array(
                [
                    [0.5, -0.5, 0.5, 0, 0, 0],
                    [0.5, 0.5, 0.5, 0, 0, 0],
                ],
                dtype=float,
            ),
        )
        response = build_orbit_density_response(library, self.grid, seed_count=1)
        target_mass = np.array([[[0.2, 0.8]]])
        target = target_mass / self.grid.volumes
        unequal_error = np.array([[[0.001, 1.0]]])
        result = solve_density_weights(
            response,
            target,
            unequal_error,
            self.fit,
            WeightModelSettings(
                mode="density_solved",
                solver="lsq_linear",
                target_normalization="unit_mass",
                regularization="l2",
                regularization_strength=0.0,
            ),
        )

        # No sum(w)=1 constraint: the target's unit-mass amplitude sets the
        # weight scale directly. The tightly measured cell dominates the fit,
        # so the total weight lands near that cell's mass fraction (0.2)
        # scaled by the response, not at 1.0.
        self.assertNotAlmostEqual(float(np.sum(result.seed_weights)), 1.0, places=4)
        self.assertAlmostEqual(float(np.sum(result.seed_weights)), 0.4, delta=1e-3)
        model_mass = result.model_density * self.grid.volumes
        self.assertAlmostEqual(model_mass[0, 0, 0], 0.2, delta=1e-3)


if __name__ == "__main__":
    unittest.main()
