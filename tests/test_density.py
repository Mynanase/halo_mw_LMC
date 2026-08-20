import unittest

import numpy as np

from halo_mw_lmc.core.config import DensityFitSettings
from halo_mw_lmc.core.density import (
    compare_density,
    density_shell_diagnostics,
    orbit_density,
)
from halo_mw_lmc.core.grids import CylindricalGrid


class CylindricalGridTests(unittest.TestCase):
    def test_cell_volumes_fill_full_cylinder(self):
        grid = CylindricalGrid.uniform(
            n_r=3,
            r_range=(0, 2),
            n_z=2,
            z_range=(-1, 1),
            n_phi=7,
        )
        self.assertAlmostEqual(grid.volumes.sum(), np.pi * 2**2 * 2)

    def test_phi_is_wrapped_at_periodic_boundary(self):
        grid = CylindricalGrid.uniform(
            n_r=1,
            r_range=(0, 2),
            n_z=1,
            z_range=(-1, 1),
            n_phi=4,
        )
        histogram = grid.histogram(
            [1, 1, 1],
            [0, 0, 0],
            [-np.pi, np.pi, 3 * np.pi],
        )
        self.assertEqual(histogram[0, 0, 0], 3)
        self.assertEqual(histogram.sum(), 3)

    def test_orbit_density_uses_exact_volume(self):
        grid = CylindricalGrid.uniform(
            n_r=1,
            r_range=(0, 2),
            n_z=1,
            z_range=(0, 1),
            n_phi=2,
        )
        _, _, phi = grid.centers
        volume = grid.volumes[0, 0]
        density = orbit_density(
            x=np.cos(phi),
            y=np.sin(phi),
            z=np.full(phi.shape, 0.5),
            weights=volume,
            grid=grid,
        )
        np.testing.assert_allclose(density, 1.0)


class DensityComparisonTests(unittest.TestCase):
    def setUp(self):
        self.grid = CylindricalGrid.uniform(
            n_r=1,
            r_range=(0, 2),
            n_z=1,
            z_range=(0, 2),
            n_phi=2,
        )
        self.settings = DensityFitSettings(
            min_abs_z=0,
            min_spherical_radius=0,
            max_spherical_radius=100,
            normalization_min_radius=0,
        )

    def test_one_global_scale_preserves_phi_contrast(self):
        data = np.array([[[1.0, 3.0]]])
        model = np.ones_like(data)
        error = np.ones_like(data)
        result = compare_density(data, error, model, self.grid, self.settings)
        self.assertAlmostEqual(result.scale, 2.0)
        self.assertAlmostEqual(result.chi2, 2.0)
        np.testing.assert_allclose(result.chi2_by_phi, [1.0, 1.0])

    def test_global_scale_recovers_common_amplitude(self):
        model = np.array([[[2.0, 5.0]]])
        data = 3.5 * model
        error = np.ones_like(data)
        result = compare_density(data, error, model, self.grid, self.settings)
        self.assertAlmostEqual(result.scale, 3.5)
        self.assertAlmostEqual(result.chi2, 0.0)

    def test_shell_diagnostics_use_left_closed_right_open_boundaries(self):
        grid = CylindricalGrid(
            r_edges=np.array([0.0, 2.0, 4.0]),
            z_edges=np.array([-1.0, 1.0]),
            phi_edges=np.array([-np.pi, 0.0, np.pi]),
        )
        data = np.ones(grid.shape)
        model = np.array([[[0.0, 1.0]], [[1.0, 3.0]]])
        comparison = compare_density(
            data,
            np.ones_like(data),
            model,
            grid,
            DensityFitSettings(
                min_abs_z=0.0,
                min_spherical_radius=0.0,
                max_spherical_radius=10.0,
                normalization_min_radius=0.0,
                normalization="none",
            ),
        )

        diagnostics = density_shell_diagnostics(
            comparison,
            [0.0, 1.0, 2.0, 10.0],
        )

        np.testing.assert_allclose(
            diagnostics.chi2_by_shell_phi,
            [[0, 0], [1, 0], [0, 4]],
        )
        np.testing.assert_array_equal(
            diagnostics.valid_bins_by_shell_phi,
            [[0, 0], [1, 1], [1, 1]],
        )
        self.assertTrue(np.isinf(diagnostics.chi2_per_bin_by_shell[0]))
        np.testing.assert_allclose(
            diagnostics.chi2_per_bin_by_shell[1:],
            [0.5, 2.0],
        )


if __name__ == "__main__":
    unittest.main()
