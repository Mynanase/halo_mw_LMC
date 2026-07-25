import unittest

import numpy as np

from halo_mw_lmc.config import DensityFitSettings
from halo_mw_lmc.density import compare_density
from halo_mw_lmc.grids import CylindricalGrid
from halo_mw_lmc.plotting import (
    _velocity_panel_values,
    isodensity_shape_profile,
)
from halo_mw_lmc.velocity import (
    SphericalVelocityGrid,
    VelocityDistributionComparison,
)


class PlottingDiagnosticsTests(unittest.TestCase):
    def test_isodensity_shape_recovers_oblate_axis_ratio(self):
        grid = CylindricalGrid.uniform(
            n_r=80,
            r_range=(0.0, 40.0),
            n_z=80,
            z_range=(0.0, 40.0),
            n_phi=2,
        )
        radius, z, _ = grid.center_mesh
        expected_q = 0.65
        density = np.exp(-np.sqrt(radius**2 + (z / expected_q) ** 2) / 8.0)
        comparison = compare_density(
            density,
            np.full_like(density, 0.05),
            density,
            grid,
            DensityFitSettings(
                min_abs_z=0.0,
                min_spherical_radius=0.0,
                max_spherical_radius=100.0,
                normalization_min_radius=0.0,
            ),
        )

        profile = isodensity_shape_profile(density, comparison, 0)

        self.assertGreaterEqual(profile.axis_ratio.size, 4)
        np.testing.assert_allclose(
            np.median(profile.axis_ratio),
            expected_q,
            atol=0.08,
        )
        self.assertTrue(np.all(np.diff(profile.radius) >= 0))

    def test_isodensity_shape_handles_empty_slice(self):
        grid = CylindricalGrid.uniform(n_r=2, n_z=2, n_phi=1)
        density = np.ones(grid.shape)
        comparison = compare_density(
            density,
            density,
            density,
            grid,
            DensityFitSettings(
                min_abs_z=0.0,
                min_spherical_radius=0.0,
                max_spherical_radius=100.0,
                normalization_min_radius=0.0,
            ),
        )
        empty = isodensity_shape_profile(
            np.zeros(grid.shape),
            comparison,
            0,
        )
        self.assertEqual(empty.radius.size, 0)

    def test_velocity_phi_average_is_occupancy_weighted(self):
        grid = SphericalVelocityGrid(
            radius_edges=np.array([0.0, 1.0]),
            theta_edges=np.array([0.0, 1.0]),
            phi_edges=np.array([-np.pi, 0.0, np.pi]),
            velocity_edges=np.array([-1.0, 0.0, 1.0]),
        )
        data_probability = np.array([[[[1.0, 0.0], [0.0, 1.0]]]])
        model_probability = np.array([[[[0.5, 0.5], [0.0, 1.0]]]])
        comparison = VelocityDistributionComparison(
            component="vr",
            grid=grid,
            data_probability=data_probability,
            data_uncertainty=np.zeros_like(data_probability),
            data_occupancy=np.array([[[1.0, 3.0]]]),
            model_probability=model_probability,
            model_occupancy=np.array([[[2.0, 2.0]]]),
        )

        data, uncertainty, model, occupancy = _velocity_panel_values(
            comparison,
            0,
            0,
            None,
        )

        np.testing.assert_allclose(data, [0.25, 0.75])
        np.testing.assert_allclose(model, [0.25, 0.75])
        np.testing.assert_allclose(
            uncertainty,
            np.sqrt(np.array([0.25, 0.75]) * np.array([0.75, 0.25]) / 4.0),
        )
        self.assertEqual(occupancy, 4.0)


if __name__ == "__main__":
    unittest.main()
