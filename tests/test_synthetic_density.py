import tempfile
import unittest
from pathlib import Path

import numpy as np

from halo_mw_lmc.configuration import (
    SyntheticDensityConfiguration,
    load_recipe_configuration,
    load_synthetic_density_configuration,
)
from halo_mw_lmc.core.grids import CylindricalGrid
from halo_mw_lmc.core.tracer_density import (
    DESI_YEAR1_KGIANTS_DENSITY,
    DesiKGiantsDensityModel,
    cell_average_cylindrical_density,
)
from halo_mw_lmc.data.density_target import read_target_density
from halo_mw_lmc.workflows.synthetic_density import generate_synthetic_density


REPOSITORY = Path(__file__).resolve().parents[1]
GENERATOR_CONFIG = (
    REPOSITORY / "configs/synthetic_density/desi_year1_kgiants.toml"
)
RECIPE_CONFIG = REPOSITORY / "configs/recipes/zhu_2026_density_solved.toml"
MODEL_SOURCE = REPOSITORY / "Desi/3D_density_profile.py"


class AnalyticTracerDensityTests(unittest.TestCase):
    def test_broken_power_law_is_continuous_at_both_breaks(self):
        model = DesiKGiantsDensityModel(
            p0=1.0,
            q0=1.0,
            phi0_rad=0.0,
            theta0_rad=0.0,
            p_coefficients=(0.0, 0.0, 0.0),
            q_coefficients=(0.0, 0.0, 0.0),
            phi_coefficients=(0.0, 0.0, 0.0),
            theta_coefficients=(0.0, 0.0, 0.0),
            break_radii_kpc=(2.0, 5.0),
            slopes=(1.0, 3.0, 6.0),
        )
        epsilon = 1e-9
        radius = np.array([2.0 - epsilon, 2.0, 5.0 - epsilon, 5.0])
        density = model(radius, np.zeros(4), np.zeros(4))

        self.assertAlmostEqual(density[0], density[1], places=7)
        self.assertAlmostEqual(density[2], density[3], places=7)
        self.assertAlmostEqual(density[1], 1.0)
        self.assertAlmostEqual(density[3], (5.0 / 2.0) ** -3)

    def test_default_desi_model_is_positive_and_azimuthally_non_axisymmetric(self):
        density = DESI_YEAR1_KGIANTS_DENSITY(
            np.array([20.0, 0.0, -20.0, 0.0]),
            np.array([0.0, 20.0, 0.0, -20.0]),
            np.full(4, 10.0),
        )

        self.assertTrue(np.all(np.isfinite(density)))
        self.assertTrue(np.all(density > 0))
        self.assertGreater(float(np.ptp(density)), 0.0)

    def test_cylindrical_quadrature_includes_the_radial_jacobian(self):
        grid = CylindricalGrid.uniform(
            n_r=2,
            r_range=(0.0, 2.0),
            n_z=1,
            z_range=(-1.0, 1.0),
            n_phi=2,
        )

        density = cell_average_cylindrical_density(
            lambda x, y, z: x * x + y * y,
            grid,
            quadrature_order=2,
        )

        expected_by_radius = 0.5 * (
            grid.r_edges[:-1] ** 2 + grid.r_edges[1:] ** 2
        )
        np.testing.assert_allclose(
            density,
            np.broadcast_to(expected_by_radius[:, None, None], grid.shape),
            atol=1e-12,
        )


class SyntheticDensityWorkflowTests(unittest.TestCase):
    def test_repository_generator_configuration_resolves_source_and_output(self):
        configuration = load_synthetic_density_configuration(GENERATOR_CONFIG)

        self.assertEqual(configuration.model_source, MODEL_SOURCE)
        self.assertEqual(configuration.grid.shape, (25, 25, 4))
        self.assertEqual(configuration.quadrature_order, 4)
        self.assertEqual(configuration.validation_order, 6)
        self.assertEqual(configuration.fractional_uncertainty, 0.1)
        self.assertEqual(
            configuration.output_path,
            REPOSITORY
            / "data_for_model/synthetic/desi_year1_kgiants_25x25x4.npz",
        )

    def test_generated_npz_round_trips_through_the_production_reader(self):
        recipe = load_recipe_configuration(RECIPE_CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "target.npz"
            configuration = SyntheticDensityConfiguration(
                source_path=GENERATOR_CONFIG,
                schema_version=1,
                recipe=recipe,
                model_name="desi_year1_kgiants_3d",
                model_source=MODEL_SOURCE,
                quadrature_order=2,
                validation_order=3,
                fractional_uncertainty=0.1,
                output_path=output,
            )

            result = generate_synthetic_density(configuration)
            density, error = read_target_density(output, configuration.grid)
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                generate_synthetic_density(configuration)
            with np.load(output, allow_pickle=False) as archive:
                model_name = str(archive["model_name"].item())
                uncertainty_semantics = str(
                    archive["uncertainty_semantics"].item()
                )

        self.assertEqual(result.grid_shape, (25, 25, 4))
        self.assertEqual(density.shape, result.grid_shape)
        self.assertTrue(np.all(np.isfinite(density)))
        self.assertTrue(np.all(density > 0))
        self.assertTrue(np.all(error >= 0.1 * density))
        self.assertEqual(model_name, "desi_year1_kgiants_3d")
        self.assertIn("fractional model error", uncertainty_semantics)


if __name__ == "__main__":
    unittest.main()
