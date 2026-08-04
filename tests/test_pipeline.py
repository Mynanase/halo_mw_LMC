import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from halo_mw_lmc.config import DensityFitSettings, ZhuComparisonConfig
from halo_mw_lmc.grids import CylindricalGrid
from halo_mw_lmc.orbits import OrbitLibrary
from skopt_oint_lamost_4phi import (
    PreparedFixedWeightData,
    evaluate_prepared_model,
)


class FixedWeightPipelineTests(unittest.TestCase):
    def test_catalogue_weights_drive_orbit_model_independently_of_target(self):
        grid = CylindricalGrid.uniform(
            n_r=1,
            r_range=(0.0, 1.0),
            n_z=1,
            z_range=(0.0, 1.0),
            n_phi=2,
        )
        config = ZhuComparisonConfig(
            density_grid=grid,
            density_fit=DensityFitSettings(
                min_abs_z=0,
                min_spherical_radius=0,
                max_spherical_radius=10,
                normalization_min_radius=0,
            ),
            orbit_samples_per_orbit=1,
            orbit_sample_divisor=1,
        )
        initial = np.array(
            [
                [0.5, -0.5, 0.5, 0, 0, 0],
                [0.5, -0.5, 0.5, 0, 0, 0],
                [0.5, 0.5, 0.5, 0, 0, 0],
            ],
            dtype=float,
        )
        seed_weights = np.array([1.0, 2.0, 4.0])
        seed_mass = grid.histogram(
            np.hypot(initial[:, 0], initial[:, 1]),
            initial[:, 2],
            np.arctan2(initial[:, 1], initial[:, 0]),
            weights=seed_weights,
        )
        target = seed_mass / grid.volumes
        prepared = PreparedFixedWeightData(
            catalog=object(),
            initial_conditions=initial,
            target_density=target,
            target_error=np.ones_like(target),
            seed_weights=seed_weights,
            config=config,
            catalog_path=Path("synthetic-catalog"),
            density_path=Path("synthetic-target"),
        )
        library = OrbitLibrary(
            seed_index=np.arange(3, dtype=np.int64),
            time=np.zeros(3),
            phase_space=initial,
        )

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("skopt_oint_lamost_4phi._build_potential", return_value=object()),
                patch(
                    "skopt_oint_lamost_4phi.integrate_agama_orbits",
                    return_value=library,
                ),
                patch(
                    "skopt_oint_lamost_4phi.plot_model_diagnostics"
                ) as plot_diagnostics,
            ):
                result = evaluate_prepared_model(
                    directory,
                    "model",
                    6.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    prepared,
                    plot=True,
                )

        self.assertAlmostEqual(result.density.scale, 1.0)
        self.assertAlmostEqual(result.density.chi2, 0.0)
        self.assertAlmostEqual(result.log_likelihood, 0.0)
        self.assertEqual(result.successful_orbits, 3)
        np.testing.assert_array_equal(result.seed_weights, seed_weights)
        plot_diagnostics.assert_called_once()
        self.assertEqual(plot_diagnostics.call_args.args[1], {})


if __name__ == "__main__":
    unittest.main()
