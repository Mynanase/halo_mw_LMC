import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from halo_mw_lmc.core.config import (
    DensityFitSettings,
    ObjectiveSettings,
    WeightModelSettings,
    ZhuComparisonConfig,
)
from halo_mw_lmc.core.grids import CylindricalGrid
from halo_mw_lmc.core.orbits import OrbitLibrary
from halo_mw_lmc.core.phase_space import cartesian_to_spherical_phase_space
from halo_mw_lmc.core.potentials import ZhuHaloParameters
from halo_mw_lmc.data.catalogue import SeedCatalogue
from halo_mw_lmc.workflows.evaluation import evaluate_prepared_model
from halo_mw_lmc.workflows.preparation import PreparedFixedWeightData


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
        catalogue = SeedCatalogue(
            initial_conditions=initial,
            seed_weights=seed_weights,
            velocity_errors={},
        )
        phase_space = cartesian_to_spherical_phase_space(
            *[initial[:, index] for index in range(6)]
        )
        prepared = PreparedFixedWeightData(
            catalogue=catalogue,
            target_density=target,
            target_error=np.ones_like(target),
            config=config,
            catalog_path=Path("synthetic-catalog"),
            density_path=Path("synthetic-target"),
            catalog_phase_space=phase_space,
        )
        library = OrbitLibrary(
            seed_index=np.arange(3, dtype=np.int64),
            time=np.zeros(3),
            phase_space=initial,
        )

        with (
            patch(
                "halo_mw_lmc.workflows.evaluation.build_potential_from_parameters",
                return_value=object(),
            ),
            patch(
                "halo_mw_lmc.workflows.evaluation.integrate_agama_orbits",
                return_value=library,
            ),
        ):
            result = evaluate_prepared_model(
                ZhuHaloParameters(
                    rho0=6.0,
                    log_rs=1.0,
                    phalo=1.0,
                    qhalo=1.0,
                    gamma=1.0,
                ),
                prepared,
            )

        self.assertAlmostEqual(result.density.scale, 1.0)
        self.assertAlmostEqual(result.density.chi2, 0.0)
        self.assertAlmostEqual(result.log_likelihood, 0.0)
        self.assertEqual(result.successful_orbits, 3)
        np.testing.assert_array_equal(prepared.seed_weights, seed_weights)


class DensitySolvedPipelineTests(unittest.TestCase):
    def test_trial_profiles_density_weights_before_scoring_velocities(self):
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
                normalization="none",
            ),
            include_velocity=True,
            orbit_samples_per_orbit=3,
            weight_model=WeightModelSettings(
                mode="density_solved",
                solver="lsq_linear",
                target_normalization="absolute",
                regularization="l2",
                regularization_strength=0.0,
            ),
            objective=ObjectiveSettings(
                mode="velocity_only",
                density_max_chi2_per_bin=1.0,
            ),
        )
        initial = np.array(
            [
                [0.5, -0.5, 0.5, 0, 0, 0],
                [0.2, 0.0, 0.5, 0, 0, 0],
                [0.5, 0.5, 0.5, 0, 0, 0],
            ],
            dtype=float,
        )
        catalogue = SeedCatalogue(
            initial_conditions=initial,
            seed_weights=None,
            velocity_errors={},
        )
        phase_space = cartesian_to_spherical_phase_space(
            *[initial[:, index] for index in range(6)]
        )
        target_mass = np.array([[[2.0, 3.0]]])
        target_density = target_mass / grid.volumes
        prepared = PreparedFixedWeightData(
            catalogue=catalogue,
            target_density=target_density,
            target_error=np.full(grid.shape, 0.01),
            config=config,
            catalog_path=Path("synthetic-catalog"),
            density_path=Path("synthetic-target"),
            catalog_phase_space=phase_space,
        )
        library = OrbitLibrary(
            seed_index=np.array([0, 0, 2, 2, 2], dtype=np.int64),
            time=np.arange(5, dtype=float),
            phase_space=np.array(
                [initial[0], initial[0], initial[2], initial[2], initial[2]]
            ),
        )

        with (
            patch(
                "halo_mw_lmc.workflows.evaluation.build_potential_from_parameters",
                return_value=object(),
            ),
            patch(
                "halo_mw_lmc.workflows.evaluation.integrate_agama_orbits",
                return_value=library,
            ),
            patch(
                "halo_mw_lmc.workflows.evaluation._score_velocities",
                return_value=(
                    {"vr": -2.0, "vphi": -1.0, "vtheta": -3.0},
                    {
                        name: np.array([-value / 2, -value / 2])
                        for name, value in (("vr", 2), ("vphi", 1), ("vtheta", 3))
                    },
                    {name: np.array([1, 1]) for name in ("vr", "vphi", "vtheta")},
                    {},
                ),
            ) as score_velocities,
        ):
            result = evaluate_prepared_model(
                ZhuHaloParameters(6.0, 1.0, 1.0, 1.0, 1.0),
                prepared,
            )

        np.testing.assert_allclose(
            result.weight_solution.seed_weights,
            [2.0, 0.0, 3.0],
            atol=1e-8,
        )
        np.testing.assert_allclose(score_velocities.call_args.args[2], np.ones(5))
        self.assertAlmostEqual(result.density.scale, 1.0)
        self.assertAlmostEqual(result.density.chi2, 0.0, places=8)
        self.assertAlmostEqual(result.objective_velocity, 6.0)
        self.assertAlmostEqual(result.objective_density_velocity, 6.0, places=8)
        self.assertAlmostEqual(result.selected_objective, 6.0)
        self.assertEqual(result.weight_mode, "density_solved")


if __name__ == "__main__":
    unittest.main()
