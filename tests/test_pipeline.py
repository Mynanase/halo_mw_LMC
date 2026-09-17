import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from halo_mw_lmc.grids import CylindricalGrid
from halo_mw_lmc.orbits import OrbitLibrary
from halo_mw_lmc.orbits import cartesian_to_spherical_phase_space
from halo_mw_lmc.potential import ZhuHaloParameters
from halo_mw_lmc.catalogue import SeedCatalogue
from halo_mw_lmc.density import build_orbit_density_response
from halo_mw_lmc.evaluate import evaluate_orbit_library, evaluate_prepared_model, score_orbit_weights
from halo_mw_lmc.weights import solve_density_weights
from halo_mw_lmc.prepare import PreparedFixedWeightData

from tests.artifact_fixture import comparison_model


class FixedWeightPipelineTests(unittest.TestCase):
    def test_catalogue_weights_drive_orbit_model_independently_of_target(self):
        grid = CylindricalGrid.uniform(
            n_r=1,
            r_range=(0.0, 1.0),
            n_z=1,
            z_range=(0.0, 1.0),
            n_phi=2,
        )
        config = comparison_model(
            grid,
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
                "halo_mw_lmc.evaluate.build_potential_from_parameters",
                return_value=object(),
            ),
            patch(
                "halo_mw_lmc.evaluate.integrate_agama_orbits",
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

    def test_shared_scoring_matches_production_entry_catalogue_fixed(self):
        grid = CylindricalGrid.uniform(
            n_r=1,
            r_range=(0.0, 1.0),
            n_z=1,
            z_range=(0.0, 1.0),
            n_phi=2,
        )
        config = comparison_model(
            grid,
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

        eval_entry = evaluate_orbit_library(library, prepared)
        eval_direct = score_orbit_weights(library, prepared, eval_entry.weight_solution)

        self.assertEqual(eval_entry.weight_mode, eval_direct.weight_mode)
        self.assertEqual(eval_entry.density_gate_passed, eval_direct.density_gate_passed)
        np.testing.assert_allclose(
            eval_entry.density.chi2, eval_direct.density.chi2, rtol=1e-12, atol=1e-12,
        )
        np.testing.assert_allclose(
            eval_entry.density_chi2_per_bin,
            eval_direct.density_chi2_per_bin,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            eval_entry.density.scale, eval_direct.density.scale, rtol=1e-12, atol=1e-12,
        )
        np.testing.assert_allclose(
            eval_entry.weight_sum, eval_direct.weight_sum, rtol=1e-12, atol=1e-12,
        )
        np.testing.assert_allclose(
            eval_entry.selected_objective,
            eval_direct.selected_objective,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_array_equal(
            eval_entry.density.fit_mask, eval_direct.density.fit_mask,
        )
        np.testing.assert_array_equal(
            eval_entry.weight_solution.seed_weights,
            eval_direct.weight_solution.seed_weights,
        )


class DensitySolvedPipelineTests(unittest.TestCase):
    def test_trial_profiles_density_weights_before_scoring_velocities(self):
        grid = CylindricalGrid.uniform(
            n_r=1,
            r_range=(0.0, 1.0),
            n_z=1,
            z_range=(0.0, 1.0),
            n_phi=2,
        )
        config = comparison_model(
            grid,
            density_fit={"normalization": "none"},
            include_velocity=True,
            orbit_samples_per_orbit=3,
            weight_model={
                "mode": "density_solved",
                "solver": "lsq_linear",
                "target_normalization": "absolute",
                "regularization": "l2",
                "regularization_strength": 0.0,
            },
            objective={
                "mode": "velocity_only",
                "density_max_chi2_per_bin": 1.0,
            },
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
                "halo_mw_lmc.evaluate.build_potential_from_parameters",
                return_value=object(),
            ),
            patch(
                "halo_mw_lmc.evaluate.integrate_agama_orbits",
                return_value=library,
            ),
            patch(
                "halo_mw_lmc.evaluate._score_velocities",
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

    def test_shared_scoring_matches_production_entry_density_solved(self):
        grid = CylindricalGrid.uniform(
            n_r=1,
            r_range=(0.0, 1.0),
            n_z=1,
            z_range=(0.0, 1.0),
            n_phi=2,
        )
        config = comparison_model(
            grid,
            density_fit={"normalization": "none"},
            include_velocity=True,
            orbit_samples_per_orbit=3,
            weight_model={
                "mode": "density_solved",
                "solver": "lsq_linear",
                "target_normalization": "absolute",
                "regularization": "l2",
                "regularization_strength": 0.0,
            },
            objective={
                "mode": "velocity_only",
                "density_max_chi2_per_bin": 1.0,
            },
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
        response = build_orbit_density_response(
            library, config["density_grid"], seed_count=prepared.initial_conditions.shape[0],
        )

        with patch(
            "halo_mw_lmc.evaluate._score_velocities",
            return_value=(
                {"vr": -2.0, "vphi": -1.0, "vtheta": -3.0},
                {
                    name: np.array([-value / 2, -value / 2])
                    for name, value in (("vr", 2), ("vphi", 1), ("vtheta", 3))
                },
                {name: np.array([1, 1]) for name in ("vr", "vphi", "vtheta")},
                {},
            ),
        ) as score_velocities:
            eval_entry = evaluate_orbit_library(library, prepared, response=response)
            weight_options = {
                key: value for key, value in config["weight_model"].items() if key != "mode"
            }
            mask_fit = {
                key: config["density_fit"][key]
                for key in ("min_abs_z", "min_spherical_radius", "max_spherical_radius", "require_positive_data")
            }
            weight_solution = solve_density_weights(
                response, prepared.target_density, prepared.target_error, mask_fit, **weight_options,
            )
            eval_direct = score_orbit_weights(library, prepared, weight_solution, response=response)

        np.testing.assert_array_equal(
            eval_entry.weight_solution.seed_weights,
            eval_direct.weight_solution.seed_weights,
        )
        self.assertEqual(eval_entry.weight_mode, eval_direct.weight_mode)
        self.assertEqual(eval_entry.density_gate_passed, eval_direct.density_gate_passed)
        np.testing.assert_allclose(
            eval_entry.density.chi2, eval_direct.density.chi2, rtol=1e-12, atol=1e-12,
        )
        np.testing.assert_allclose(
            eval_entry.density_chi2_per_bin,
            eval_direct.density_chi2_per_bin,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            eval_entry.density.scale, eval_direct.density.scale, rtol=1e-12, atol=1e-12,
        )
        np.testing.assert_allclose(
            eval_entry.objective_velocity, eval_direct.objective_velocity, rtol=1e-12, atol=1e-12,
        )
        np.testing.assert_allclose(
            eval_entry.weight_sum, eval_direct.weight_sum, rtol=1e-12, atol=1e-12,
        )
        np.testing.assert_allclose(
            eval_entry.selected_objective,
            eval_direct.selected_objective,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_array_equal(
            eval_entry.density.fit_mask, eval_direct.density.fit_mask,
        )
        self.assertEqual(
            eval_entry.orbit_support_audit.density_supported_orbit_count,
            eval_direct.orbit_support_audit.density_supported_orbit_count,
        )
        self.assertEqual(
            eval_entry.orbit_support_audit.velocity_supported_orbit_count,
            eval_direct.orbit_support_audit.velocity_supported_orbit_count,
        )
        self.assertEqual(
            eval_entry.orbit_support_audit.zero_density_response_velocity_orbit_count,
            eval_direct.orbit_support_audit.zero_density_response_velocity_orbit_count,
        )
        np.testing.assert_allclose(
            eval_entry.orbit_support_audit.zero_density_response_velocity_weight_sum,
            eval_direct.orbit_support_audit.zero_density_response_velocity_weight_sum,
            rtol=1e-12,
            atol=1e-12,
        )
        self.assertEqual(score_velocities.call_count, 2)


if __name__ == "__main__":
    unittest.main()
