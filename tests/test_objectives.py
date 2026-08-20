import unittest

import numpy as np

from halo_mw_lmc.core.config import DensityFitSettings
from halo_mw_lmc.core.density import DensityShellDiagnostics, compare_density
from halo_mw_lmc.core.grids import CylindricalGrid
from halo_mw_lmc.core.weight_solver import WeightSolution
from halo_mw_lmc.workflows.evaluation import (
    INVALID_TRIAL_PENALTY,
    ModelEvaluation,
)


class ProfileObjectiveTests(unittest.TestCase):
    def _evaluation(
        self,
        *,
        mode,
        density_limit=None,
        converged=True,
        shell_values=None,
        shell_counts=None,
        shell_limit=None,
    ):
        grid = CylindricalGrid.uniform(
            n_r=1,
            r_range=(0, 1),
            n_z=1,
            z_range=(0, 1),
            n_phi=2,
        )
        data = np.array([[[2.0, 4.0]]])
        model = np.array([[[1.0, 2.0]]])
        density = compare_density(
            data,
            np.ones_like(data),
            model,
            grid,
            DensityFitSettings(
                min_abs_z=0,
                min_spherical_radius=0,
                max_spherical_radius=10,
                normalization_min_radius=0,
                normalization="none",
            ),
        )
        solution = WeightSolution(
            seed_weights=np.array([1.0]),
            model_density=model,
            target_density=data,
            target_error=np.ones_like(data),
            inner_objective=density.chi2,
            regularization_penalty=0.0,
            effective_orbit_count=1.0,
            maximum_weight_fraction=1.0,
            active_orbit_count=1,
            converged=converged,
            status=1 if converged else 0,
            message="synthetic",
        )
        shells = None
        if shell_values is not None:
            values = np.asarray(shell_values, dtype=float)
            counts = np.asarray(shell_counts, dtype=np.int64)
            shells = DensityShellDiagnostics(
                radius_edges=np.arange(values.shape[0] + 1, dtype=float),
                chi2_by_shell=np.sum(values * counts, axis=1),
                valid_bins_by_shell=np.sum(counts, axis=1),
                chi2_by_shell_phi=values * counts,
                valid_bins_by_shell_phi=counts,
            )
        return ModelEvaluation(
            density=density,
            velocity_loglike={"vr": -3.0, "vphi": -2.0, "vtheta": -1.0},
            velocity_loglike_by_phi={},
            velocity_stars_by_phi={},
            velocity_distributions={},
            successful_orbits=1,
            weight_mode="density_solved",
            weight_solution=solution,
            objective_mode=mode,
            density_max_chi2_per_bin=density_limit,
            density_shells=shells,
            density_shell_phi_max_chi2_per_bin=shell_limit,
        )

    def test_both_objectives_are_available_from_one_evaluation(self):
        evaluation = self._evaluation(mode="density_velocity")

        self.assertEqual(evaluation.objective_velocity, 6.0)
        self.assertEqual(evaluation.objective_density_velocity, 8.5)
        self.assertEqual(evaluation.selected_objective, 8.5)

    def test_velocity_only_rejects_a_poor_density_profile(self):
        evaluation = self._evaluation(
            mode="velocity_only",
            density_limit=2.0,
        )

        self.assertEqual(evaluation.density_chi2_per_bin, 2.5)
        self.assertGreaterEqual(evaluation.selected_objective, INVALID_TRIAL_PENALTY)

    def test_nonconverged_weight_solution_is_never_ranked_normally(self):
        evaluation = self._evaluation(
            mode="density_velocity",
            converged=False,
        )

        self.assertGreaterEqual(evaluation.selected_objective, INVALID_TRIAL_PENALTY)

    def test_shell_phi_gate_accepts_only_when_every_cell_passes(self):
        evaluation = self._evaluation(
            mode="velocity_only",
            density_limit=3.0,
            shell_values=[[1.0, 2.0], [1.5, 1.9]],
            shell_counts=[[2, 2], [1, 1]],
            shell_limit=2.0,
        )

        self.assertTrue(evaluation.density_shell_phi_gate_passed)
        self.assertTrue(evaluation.density_gate_passed)
        self.assertEqual(evaluation.selected_objective, 6.0)

    def test_shell_phi_gate_rejects_excess_empty_and_nonfinite_cells(self):
        cases = (
            ([[1.0, 2.1]], [[1, 1]]),
            ([[1.0, 0.0]], [[1, 0]]),
            ([[1.0, np.nan]], [[1, 1]]),
        )
        for values, counts in cases:
            with self.subTest(values=values, counts=counts):
                evaluation = self._evaluation(
                    mode="velocity_only",
                    density_limit=3.0,
                    shell_values=values,
                    shell_counts=counts,
                    shell_limit=2.0,
                )
                self.assertFalse(evaluation.density_shell_phi_gate_passed)
                self.assertTrue(np.isfinite(evaluation.selected_objective))
                self.assertGreaterEqual(
                    evaluation.selected_objective,
                    INVALID_TRIAL_PENALTY,
                )

    def test_orbit_and_weight_diagnostics_are_derived_from_saved_slots(self):
        evaluation = self._evaluation(mode="density_velocity")

        self.assertEqual(evaluation.seed_orbits, 1)
        self.assertEqual(evaluation.successful_orbits, 1)
        self.assertEqual(evaluation.failed_orbits, 0)
        self.assertEqual(evaluation.weight_sum, 1.0)


if __name__ == "__main__":
    unittest.main()
