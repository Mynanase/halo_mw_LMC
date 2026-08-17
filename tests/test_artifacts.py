import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from halo_mw_lmc.artifacts import (
    BEST_EVALUATION_SCHEMA_VERSION,
    discover_runs,
    load_best_evaluation,
    load_run_summary,
    save_best_evaluation,
    write_resolved_config,
)
from halo_mw_lmc.core.config import DensityFitSettings
from halo_mw_lmc.core.density import compare_density
from halo_mw_lmc.core.grids import CylindricalGrid
from halo_mw_lmc.core.potentials import ZhuHaloParameters
from halo_mw_lmc.core.weight_solver import WeightSolution


class RunArtifactTests(unittest.TestCase):
    def _evaluation(self):
        grid = CylindricalGrid.uniform(
            n_r=1,
            r_range=(0, 2),
            n_z=1,
            z_range=(0, 2),
            n_phi=2,
        )
        target = np.array([[[2.0, 3.0]]])
        density = compare_density(
            target,
            np.ones_like(target),
            target,
            grid,
            DensityFitSettings(
                min_abs_z=0,
                min_spherical_radius=0,
                max_spherical_radius=10,
                normalization_min_radius=0,
            ),
        )
        weight_solution = WeightSolution(
            seed_weights=np.array([1.0, 2.0]),
            model_density=density.raw_model_density,
            target_density=density.data_density,
            target_error=density.data_error,
            inner_objective=0.0,
            regularization_penalty=0.0,
            effective_orbit_count=1.8,
            maximum_weight_fraction=2 / 3,
            active_orbit_count=2,
            converged=True,
            status=0,
            message="catalogue-fixed weights",
        )
        return SimpleNamespace(
            density=density,
            velocity_loglike={},
            velocity_loglike_by_phi={},
            velocity_stars_by_phi={},
            velocity_distributions={},
            successful_orbits=2,
            weight_mode="catalogue_fixed",
            weight_solution=weight_solution,
            objective_mode="density_velocity",
            objective_velocity=0.0,
            objective_density_velocity=0.0,
            density_chi2_per_bin=0.0,
            density_max_chi2_per_bin=None,
        )

    def test_best_evaluation_round_trip_uses_only_npz_and_json(self):
        parameters = ZhuHaloParameters(6.2, 1.8, 0.8, 0.92, 1.0)
        with tempfile.TemporaryDirectory() as directory:
            save_best_evaluation(
                directory,
                self._evaluation(),
                parameters,
                iteration=4,
                objective=12.5,
            )
            stored = load_best_evaluation(directory)

        self.assertEqual(stored.metadata["iteration"], 4)
        self.assertEqual(stored.metadata["seed_orbits"], 2)
        self.assertEqual(stored.metadata["successful_orbits"], 2)
        self.assertEqual(stored.metadata["failed_orbits"], 0)
        self.assertEqual(stored.metadata["weight_sum"], 3.0)
        self.assertEqual(stored.density.grid.shape, (1, 1, 2))
        np.testing.assert_allclose(stored.density.model_density, [[[2.0, 3.0]]])
        self.assertEqual(stored.velocity_distributions, {})
        np.testing.assert_allclose(stored.weight_solution.seed_weights, [1.0, 2.0])

    def test_run_summary_is_portable_and_discoverable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "one"
            run.mkdir()
            write_resolved_config(run / "resolved_config.json", {"run": {"id": "one"}})
            (run / "sample.dat").write_text(
                "# iteration objective\n0 2.0\n1 1.0\n"
            )
            np.savez_compressed(run / "fixed_seed_weights.npz", total_weight=3.0)
            summary = load_run_summary(run)
            discovered = discover_runs(root)

        self.assertEqual(discovered, [run.resolve()])
        self.assertEqual(summary.samples.shape, (2,))
        self.assertEqual(float(summary.weight_audit["total_weight"]), 3.0)

    def test_unknown_best_metadata_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            save_best_evaluation(
                directory,
                self._evaluation(),
                ZhuHaloParameters(6.2, 1.8, 0.8, 0.92, 1.0),
                iteration=0,
                objective=1.0,
            )
            metadata = Path(directory) / "best/metadata.json"
            document = json.loads(metadata.read_text())
            document["schema_version"] = 999
            metadata.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "unsupported"):
                load_best_evaluation(directory)

    def test_summary_rejects_a_stale_best_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            write_resolved_config(run / "resolved_config.json", {"run": {"id": "x"}})
            (run / "sample.dat").write_text(
                "# iteration objective\n0 2.0\n1 1.0\n"
            )
            best = run / "best"
            best.mkdir()
            (best / "metadata.json").write_text(
                json.dumps(
                    {
                        "schema_version": BEST_EVALUATION_SCHEMA_VERSION,
                        "iteration": 0,
                        "objective": 2.0,
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_run_summary(run)

    def test_best_loader_rejects_mixed_metadata_and_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            save_best_evaluation(
                directory,
                self._evaluation(),
                ZhuHaloParameters(6.2, 1.8, 0.8, 0.92, 1.0),
                iteration=0,
                objective=1.0,
            )
            metadata_path = Path(directory) / "best/metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["generation"] = "different-generation"
            metadata_path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "do not match"):
                load_best_evaluation(directory)

    def test_summary_rejects_unknown_resolved_config_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "resolved_config.json").write_text(
                json.dumps({"schema_version": 999})
            )
            (run / "sample.dat").write_text("# iteration objective\n0 1.0\n")
            with self.assertRaisesRegex(ValueError, "resolved-config schema"):
                load_run_summary(run)


if __name__ == "__main__":
    unittest.main()
