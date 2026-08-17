import tempfile
import unittest
from pathlib import Path

import numpy as np

from halo_mw_lmc.core.grids import CylindricalGrid
from halo_mw_lmc.data.catalogue import read_seed_catalogue
from halo_mw_lmc.data.density_target import read_target_density


class DataAdapterTests(unittest.TestCase):
    def test_seed_catalogue_becomes_typed_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.txt"
            path.write_text(
                "x_gc y_gc z_gc vx_gc vy_gc vz_gc w vr_err vphi_err vthe_err\n"
                "1 2 3 4 5 6 0.5 10 11 12\n"
                "7 8 9 10 11 12 1.5 13 14 15\n"
            )
            catalogue = read_seed_catalogue(path, include_velocity=True)

        self.assertEqual(catalogue.initial_conditions.shape, (2, 6))
        np.testing.assert_allclose(catalogue.seed_weights, [0.5, 1.5])
        np.testing.assert_allclose(catalogue.velocity_errors["vtheta"], [12, 15])

    def test_density_solved_catalogue_does_not_require_a_weight_column(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.txt"
            path.write_text(
                "x_gc y_gc z_gc vx_gc vy_gc vz_gc vr_err vphi_err vthe_err\n"
                "1 2 3 4 5 6 10 11 12\n"
                "7 8 9 10 11 12 13 14 15\n"
            )
            catalogue = read_seed_catalogue(
                path,
                include_velocity=True,
                require_weights=False,
            )

        self.assertIsNone(catalogue.seed_weights)
        self.assertEqual(catalogue.initial_conditions.shape, (2, 6))
        np.testing.assert_allclose(catalogue.velocity_errors["vr"], [10, 13])

    def test_legacy_z_r_phi_flattening_is_transposed_once(self):
        grid = CylindricalGrid.uniform(
            n_r=25,
            r_range=(0, 50),
            n_z=25,
            z_range=(0, 50),
            n_phi=4,
        )
        source = np.arange(2500, dtype=float).reshape(25, 25, 4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "density.txt"
            rows = ["den den_srr"]
            rows.extend(
                f"{value:.1f} {value + 100:.1f}" for value in source.ravel()
            )
            path.write_text("\n".join(rows) + "\n")
            density, error = read_target_density(path, grid)

        self.assertEqual(density.shape, (25, 25, 4))
        np.testing.assert_array_equal(density, np.transpose(source, (1, 0, 2)))
        np.testing.assert_array_equal(error, np.transpose(source + 100, (1, 0, 2)))

    def test_target_size_mismatch_is_rejected(self):
        grid = CylindricalGrid.uniform(n_r=25, n_z=25, n_phi=4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "density.txt"
            path.write_text("den den_srr\n1 1\n")
            with self.assertRaisesRegex(ValueError, "expected 2500"):
                read_target_density(path, grid)

    def test_custom_grid_npz_requires_matching_edges(self):
        grid = CylindricalGrid.uniform(
            n_r=2,
            r_range=(0, 4),
            n_z=3,
            z_range=(-2, 4),
            n_phi=2,
        )
        density = np.arange(12, dtype=float).reshape(grid.shape)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "target.npz"
            np.savez_compressed(
                path,
                target_density=density,
                target_error=np.ones_like(density),
                r_edges=grid.r_edges,
                z_edges=grid.z_edges,
                phi_edges=grid.phi_edges,
            )
            loaded, error = read_target_density(path, grid)

        np.testing.assert_array_equal(loaded, density)
        np.testing.assert_array_equal(error, np.ones_like(density))

    def test_custom_grid_rejects_metadata_free_ascii(self):
        grid = CylindricalGrid.uniform(n_r=1, n_z=1, n_phi=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "target.txt"
            path.write_text("den den_srr\n1 1\n2 1\n")
            with self.assertRaisesRegex(ValueError, "metadata-free ASCII"):
                read_target_density(path, grid)


if __name__ == "__main__":
    unittest.main()
