import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from halo_mw_lmc.data_coverage import build_data_coverage
from halo_mw_lmc.grids import CylindricalGrid
from plot_data_coverage import _read_initial_conditions, build_parser


class DataCoverageTests(unittest.TestCase):
    def setUp(self):
        self.initial = np.array(
            [
                [1.0, 0.0, 1.0, 10.0, 20.0, 30.0],
                [-1.0, 0.0, 1.0, -10.0, 5.0, 15.0],
                [0.0, 2.0, 3.0, 40.0, -20.0, 10.0],
                [np.nan, 0.0, 1.0, 10.0, 20.0, 30.0],
                [1.0, 1.0, 1.0, np.nan, 20.0, 30.0],
            ]
        )
        self.grid = CylindricalGrid.uniform(
            n_r=2,
            r_range=(0.0, 4.0),
            n_z=2,
            z_range=(0.0, 4.0),
            n_phi=4,
        )

    def test_complete_rows_drive_spatial_coverage(self):
        coverage = build_data_coverage(
            self.initial,
            rzphi_grid=self.grid,
            spherical_radius_edges=np.array([0.0, 2.0, 4.0]),
            theta_edges=np.array([0.0, np.pi / 4, np.pi / 2]),
        )

        self.assertEqual(coverage.input_rows, 5)
        self.assertEqual(coverage.position_finite_rows, 4)
        self.assertEqual(coverage.complete_phase_space_rows, 3)
        self.assertEqual(int(coverage.rzphi_counts.sum()), 3)
        self.assertEqual(int(coverage.rtheta_phi_counts.sum()), 3)
        self.assertEqual(coverage.rzphi_counts.shape, (2, 2, 4))
        self.assertEqual(coverage.rtheta_phi_counts.shape, (2, 2, 4))

    def test_summary_reports_empty_and_low_occupancy_cells(self):
        coverage = build_data_coverage(
            self.initial,
            rzphi_grid=self.grid,
            spherical_radius_edges=np.array([0.0, 2.0, 4.0]),
            theta_edges=np.array([0.0, np.pi / 4, np.pi / 2]),
        )

        summary = coverage.summary()
        self.assertEqual(summary["complete_6d_rows"], 3)
        self.assertAlmostEqual(summary["complete_6d_fraction"], 3 / 5)
        self.assertGreater(summary["rzphi"]["empty_cells"], 0)
        self.assertGreater(summary["rtheta_phi"]["empty_cells"], 0)
        self.assertEqual(sum(summary["rzphi"]["rows_by_phi"]), 3)
        self.assertEqual(
            summary["rzphi"]["in_grid_rows"]
            + summary["rzphi"]["outside_grid_rows"],
            3,
        )

    def test_sampling_density_preserves_counts_after_volume_integration(self):
        coverage = build_data_coverage(
            self.initial,
            rzphi_grid=self.grid,
            spherical_radius_edges=np.array([0.0, 2.0, 4.0]),
            theta_edges=np.array([0.0, np.pi / 4, np.pi / 2]),
        )

        self.assertAlmostEqual(
            float(np.sum(coverage.rzphi_sampling_density * self.grid.volumes)),
            3.0,
        )
        self.assertAlmostEqual(
            float(
                np.sum(
                    coverage.rtheta_phi_sampling_density
                    * coverage.spherical_cell_volumes
                )
            ),
            3.0,
        )

    def test_no_complete_rows_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no complete"):
            build_data_coverage(
                np.full((2, 6), np.nan),
                rzphi_grid=self.grid,
            )

    def test_cli_defaults_match_the_active_spatial_grid(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.nphi, 4)
        self.assertEqual(args.n_rz, 25)
        self.assertEqual(args.rz_max, 50.0)
        self.assertEqual(args.z_min, 0.0)
        self.assertEqual(args.z_max, 50.0)

    def test_ascii_catalogue_can_be_read_without_pipeline_setup(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.txt"
            path.write_text(
                "x_gc y_gc z_gc vx_gc vy_gc vz_gc\n"
                "1 2 3 4 5 6\n"
                "7 8 9 10 11 12\n"
            )
            initial = _read_initial_conditions(path)

        np.testing.assert_allclose(
            initial,
            [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
        )


if __name__ == "__main__":
    unittest.main()
