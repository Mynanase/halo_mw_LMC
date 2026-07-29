import tempfile
import unittest
from pathlib import Path

from halo_mw_lmc.samples import (
    SampleFileError,
    best_sample,
    load_sample_table,
)
from plot_best_fit import _best_row


HEADER = (
    "# iteration qhalo phalo rho0 rho0_plus_2logrs gamma "
    "objective chi2 density_scale successful_orbits "
    "chi2_phi0 chi2_phi1 chi2_phi2\n"
)


class SampleFileTests(unittest.TestCase):
    def _write_sample(self, directory: str, rows: str) -> Path:
        path = Path(directory) / "sample.dat"
        path.write_text(HEADER + rows)
        return path

    def test_single_row_is_kept_one_dimensional(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_sample(
                directory,
                "0 1.0 0.8 6.0 9.8 1.0 12.0 24.0 1.0 10 1 2 3\n",
            )
            data = load_sample_table(path, required_columns=("objective",))

        self.assertEqual(data.shape, (1,))
        self.assertEqual(float(best_sample(data)["objective"]), 12.0)

    def test_missing_required_column_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_sample(
                directory,
                "0 1.0 0.8 6.0 9.8 1.0 12.0 24.0 1.0 10 1 2 3\n",
            )
            with self.assertRaisesRegex(SampleFileError, "missing required"):
                load_sample_table(path, required_columns=("not_a_column",))

    def test_best_row_uses_requested_phi_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_sample(
                directory,
                "0 1.0 0.8 6.0 9.8 1.0 12.0 24.0 1.0 10 1 2 3\n",
            )
            data = load_sample_table(path)

        best = _best_row(data, nphi=3)
        self.assertEqual(best["chi2_by_phi"], [1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
