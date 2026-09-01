import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from halo_mw_lmc.inspection import inspect_run
from halo_mw_lmc.workflows.reporting import generate_report_from_run

from tests.artifact_fixture import write_complete_run


class ManagedReportTests(unittest.TestCase):
    def test_artifact_only_report_renders_nonempty_pdfs_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            run = write_complete_run(
                Path(directory) / "run",
                include_velocity=True,
            )
            (run / "fixed_seed_weights.npz").write_bytes(b"not-report-input")
            written = generate_report_from_run(run)
            manifest = json.loads((run / "report/manifest.json").read_text())
            inspection = inspect_run(run)

            expected = (
                run / "report/convergence.pdf",
                run / "report/parameter_constraints.pdf",
                run / "report/density/overview.pdf",
                run / "report/density/flattening.pdf",
                run / "report/density/shell_phi_gate.pdf",
                run / "report/density/phi_00.pdf",
                run / "report/velocity/phi_average.pdf",
                run / "report/velocity/phi_00.pdf",
                run / "report/weights/distribution.pdf",
            )
            for path in expected:
                self.assertTrue(path.is_file(), path)
                self.assertGreater(path.stat().st_size, 0)

        self.assertTrue(written)
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["resolved_config_schema"], 7)
        self.assertNotIn(
            "velocity/: no saved velocity distributions",
            manifest["omitted"],
        )
        self.assertEqual(inspection.report_status, "current")

    def test_default_refuses_existing_report_and_overwrite_replaces_it(self):
        with tempfile.TemporaryDirectory() as directory:
            run = write_complete_run(Path(directory) / "run")
            generate_report_from_run(run)
            marker = run / "report/old-marker.txt"
            marker.write_text("old")
            with self.assertRaises(FileExistsError):
                generate_report_from_run(run)

            generate_report_from_run(run, overwrite=True)

            self.assertFalse(marker.exists())
            self.assertTrue((run / "report/manifest.json").exists())

    def test_failed_overwrite_preserves_published_report(self):
        with tempfile.TemporaryDirectory() as directory:
            run = write_complete_run(Path(directory) / "run")
            generate_report_from_run(run)
            manifest_before = (run / "report/manifest.json").read_bytes()

            with patch(
                "halo_mw_lmc.workflows.reporting._render_report",
                side_effect=RuntimeError("render failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "render failed"):
                    generate_report_from_run(run, overwrite=True)

            self.assertEqual(
                (run / "report/manifest.json").read_bytes(),
                manifest_before,
            )


if __name__ == "__main__":
    unittest.main()
