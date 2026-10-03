import json
import tempfile
import unittest
from pathlib import Path

from halo_mw_lmc.inspection import inspect_run, save_inspection

from tests.artifact_fixture import write_complete_run


class RunInspectionTests(unittest.TestCase):
    def test_complete_numerical_run_with_missing_report(self):
        with tempfile.TemporaryDirectory() as directory:
            run = write_complete_run(Path(directory) / "run")
            inspection = inspect_run(run)

        self.assertEqual(inspection.numerical_status, "complete")
        self.assertEqual(inspection.report_status, "missing")
        self.assertEqual(inspection.document["trials"], {"planned": 1, "completed": 1})
        self.assertTrue(inspection.document["artifacts"]["best_matches_sample"])

    def test_partial_sample_and_missing_best_are_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            run = write_complete_run(Path(directory) / "run", iterations=2)
            (run / "sample.dat").write_text("# iteration objective\n0 2.0\n")
            for path in (run / "best").iterdir():
                path.unlink()
            inspection = inspect_run(run)

        self.assertEqual(inspection.numerical_status, "incomplete")
        self.assertEqual(inspection.document["trials"]["completed"], 1)

    def test_mixed_best_generation_is_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            run = write_complete_run(Path(directory) / "run")
            metadata = run / "best/metadata.json"
            document = json.loads(metadata.read_text())
            document["generation"] = "not-the-array-generation"
            metadata.write_text(json.dumps(document))
            inspection = inspect_run(run)

        self.assertEqual(inspection.numerical_status, "invalid")
        self.assertTrue(inspection.document["errors"])

    def test_unknown_resolved_schema_is_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            run = write_complete_run(Path(directory) / "run")
            config = run / "resolved_config.json"
            document = json.loads(config.read_text())
            document["schema_version"] = 999
            config.write_text(json.dumps(document))
            inspection = inspect_run(run)

        self.assertEqual(inspection.numerical_status, "invalid")

    def test_report_states_and_atomic_save(self):
        with tempfile.TemporaryDirectory() as directory:
            run = write_complete_run(Path(directory) / "run")
            generation = json.loads((run / "best/metadata.json").read_text())["generation"]
            report = run / "report"
            report.mkdir()
            (report / "summary.md").write_text("ok\n")
            manifest = {
                "schema_version": 1,
                "best_generation": generation,
                "files": ["summary.md"],
            }
            (report / "manifest.json").write_text(json.dumps(manifest))

            current = inspect_run(run)
            destination = save_inspection(current)
            manifest["best_generation"] = "old"
            (report / "manifest.json").write_text(json.dumps(manifest))
            stale = inspect_run(run)
            failed = inspect_run(run, report_failure="render failed")

        self.assertEqual(current.report_status, "current")
        self.assertTrue(destination.name == "inspection.json")
        self.assertEqual(stale.report_status, "stale")
        self.assertEqual(failed.report_status, "failed")


if __name__ == "__main__":
    unittest.main()
