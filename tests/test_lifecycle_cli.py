import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from halo_mw_lmc.cli import main

from tests.artifact_fixture import write_complete_run


RUN_CONFIG = Path(__file__).resolve().parents[1] / "configs/runs/fix_weight.toml"


class LifecycleCliTests(unittest.TestCase):
    def test_validate_json_writes_only_structured_stdout(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = main(["validate", str(RUN_CONFIG), "--json"])

        document = json.loads(output.getvalue())
        self.assertEqual(status, 0)
        self.assertTrue(document["valid"])

    def test_inspect_incomplete_returns_zero_but_invalid_returns_one(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = write_complete_run(root / "run", iterations=2)
            (run / "sample.dat").write_text("# iteration objective\n0 2.0\n")
            complete_output = io.StringIO()
            with contextlib.redirect_stdout(complete_output):
                incomplete_status = main(["inspect", str(run), "--json"])
            invalid_output = io.StringIO()
            with contextlib.redirect_stdout(invalid_output):
                invalid_status = main(["inspect", str(root / "missing"), "--json"])

        self.assertEqual(incomplete_status, 0)
        self.assertEqual(json.loads(complete_output.getvalue())["numerical_status"], "incomplete")
        self.assertEqual(invalid_status, 1)
        self.assertEqual(json.loads(invalid_output.getvalue())["numerical_status"], "invalid")

    def test_usage_error_uses_argparse_exit_code_two(self):
        with self.assertRaises(SystemExit) as raised:
            main(["preflight", str(RUN_CONFIG), "--stage", "unknown"])
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
