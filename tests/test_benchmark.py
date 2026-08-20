import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from halo_mw_lmc.benchmark import validate_benchmark_preflight
from halo_mw_lmc.configuration import load_run_configuration


REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG = REPOSITORY / "configs/runs/density_solved_r8_40_benchmark.toml"
LAUNCHER = REPOSITORY / "scripts/run_density_solved_r8_40_case.sh"


class BenchmarkPreflightTests(unittest.TestCase):
    def _time_version(self, *args, **kwargs):
        return SimpleNamespace(stdout="GNU time 1.9\n", stderr="")

    @patch("halo_mw_lmc.benchmark.Path.is_file", return_value=True)
    @patch("halo_mw_lmc.benchmark.os.access", return_value=True)
    @patch("halo_mw_lmc.benchmark.subprocess.run")
    def test_valid_case_passes(
        self,
        run,
        _access,
        _is_file,
    ):
        run.side_effect = self._time_version

        result = validate_benchmark_preflight(REPOSITORY, CONFIG)

        self.assertEqual(result.configuration.iterations, 1)
        run.assert_called_once_with(
            ["/usr/bin/time", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )

    @patch("halo_mw_lmc.benchmark.Path.is_file", return_value=False)
    def test_missing_gnu_time_fails(self, _is_file):
        with self.assertRaisesRegex(RuntimeError, "GNU time executable not found"):
            validate_benchmark_preflight(REPOSITORY, CONFIG)

    @patch("halo_mw_lmc.benchmark.Path.is_file", return_value=True)
    @patch("halo_mw_lmc.benchmark.os.access", return_value=True)
    @patch(
        "halo_mw_lmc.benchmark.subprocess.run",
        return_value=SimpleNamespace(stdout="BSD time\n", stderr=""),
    )
    def test_non_gnu_time_fails(self, _run, _access, _is_file):
        with self.assertRaisesRegex(RuntimeError, "requires GNU time"):
            validate_benchmark_preflight(REPOSITORY, CONFIG)

    @patch("halo_mw_lmc.benchmark.Path.is_file", return_value=True)
    @patch("halo_mw_lmc.benchmark.os.access", return_value=True)
    @patch("halo_mw_lmc.benchmark.subprocess.run")
    def test_existing_output_directory_fails_before_execution(
        self,
        run,
        _access,
        _is_file,
    ):
        run.side_effect = self._time_version
        configuration = load_run_configuration(CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            configuration = replace(
                configuration,
                run=replace(configuration.run, output_dir=Path(directory)),
            )
            with patch(
                "halo_mw_lmc.benchmark.load_run_configuration",
                return_value=configuration,
            ):
                with self.assertRaisesRegex(RuntimeError, "already exists"):
                    validate_benchmark_preflight(REPOSITORY, CONFIG)


class BenchmarkLauncherPolicyTests(unittest.TestCase):
    def test_launcher_records_git_state_without_locking_it(self):
        source = LAUNCHER.read_text()

        self.assertIn('git rev-parse HEAD > "$STAGING/git-head.txt"', source)
        self.assertIn(
            'git status --porcelain --untracked-files=all > '
            '"$STAGING/git-status.txt"',
            source,
        )
        self.assertNotIn("LOCKED_GIT_SHA", source)
        self.assertNotIn("git diff", source)

    def test_launcher_usage_has_no_sha_argument(self):
        completed = subprocess.run(
            ["bash", str(LAUNCHER)],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("RUN_CONFIG [--preflight-only]", completed.stderr)
        self.assertNotIn("GIT_SHA", completed.stderr)


if __name__ == "__main__":
    unittest.main()
