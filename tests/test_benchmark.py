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
COMMIT = "a" * 40


class BenchmarkPreflightTests(unittest.TestCase):
    def _time_version(self, *args, **kwargs):
        return SimpleNamespace(stdout="GNU time 1.9\n", stderr="")

    @patch("halo_mw_lmc.benchmark.Path.is_file", return_value=True)
    @patch("halo_mw_lmc.benchmark.os.access", return_value=True)
    @patch("halo_mw_lmc.benchmark.subprocess.run")
    @patch("halo_mw_lmc.benchmark._git")
    def test_clean_locked_case_passes(
        self,
        git,
        run,
        _access,
        _is_file,
    ):
        git.side_effect = [COMMIT, ""]
        run.side_effect = self._time_version

        result = validate_benchmark_preflight(REPOSITORY, CONFIG, COMMIT)

        self.assertEqual(result.git_commit, COMMIT)
        self.assertEqual(result.configuration.iterations, 1)

    @patch("halo_mw_lmc.benchmark._git", return_value="b" * 40)
    def test_mismatched_commit_fails(self, _git):
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            validate_benchmark_preflight(REPOSITORY, CONFIG, COMMIT)

    @patch("halo_mw_lmc.benchmark._git")
    def test_dirty_worktree_fails(self, git):
        git.side_effect = [COMMIT, " M halo_mw_lmc/core/density.py"]
        with self.assertRaisesRegex(RuntimeError, "not clean"):
            validate_benchmark_preflight(REPOSITORY, CONFIG, COMMIT)

    @patch("halo_mw_lmc.benchmark.Path.is_file", return_value=False)
    @patch("halo_mw_lmc.benchmark._git")
    def test_missing_gnu_time_fails(self, git, _is_file):
        git.side_effect = [COMMIT, ""]
        with self.assertRaisesRegex(RuntimeError, "GNU time executable not found"):
            validate_benchmark_preflight(REPOSITORY, CONFIG, COMMIT)

    @patch("halo_mw_lmc.benchmark.Path.is_file", return_value=True)
    @patch("halo_mw_lmc.benchmark.os.access", return_value=True)
    @patch(
        "halo_mw_lmc.benchmark.subprocess.run",
        return_value=SimpleNamespace(stdout="BSD time\n", stderr=""),
    )
    @patch("halo_mw_lmc.benchmark._git")
    def test_non_gnu_time_fails(self, git, _run, _access, _is_file):
        git.side_effect = [COMMIT, ""]
        with self.assertRaisesRegex(RuntimeError, "requires GNU time"):
            validate_benchmark_preflight(REPOSITORY, CONFIG, COMMIT)

    @patch("halo_mw_lmc.benchmark.Path.is_file", return_value=True)
    @patch("halo_mw_lmc.benchmark.os.access", return_value=True)
    @patch("halo_mw_lmc.benchmark.subprocess.run")
    @patch("halo_mw_lmc.benchmark._git")
    def test_existing_output_directory_fails_before_execution(
        self,
        git,
        run,
        _access,
        _is_file,
    ):
        git.side_effect = [COMMIT, ""]
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
                    validate_benchmark_preflight(REPOSITORY, CONFIG, COMMIT)


if __name__ == "__main__":
    unittest.main()
