import json
import tempfile
import unittest
from pathlib import Path

from halo_mw_lmc.weight_solver_benchmark import compare_solver_runs


HEADER = (
    "# iteration objective objective_velocity inner_weight_objective "
    "density_chi2_per_bin effective_orbit_count max_weight_fraction "
    "active_orbit_count zero_weight_fraction "
    "weight_solver_converged weight_solver_kkt_residual "
    "weight_solver_wall_seconds density_shell_phi_gate_passed\n"
)


class WeightSolverBenchmarkTests(unittest.TestCase):
    def _run(
        self,
        root: Path,
        backend: str,
        *,
        wall: float,
        rss: int,
        kkt: float,
        fingerprint: str = "same-problem",
        directory_name: str | None = None,
        repeats: int = 1,
        inner: float = 4.0,
    ) -> Path:
        run = root / (directory_name or backend)
        (run / "best").mkdir(parents=True)
        (run / "benchmark_metadata").mkdir()
        (run / "resolved_config.json").write_text(
            json.dumps(
                {
                    "schema_version": 7,
                    "weight_model": {
                        "solver": backend,
                        "solver_tolerance": 1e-8,
                    },
                    "objective": {"density_max_chi2_per_bin": 2.0},
                }
            )
        )
        rows = "".join(
            f"{iteration} 12.0 10.0 {inner} 1.0 4.5 0.2 5 0.5 1 {kkt} "
            f"{wall + 0.1 * iteration} 1\n"
            for iteration in range(repeats)
        )
        (run / "sample.dat").write_text(HEADER + rows)
        (run / "best/metadata.json").write_text(
            json.dumps(
                {
                    "schema_version": 4,
                    "iteration": 0,
                    "objective": 12.0,
                    "weight_problem_fingerprint": fingerprint,
                }
            )
        )
        (run / "benchmark_metadata/time-v.txt").write_text(
            f"Maximum resident set size (kbytes): {rss}\n"
        )
        return run

    def _three_runs(
        self,
        root: Path,
        backend: str,
        *,
        wall: float,
        rss: int,
        kkt: float,
        fingerprint: str = "same-problem",
        inner: float = 4.0,
    ) -> list[Path]:
        return [
            self._run(
                root,
                backend,
                wall=wall + repeat,
                rss=rss + repeat,
                kkt=kkt,
                fingerprint=fingerprint,
                directory_name=f"{backend}-{repeat}",
                inner=inner,
            )
            for repeat in range(3)
        ]

    def test_selects_lower_memory_backend_within_twenty_percent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = (
                self._three_runs(
                    root, "lsq_linear", wall=100.0, rss=50, kkt=1e-3
                )
                + self._three_runs(
                    root, "dense_nnls", wall=10.0, rss=200, kkt=1e-10
                )
                + self._three_runs(
                    root, "dual_ridge", wall=11.0, rss=100, kkt=1e-10
                )
            )

            result = compare_solver_runs(runs)

        self.assertTrue(result["same_problem_fingerprint"])
        self.assertEqual(result["winner"], "dual_ridge")
        self.assertFalse(result["production_ready"])
        self.assertFalse(
            result["runs"]["lsq_linear"]["qualified_for_speed_selection"]
        )
        self.assertTrue(
            result["runs"]["dual_ridge"]["qualified_for_speed_selection"]
        )

    def test_mismatched_problem_fingerprint_blocks_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = (
                self._three_runs(
                    root, "lsq_linear", wall=100.0, rss=50, kkt=1e-3
                )
                + self._three_runs(
                    root,
                    "dense_nnls",
                    wall=10.0,
                    rss=200,
                    kkt=1e-10,
                    fingerprint="dense-problem",
                )
                + self._three_runs(
                    root,
                    "dual_ridge",
                    wall=11.0,
                    rss=100,
                    kkt=1e-10,
                    fingerprint="dual-problem",
                )
            )

            result = compare_solver_runs(runs)

        self.assertFalse(result["same_problem_fingerprint"])
        self.assertIsNone(result["winner"])

    def test_baseline_remains_winner_when_it_is_qualified_and_fastest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = (
                self._three_runs(
                    root, "lsq_linear", wall=5.0, rss=50, kkt=1e-10
                )
                + self._three_runs(
                    root, "dense_nnls", wall=10.0, rss=200, kkt=1e-10
                )
                + self._three_runs(
                    root, "dual_ridge", wall=11.0, rss=100, kkt=1e-10
                )
            )

            result = compare_solver_runs(runs)

        self.assertEqual(result["winner"], "lsq_linear")
        self.assertTrue(
            result["runs"]["lsq_linear"]["qualified_for_speed_selection"]
        )
        self.assertEqual(
            result["selection_reason"],
            "selected fastest qualified backend",
        )

    def test_aggregates_three_independent_cold_start_runs_per_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = []
            for backend, wall, rss, kkt in (
                ("lsq_linear", 100.0, 50, 1e-3),
                ("dense_nnls", 10.0, 200, 1e-10),
                ("dual_ridge", 8.0, 100, 1e-10),
            ):
                for repeat in range(3):
                    runs.append(
                        self._run(
                            root,
                            backend,
                            wall=wall + repeat,
                            rss=rss + repeat,
                            kkt=kkt,
                            directory_name=f"{backend}-{repeat}",
                        )
                    )

            result = compare_solver_runs(runs)

        self.assertEqual(result["winner"], "dual_ridge")
        self.assertEqual(result["runs"]["dense_nnls"]["repeats"], 3)
        self.assertEqual(
            result["runs"]["dual_ridge"]["median_wall_seconds"],
            9.0,
        )

    def test_rejects_multiple_samples_from_one_adaptive_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = self._run(
                root,
                "lsq_linear",
                wall=1.0,
                rss=50,
                kkt=1e-10,
                repeats=3,
            )

            with self.assertRaisesRegex(ValueError, "exactly one cold-start"):
                compare_solver_runs([run])


if __name__ == "__main__":
    unittest.main()
