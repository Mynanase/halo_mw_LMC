import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPOSITORY / "scripts/compare_density_solved_r8_40_potential_ranking.py"
)
SPEC = importlib.util.spec_from_file_location("potential_ranking_comparison", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PotentialRankingComparisonTests(unittest.TestCase):
    coordinates = (
        (0.920, 0.800, 6.200, 9.890, 1.000),
        (0.820, 0.700, 6.200, 9.890, 1.000),
        (1.020, 0.950, 6.200, 9.890, 1.000),
        (0.920, 0.800, 6.500, 9.800, 1.200),
        (0.920, 0.800, 5.900, 10.050, 0.800),
    )

    def _write_run(
        self, root: Path, name: str, objectives: list[float], *,
        chi2: list[float] | None = None, gate: bool = True,
        converged: bool = True, failed_orbits: int = 0,
    ) -> None:
        run = root / name
        (run / "benchmark_metadata").mkdir(parents=True)
        lines = [
            "# iteration qhalo phalo rho0 rho0_plus_2logrs gamma "
            "objective objective_velocity objective_density_velocity chi2 "
            "density_shell_phi_gate_passed "
            "weight_solver_converged failed_orbits"
        ]
        for index, (coordinates, objective) in enumerate(
            zip(self.coordinates, objectives)
        ):
            values = " ".join(str(value) for value in coordinates)
            density_chi2 = 0.0 if chi2 is None else chi2[index]
            joint = objective + 0.5 * density_chi2
            selected = objective if gate and converged else 1e30
            lines.append(
                f"{index} {values} {selected} {objective} {joint} {density_chi2} "
                f"{int(gate)} {int(converged)} {failed_orbits}"
            )
        (run / "sample.dat").write_text("\n".join(lines) + "\n")
        tolerance = 1e-7 if "tol1e7" in name else 1e-8
        (run / "resolved_config.json").write_text(
            json.dumps(
                {
                    "git_commit": "a" * 40,
                    "git_dirty": False,
                    "optimizer": {
                        "schedule": "fixed_points",
                        "fixed_points": self.coordinates,
                    },
                    "weight_model": {"lsmr_tol": tolerance},
                    "objective": {"mode": "velocity_only"},
                }
            )
        )
        (run / "benchmark_metadata/input-sha256.txt").write_text(
            "same inputs\n"
        )

    def test_constant_tolerance_offset_preserves_ranking(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = [100.0, 200.0, 300.0, 400.0, 500.0]
            right = [value + 200.0 for value in left]
            self._write_run(root, MODULE.RUN_NAMES["tol1e7"], left)
            self._write_run(root, MODULE.RUN_NAMES["tol1e8"], right)

            result = MODULE.compare_runs(root)

        self.assertTrue(result["criteria"]["ranking_stable"])
        self.assertEqual(
            result["criteria"]["max_differential_shift_fraction_of_span"],
            0.0,
        )

    def test_changed_best_point_fails_ranking_stability(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_run(
                root,
                MODULE.RUN_NAMES["tol1e7"],
                [100.0, 200.0, 300.0, 400.0, 500.0],
            )
            self._write_run(
                root,
                MODULE.RUN_NAMES["tol1e8"],
                [550.0, 250.0, 350.0, 450.0, 150.0],
            )

            result = MODULE.compare_runs(root)

        self.assertFalse(result["criteria"]["same_best_point"])
        self.assertFalse(result["criteria"]["ranking_stable"])

    def test_joint_rescoring_uses_density_and_recovers_gate_rejected_points(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = [100.0, 200.0, 300.0, 400.0, 500.0]
            for tag, offset in (("tol1e7", 0), ("tol1e8", 200)):
                self._write_run(
                    root, MODULE.RUN_NAMES[tag], [v + offset for v in left],
                    chi2=[1000.0, 0.0, 0.0, 0.0, 0.0], gate=False,
                )
            legacy = MODULE.compare_runs(root)
            joint = MODULE.compare_runs(root, objective_mode="density_velocity")

        self.assertFalse(legacy["criteria"]["all_points_valid"])
        self.assertTrue(joint["criteria"]["ranking_stable"])
        self.assertEqual(joint["points"][0]["objective_tol1e7"], 600.0)
        self.assertEqual(joint["points"][0]["density_term_tol1e7"], 500.0)
        self.assertEqual(joint["points"][1]["rank_tol1e7"], 0)
        self.assertFalse(joint["runs"]["tol1e7"]["density_gate_applied"])
        self.assertEqual(
            joint["runs"]["tol1e7"]["recorded_density_gate_by_point"], [False] * 5
        )

    def test_joint_rescoring_keeps_solver_and_failed_orbit_rejections(self):
        for failure in ({"converged": False}, {"failed_orbits": 1}):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for name in MODULE.RUN_NAMES.values():
                    self._write_run(root, name, [1, 2, 3, 4, 5], **failure)
                result = MODULE.compare_runs(root, objective_mode="density_velocity")
            self.assertFalse(result["criteria"]["all_points_valid"])
            self.assertFalse(result["criteria"]["ranking_stable"])

    def test_joint_decomposition_must_match_saved_components(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in MODULE.RUN_NAMES.values():
                self._write_run(root, name, [100, 200, 300, 400, 500])
            sample = root / MODULE.RUN_NAMES["tol1e7"] / "sample.dat"
            lines = sample.read_text().splitlines()
            columns = lines[0].lstrip("# ").split()
            values = lines[1].split()
            values[columns.index("objective_density_velocity")] = "101.0"
            lines[1] = " ".join(values)
            sample.write_text("\n".join(lines) + "\n")
            with self.assertRaisesRegex(ValueError, "joint objective does not equal"):
                MODULE.compare_runs(root, objective_mode="density_velocity")

    def test_joint_rescoring_rejects_nonfinite_components(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in MODULE.RUN_NAMES.values():
                self._write_run(
                    root, name, [100, 200, 300, 400, 500],
                    chi2=[float("nan"), 0, 0, 0, 0],
                )
            with self.assertRaisesRegex(ValueError, "non-finite"):
                MODULE.compare_runs(root, objective_mode="density_velocity")

    def test_screening_can_pass_with_one_discordant_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_run(root, MODULE.RUN_NAMES["tol1e7"], [100, 101, 200, 300, 50])
            self._write_run(root, MODULE.RUN_NAMES["tol1e8"], [101, 100, 200, 300, 50])
            result = MODULE.compare_runs(root, objective_mode="density_velocity")
        self.assertTrue(result["criteria"]["ranking_stable"])
        self.assertAlmostEqual(result["criteria"]["spearman_rank_correlation"], 0.9)
        self.assertEqual(result["criteria"]["pairwise_order_agreement"], 0.9)


if __name__ == "__main__":
    unittest.main()
