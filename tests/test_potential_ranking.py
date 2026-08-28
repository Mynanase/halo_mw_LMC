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

    def _write_run(self, root: Path, name: str, objectives: list[float]) -> None:
        run = root / name
        (run / "benchmark_metadata").mkdir(parents=True)
        lines = [
            "# iteration qhalo phalo rho0 rho0_plus_2logrs gamma "
            "objective objective_velocity density_shell_phi_gate_passed "
            "weight_solver_converged failed_orbits"
        ]
        for index, (coordinates, objective) in enumerate(
            zip(self.coordinates, objectives)
        ):
            values = " ".join(str(value) for value in coordinates)
            lines.append(
                f"{index} {values} {objective} {objective} 1 1 0"
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


if __name__ == "__main__":
    unittest.main()
