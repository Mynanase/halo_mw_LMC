import ast
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
PACKAGE = REPOSITORY / "halo_mw_lmc"

NUMERICAL = ("potential", "orbits", "grids", "density", "weights", "velocity")
STAGES = (
    "catalogue",
    "config",
    "prepare",
    "evaluate",
    "optimize",
    "run",
    "report",
    "inspection",
    "artifacts",
    "benchmark",
    "synthetic_density",
    "solver_budget",
    "weight_solver_benchmark",
    "cli",
)
PLOTS = ("plot_model", "plot_coverage", "plot_constraints", "plot_weights", "plot_convergence")


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def flat_module(name: str) -> str:
    """Normalize halo_mw_lmc.X / .X / X import spellings to the flat name."""
    for prefix in ("halo_mw_lmc.", "."):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    return name.split(".")[0]


class DependencyDirectionTests(unittest.TestCase):
    def test_numerical_modules_do_not_depend_on_data_execution_or_display(self):
        forbidden = set(STAGES) | set(PLOTS) | {
            "astropy",
            "matplotlib",
            "marimo",
            "skopt",
        }
        forbidden.discard("config")  # config.py is stdlib-only value definitions
        violations = []
        for name in NUMERICAL:
            for module in imported_modules(PACKAGE / f"{name}.py"):
                if flat_module(module) in forbidden:
                    violations.append(f"{name}.py: {module}")
        self.assertEqual(violations, [])

    def test_optimizer_does_not_import_display_or_reporting(self):
        imports = imported_modules(PACKAGE / "optimize.py")
        forbidden = set(PLOTS) | {"report", "marimo"}
        violations = [
            module
            for module in imports
            if flat_module(module) in forbidden
        ]
        self.assertEqual(violations, [])

    def test_inspection_reads_artifacts_only(self):
        forbidden = set(NUMERICAL) | set(PLOTS) | {
            "catalogue",
            "prepare",
            "evaluate",
            "optimize",
            "report",
            "agama",
            "skopt",
        }
        violations = [
            module
            for module in imported_modules(PACKAGE / "inspection.py")
            if flat_module(module) in forbidden
        ]
        self.assertEqual(violations, [])

    def test_report_never_reintegrates_or_reoptimizes(self):
        forbidden = {"evaluate", "optimize", "agama", "skopt", "marimo"}
        violations = [
            module
            for module in imported_modules(PACKAGE / "report.py")
            if flat_module(module) in forbidden
        ]
        self.assertEqual(violations, [])

    def test_flat_layout_has_no_layer_directories(self):
        for relative in ("core", "data", "workflows", "visualization"):
            self.assertFalse(
                (PACKAGE / relative).exists(),
                f"layer directory {relative}/ must not exist on the flat layout",
            )


if __name__ == "__main__":
    unittest.main()
