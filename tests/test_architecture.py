import ast
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
CORE = REPOSITORY / "halo_mw_lmc/core"


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


class DependencyDirectionTests(unittest.TestCase):
    def test_core_does_not_depend_on_operational_or_display_layers(self):
        forbidden = (
            "halo_mw_lmc.configuration",
            "halo_mw_lmc.data",
            "halo_mw_lmc.workflows",
            "halo_mw_lmc.visualization",
            "astropy",
            "matplotlib",
            "marimo",
            "skopt",
        )
        violations = []
        for path in CORE.glob("*.py"):
            for module in imported_modules(path):
                if module.startswith(forbidden):
                    violations.append(f"{path.name}: {module}")
        self.assertEqual(violations, [])

    def test_optimizer_does_not_import_visualization_or_reporting(self):
        path = REPOSITORY / "halo_mw_lmc/workflows/optimization.py"
        imports = imported_modules(path)
        self.assertFalse(
            any(
                module.startswith(
                    ("halo_mw_lmc.visualization", ".visualization", "reporting")
                )
                for module in imports
            )
        )

    def test_reporting_and_inspection_do_not_import_execution_or_source_data(self):
        for relative in (
            "halo_mw_lmc/workflows/reporting.py",
            "halo_mw_lmc/inspection.py",
        ):
            with self.subTest(path=relative):
                imports = imported_modules(REPOSITORY / relative)
                self.assertFalse(
                    any(
                        module.startswith(
                            (
                                "halo_mw_lmc.data",
                                ".data",
                                "optimization",
                                ".optimization",
                                "agama",
                                "skopt",
                            )
                        )
                        for module in imports
                    )
                )


if __name__ == "__main__":
    unittest.main()
