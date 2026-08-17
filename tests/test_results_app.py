import ast
import unittest
from pathlib import Path


APP_PATH = Path(__file__).resolve().parents[1] / "apps" / "results.py"


class ResultsAppBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = APP_PATH.read_text()
        cls.tree = ast.parse(cls.source, filename=str(APP_PATH))

    def test_app_uses_only_the_read_side_of_the_artifact_api(self):
        for api_name in (
            "discover_runs",
            "load_run_summary",
            "load_best_evaluation",
        ):
            self.assertIn(api_name, self.source)

    def test_app_does_not_import_execution_or_physics_modules(self):
        imported = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)

        forbidden_prefixes = (
            "agama",
            "astropy",
            "skopt",
            "skopt_oint_lamost_4phi",
            "run_skopt_lamost_4phi",
            "plot_best_fit",
            "halo_mw_lmc.workflows",
            "halo_mw_lmc.orbits",
            "halo_mw_lmc.potentials",
            "halo_mw_lmc.core.orbits",
            "halo_mw_lmc.core.potentials",
        )
        violations = sorted(
            module
            for module in imported
            if any(
                module == prefix or module.startswith(prefix + ".")
                for prefix in forbidden_prefixes
            )
        )
        self.assertEqual(violations, [])

    def test_app_contains_no_write_or_model_execution_calls(self):
        forbidden_attributes = {
            "dump",
            "mkdir",
            "rename",
            "replace",
            "save",
            "savefig",
            "savetxt",
            "savez",
            "savez_compressed",
            "touch",
            "unlink",
            "write_bytes",
            "write_text",
        }
        forbidden_names = {
            "evaluate_one_model",
            "evaluate_prepared_model",
            "integrate_agama_orbits",
            "int_one_model",
        }
        violations = []
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if isinstance(function, ast.Attribute) and function.attr in forbidden_attributes:
                violations.append(function.attr)
            elif isinstance(function, ast.Name) and function.id in forbidden_names:
                violations.append(function.id)
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
