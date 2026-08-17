import unittest
from pathlib import Path
from unittest.mock import patch

from halo_mw_lmc.workflows.run import run_full_workflow


class FullRunWorkflowTests(unittest.TestCase):
    @patch("halo_mw_lmc.workflows.run.generate_report")
    @patch("halo_mw_lmc.workflows.run.run_optimization")
    @patch("halo_mw_lmc.workflows.run.importlib.util.find_spec")
    def test_default_workflow_optimizes_then_reports(
        self,
        find_spec,
        run_optimization,
        generate_report,
    ):
        configuration = object()
        run_directory = Path("run-output")
        report_paths = [run_directory / "figures" / "convergence.pdf"]
        find_spec.return_value = object()
        run_optimization.return_value = run_directory
        generate_report.return_value = report_paths

        result = run_full_workflow(configuration)

        run_optimization.assert_called_once_with(configuration)
        generate_report.assert_called_once_with(configuration)
        self.assertEqual(result.run_directory, run_directory)
        self.assertEqual(result.report_paths, tuple(report_paths))

    @patch("halo_mw_lmc.workflows.run.generate_report")
    @patch("halo_mw_lmc.workflows.run.run_optimization")
    @patch("halo_mw_lmc.workflows.run.importlib.util.find_spec", return_value=None)
    def test_default_workflow_preflights_report_dependency(
        self,
        _find_spec,
        run_optimization,
        generate_report,
    ):
        with self.assertRaisesRegex(RuntimeError, "Matplotlib"):
            run_full_workflow(object())

        run_optimization.assert_not_called()
        generate_report.assert_not_called()
