import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from halo_mw_lmc.workflows.run import run_full_workflow


class FullRunWorkflowTests(unittest.TestCase):
    @patch("halo_mw_lmc.workflows.run.save_inspection")
    @patch("halo_mw_lmc.workflows.run.inspect_run")
    @patch("halo_mw_lmc.workflows.run.generate_report_from_run")
    @patch("halo_mw_lmc.workflows.run.run_optimization")
    @patch("halo_mw_lmc.workflows.run.require_preflight")
    @patch("halo_mw_lmc.workflows.run.preflight_and_prepare")
    def test_default_workflow_prepares_once_then_reports(
        self,
        preflight_and_prepare,
        require_preflight,
        run_optimization,
        generate_report,
        inspect_run,
        save_inspection,
    ):
        configuration = SimpleNamespace(output_dir=Path("run-output"))
        prepared = object()
        preflight = SimpleNamespace(
            execution=prepared,
            numerical_stage="optimize",
        )
        require_preflight.return_value = preflight
        run_directory = Path("run-output")
        report_paths = [run_directory / "report" / "convergence.pdf"]
        run_optimization.return_value = run_directory
        generate_report.return_value = report_paths
        inspection = SimpleNamespace(numerical_status="complete")
        inspect_run.return_value = inspection
        save_inspection.return_value = run_directory / "inspection.json"

        result = run_full_workflow(configuration)

        preflight_and_prepare.assert_called_once_with(configuration, stage="run")
        run_optimization.assert_called_once_with(configuration, prepared)
        generate_report.assert_called_once_with(run_directory)
        self.assertEqual(result.run_directory, run_directory)
        self.assertEqual(result.report_paths, tuple(report_paths))
        self.assertEqual(result.inspection_path, run_directory / "inspection.json")

    @patch("halo_mw_lmc.workflows.run.run_optimization")
    @patch("halo_mw_lmc.workflows.run.require_preflight")
    @patch("halo_mw_lmc.workflows.run.preflight_and_prepare")
    def test_preflight_failure_creates_no_run(
        self,
        preflight_and_prepare,
        require_preflight,
        run_optimization,
    ):
        require_preflight.side_effect = RuntimeError("preflight failed")
        with self.assertRaisesRegex(RuntimeError, "preflight failed"):
            run_full_workflow(SimpleNamespace(output_dir=Path("unused")))
        run_optimization.assert_not_called()

    @patch("halo_mw_lmc.workflows.run.save_inspection")
    @patch("halo_mw_lmc.workflows.run.inspect_run")
    @patch(
        "halo_mw_lmc.workflows.run.generate_report_from_run",
        side_effect=RuntimeError("report failed"),
    )
    @patch("halo_mw_lmc.workflows.run.run_optimization")
    @patch("halo_mw_lmc.workflows.run.require_preflight")
    @patch("halo_mw_lmc.workflows.run.preflight_and_prepare")
    def test_report_failure_preserves_complete_numerical_status(
        self,
        _preflight_and_prepare,
        require_preflight,
        run_optimization,
        _generate_report,
        inspect_run,
        save_inspection,
    ):
        configuration = SimpleNamespace(output_dir=Path("run-output"))
        prepared = object()
        require_preflight.return_value = SimpleNamespace(
            execution=prepared,
            numerical_stage="optimize",
        )
        run_optimization.return_value = configuration.output_dir
        complete = SimpleNamespace(numerical_status="complete")
        failed = SimpleNamespace(numerical_status="complete")
        inspect_run.side_effect = [complete, failed]

        with self.assertRaisesRegex(RuntimeError, "report failed"):
            run_full_workflow(configuration)

        inspect_run.assert_called_with(
            configuration.output_dir,
            report_failure="report failed",
        )
        save_inspection.assert_called_with(failed)


if __name__ == "__main__":
    unittest.main()
