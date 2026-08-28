import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from halo_mw_lmc.workflows.reporting import generate_report


class ReportingTests(unittest.TestCase):
    @patch("halo_mw_lmc.workflows.reporting.build_parameter_constraints_figure")
    @patch("halo_mw_lmc.workflows.reporting.build_convergence_figure")
    @patch("halo_mw_lmc.workflows.reporting.plot_model_diagnostics")
    @patch("halo_mw_lmc.workflows.reporting.load_best_evaluation")
    @patch("halo_mw_lmc.workflows.reporting.load_run_summary")
    def test_static_report_writes_parameter_constraint_pdf(
        self,
        load_summary,
        load_best,
        plot_model,
        build_convergence,
        build_constraints,
    ):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.skipTest("Matplotlib is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "figures" / "best").mkdir(parents=True)
            samples = object()
            bounds = {
                "qhalo": [0.7, 1.1],
                "phalo": [0.7, 1.1],
                "rho0": [5.5, 7.0],
                "rho0_plus_2logrs": [9.0, 11.0],
                "gamma": [0.0, 2.0],
            }
            load_summary.return_value = SimpleNamespace(
                samples=samples,
                config={"optimizer": {"bounds": bounds}},
            )
            load_best.return_value = SimpleNamespace(
                density=object(),
                velocity_distributions=object(),
                density_shells=None,
                orbit_support_audit=None,
                metadata={},
            )
            plot_model.return_value = []
            convergence = plt.figure()
            constraints = plt.figure()
            convergence.savefig = MagicMock()
            constraints.savefig = MagicMock()
            build_convergence.return_value = convergence
            build_constraints.return_value = constraints
            configuration = SimpleNamespace(
                output_dir=run,
                report=SimpleNamespace(velocity_bin_factor=3),
            )

            written = generate_report(configuration)

        build_constraints.assert_called_once_with(
            samples,
            {name: tuple(interval) for name, interval in bounds.items()},
        )
        constraints.savefig.assert_called_once()
        saved_path = constraints.savefig.call_args.args[0]
        self.assertEqual(saved_path.name, "parameter_constraints.pdf")
        self.assertIn(saved_path, written)


if __name__ == "__main__":
    unittest.main()
