import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from halo_mw_lmc.configuration import load_run_configuration
from halo_mw_lmc.workflows.preflight import PreflightError, preflight_and_prepare


RUN_CONFIG = Path(__file__).resolve().parents[1] / "configs/runs/fix_weight.toml"


class PreflightTests(unittest.TestCase):
    def _temporary_configuration(self, root: Path):
        configuration = load_run_configuration(RUN_CONFIG)
        catalog = root / "catalog.txt"
        target = root / "target.npz"
        catalog.write_text("placeholder")
        target.write_text("placeholder")
        return replace(
            configuration,
            data=replace(
                configuration.data,
                catalog=catalog,
                target_density=target,
            ),
            run=replace(configuration.run, output_dir=root / "run"),
            coverage=replace(configuration.coverage, output_dir=root / "coverage"),
        )

    def test_run_prepares_catalogue_and_target_once_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configuration = self._temporary_configuration(root)
            prepared = SimpleNamespace(
                initial_conditions=np.zeros((2, 6)),
                seed_weights=np.ones(2),
                target_density=np.ones((1, 1, 1)),
            )
            with (
                patch(
                    "halo_mw_lmc.workflows.preflight.importlib.util.find_spec",
                    return_value=object(),
                ),
                patch(
                    "halo_mw_lmc.workflows.preflight.prepare_model_data",
                    return_value=prepared,
                ) as prepare,
                patch(
                    "halo_mw_lmc.workflows.preflight.catalogue_weight_audit",
                    return_value={"total_weight": np.asarray(2.0)},
                ) as audit,
            ):
                result = preflight_and_prepare(configuration, stage="run")

            self.assertTrue(result.ok)
            self.assertIsNotNone(result.execution)
            prepare.assert_called_once()
            audit.assert_called_once()
            self.assertFalse(configuration.output_dir.exists())

    def test_coverage_is_catalogue_only_and_never_probes_numerical_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configuration = self._temporary_configuration(root)
            configuration.data.target_density.unlink()
            dependencies = []

            def find_spec(name):
                dependencies.append(name)
                return object() if name == "matplotlib" else None

            coverage = SimpleNamespace(input_rows=3)
            with (
                patch(
                    "halo_mw_lmc.workflows.preflight.importlib.util.find_spec",
                    side_effect=find_spec,
                ),
                patch(
                    "halo_mw_lmc.workflows.preflight.read_phase_space_catalogue",
                    return_value=np.zeros((3, 6)),
                ) as read,
                patch(
                    "halo_mw_lmc.workflows.preflight.build_data_coverage",
                    return_value=coverage,
                ),
            ):
                result = preflight_and_prepare(configuration, stage="coverage")

            self.assertTrue(result.ok)
            read.assert_called_once_with(configuration.data.catalog)
            self.assertEqual(dependencies, ["astropy", "matplotlib"])
            self.assertFalse(configuration.coverage.output_dir.exists())

    def test_evaluate_rejects_adaptive_configuration(self):
        configuration = load_run_configuration(RUN_CONFIG)
        with self.assertRaisesRegex(PreflightError, "fixed_points"):
            preflight_and_prepare(configuration, stage="evaluate")


if __name__ == "__main__":
    unittest.main()
