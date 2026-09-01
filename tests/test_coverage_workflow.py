import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from halo_mw_lmc.configuration import load_run_configuration
from halo_mw_lmc.workflows.coverage import generate_coverage_report


RUN_CONFIG = Path(__file__).resolve().parents[1] / "configs/runs/fix_weight.toml"


class CoverageWorkflowTests(unittest.TestCase):
    def test_coverage_uses_the_recipe_velocity_spatial_grid(self):
        configuration = load_run_configuration(RUN_CONFIG)
        custom_velocity = replace(
            configuration.recipe.velocity_fit,
            radius_edges_kpc=(5.0, 9.0, 20.0),
            theta_edges_deg=(0.0, 30.0, 90.0),
        )
        configuration = replace(
            configuration,
            recipe=replace(configuration.recipe, velocity_fit=custom_velocity),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.txt"
            catalog.write_text("placeholder")
            configuration = replace(
                configuration,
                data=replace(configuration.data, catalog=catalog),
                coverage=replace(
                    configuration.coverage,
                    output_dir=root / "coverage",
                ),
            )
            with (
                patch(
                    "halo_mw_lmc.workflows.preflight.read_phase_space_catalogue",
                    return_value=np.zeros((1, 6)),
                ),
                patch(
                    "halo_mw_lmc.workflows.preflight.build_data_coverage",
                    side_effect=RuntimeError("stop after boundary check"),
                ) as build,
            ):
                with self.assertRaisesRegex(RuntimeError, "boundary check"):
                    generate_coverage_report(configuration)

        call = build.call_args
        np.testing.assert_allclose(call.kwargs["spherical_radius_edges"], [5, 9, 20])
        np.testing.assert_allclose(
            call.kwargs["theta_edges"],
            np.deg2rad([0, 30, 90]),
        )


if __name__ == "__main__":
    unittest.main()
