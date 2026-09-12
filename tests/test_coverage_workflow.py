import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from halo_mw_lmc.config import load_run_configuration
from halo_mw_lmc.report import generate_coverage_report
from halo_mw_lmc.velocity import SphericalVelocityGrid


RUN_CONFIG = Path(__file__).resolve().parents[1] / "configs/runs/fix_weight.toml"


class CoverageWorkflowTests(unittest.TestCase):
    def test_coverage_uses_the_recipe_velocity_spatial_grid(self):
        configuration = load_run_configuration(RUN_CONFIG)
        recipe_grid = configuration["recipe"]["velocity_grid"]
        configuration["recipe"]["velocity_grid"] = SphericalVelocityGrid(
            radius_edges=np.array([5.0, 9.0, 20.0]),
            theta_edges=np.deg2rad(np.array([0.0, 30.0, 90.0])),
            phi_edges=recipe_grid.phi_edges,
            velocity_edges=recipe_grid.velocity_edges,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.txt"
            catalog.write_text("placeholder")
            configuration["data"] = {**configuration["data"], "catalog": catalog}
            configuration["coverage"] = {
                **configuration["coverage"],
                "output_dir": root / "coverage",
            }
            with (
                patch(
                    "halo_mw_lmc.prepare.read_phase_space_catalogue",
                    return_value=np.zeros((1, 6)),
                ),
                patch(
                    "halo_mw_lmc.prepare.build_data_coverage",
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
