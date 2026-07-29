import math
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from halo_mw_lmc.potentials import (
    ZHU_2026_BEST_FIT,
    build_zhu_2026_potential,
    zhu_2026_component_parameters,
)


class ZhuPotentialTests(unittest.TestCase):
    def test_best_fit_components_match_paper_model(self):
        components = zhu_2026_component_parameters(
            ZHU_2026_BEST_FIT["rho0"],
            ZHU_2026_BEST_FIT["log_rs"],
            ZHU_2026_BEST_FIT["phalo"],
            ZHU_2026_BEST_FIT["qhalo"],
            ZHU_2026_BEST_FIT["gamma"],
        )

        self.assertEqual(len(components), 4)
        self.assertEqual(
            components[0],
            {
                "type": "Ferrers",
                "mass": 1.6e10,
                "scaleRadius": 3.5,
                "p": 0.44,
                "q": 0.31,
            },
        )
        self.assertEqual(components[1]["type"], "Disk")
        self.assertEqual(components[1]["mass"], 3.16e10)
        self.assertEqual(components[1]["innerCutoffRadius"], 7.0)
        self.assertEqual(components[1]["scaleHeight"], 0.3)
        self.assertEqual(components[2]["type"], "Disk")
        self.assertEqual(components[2]["mass"], 6.0e9)
        self.assertEqual(components[2]["scaleHeight"], 0.9)
        self.assertEqual(components[3]["type"], "Spheroid")
        self.assertAlmostEqual(components[3]["densityNorm"], 10**6.2)
        self.assertAlmostEqual(components[3]["scaleRadius"], 70.0)
        self.assertEqual(components[3]["p"], 0.8)
        self.assertEqual(components[3]["q"], 0.92)
        self.assertEqual(components[3]["outerCutoffRadius"], 500.0)
        self.assertEqual(components[3]["cutoffStrength"], 5.0)

    def test_invalid_halo_parameters_are_rejected_instead_of_clipped(self):
        with self.assertRaisesRegex(ValueError, "gamma"):
            zhu_2026_component_parameters(6.2, math.log10(70), 0.8, 0.92, 3.0)
        with self.assertRaisesRegex(ValueError, "axis ratios"):
            zhu_2026_component_parameters(6.2, math.log10(70), 0.0, 0.92, 1.0)
        with self.assertRaises(NotImplementedError):
            zhu_2026_component_parameters(
                6.2,
                math.log10(70),
                0.8,
                0.92,
                1.0,
                alpha_halo=1.0,
            )

    def test_builder_passes_all_components_to_agama_in_one_call(self):
        calls = {}

        def set_units(**kwargs):
            calls["units"] = kwargs

        def potential(*args, **kwargs):
            calls["potential_args"] = args
            calls["potential_kwargs"] = kwargs
            return "constructed"

        fake_agama = SimpleNamespace(setUnits=set_units, Potential=potential)
        with patch.dict(sys.modules, {"agama": fake_agama}):
            result = build_zhu_2026_potential(6.2, math.log10(70), 0.8, 0.92, 1.0)

        self.assertEqual(result, "constructed")
        self.assertEqual(calls["units"], {"length": 1, "velocity": 1, "mass": 1})
        self.assertEqual(len(calls["potential_args"]), 4)
        self.assertEqual(calls["potential_kwargs"], {})


if __name__ == "__main__":
    unittest.main()
