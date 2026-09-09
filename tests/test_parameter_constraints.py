import unittest

import numpy as np

from halo_mw_lmc.visualization.parameter_constraints import (
    CORNER_PANELS,
    DISPLAY_ORDER,
    PANELS,
    PARAMETER_NAMES,
    ProfileSettings,
    ProfileSurface,
    _corner_nuisance_params,
    _corner_param_limits,
    _embed_panel_points,
    _fit_surrogates,
    _logical_to_internal_norm,
    build_parameter_constraints_corner_figure,
    build_parameter_constraints_figure,
    deterministic_maximin_indices,
    persist_corner_surfaces,
    prepare_constraint_samples,
    profile_surrogate_surface,
    scale_radius_kpc,
    search_bounds_from_resolved_config,
    shared_sobol_points,
)


BOUNDS = {
    "qhalo": (0.0, 1.0),
    "phalo": (0.0, 1.0),
    "rho0": (5.0, 7.0),
    "rho0_plus_2logrs": (8.0, 11.0),
    "gamma": (0.0, 2.0),
}


def sample_table(coordinates, total, velocity=None, density=None):
    coordinates = np.asarray(coordinates, dtype=float)
    total = np.asarray(total, dtype=float)
    velocity = total * 0.4 if velocity is None else np.asarray(velocity, dtype=float)
    density = total - velocity if density is None else np.asarray(density, dtype=float)
    dtype = [(name, "f8") for name in PARAMETER_NAMES] + [
        ("objective_velocity", "f8"),
        ("objective_density_velocity", "f8"),
        ("chi2", "f8"),
        ("weight_solver_converged", "i8"),
        ("failed_orbits", "i8"),
    ]
    result = np.zeros(coordinates.shape[0], dtype=dtype)
    for index, name in enumerate(PARAMETER_NAMES):
        result[name] = coordinates[:, index]
    result["objective_velocity"] = velocity / 2.0
    result["objective_density_velocity"] = total / 2.0
    result["chi2"] = density
    result["weight_solver_converged"] = 1
    return result


class QuadraticSurrogate:
    def __init__(self, center, scale=20.0, standard_deviation=0.05):
        self.center = np.asarray(center, dtype=float)
        self.scale = float(scale)
        self.standard_deviation = float(standard_deviation)

    def predict(self, points, *, return_std=False):
        values = self.scale * np.sum(
            (np.asarray(points, dtype=float) - self.center) ** 2,
            axis=1,
        )
        if return_std:
            return values, np.full(values.shape, self.standard_deviation)
        return values


class ParameterConstraintTests(unittest.TestCase):
    def test_scale_radius_uses_persisted_combined_coordinate(self):
        coordinates = np.array([[0.8, 0.9, 6.0, 10.0, 1.0]])
        np.testing.assert_allclose(scale_radius_kpc(coordinates), [100.0])

    def test_deduplication_uses_componentwise_medians(self):
        coordinates = np.array(
            [
                [0.2, 0.3, 5.5, 9.0, 0.4],
                [0.2, 0.3, 5.5, 9.0, 0.4],
                [0.7, 0.8, 6.5, 10.5, 1.4],
            ]
        )
        samples = sample_table(
            coordinates,
            total=[6.0, 2.0, 10.0],
            velocity=[2.0, 4.0, 3.0],
            density=[4.0, 8.0, 7.0],
        )

        prepared = prepare_constraint_samples(
            samples,
            BOUNDS,
            settings=ProfileSettings(minimum_samples=1),
        )

        self.assertEqual(prepared.coordinates.shape[0], 2)
        np.testing.assert_allclose(prepared.objectives["total"], [4.0, 10.0])
        np.testing.assert_allclose(prepared.objectives["velocity"], [3.0, 3.0])
        np.testing.assert_allclose(prepared.objectives["density"], [6.0, 7.0])

    def test_maximin_and_shared_sobol_design_are_deterministic(self):
        rng = np.random.default_rng(4)
        points = rng.random((30, 5))
        ranking = rng.random(30)
        first = deterministic_maximin_indices(
            points,
            ranking,
            maximum=12,
            retain_best=4,
        )
        second = deterministic_maximin_indices(
            points,
            ranking,
            maximum=12,
            retain_best=4,
        )
        np.testing.assert_array_equal(first, second)
        expected_best = set(np.argsort(ranking, kind="stable")[:4])
        self.assertTrue(expected_best.issubset(set(first)))

        settings = ProfileSettings(sobol_count=16, local_starts=2)
        np.testing.assert_array_equal(
            shared_sobol_points(settings),
            shared_sobol_points(settings),
        )

    def test_bounded_profile_recovers_quadratic_nuisance_minimum(self):
        settings = ProfileSettings(
            grid_size=7,
            sobol_count=16,
            local_starts=2,
            local_maxiter=30,
            minimum_samples=1,
            maximum_predictive_std=1.0,
        )
        center = np.array([0.35, 0.65, 0.45, 0.25, 0.75])
        surrogate = QuadraticSurrogate(center)
        rng = np.random.default_rng(8)
        training = np.vstack((rng.random((120, 5)), center))
        surface = profile_surrogate_surface(
            surrogate,
            training,
            np.asarray([BOUNDS[name] for name in PARAMETER_NAMES]),
            PANELS[0],
            shared_sobol_points(settings),
            settings=settings,
        )

        minimum = np.unravel_index(np.nanargmin(surface.delta_chi2), surface.delta_chi2.shape)
        self.assertAlmostEqual(surface.x[minimum[1]], center[4] * 2.0, delta=0.35)
        self.assertAlmostEqual(surface.y[minimum[0]], 5.0 + center[2] * 2.0, delta=0.35)
        np.testing.assert_allclose(
            surface.minimizers[minimum][[0, 1, 3]],
            center[[0, 1, 3]],
            atol=2e-3,
        )

    def test_nonuniform_samples_support_five_dimensional_gp_profile(self):
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn is unavailable")
        settings = ProfileSettings(
            grid_size=5,
            sobol_count=8,
            local_starts=2,
            local_maxiter=12,
            minimum_samples=20,
            maximum_predictive_std=100.0,
        )
        rng = np.random.default_rng(12)
        normalized = rng.beta(1.5, 3.0, size=(70, 5))
        center = np.array([0.35, 0.45, 0.55, 0.50, 0.40])
        normalized = np.vstack((normalized, center))
        bound_array = np.asarray([BOUNDS[name] for name in PARAMETER_NAMES])
        coordinates = bound_array[:, 0] + normalized * np.diff(bound_array, axis=1)[:, 0]
        objective = 25.0 * np.sum((normalized - center) ** 2, axis=1)
        prepared = prepare_constraint_samples(
            sample_table(coordinates, objective),
            BOUNDS,
            settings=settings,
        )
        surrogate = _fit_surrogates(prepared)["total"]

        surface = profile_surrogate_surface(
            surrogate,
            prepared.normalized_coordinates,
            prepared.bounds,
            PANELS[0],
            shared_sobol_points(settings),
            settings=settings,
        )

        self.assertTrue(np.any(np.isfinite(surface.delta_chi2)))
        minimum = np.unravel_index(np.nanargmin(surface.delta_chi2), surface.delta_chi2.shape)
        self.assertAlmostEqual(surface.x[minimum[1]], center[4] * 2.0, delta=0.55)
        self.assertAlmostEqual(surface.y[minimum[0]], 5.0 + center[2] * 2.0, delta=0.55)

    def test_support_mask_rejects_profiled_extrapolation(self):
        settings = ProfileSettings(
            grid_size=5,
            sobol_count=8,
            local_starts=1,
            local_maxiter=20,
            minimum_samples=1,
            maximum_predictive_std=1.0,
        )
        center = np.full(5, 0.5)
        rng = np.random.default_rng(22)
        support = np.clip(
            np.vstack((center, center + rng.normal(0.0, 0.03, size=(80, 5)))),
            0.0,
            1.0,
        )
        surface = profile_surrogate_surface(
            QuadraticSurrogate(center),
            support,
            np.asarray([BOUNDS[name] for name in PARAMETER_NAMES]),
            PANELS[0],
            shared_sobol_points(settings),
            settings=settings,
        )

        self.assertTrue(surface.reliable[2, 2])
        self.assertFalse(surface.reliable[0, 0])
        self.assertFalse(surface.reliable[-1, -1])

    def test_short_run_renders_annotated_scatter_only_figure(self):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.skipTest("Matplotlib is unavailable")
        rng = np.random.default_rng(16)
        normalized = rng.random((12, 5))
        bound_array = np.asarray([BOUNDS[name] for name in PARAMETER_NAMES])
        coordinates = bound_array[:, 0] + normalized * np.diff(bound_array, axis=1)[:, 0]
        figure = build_parameter_constraints_figure(
            sample_table(coordinates, np.arange(12.0)),
            BOUNDS,
        )

        self.assertTrue(
            any("at least 50 required" in text.get_text() for text in figure.texts)
        )
        plt.close(figure)

    def test_search_bounds_are_read_from_persisted_config(self):
        document = {
            "optimizer": {
                "bounds": {name: list(interval) for name, interval in BOUNDS.items()}
            }
        }
        self.assertEqual(search_bounds_from_resolved_config(document), BOUNDS)


def _logical_center():
    """A physically valid display-space center for the corner panels."""

    return {
        "rho0": 6.0,
        "rs": 30.0,
        "gamma": 1.0,
        "qhalo": 0.8,
        "phalo": 0.85,
    }


class CornerConstraintTests(unittest.TestCase):
    def test_corner_panels_enumerate_all_unordered_pairs(self):
        expected = {
            frozenset((a, b)) for a in DISPLAY_ORDER for b in DISPLAY_ORDER if a != b
        }
        actual = {
            frozenset((panel.x_param, panel.y_param)) for panel in CORNER_PANELS
        }
        self.assertEqual(len(CORNER_PANELS), 10)
        self.assertEqual(actual, expected)

    def test_corner_panel_row_col_match_display_order(self):
        for panel in CORNER_PANELS:
            self.assertEqual(panel.x_param, DISPLAY_ORDER[panel.col_index])
            self.assertEqual(panel.y_param, DISPLAY_ORDER[panel.row_index])
            self.assertGreater(panel.row_index, panel.col_index)

    def test_logical_to_internal_couples_radius_and_rho0(self):
        bound_array = np.asarray([BOUNDS[name] for name in PARAMETER_NAMES])
        normalized = _logical_to_internal_norm(_logical_center(), bound_array)
        self.assertIsNotNone(normalized)
        internal = normalized * np.diff(bound_array, axis=1)[:, 0] + bound_array[:, 0]
        self.assertAlmostEqual(
            internal[3],
            6.0 + 2.0 * np.log10(30.0),
            places=9,
        )

    def test_corner_embed_round_trips_all_panels(self):
        bound_array = np.asarray([BOUNDS[name] for name in PARAMETER_NAMES])
        center = _logical_center()
        for panel in CORNER_PANELS:
            nuisance = [0.5, 0.5, 0.5]
            embedded = _embed_panel_points(
                panel,
                center[panel.x_param],
                center[panel.y_param],
                np.asarray([nuisance], dtype=float),
                bound_array,
            )
            self.assertIsNotNone(embedded, f"panel {panel.name} failed to embed")
            self.assertEqual(embedded.shape, (1, len(PARAMETER_NAMES)))
            self.assertTrue(np.all((embedded >= 0.0) & (embedded <= 1.0)))

    def test_corner_surface_profiles_on_quadratic_surrogate(self):
        settings = ProfileSettings(
            grid_size=7,
            sobol_count=16,
            local_starts=2,
            local_maxiter=30,
            minimum_samples=1,
            maximum_predictive_std=100.0,
        )
        bound_array = np.asarray([BOUNDS[name] for name in PARAMETER_NAMES])
        center = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        surrogate = QuadraticSurrogate(center, standard_deviation=0.05)
        panel = next(
            panel for panel in CORNER_PANELS
            if {panel.x_param, panel.y_param} == {"gamma", "rho0"}
        )
        support = np.clip(
            np.vstack(
                (
                    center,
                    np.random.default_rng(3).normal(0.0, 0.03, size=(90, 5)) + center,
                )
            ),
            0.0,
            1.0,
        )
        surface = profile_surrogate_surface(
            surrogate,
            support,
            bound_array,
            panel,
            shared_sobol_points(settings),
            settings=settings,
        )

        self.assertTrue(np.any(np.isfinite(surface.delta_chi2)))
        minimum = np.unravel_index(
            np.nanargmin(surface.delta_chi2),
            surface.delta_chi2.shape,
        )
        x_limits = _corner_param_limits(panel.x_param, bound_array)
        y_limits = _corner_param_limits(panel.y_param, bound_array)
        expected_x = x_limits[0] + 0.5 * (x_limits[1] - x_limits[0])
        expected_y = y_limits[0] + 0.5 * (y_limits[1] - y_limits[0])
        self.assertAlmostEqual(surface.x[minimum[1]], expected_x, delta=0.35)
        self.assertAlmostEqual(surface.y[minimum[0]], expected_y, delta=0.35)

    def test_corner_short_run_renders_degraded_figure(self):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.skipTest("Matplotlib is unavailable")
        rng = np.random.default_rng(16)
        normalized = rng.random((12, 5))
        bound_array = np.asarray([BOUNDS[name] for name in PARAMETER_NAMES])
        coordinates = bound_array[:, 0] + normalized * np.diff(bound_array, axis=1)[:, 0]
        figure = build_parameter_constraints_corner_figure(
            sample_table(coordinates, np.arange(12.0)),
            BOUNDS,
        )

        self.assertGreaterEqual(len(figure.axes), 25)
        self.assertTrue(
            any("at least 50 required" in text.get_text() for text in figure.texts)
        )
        plt.close(figure)

    def test_corner_invalid_solver_samples_render_empty_figure(self):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.skipTest("Matplotlib is unavailable")
        rng = np.random.default_rng(17)
        normalized = rng.random((60, 5))
        bound_array = np.asarray([BOUNDS[name] for name in PARAMETER_NAMES])
        coordinates = bound_array[:, 0] + normalized * np.diff(bound_array, axis=1)[:, 0]
        table = sample_table(coordinates, np.arange(60.0))
        table["weight_solver_converged"] = 0
        figure, surfaces = build_parameter_constraints_corner_figure(
            table,
            BOUNDS,
            return_artifacts=True,
        )

        self.assertEqual(surfaces, {})
        self.assertTrue(
            any(
                "parameter constraints unavailable" in text.get_text()
                for text in figure.texts
            )
        )
        plt.close(figure)

    def test_persist_corner_surfaces_roundtrip(self):
        import os
        import tempfile

        x = np.linspace(0.0, 1.0, 4)
        y = np.linspace(0.0, 1.0, 4)
        surface = ProfileSurface(
            x=x,
            y=y,
            delta_chi2=np.zeros((4, 4)),
            reliable=np.ones((4, 4), dtype=bool),
            minimizers=np.zeros((4, 4, 5)),
            predictive_std=np.zeros((4, 4)),
            support_distance=np.zeros((4, 4)),
        )
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as handle:
            handle.close()
            path = handle.name
        try:
            persist_corner_surfaces({"corner_1_0": {"total": surface}}, path)
            loaded = np.load(path)
            np.testing.assert_array_equal(loaded["corner_1_0__total__x"], x)
            np.testing.assert_array_equal(
                loaded["corner_1_0__total__reliable"],
                np.ones((4, 4), dtype=bool),
            )
            self.assertIn("panel_names", loaded)
            self.assertIn("corner_1_0", set(loaded["panel_names"]))
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
