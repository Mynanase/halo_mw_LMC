import unittest

import numpy as np

from halo_mw_lmc.visualization.parameter_constraints import (
    CORNER_PANELS,
    DISPLAY_ORDER,
    DIAGNOSTIC_LEVELS,
    PANELS,
    PARAMETER_NAMES,
    ProfileSettings,
    ProfileSurface,
    _baseline_by_shared_reference,
    _cleaned_reliable,
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
            objective_scale=surrogate.scale,
        )

        self.assertTrue(np.any(np.isfinite(surface.delta_chi2)))
        minimum = np.unravel_index(np.nanargmin(surface.delta_chi2), surface.delta_chi2.shape)
        self.assertAlmostEqual(surface.x[minimum[1]], center[4] * 2.0, delta=0.55)
        self.assertAlmostEqual(surface.y[minimum[0]], 5.0 + center[2] * 2.0, delta=0.55)
        # Pin the physical-unit restoration: the profiled surface must live on
        # the training-objective scale, not the internal GP-fit scale (where
        # the spread is 1).  GP overshoot far from the support may exceed the
        # training spread, so only the lower bound is tight.
        finite = surface.delta_chi2[np.isfinite(surface.delta_chi2)]
        training_spread = float(np.std(prepared.objectives["total"] - np.min(prepared.objectives["total"])))
        self.assertGreater(float(np.max(finite)), 0.2 * training_spread)

    def test_fitted_scale_is_training_spread(self):
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn is unavailable")
        settings = ProfileSettings(minimum_samples=1)
        rng = np.random.default_rng(23)
        normalized = rng.random((40, 5))
        bound_array = np.asarray([BOUNDS[name] for name in PARAMETER_NAMES])
        coordinates = bound_array[:, 0] + normalized * np.diff(bound_array, axis=1)[:, 0]
        objective = 120.0 * (0.5 + np.sum(normalized, axis=1)) + rng.normal(0.0, 3.0, 40)
        prepared = prepare_constraint_samples(
            sample_table(coordinates, objective),
            BOUNDS,
            settings=settings,
        )
        surrogate = _fit_surrogates(prepared)["total"]
        expected_scale = float(
            np.std(prepared.objectives["total"] - np.min(prepared.objectives["total"]))
        )
        self.assertGreater(surrogate.scale, 0.0)
        self.assertAlmostEqual(surrogate.scale, expected_scale, delta=1e-6)

    def test_predictive_std_mask_uses_scaled_units(self):
        """The std mask interprets std as a multiple of the objective spread."""
        settings = ProfileSettings(
            grid_size=5,
            sobol_count=8,
            local_starts=1,
            local_maxiter=20,
            minimum_samples=1,
            maximum_predictive_std=1.0,
        )
        bound_array = np.asarray([BOUNDS[name] for name in PARAMETER_NAMES])
        center = np.full(5, 0.5)

        class LargeStdSurrogate:
            def __init__(self, std):
                self.std = float(std)

            def predict(self, points, *, return_std=False):
                mean = 10.0 * np.sum((np.asarray(points) - center) ** 2, axis=1)
                if return_std:
                    return mean, np.full(points.shape[0], self.std)
                return mean

        support = np.clip(
            np.vstack((center, np.random.default_rng(24).normal(0.0, 0.03, size=(80, 5)) + center)),
            0.0,
            1.0,
        )
        # std in absolute units may be far above the 1.0 threshold; only the
        # scaled comparison against the objective_scale decides.
        too_coarse = profile_surrogate_surface(
            LargeStdSurrogate(std=50.0),
            support,
            bound_array,
            PANELS[0],
            shared_sobol_points(settings),
            settings=settings,
        )
        self.assertFalse(np.any(too_coarse.reliable))
        fine = profile_surrogate_surface(
            LargeStdSurrogate(std=0.05),
            support,
            bound_array,
            PANELS[0],
            shared_sobol_points(settings),
            settings=settings,
        )
        self.assertTrue(np.any(fine.reliable))

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


class _AdmissionTableBuilder:
    """Build a valid sample table with an optional successful_orbits column."""

    @staticmethod
    def build(rows: int, *, with_successful: bool = True) -> np.ndarray:
        rng = np.random.default_rng(42)
        bound_array = np.asarray([BOUNDS[name] for name in PARAMETER_NAMES])
        coordinates = bound_array[:, 0] + rng.random((rows, 5)) * np.diff(
            bound_array, axis=1
        )[:, 0]
        dtype = [(name, "f8") for name in PARAMETER_NAMES] + [
            ("objective_velocity", "f8"),
            ("objective_density_velocity", "f8"),
            ("chi2", "f8"),
            ("weight_solver_converged", "i8"),
            ("failed_orbits", "i8"),
        ]
        if with_successful:
            dtype.append(("successful_orbits", "i8"))
        table = np.zeros(rows, dtype=dtype)
        for index, name in enumerate(PARAMETER_NAMES):
            table[name] = coordinates[:, index]
        table["objective_density_velocity"] = 100.0
        table["objective_velocity"] = 40.0
        table["chi2"] = 60.0
        table["weight_solver_converged"] = 1
        return table


class FailedOrbitAdmissionTests(unittest.TestCase):
    """Admission gate bounds orbit-library loss rather than solver success."""

    def test_fraction_threshold_retains_small_loss_and_rejects_large_loss(self):
        table = _AdmissionTableBuilder.build(20, with_successful=True)
        kept = [0, 1, 2, 3, 4]  # failed / successful ~ 0.03 (within the 0.05 budget)
        dropped = [5, 6, 7, 8, 9]  # failed / successful ~ 0.08 (over budget)
        clean = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]  # zero failed

        table["successful_orbits"] = 1000
        table["failed_orbits"] = 0
        table["failed_orbits"][kept] = 30
        table["failed_orbits"][dropped] = 80

        data = prepare_constraint_samples(table, BOUNDS)
        retained = data.display_coordinates.shape[0]
        # Clean rows plus the 3% rows survive; the 8% rows fail the gate.
        self.assertEqual(retained, len(clean) + len(kept))

    def test_missing_successful_orbits_falls_back_to_strict_rule(self):
        table = _AdmissionTableBuilder.build(15, with_successful=False)
        table["failed_orbits"] = 0
        table["failed_orbits"][3] = 7  # no ratio available -> strict fallback
        data = prepare_constraint_samples(table, BOUNDS)
        self.assertEqual(data.display_coordinates.shape[0], 14)

    def test_tighter_threshold_rejects_intermediate_loss(self):
        table = _AdmissionTableBuilder.build(15, with_successful=True)
        table["successful_orbits"] = 1000
        table["failed_orbits"] = 0
        table["failed_orbits"][:5] = 30  # 3% (intended to be kept)
        table["failed_orbits"][5:10] = 30
        settings = ProfileSettings(maximum_failed_orbit_fraction=0.01)
        data = prepare_constraint_samples(table, BOUNDS, settings=settings)
        # At 1% the 3% rows are rejected; only the five clean rows remain.
        self.assertEqual(data.display_coordinates.shape[0], 5)

    def test_validate_rejects_non_strict_fractions(self):
        for value in (0.0, 1.0, -0.1, 1.2):
            with self.assertRaises(ValueError):
                ProfileSettings(maximum_failed_orbit_fraction=value).validate()
        ProfileSettings(maximum_failed_orbit_fraction=0.05).validate()


class DiagnosticLevelTests(unittest.TestCase):
    """Diagnostic-level contouring and cross-panel zero-point behaviour."""

    @staticmethod
    def _surface(values: np.ndarray, reliable: np.ndarray | None = None) -> ProfileSurface:
        """Build a small ProfileSurface with uniform x/y and a value grid."""

        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values.reshape((int(np.sqrt(values.size)), -1))
        rows, cols = values.shape
        if reliable is None:
            reliable = np.isfinite(values)
        else:
            reliable = np.asarray(reliable, dtype=bool)
        return ProfileSurface(
            x=np.linspace(0.0, 1.0, cols),
            y=np.linspace(0.0, 1.0, rows),
            delta_chi2=values,
            reliable=reliable,
            minimizers=np.zeros((rows, cols, 5)),
            predictive_std=np.zeros((rows, cols)),
            support_distance=np.zeros((rows, cols)),
        )

    def test_level_selection_skips_out_of_range_levels(self):
        # Values form a Manhattan bowl centred in the grid (so the minimum
        # survives neighbour cleaning); after cleaning the surviving maximum is
        # about 9100, so 230 and 2300 cross while 2.30 stays well below the
        # minimum.
        drawn = self._draw_bowl(base=100.0, scale=1000.0)
        self.assertTrue(drawn[230.0])
        self.assertTrue(drawn[2300.0])
        self.assertFalse(drawn[2.30])

    def test_level_selection_when_only_2_30_crosses(self):
        # A shallow bowl spanning about [1, 6]: only 2.30 crosses.
        drawn = self._draw_bowl(base=1.0, scale=1.0)
        self.assertTrue(drawn[2.30])
        self.assertFalse(drawn[230.0])
        self.assertFalse(drawn[2300.0])

    def _draw_bowl(self, *, base: float, scale: float):
        from halo_mw_lmc.visualization.parameter_constraints import _draw_profile_contour

        class Axis:
            def __init__(self):
                self.calls = []

            def contour(self, *args, **kwargs):
                self.calls.append(kwargs.get("levels"))

        n = 11
        center = (n - 1) / 2.0
        rows, cols = np.mgrid[0:n, 0:n]
        values = base + scale * (
            np.abs(rows - center) + np.abs(cols - center)
        )
        surface = self._surface(values)
        axis = Axis()
        drawn = _draw_profile_contour(
            axis, surface, color="#9b0000", linestyle="solid"
        )
        self.assertTrue(hasattr(axis, "calls"))
        return drawn

    def test_neighbour_cleaning_removes_isolated_pixels(self):
        # A 5x5 reliable block should survive; an isolated pixel alone drops.
        reliable = np.zeros((10, 10), dtype=bool)
        reliable[1:6, 1:6] = True  # 5x5 block (interior pixels have >= 5 neighbours)
        reliable[8, 8] = True  # isolated single pixel
        clean = _cleaned_reliable(reliable)
        self.assertTrue(clean[3, 3])
        self.assertTrue(clean[2, 2])
        self.assertFalse(clean[8, 8])

    def test_neighbour_cleaning_drops_thin_crossing_centre(self):
        # The centre of a 5-pixel plus/cross shape has exactly four reliable
        # neighbours and must be dropped: the filter counts eight neighbours,
        # never the centre pixel itself.
        reliable = np.zeros((9, 9), dtype=bool)
        reliable[4, 3:6] = True
        reliable[3:6, 4] = True
        clean = _cleaned_reliable(reliable)
        self.assertFalse(clean[4, 4])
        self.assertFalse(clean.any())

    def test_shared_reference_uses_global_reliable_min(self):
        panel_a = self._surface(np.array([[10.0, 20.0], [30.0, 40.0]]), np.ones((2, 2), dtype=bool))
        panel_b = self._surface(np.array([[110.0, 120.0], [130.0, 140.0]]), np.ones((2, 2), dtype=bool))
        surfaces = {"a": {"total": panel_a}, "b": {"total": panel_b}}
        baselined = _baseline_by_shared_reference(surfaces)
        # The shared reference is the global reliable minimum (10.0); subtract it.
        np.testing.assert_allclose(baselined["a"]["total"].delta_chi2, panel_a.delta_chi2 - 10.0)
        np.testing.assert_allclose(baselined["b"]["total"].delta_chi2, panel_b.delta_chi2 - 10.0)
        # The input mapping is left unchanged.
        np.testing.assert_allclose(surfaces["a"]["total"].delta_chi2, np.array([[10.0, 20.0], [30.0, 40.0]]))
        # New instances: other frozen fields are preserved unchanged.
        self.assertIs(baselined["a"]["total"].reliable, panel_a.reliable)
        self.assertIs(baselined["a"]["total"].minimizers, panel_a.minimizers)
        self.assertIs(baselined["a"]["total"].x, panel_a.x)
        self.assertIsNot(baselined["a"]["total"], panel_a)

    def test_unresolved_annotation_present_when_2_30_not_crossed(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        surfaces = {"corner_1_0": {"total": self._surface(np.linspace(100.0, 5000.0, 49))}}
        figure, axes = plt.subplots()
        from halo_mw_lmc.visualization.parameter_constraints import (
            _contour_touches_boundary,
            _draw_panel_annotations,
            _draw_profile_contour,
        )

        drawn = _draw_profile_contour(axes, surfaces["corner_1_0"]["total"], color="#9b0000", linestyle="solid")
        _draw_panel_annotations(axes, drawn, truncated=_contour_touches_boundary(surfaces["corner_1_0"]["total"]))
        texts = [text.get_text() for text in axes.texts]
        self.assertTrue(any("2.30 unresolved" in text for text in texts))
        plt.close(figure)

    def test_diagnostic_levels_constant(self):
        self.assertEqual(DIAGNOSTIC_LEVELS, (2.30, 230.0, 2300.0))


if __name__ == "__main__":
    unittest.main()
