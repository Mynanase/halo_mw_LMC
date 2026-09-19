"""Small-array acceptance tests for the nphi1 bundling repair math.

Covers the plan contract in ``docs/nphi1_bundling_repair_plan.md`` §2.2/§2.1:
the two backfill identities, non-contiguous seed numbering, empty bundles, a
deliberately wrong mapping that the identities must catch, and the amplitude
oracle pinning scale=1 scoring against volume normalization.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import scipy.sparse

from halo_mw_lmc.catalogue import SeedCatalogue
from halo_mw_lmc.density import CylindricalGrid, build_orbit_density_response, compare_density
from halo_mw_lmc.evaluate import score_orbit_weights
from halo_mw_lmc.orbits import OrbitLibrary, cartesian_to_spherical_phase_space
from halo_mw_lmc.prepare import PreparedFixedWeightData
from halo_mw_lmc.weights import _build_weight_problem, _normalized_target, _primal_kkt_residual
from tests.artifact_fixture import comparison_model

REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "benchmark_nphi1_bundling", REPO / "scripts" / "benchmark_nphi1_bundling.py",
)
bundling = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bundling)


def _three_bundle_fixture():
    """Design (6 rows, 6 active orbits) with bundles of sizes [3, 1, 2]."""
    # Bundle 0: orbits 0,1,2 (orbit 1 has an all-zero response column);
    # bundle 1: orbit 3; bundle 2: orbits 4,5.
    assignments = np.array([0, 0, 0, 1, 2, 2])
    design = scipy.sparse.csr_matrix(
        np.array(
            [
                [1.0, 0.0, 2.0, 3.0, 0.5, 1.5],
                [2.0, 0.0, 1.0, 1.0, 2.0, 0.5],
                [0.0, 0.0, 4.0, 2.0, 1.0, 1.0],
                [1.0, 0.0, 0.5, 4.0, 0.5, 2.0],
                [3.0, 0.0, 1.0, 0.5, 2.0, 1.0],
                [0.5, 0.0, 2.0, 1.0, 1.0, 3.0],
            ],
        ),
    )
    observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    return design, observed, assignments


def _indicator(assignments, k_total):
    indicator = np.zeros((assignments.size, k_total))
    indicator[np.arange(assignments.size), assignments] = 1.0
    return scipy.sparse.csr_matrix(indicator)


def _orbit_weights(w, successful, active):
    """Per-active-orbit weights extracted from the full seed array."""
    return w[successful[active]]


class BundledSolveIdentityTests(np.testing.TestCase):
    def test_backfill_identity_product(self):
        """A @ w == (A @ S) @ u with unequal bundle sizes and a zero column."""
        design, observed, assignments = _three_bundle_fixture()
        u = bundling.solve_bundled_weights(design, observed, assignments, 1e-6)
        successful = np.arange(6, dtype=np.int64)
        active = np.arange(6, dtype=np.int64)
        w = bundling.map_bundle_weights_to_seeds(u, assignments, successful, active, 6)
        reduced = design @ _indicator(assignments, 3)
        np.testing.assert_allclose(
            design @ _orbit_weights(w, successful, active), reduced @ u, rtol=1e-12, atol=1e-12,
        )

    def test_backfill_identity_regularization(self):
        """lambda * dot(w, w) == lambda * sum_k n_k u_k^2."""
        design, observed, assignments = _three_bundle_fixture()
        lam = 1e-6
        u = bundling.solve_bundled_weights(design, observed, assignments, lam)
        w = bundling.map_bundle_weights_to_seeds(u, assignments, np.arange(6, dtype=np.int64), np.arange(6, dtype=np.int64), 6)
        member_count = np.bincount(assignments, minlength=3).astype(float)
        np.testing.assert_allclose(
            lam * float(np.dot(w, w)), lam * float(np.sum(member_count * u**2)),
            rtol=1e-12, atol=1e-12,
        )

    def test_non_contiguous_seed_backfill(self):
        """Backfill honors shuffled successful seed numbering [9, 2, 7]."""
        design, observed, assignments = _three_bundle_fixture()
        u = bundling.solve_bundled_weights(design, observed, assignments, 1e-6)
        successful = np.array([9, 2, 7, 4, 1, 3], dtype=np.int64)
        active = np.array([5, 0, 3, 1, 2, 4], dtype=np.int64)
        w = bundling.map_bundle_weights_to_seeds(u, assignments, successful, active, 10)
        expected_seeds = successful[active]
        expected = np.zeros(10)
        expected[expected_seeds] = u[assignments]
        np.testing.assert_array_equal(w, expected)
        # Every nonzero weight landed exactly on the mapped seed slot.
        np.testing.assert_array_equal(np.nonzero(w)[0], np.sort(expected_seeds[u[assignments] > 0]))

    def test_empty_bundle_forced_zero(self):
        """An empty bundle keeps u = 0 and both identities still hold."""
        design, observed, _ = _three_bundle_fixture()
        # Re-bucket the same six orbits into 4 bundles, leaving bundle 2 empty.
        assignments = np.array([0, 0, 0, 1, 3, 3])
        lam = 1e-6
        u = bundling.solve_bundled_weights(design, observed, assignments, lam)
        self.assertEqual(u[2], 0.0)
        successful = np.arange(6, dtype=np.int64)
        active = np.arange(6, dtype=np.int64)
        w = bundling.map_bundle_weights_to_seeds(u, assignments, successful, active, 6)
        reduced = design @ _indicator(assignments, 4)
        np.testing.assert_allclose(
            design @ _orbit_weights(w, successful, active), reduced @ u, rtol=1e-12, atol=1e-12,
        )
        member_count = np.bincount(assignments, minlength=4).astype(float)
        np.testing.assert_allclose(
            lam * float(np.dot(w, w)), lam * float(np.sum(member_count * u**2)),
            rtol=1e-12, atol=1e-12,
        )

    def test_wrong_mapping_fails_identity(self):
        """Backfilling through a permuted mapping must break identity 1."""
        design, observed, assignments = _three_bundle_fixture()
        u = bundling.solve_bundled_weights(design, observed, assignments, 1e-6)
        successful = np.arange(6, dtype=np.int64)
        active = np.arange(6, dtype=np.int64)
        wrong_assignments = np.array([1, 2, 0, 0, 2, 1])  # permuted, unequal sizes
        w_wrong = bundling.map_bundle_weights_to_seeds(u, wrong_assignments, successful, active, 6)
        reduced = design @ _indicator(assignments, 3)
        assert not np.allclose(
            design @ _orbit_weights(w_wrong, successful, active), reduced @ u, rtol=1e-12, atol=1e-12,
        )


class AmplitudeOracleTests(np.testing.TestCase):
    def test_unit_mass_scale_one_scoring(self):
        """scale=1 scoring equals the direct normalized formula; volume differs."""
        grid = CylindricalGrid.uniform(n_r=1, r_range=(0.0, 1.0), n_z=1, z_range=(0.0, 1.0), n_phi=1)
        fit_options = {
            "min_abs_z": 0.0, "min_spherical_radius": 0.0, "max_spherical_radius": 10.0,
            "require_positive_data": False,
        }
        target_density = np.full(grid.shape, 1.0)
        target_error = np.full(grid.shape, 0.1)
        model = np.full(grid.shape, 2.0)

        # _normalized_target reads volumes through the response's grid.
        response_stub = SimpleNamespace(grid=grid)
        target_n, error_n = _normalized_target(
            target_density, target_error, np.ones(grid.shape, dtype=bool), response_stub, "unit_mass",
        )
        mass = float(np.sum(target_density * grid.volumes))
        density = compare_density(target_n, error_n, model, grid, normalization="none", **fit_options)
        self.assertEqual(density.scale, 1.0)
        expected_chi2 = float(np.sum(((model - target_density / mass) / (target_error / mass)) ** 2))
        self.assertEqual(density.chi2, expected_chi2)

        volume_density = compare_density(
            target_density, target_error, model, grid, normalization="volume",
            normalization_min_radius=0.0, **fit_options,
        )
        self.assertNotAlmostEqual(volume_density.chi2, density.chi2)


class ArtifactRoundTripTests(np.testing.TestCase):
    def test_save_attempt_load_attempt_roundtrip(self):
        """case.json and evaluation.npz survive a write/read cycle exactly."""

        case = {
            "method": "bundled",
            "objective_velocity": 137503.6699630702,
            "density_chi2_per_bin": 0.7486127667712524,
            "density_gate_passed": True,
            "iterations": None,
            "timings_seconds": {"solve": 0.8071084823459387, "total_outer": 1.5},
        }
        arrays = {
            "seed_weights": np.array([0.0, 1.5, 0.0, 2.25]),
            "orbit_weights": np.array([1.5, 1.5, 2.25, 2.25, 0.0]),
            "model_density": np.array([[[1.0, 2.0]], [[3.0, 4.0]]]),
            "residual": np.array([[[0.5, -0.5]], [[1.0, -1.0]]]),
            "fit_mask": np.array([[[True, True]], [[True, False]]]),
        }
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            bundling.save_attempt(directory, "bundled", case, arrays)
            loaded_case, loaded_arrays = bundling.load_attempt(directory, "bundled")
        self.assertEqual(loaded_case["method"], case["method"])
        self.assertIsNone(loaded_case["iterations"])
        self.assertIs(loaded_case["density_gate_passed"], True)
        for key, value in case["timings_seconds"].items():
            self.assertEqual(loaded_case["timings_seconds"][key], value)
        for key, value in arrays.items():
            np.testing.assert_allclose(loaded_arrays[key], value, rtol=1e-12, atol=1e-12)


class GroupingVariableTests(np.testing.TestCase):
    def test_dominant_frequency_recovers_synthetic_peak(self):
        """The FFT estimator recovers a known frequency to span-limited precision.

        Parabolic interpolation of a Hann-windowed peak has error scaling
        with the inverse square of the cycle count; ~3.7 cycles over the
        span (the same order as each frozen orbit's 10 periods) keeps the
        recovery at the percent level, not the machine level.
        """
        n, dt, truth = 1000, 0.01, 0.37
        t = np.arange(n) * dt
        rng = np.random.default_rng(0)
        series = np.stack([
            np.sin(2.0 * np.pi * truth * t + rng.uniform(0.0, 2.0 * np.pi)) + 0.3 * rng.standard_normal(n)
            for _ in range(40)
        ])
        recovered = bundling.dominant_frequency(series, dt)
        np.testing.assert_allclose(recovered, truth, rtol=0.03)

    def test_dominant_frequency_per_orbit_dt(self):
        """Per-orbit dt arrays scale the same peak bins to the right frequency."""
        n, dt, truth = 600, 0.02, 1.7
        t = np.arange(n) * dt
        series = np.sin(2.0 * np.pi * truth * t)[None, :]
        same = bundling.dominant_frequency(series, dt)
        halved = bundling.dominant_frequency(series, np.full(1, 2.0 * dt))
        np.testing.assert_allclose(same, truth, rtol=1e-2)
        np.testing.assert_allclose(halved[0], truth / 2.0, rtol=1e-2)

    def test_resolve_grouping_variables_default_and_contracts(self):
        """Missing key defaults to the audited (lam_z, energy) pair."""
        self.assertEqual(bundling.resolve_grouping_variables({}), ["lam_z", "energy"])
        self.assertEqual(
            bundling.resolve_grouping_variables({"grouping_variables": ["energy", "omega_z"]}),
            ["energy", "omega_z"],
        )
        with self.assertRaises(ValueError):
            bundling.resolve_grouping_variables({"grouping_variables": ["energy"]})
        with self.assertRaises(ValueError):
            bundling.resolve_grouping_variables({"grouping_variables": ["energy", "ecc"]})

    def test_quantile_grid_is_variable_agnostic(self):
        """The same partition code serves any two variables with tied ranks broken."""
        first = np.array([0.1, 0.2, 0.3, 0.4, -1.0, 5.0])
        second = np.array([3.0, 2.0, 1.0, 0.5, -2.0, 10.0])
        assignments, k_total = bundling.quantile_bundle_grid(first, second, 2, 3)
        self.assertEqual(k_total, 6)
        self.assertEqual(assignments.min(), 0)
        self.assertEqual(assignments.max(), 5)


    def test_project_weights_to_bundles_is_member_mean(self):
        """The projection assigns every member its bundle's active-weight mean."""
        active_weights = np.array([2.0, 4.0, 0.0, 3.0, 9.0])
        assignments = np.array([0, 0, 1, 1, 2])  # bundle 1 would be empty if dropped
        u = bundling.project_weights_to_bundles(active_weights, assignments)
        np.testing.assert_allclose(u, [(2.0 + 4.0) / 2, (0.0 + 3.0) / 2, 9.0])
        # Backfilled weights are bundle-constant and stay on mapped seeds only.
        successful = np.array([9, 2, 7, 4, 1], dtype=np.int64)
        active = np.arange(5, dtype=np.int64)
        w = bundling.map_bundle_weights_to_seeds(u, assignments, successful, active, 10)
        expected = np.zeros(10)
        expected[successful[active]] = u[assignments]
        np.testing.assert_array_equal(w, expected)
        # An empty bundle keeps u = 0 and never enters the projection.
        assignments_with_gap = np.array([0, 0, 2, 2, 3])
        u_gap = bundling.project_weights_to_bundles(active_weights, assignments_with_gap)
        self.assertEqual(u_gap[1], 0.0)
        np.testing.assert_allclose(u_gap[[0, 2, 3]], [(2.0 + 4.0) / 2, 1.5, 9.0])


class BundledAssemblyParityTests(np.testing.TestCase):
    def test_assembled_solution_scores_like_direct_compare(self):
        """score_orbit_weights on an assembled bundled solution equals the
        direct compare_density chi2/bin on the same seed weights."""

        grid = CylindricalGrid.uniform(n_r=1, r_range=(0.0, 1.0), n_z=1, z_range=(0.0, 1.0), n_phi=1)
        config = comparison_model(
            grid,
            density_fit={"normalization": "none"},
            include_velocity=True,
            orbit_samples_per_orbit=1,
            orbit_sample_divisor=1,
            weight_model={
                "mode": "density_solved",
                "solver": "lsq_linear",
                "target_normalization": "absolute",
                "regularization": "l2",
                "regularization_strength": 1e-6,
            },
            objective={"mode": "velocity_only", "density_max_chi2_per_bin": 2.0},
        )
        initial = np.array(
            [
                [0.5, -0.5, 0.5, 0, 0, 0],
                [0.2, 0.0, 0.5, 0, 0, 0],
                [0.5, 0.5, 0.5, 0, 0, 0],
            ],
            dtype=float,
        )
        catalogue = SeedCatalogue(initial_conditions=initial, seed_weights=None, velocity_errors={})
        phase_space = cartesian_to_spherical_phase_space(*[initial[:, index] for index in range(6)])
        target_density = np.full(grid.shape, 2.0)
        prepared = PreparedFixedWeightData(
            catalogue=catalogue,
            target_density=target_density,
            target_error=np.full(grid.shape, 0.01),
            config=config,
            catalog_path=Path("synthetic-catalog"),
            density_path=Path("synthetic-target"),
            catalog_phase_space=phase_space,
        )
        library = OrbitLibrary(
            seed_index=np.array([0, 0, 2, 2, 2], dtype=np.int64),
            time=np.arange(5, dtype=float),
            phase_space=np.array([initial[0], initial[0], initial[2], initial[2], initial[2]]),
        )
        response = build_orbit_density_response(
            library, config["density_grid"], seed_count=prepared.initial_conditions.shape[0],
        )
        fit_keys = {
            key: config["density_fit"][key]
            for key in ("min_abs_z", "min_spherical_radius", "max_spherical_radius", "require_positive_data")
        }
        from halo_mw_lmc.density import density_fit_mask

        mask = density_fit_mask(target_density, prepared.target_error, grid, **fit_keys)
        target_n, error_n = _normalized_target(
            target_density, prepared.target_error, mask, response, "absolute",
        )
        problem = _build_weight_problem(response, target_n, error_n, mask, 1e-6)
        # One assignment per active orbit column; the two samples of seed 0
        # share one column, so there are two active columns here.
        n_active = problem.design.shape[1]
        assignments = np.arange(n_active) % 2
        u = bundling.solve_bundled_weights(problem.design, problem.observed, assignments, 1e-6)
        solution, kkt = bundling.assemble_bundled_solution(
            response, problem, u, assignments, target_n, error_n, 0.0,
        )
        with patch(
            "halo_mw_lmc.evaluate._score_velocities",
            return_value=({}, {}, {}, {}),
        ):
            evaluation = score_orbit_weights(library, prepared, solution, response=response)
        direct = compare_density(
            target_n, error_n, response.model_density(solution.seed_weights), grid, **config["density_fit"],
        )
        expected = direct.chi2 / int(np.count_nonzero(direct.fit_mask))
        np.testing.assert_allclose(
            evaluation.density_chi2_per_bin, expected, rtol=1e-12, atol=1e-12,
        )
        self.assertEqual(evaluation.density.scale, 1.0)
        # The reduced-space KKT reported by the assembly must match a direct
        # evaluation of the same residual on the rebuilt augmented problem.
        reduced = bundling._reduced_augmented_problem(problem, assignments)
        raw, normalized = _primal_kkt_residual(reduced, u)
        self.assertEqual(kkt["reduced_raw"], raw)
        self.assertEqual(kkt["reduced_normalized"], normalized)


class JzOverJtotVariableTests(np.testing.TestCase):
    def _library_and_stub(self):
        """Two successful seeds (3 samples each) plus one excluded seed 5."""

        library = SimpleNamespace(
            seed_index=np.array([0, 0, 0, 2, 2, 2, 5, 5], dtype=np.int64),
            time=np.zeros((8, 2)),
            phase_space=np.arange(48, dtype=float).reshape(8, 6),
        )
        successful = np.array([0, 2], dtype=np.int64)
        # (Jr, Jz, Jphi) per successful sample in library order; Jz and |Jphi|
        # vary within seed 0 so orbit means, not sample values, decide the ratio.
        actions = np.array([
            [1.0, 3.0, -2.0], [1.0, 0.0, 2.0], [2.0, 3.0, 4.0],
            [4.0, 1.0, -3.0], [4.0, 1.0, -3.0], [4.0, 1.0, -3.0],
        ])
        agama_stub = types.ModuleType("agama")
        agama_stub.Potential = lambda **kwargs: object()
        agama_stub.ActionFinder = lambda potential: (lambda samples: actions)
        return library, successful, agama_stub

    def test_jz_over_jtot_is_ratio_of_orbit_means(self):
        """jz_over_jtot_phi = mean(Jz) / mean(Jr + Jz + |Jphi|), |Jphi| per sample."""

        library, successful, agama_stub = self._library_and_stub()
        with patch.dict(sys.modules, {"agama": agama_stub}):
            values, _ = bundling.compute_orbit_variables(
                ["jz_over_jtot_phi", "jr_phi", "jz_phi", "jphi_phi"], library, successful, None,
            )
        # seed 0: mean Jz = 2, mean Jr = 4/3, mean |Jphi| = 8/3, so the
        # denominator is the per-sample jtot mean 20/3, not |mean Jphi| = 4/3.
        np.testing.assert_allclose(values["jr_phi"], [4.0 / 3.0, 4.0])
        np.testing.assert_allclose(values["jz_phi"], [2.0, 1.0])
        np.testing.assert_allclose(values["jphi_phi"], [4.0 / 3.0, -3.0])
        np.testing.assert_allclose(values["jz_over_jtot_phi"], [2.0 / 6.0, 1.0 / 8.0])

    def test_jz_over_jtot_computed_when_requested_alone(self):
        """The action-finder block triggers for the ratio without the raw actions."""

        library, successful, agama_stub = self._library_and_stub()
        with patch.dict(sys.modules, {"agama": agama_stub}):
            values, _ = bundling.compute_orbit_variables(
                ["jz_over_jtot_phi"], library, successful, None,
            )
        np.testing.assert_allclose(values["jz_over_jtot_phi"], [2.0 / 6.0, 1.0 / 8.0])
        self.assertEqual(list(values), ["jz_over_jtot_phi"])


class ResponseKmeansGroupingTests(unittest.TestCase):
    def test_resolve_grouping_mode_defaults_and_contracts(self):
        self.assertEqual(
            bundling.resolve_grouping_mode({"n_lambda": 8, "n_energy": 4}),
            {
                "mode": "variables", "n_first": 8, "n_second": 4,
                "variables": ["lam_z", "energy"], "n_third": 1, "k": None, "seed": 0,
                "minibatch_size": None, "minibatch_rounds": None,
                "pca_components": None, "warmstart_iterations": None,
            },
        )
        self.assertEqual(
            bundling.resolve_grouping_mode(
                {"grouping_mode": "variables", "n_lambda": 64, "n_energy": 64}
            ),
            {
                "mode": "variables", "n_first": 64, "n_second": 64,
                "variables": ["lam_z", "energy"], "n_third": 1, "k": None, "seed": 0,
                "minibatch_size": None, "minibatch_rounds": None,
                "pca_components": None, "warmstart_iterations": None,
            },
        )
        self.assertEqual(
            bundling.resolve_grouping_mode(
                {"grouping_mode": "response_kmeans", "kmeans_bundles": 1024}
            ),
            {
                "mode": "response_kmeans", "n_first": None, "n_second": None,
                "variables": None, "n_third": None, "k": 1024, "seed": 0,
                "minibatch_size": None, "minibatch_rounds": None,
                "pca_components": None, "warmstart_iterations": None,
            },
        )
        self.assertEqual(
            bundling.resolve_grouping_mode(
                {"grouping_mode": "response_kmeans", "kmeans_bundles": 2304, "kmeans_seed": 7}
            ),
            {
                "mode": "response_kmeans", "n_first": None, "n_second": None,
                "variables": None, "n_third": None, "k": 2304, "seed": 7,
                "minibatch_size": None, "minibatch_rounds": None,
                "pca_components": None, "warmstart_iterations": None,
            },
        )
        with self.assertRaises(ValueError):
            bundling.resolve_grouping_mode({"grouping_mode": "response_kmeans", "kmeans_bundles": 0})
        with self.assertRaises(ValueError):
            bundling.resolve_grouping_mode({"grouping_mode": "spectral"})

    def test_resolve_grouping_mode_section13_families(self):
        self.assertEqual(
            bundling.resolve_grouping_mode(
                {"grouping_mode": "response_kmeans_minibatch", "kmeans_bundles": 2304}
            )["minibatch_size"],
            1024,
        )
        self.assertEqual(
            bundling.resolve_grouping_mode(
                {"grouping_mode": "response_kmeans_minibatch", "kmeans_bundles": 2304}
            )["minibatch_rounds"],
            100,
        )
        self.assertEqual(
            bundling.resolve_grouping_mode(
                {"grouping_mode": "response_pca_grid", "pca_first_bins": 48, "pca_second_bins": 48}
            ),
            {
                "mode": "response_pca_grid", "n_first": 48, "n_second": 48,
                "variables": None, "n_third": None, "k": None, "seed": 0,
                "minibatch_size": None, "minibatch_rounds": None,
                "pca_components": 2, "warmstart_iterations": None,
            },
        )
        self.assertEqual(
            bundling.resolve_grouping_mode(
                {"grouping_mode": "response_kmeans_warmstart", "n_lambda": 48, "n_energy": 48}
            ),
            {
                "mode": "response_kmeans_warmstart", "n_first": 48, "n_second": 48,
                "variables": ["jz_over_jtot_phi", "energy"], "n_third": None, "k": None, "seed": 0,
                "minibatch_size": None, "minibatch_rounds": None,
                "pca_components": None, "warmstart_iterations": 5,
            },
        )
        with self.assertRaises(ValueError):
            bundling.resolve_grouping_mode(
                {"grouping_mode": "response_pca_grid", "pca_components": 1,
                 "pca_first_bins": 8, "pca_second_bins": 8}
            )
        with self.assertRaises(ValueError):
            bundling.resolve_grouping_mode(
                {"grouping_mode": "response_kmeans_warmstart", "n_lambda": 8,
                 "n_energy": 8, "warmstart_iterations": -1}
            )
        with self.assertRaises(ValueError) as caught:
            bundling.resolve_grouping_mode({"grouping_mode": "ward"})
        self.assertIn("response_kmeans_minibatch", str(caught.exception))

    def test_resolve_grouping_mode_three_variable_contracts(self):
        config = {
            "grouping_variables": ["jz_over_jtot_phi", "energy", "lam_z"],
            "n_lambda": 16, "n_energy": 16, "n_third": 9,
        }
        spec = bundling.resolve_grouping_mode(config)
        self.assertEqual(spec["variables"], ["jz_over_jtot_phi", "energy", "lam_z"])
        self.assertEqual((spec["n_first"], spec["n_second"], spec["n_third"]), (16, 16, 9))
        with self.assertRaises(ValueError):
            bundling.resolve_grouping_mode({**config, "n_third": 1})
        with self.assertRaises(ValueError):
            bundling.resolve_grouping_mode(
                {"grouping_variables": ["jz_over_jtot_phi", "energy"],
                 "n_lambda": 8, "n_energy": 8, "n_third": 4}
            )
        with self.assertRaises(ValueError):
            bundling.resolve_grouping_mode(
                {"grouping_variables": ["lam_z", "energy", "omega_z", "jr_phi"],
                 "n_lambda": 8, "n_energy": 8, "n_third": 4}
            )

    def test_response_kmeans_groups_identical_column_pairs_deterministically(self):
        """Two clusters of exactly identical columns separate under any seed."""

        design = scipy.sparse.csr_matrix(
            np.array(
                [
                    [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                    [0.0, 0.0, 0.0, 0.2, 0.2, 0.2],
                ]
            )
        )
        assignments, distortion = bundling.response_kmeans_assignments(design, 2, seed=0)
        self.assertEqual(assignments.shape, (6,))
        self.assertTrue(np.all(assignments[:3] == assignments[0]))
        self.assertTrue(np.all(assignments[3:] == assignments[3]))
        self.assertNotEqual(assignments[0], assignments[3])
        again, _ = bundling.response_kmeans_assignments(design, 2, seed=0)
        self.assertTrue(np.array_equal(assignments, again))
        self.assertEqual(distortion["populated_bundles"], 2)
        # Identical columns within every bundle: the equal-weight column mean
        # reproduces each column exactly, so the audit-definition D is zero.
        self.assertAlmostEqual(distortion["equal_weight_distortion"], 0.0, places=12)

    def test_equal_weight_distortion_of_zero_for_one_bundle_of_identical_columns(self):
        design = scipy.sparse.csr_matrix(np.tile(np.array([[1.0], [2.0], [0.5]]), (1, 3)))
        record = bundling.equal_weight_distortion_of(design, np.zeros(3, dtype=np.int64))
        self.assertEqual(record["populated_bundles"], 1)
        self.assertAlmostEqual(record["equal_weight_distortion"], 0.0, places=12)


class BundleAlternativesGroupingTests(unittest.TestCase):
    """Small-array acceptance for the §13 grouping families and concentration."""

    @staticmethod
    def _three_cluster_design():
        # Three well-separated response directions (rows 0-2), each carrying
        # four orbits with tiny per-orbit jitter so k-means++ never runs out
        # of distinct points.
        rng = np.random.default_rng(3)
        centers = np.array(
            [
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 10.0],
            ]
        )
        columns = np.vstack([center + 1e-3 * rng.normal(size=(4, 3)) for center in centers])
        return scipy.sparse.csr_matrix(columns.T)

    def test_minibatch_kmeans_recovers_separated_clusters_deterministically(self):
        design = self._three_cluster_design()
        assignments, record = bundling.minibatch_kmeans_assignments(
            design, 3, seed=0, batch_size=6, rounds=10,
        )
        self.assertEqual(assignments.shape, (12,))
        for start in (0, 4, 8):
            block = assignments[start:start + 4]
            self.assertTrue(np.all(block == block[0]), f"cluster at {start} split")
        self.assertEqual(len(np.unique(assignments)), 3)
        self.assertEqual(record["populated_bundles"], 3)
        again, _ = bundling.minibatch_kmeans_assignments(design, 3, seed=0, batch_size=6, rounds=10)
        self.assertTrue(np.array_equal(assignments, again))

    def test_response_pca_grid_is_deterministic_and_sign_invariant(self):
        rng = np.random.default_rng(5)
        base = rng.normal(size=(5, 40))
        base[:2] *= 8.0  # leading two directions dominate the spectrum
        design = scipy.sparse.csr_matrix(base)
        assignments, record = bundling.response_pca_assignments(design, 4, 5, 2)
        again, _ = bundling.response_pca_assignments(design, 4, 5, 2)
        self.assertTrue(np.array_equal(assignments, again))
        self.assertEqual(record["pca_components"], 2)
        # Negating the design flips principal-direction signs; the quantile
        # grid only relabels axes, so the partition structure is unchanged.
        flipped, flipped_record = bundling.response_pca_assignments(
            scipy.sparse.csr_matrix(-base), 4, 5, 2,
        )
        self.assertTrue(
            np.array_equal(np.sort(np.bincount(assignments)), np.sort(np.bincount(flipped))),
        )
        self.assertAlmostEqual(
            record["pca_explained_variance_share"][0],
            flipped_record["pca_explained_variance_share"][0],
        )

    def test_warmstart_zero_iterations_is_the_init_partition(self):
        design = self._three_cluster_design()
        points = np.asarray(design.todense(), dtype=float).T
        init_centers = points[[0, 4, 8]]
        assignments, record = bundling.warmstart_kmeans_assignments(design, init_centers, 0)
        self.assertEqual(record["warmstart_iterations_used"], 0)
        self.assertEqual(record["warmstart_init_centers"], 3)
        for start in (0, 4, 8):
            block = assignments[start:start + 4]
            self.assertTrue(np.all(block == block[0]))
        refined, refined_record = bundling.warmstart_kmeans_assignments(design, init_centers, 5)
        self.assertLessEqual(refined_record["warmstart_iterations_used"], 5)
        # Centroid-initialized Lloyd is converged after one step on tight
        # clusters, so further refinement must not move any orbit.
        self.assertTrue(np.array_equal(assignments, refined))

    def test_weight_concentration_known_vector(self):
        weights = np.array([0.5, 0.25, 0.125, 0.125])
        record = bundling.weight_concentration(weights, np.array([0, 0, 1, 1]))
        self.assertAlmostEqual(record["n_eff"], 1.0 / 0.34375, places=12)
        self.assertAlmostEqual(record["max_fraction"], 0.5, places=12)
        self.assertAlmostEqual(record["hhi"], 0.34375, places=12)
        self.assertEqual(record["n90"], 4)
        self.assertEqual(record["n99"], 4)
        self.assertEqual(record["nonzero_orbits"], 4)
        bundle = record["bundle"]
        self.assertEqual(bundle["populated_bundles"], 2)
        self.assertAlmostEqual(bundle["n_eff"], 1.0 / 0.625, places=12)
        self.assertAlmostEqual(bundle["max_fraction"], 0.75, places=12)
        self.assertAlmostEqual(bundle["hhi"], 0.625, places=12)
        self.assertEqual(bundle["n90"], 2)
        with self.assertRaises(ValueError):
            bundling.weight_concentration(np.zeros(4))
        with self.assertRaises(ValueError):
            bundling.weight_concentration(
                np.ones(6), np.array([0, 0, 1, 1]),
            )

    def test_quantile_bundle_partition_matches_grid_and_composes_third_axis(self):
        rng = np.random.default_rng(9)
        first = rng.normal(size=600)
        second = rng.normal(size=600)
        third = rng.normal(size=600)
        grid_assignments, grid_total = bundling.quantile_bundle_grid(first, second, 8, 6)
        partition_assignments, partition_total = bundling.quantile_bundle_partition(
            [first, second], [8, 6],
        )
        self.assertTrue(np.array_equal(grid_assignments, partition_assignments))
        self.assertEqual(grid_total, partition_total)
        # Third axis composes row-major: the composite index of a 3-axis
        # partition equals the 2-axis index of the first two axes times the
        # third bin count plus the third-axis bin.
        three_assignments, three_total = bundling.quantile_bundle_partition(
            [first, second, third], [8, 6, 4],
        )
        self.assertEqual(three_total, 8 * 6 * 4)
        self.assertTrue(np.array_equal(three_assignments // 4, partition_assignments))
        self.assertEqual(np.max(three_assignments), three_total - 1)

    def test_xy_angle_recovers_ellipse_orientation_with_pi_periodicity(self):
        angles = np.array([0.0, np.pi / 3, np.pi / 3 + np.pi, -np.pi / 6])
        t = np.linspace(0.0, 4.0 * np.pi, 128)
        phase = np.zeros((angles.size, t.size, 6))
        for row, theta in enumerate(angles):
            phase[row, :, 0] = 3.0 * np.cos(t) * np.cos(theta) - 0.5 * np.sin(t) * np.sin(theta)
            phase[row, :, 1] = 3.0 * np.cos(t) * np.sin(theta) + 0.5 * np.sin(t) * np.cos(theta)
        library = SimpleNamespace(
            seed_index=np.repeat(np.arange(angles.size), t.size),
            phase_space=phase.reshape(-1, 6),
        )
        successful = np.arange(angles.size)
        values, _ = bundling.compute_orbit_variables(
            ["xy_angle"], library, successful, None,
        )
        recovered = values["xy_angle"]
        # The principal-axis angle has period pi: theta + pi is the same axis.
        wrapped = np.angle(np.exp(2j * (recovered - angles))) / 2.0
        self.assertTrue(np.all(np.abs(wrapped) < 1e-9))


if __name__ == "__main__":
    unittest.main()
