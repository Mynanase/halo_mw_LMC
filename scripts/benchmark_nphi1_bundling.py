#!/usr/bin/env python3
"""nphi1 bundling repair experiment: core math and experiment driver.

Experimental-script companion of ``docs/nphi1_bundling_repair_plan.md``:
§2.2 quantile bundle assignment in two configurable orbit variables (the
audited default is (lambda_z, E); §6 adds (E, omega_z) and (Jr, Jz)), the
reduced bundled NNLS problem ``min(u>=0) ||A S u - b||^2 + lambda *
sum_k(n_k * u_k^2)``, and the strict active-orbit -> successful-seed ->
full-seed backfill of bundle weights; plus the stage-2 driver that runs
full/bundled/random on one frozen orbit library through the shared
production scoring boundary.
Section 12 adds the response-kmeans grouping mode: bundles from k-means on
the error-normalized density design columns (the plan §11 audit oracle),
re-solved through the same shared solve/score path.
Section 13 adds three cheaper grouping families on the same orbit points:
minibatch k-means, a deterministic PCA quantile grid, and capped Lloyd
refinement warm-started from physical-grid centroids; every case also records
orbit-level and bundle-mass-level weight-concentration diagnostics.

The bundled solve solves for one weight u_k per bundle; the per-orbit weight
is ``w = u[assignment]``, so the L2 penalty on w maps to ``sum_k n_k u_k^2``
with n_k the bundle member count. The member count is derived inside
``solve_bundled_weights`` from the same assignment array passed to the solve,
so the mapping and the regularization can never disagree. The driver keeps
one ``assignments`` variable per case from the quantile partition through
both the solve and the backfill, and the random control maps its own
permuted assignments through the same pair of calls.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import tomllib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

# Iteration budget for scipy.optimize.nnls, inherited from the audited
# prototype (.agent-local/benchmarks/nphi1_full_iteration_bundled.py);
# 100 * k_total.
NNLS_MAXITER_PER_BUNDLE = 100

# Reduced-space KKT acceptance for the bundled NNLS, fixed before the run
# (plan §2.3); also mirrored in configs/benchmarks/nphi1_bundling.toml.
REDUCED_KKT_TOLERANCE = 1e-8


def quantile_bundle_grid(first, second, n_first, n_second):
    """Quantile bundle partition in two orbit variables; return (assignments, k_total).

    Edges are linspace(0, 1, n+1) quantiles with open outer edges
    (-inf/inf); bins use searchsorted(side="right") - 1 clipped to the
    interior, so every orbit lands in exactly one of k_total = n_first *
    n_second bundles. Variable-agnostic: the audited (lambda_z, E) grouping
    and the §6 two-dimensional alternatives pass through the same code.
    """

    first_edges = np.quantile(first, np.linspace(0.0, 1.0, n_first + 1))
    first_edges[0], first_edges[-1] = -np.inf, np.inf
    second_edges = np.quantile(second, np.linspace(0.0, 1.0, n_second + 1))
    second_edges[0], second_edges[-1] = -np.inf, np.inf
    first_bin = np.clip(np.searchsorted(first_edges, first, side="right") - 1, 0, n_first - 1)
    second_bin = np.clip(np.searchsorted(second_edges, second, side="right") - 1, 0, n_second - 1)
    assignments = first_bin * n_second + second_bin
    return assignments, n_first * n_second


def solve_bundled_weights(design, observed, assignments, regularization):
    """Solve min(u>=0) ||A S u - b||^2 + lambda * sum_k n_k u_k^2; return u.

    ``design`` is the (rows, n_active) error-normalized orbit design matrix
    and ``assignments`` maps each active orbit column to its bundle. The
    member counts n_k are computed here from ``assignments`` so the mapping
    and the regularization diagonal cannot diverge. Empty bundles keep
    u_k = 0 (they are not identifiable). The augmented matrix is
    [[A S]; [sqrt(lambda * n_k) I]] against [b; 0], solved by dense NNLS
    with maxiter = NNLS_MAXITER_PER_BUNDLE * k_total.
    """

    rows, n_active = design.shape
    k_total = int(np.max(assignments)) + 1
    member_count = np.bincount(assignments, minlength=k_total).astype(float)
    indicator = np.zeros((n_active, k_total))
    indicator[np.arange(n_active), assignments] = 1.0
    reduced = design @ indicator
    dense = np.zeros((rows + k_total, k_total), dtype=float, order="F")
    dense[:rows, :] = reduced
    dense[rows:, :].flat[:: k_total + 1] = np.sqrt(regularization * member_count)

    from scipy.optimize import nnls

    bundle_weights, _ = nnls(
        dense, np.concatenate([observed, np.zeros(k_total)]),
        maxiter=NNLS_MAXITER_PER_BUNDLE * k_total,
    )
    bundle_weights = np.asarray(bundle_weights, dtype=float)
    bundle_weights[member_count == 0] = 0.0
    return bundle_weights


def map_bundle_weights_to_seeds(bundle_weights, assignments, successful_seed_index, active_columns, seed_count):
    """Backfill bundle weights u into a full seed-weight array.

    Strict order: active orbit columns -> successful seeds via
    ``successful_seed_index[active_columns]`` -> the full seed array of
    length ``seed_count``. No assumption of contiguous, sorted, or complete
    seed numbering; seeds outside the successful set stay zero.
    """

    seed_weights = np.zeros(seed_count, dtype=float)
    seed_weights[successful_seed_index[active_columns]] = bundle_weights[assignments]
    return seed_weights


def project_weights_to_bundles(active_weights, assignments):
    """L2 bundle-mean projection of active-orbit weights; return bundle weights u.

    The closest (in L2 on the active columns) bundle-constant weight vector
    assigns every member the mean of its bundle's active weights. Unlike a
    bundled re-solve this touches no solver: it isolates the effect of the
    equal-weight subspace direction itself. Empty bundles stay zero.
    """

    k_total = int(np.max(assignments)) + 1
    member_count = np.bincount(assignments, minlength=k_total)
    weight_total = np.bincount(assignments, weights=active_weights, minlength=k_total)
    projection = np.zeros(k_total, dtype=float)
    populated = member_count > 0
    projection[populated] = weight_total[populated] / member_count[populated]
    return projection


# -----------------------------------------------------------------------
# Driver support: hashing, reduced KKT, assembly, artifact IO.
# -----------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _assignments_fingerprint(assignments: np.ndarray) -> str:
    digest = hashlib.sha256()
    contiguous = np.ascontiguousarray(assignments, dtype=np.int64)
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _reduced_augmented_problem(problem, assignments: np.ndarray):
    """Rebuild [[A S]; [sqrt(lambda n_k) I]] vs [b; 0] for the KKT audit only.

    Mirrors the construction inside ``solve_bundled_weights``; used with
    ``_primal_kkt_residual`` (regularization zero) to report the reduced-space
    KKT of the bundled solution. The reduced problem is a restricted
    approximation of the original per-orbit problem, never its equivalent.
    """

    from halo_mw_lmc.weights import _WeightProblem

    rows, n_active = problem.design.shape
    k_total = int(np.max(assignments)) + 1
    member_count = np.bincount(assignments, minlength=k_total).astype(float)
    indicator = np.zeros((n_active, k_total))
    indicator[np.arange(n_active), assignments] = 1.0
    dense = np.zeros((rows + k_total, k_total), dtype=float)
    dense[:rows, :] = problem.design @ indicator
    dense[rows:, :].flat[:: k_total + 1] = np.sqrt(problem.regularization * member_count)
    observed = np.concatenate([problem.observed, np.zeros(k_total)])
    reduced = _WeightProblem(
        design=dense,
        observed=observed,
        active_columns=np.ones(k_total, dtype=bool),
        successful_orbit_count=problem.successful_orbit_count,
        regularization=0.0,
        fingerprint="",
    )
    return reduced


def _circularity_scale(potential):
    """Lz_circ(E) interpolant grid, replicating the audited prototype math
    (.agent-local/benchmarks/lambdaz_bundle_prototype.py::circularity_scale).
    """

    phi_center = float(potential.potential(np.array([[0.0, 0.0, 0.0]]))[0])
    energies = np.linspace(phi_center * (1.0 - 1e-9), -1e-4, 4096)
    try:
        radius = np.asarray(potential.Rcirc(E=energies), dtype=float)
    except Exception:
        radius = np.array([float(potential.Rcirc(E=float(value))) for value in energies])
    finite = np.isfinite(radius) & (radius > 0)
    radius, energies = radius[finite], energies[finite]
    points = np.column_stack([radius, np.zeros_like(radius), np.zeros_like(radius)])
    force_r = np.asarray(potential.force(points), dtype=float)[:, 0]
    v_circ = np.sqrt(radius * np.abs(force_r))
    lz_circ = radius * v_circ
    order = np.argsort(energies)
    return energies[order], lz_circ[order]


SUPPORTED_GROUPING_VARIABLES = (
    "lam_z", "energy", "omega_z", "jr_phi", "jz_phi", "jphi_phi", "jz_over_jtot_phi",
)


def resolve_grouping_variables(experiment: dict) -> list[str]:
    """Read ``grouping_variables`` from the experiment dict; default (lam_z, energy)."""

    names = [str(name) for name in experiment.get("grouping_variables", ["lam_z", "energy"])]
    if len(names) != 2:
        raise ValueError("grouping_variables must name exactly two orbit variables")
    unknown = [name for name in names if name not in SUPPORTED_GROUPING_VARIABLES]
    if unknown:
        raise ValueError(
            f"unsupported grouping variables: {', '.join(unknown)}; "
            f"supported: {', '.join(SUPPORTED_GROUPING_VARIABLES)}"
        )
    return names


def resolve_grouping_mode(experiment: dict) -> dict:
    """Read `grouping_mode` (default "variables") and its mode-specific keys.

    "variables" is the §2/§6/§10 quantile partition in two orbit variables
    (`n_lambda` x `n_energy`). "response_kmeans" is the §12 oracle bundling:
    k-means on the error-normalized response columns with `kmeans_bundles`
    centers and `kmeans_seed` (default 0, the §11 audit seed, so the same
    design reproduces the audit oracle assignments bit-for-bit).
    Section 13 adds three cheaper families on the same points:
    "response_kmeans_minibatch" (k-means++ on a subsample plus Sculley-style
    minibatch updates; keys `kmeans_bundles`, `kmeans_seed`,
    `minibatch_size` default 1024, `minibatch_rounds` default 100),
    "response_pca_grid" (deterministic SVD reduction; keys
    `pca_components` default 2, `pca_first_bins`, `pca_second_bins`),
    and "response_kmeans_warmstart" (Lloyd from `warmstart_variables`
    quantile-grid centroids, default (jz_over_jtot_phi, energy) at
    `n_lambda` x `n_energy` cells, capped at `warmstart_iterations`
    default 5; 0 iterations is the pure grid reference). Returns a spec
    dict; the driver never mutates it.
    """

    mode = str(experiment.get("grouping_mode", "variables"))
    spec = {
        "mode": mode, "n_first": None, "n_second": None, "variables": None,
        "k": None, "seed": 0, "minibatch_size": None, "minibatch_rounds": None,
        "pca_components": None, "warmstart_iterations": None,
    }
    if mode == "variables":
        spec.update(
            variables=resolve_grouping_variables(experiment),
            n_first=int(experiment["n_lambda"]), n_second=int(experiment["n_energy"]),
        )
        return spec
    if mode == "response_kmeans":
        bundles = int(experiment["kmeans_bundles"])
        if bundles <= 0:
            raise ValueError("kmeans_bundles must be a positive integer")
        spec.update(k=bundles, seed=int(experiment.get("kmeans_seed", 0)))
        return spec
    if mode == "response_kmeans_minibatch":
        bundles = int(experiment["kmeans_bundles"])
        if bundles <= 0:
            raise ValueError("kmeans_bundles must be a positive integer")
        size = int(experiment.get("minibatch_size", 1024))
        rounds = int(experiment.get("minibatch_rounds", 100))
        if size <= 0 or rounds <= 0:
            raise ValueError("minibatch_size and minibatch_rounds must be positive integers")
        spec.update(
            k=bundles, seed=int(experiment.get("kmeans_seed", 0)),
            minibatch_size=size, minibatch_rounds=rounds,
        )
        return spec
    if mode == "response_pca_grid":
        components = int(experiment.get("pca_components", 2))
        if components < 2:
            raise ValueError("pca_components must be at least 2 (the grid is two-dimensional)")
        n_first = int(experiment["pca_first_bins"])
        n_second = int(experiment["pca_second_bins"])
        if n_first <= 0 or n_second <= 0:
            raise ValueError("pca_first_bins and pca_second_bins must be positive integers")
        spec.update(pca_components=components, n_first=n_first, n_second=n_second)
        return spec
    if mode == "response_kmeans_warmstart":
        names = [str(name) for name in experiment.get(
            "warmstart_variables", ["jz_over_jtot_phi", "energy"],
        )]
        if len(names) != 2:
            raise ValueError("warmstart_variables must name exactly two orbit variables")
        unknown = [name for name in names if name not in SUPPORTED_GROUPING_VARIABLES]
        if unknown:
            raise ValueError(
                f"unsupported warmstart variables: {', '.join(unknown)}; "
                f"supported: {', '.join(SUPPORTED_GROUPING_VARIABLES)}"
            )
        iterations = int(experiment.get("warmstart_iterations", 5))
        if iterations < 0:
            raise ValueError("warmstart_iterations must be >= 0 (0 = the pure grid reference)")
        spec.update(
            variables=names, n_first=int(experiment["n_lambda"]),
            n_second=int(experiment["n_energy"]), warmstart_iterations=iterations,
        )
        return spec
    raise ValueError(
        f"unsupported grouping_mode: {mode}; supported: variables, response_kmeans, "
        "response_kmeans_minibatch, response_pca_grid, response_kmeans_warmstart"
    )


def response_kmeans_assignments(design, bundle_count, seed=0):
    """Equal-weight bundles from k-means on the error-normalized response columns.

    Groups the active-orbit design columns with the §11 audit's kmeans++/Lloyd
    implementation (same seed and iteration budget, so the same design
    reproduces the audit oracle assignments bit-for-bit); returns the
    assignments plus the audit-definition equal-weight distortion record.
    The grouping input is the density design only -- the velocity objective
    never enters the grouping.
    """

    sys.path.insert(0, str(REPO / "scripts"))
    from audit_third_invariant import kmeans_columns
    from review_fz_energy_basis import equal_weight_distortion

    design_dense = np.asarray(design.todense(), dtype=float)
    assignments = np.asarray(
        kmeans_columns(design_dense.T.copy(), int(bundle_count), seed=seed), dtype=np.int64,
    )
    distortion = equal_weight_distortion(design_dense, assignments, int(bundle_count))
    return assignments, {
        "equal_weight_distortion": distortion["distortion"],
        "populated_bundles": distortion["populated_bundles"],
    }


def _design_points(design):
    """Active design columns as orbit points (n_orbits, n_rows) for clustering."""

    return np.asarray(design.todense(), dtype=float).T.copy()


def _nearest_labels(points, centers):
    """Index of the nearest center per point (squared-distance expansion)."""

    return np.argmin(
        np.sum(points ** 2, axis=1)[:, None]
        - 2.0 * (points @ centers.T)
        + np.sum(centers ** 2, axis=1)[None, :],
        axis=1,
    )


def _kmeanspp_centers(points, bundle_count, rng):
    """k-means++ seeding with BLAS-form D^2 updates (expansion identity).

    Same sampling sequence as the §11 audit's broadcasting form, but each new
    center's distances are computed as ||p||^2 - 2 p.c + ||c||^2 (a matvec
    plus vector ops) instead of materializing a (points, features) temporary;
    on the nphi4 library this moves the seeding cost from memory-bound to
    compute-bound. Tiny floating-point differences relative to the audit form
    are acceptable: this path feeds the §13 minibatch mode only.
    """

    count = points.shape[0]
    if count < bundle_count:
        raise ValueError("k-means++ sample is smaller than the requested bundle count")
    centers = points[rng.integers(count)].reshape(1, -1).copy()
    min_distance = np.full(count, np.inf)
    point_sq = np.sum(points ** 2, axis=1)
    while centers.shape[0] < bundle_count:
        center = centers[-1]
        distance = np.maximum(point_sq - 2.0 * (points @ center) + float(np.dot(center, center)), 0.0)
        min_distance = np.minimum(min_distance, distance)
        min_distance[~np.isfinite(min_distance)] = 0.0
        total = min_distance.sum()
        if total <= 0:
            raise ValueError("kmeans++ ran out of distinct points")
        centers = np.vstack([centers, points[rng.choice(count, p=min_distance / total)]])
    return centers


def minibatch_kmeans_assignments(design, bundle_count, seed=0, batch_size=1024, rounds=100):
    """Equal-weight bundles from minibatch k-means (§13 cheap-oracle variant).

    Same orbit points and objective family as the §12 oracle -- k-means++
    seeding plus Sculley-style per-center count updates on small batches --
    intended to test whether oracle-grade grouping survives at a fraction of
    the full Lloyd cost. Deterministic for a fixed seed. The grouping input is
    the density design only; the velocity objective never enters the grouping.
    """

    points = _design_points(design)
    rng = np.random.default_rng(seed)
    sample_size = min(points.shape[0], max(3 * int(bundle_count), 4096))
    init_sample = points[rng.choice(points.shape[0], size=sample_size, replace=False)]
    centers = _kmeanspp_centers(init_sample, int(bundle_count), rng)
    counts = np.zeros(int(bundle_count), dtype=float)
    batch_size = min(int(batch_size), points.shape[0])
    for _ in range(int(rounds)):
        batch = points[rng.choice(points.shape[0], size=batch_size, replace=False)]
        labels = _nearest_labels(batch, centers)
        sums = np.zeros_like(centers)
        np.add.at(sums, labels, batch)
        batch_counts = np.bincount(labels, minlength=centers.shape[0]).astype(float)
        active = batch_counts > 0
        means = sums[active] / batch_counts[active][:, None]
        previous = counts[active]
        factor = np.where(
            previous > 0, previous / np.maximum(previous + batch_counts[active], 1e-300), 0.0,
        )[:, None]
        centers[active] = np.where(
            previous[:, None] > 0,
            centers[active] + factor * (means - centers[active]),
            means,
        )
        counts[active] += batch_counts[active]
    assignments = np.asarray(_nearest_labels(points, centers), dtype=np.int64)
    populated = np.bincount(assignments, minlength=int(bundle_count))
    record = {
        "minibatch_init_sample": int(sample_size),
        "minibatch_batch_size": int(batch_size),
        "minibatch_rounds": int(rounds),
        "populated_bundles": int(np.count_nonzero(populated)),
    }
    return assignments, record


def response_pca_assignments(design, n_first, n_second, components):
    """Equal-weight bundles from a quantile grid on leading response-PCA scores (§13).

    Deterministic linear reduction of the error-normalized design columns:
    center the orbit points, keep the leading principal directions, and
    partition the leading two scores with the §2.2 variable-agnostic quantile
    grid. No iterative clustering and no random draws; the grid is invariant
    to per-axis sign flips (a rank-preserving relabeling).
    """

    points = _design_points(design)
    centered = points - points.mean(axis=0, keepdims=True)
    _, singular, right_vectors = np.linalg.svd(centered, full_matrices=False)
    scores = centered @ right_vectors[: int(components)].T
    total_energy = float(np.sum(singular ** 2))
    explained = [float(value ** 2 / total_energy) for value in singular[: int(components)]]
    assignments, k_total = quantile_bundle_grid(scores[:, 0], scores[:, 1], int(n_first), int(n_second))
    populated = np.bincount(assignments, minlength=int(k_total))
    record = {
        "pca_components": int(components),
        "pca_explained_variance_share": explained,
        "populated_bundles": int(np.count_nonzero(populated)),
    }
    return assignments, record


def warmstart_kmeans_assignments(design, init_centers, iterations):
    """Lloyd refinement from provided centroids, iteration-capped (§13).

    Starts from physical-grid cell centroids instead of k-means++ seeding, so
    the cost is one distance pass per iteration; tests how far a few Lloyd
    steps move the (f_z,E) grid toward the oracle partition. Deterministic:
    no random draws, and centers that lose all members keep their position.
    """

    points = _design_points(design)
    centers = np.asarray(init_centers, dtype=float).copy()
    used = 0
    for iteration in range(int(iterations)):
        labels = _nearest_labels(points, centers)
        sums = np.zeros_like(centers)
        np.add.at(sums, labels, points)
        counts = np.bincount(labels, minlength=centers.shape[0]).astype(float)
        populated = counts > 0
        new_centers = centers.copy()
        new_centers[populated] = sums[populated] / counts[populated][:, None]
        shift = float(np.max(np.sum((new_centers - centers) ** 2, axis=1)))
        centers = new_centers
        used = iteration + 1
        if shift < 1e-12:
            break
    assignments = np.asarray(_nearest_labels(points, centers), dtype=np.int64)
    populated = np.bincount(assignments, minlength=centers.shape[0])
    record = {
        "warmstart_iterations_used": used,
        "warmstart_init_centers": int(centers.shape[0]),
        "populated_bundles": int(np.count_nonzero(populated)),
    }
    return assignments, record


def weight_concentration(weights, assignments=None, k_total=None):
    """Orbit-level and (for bundled cases) bundle-mass-level concentration (§13).

    Orbit level: shares of the per-orbit (backfilled) weights over the total.
    Bundle level (when `assignments` is given): shares of the bundle masses
    n_k u_k, the quantity §12 reported as the maximum bundle weight fraction.
    n90/n99 are the counts of orbits/bundles carrying 90/99% of the mass.
    """

    w = np.asarray(weights, dtype=float).ravel()
    total = float(np.sum(w))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("weight_concentration needs a finite positive total weight")
    if assignments is not None and np.asarray(assignments).shape[0] != w.shape[0]:
        raise ValueError(
            "weight_concentration: assignments must index the same orbit vector as the weights"
        )
    shares = np.sort(w / total)[::-1]
    cumulative = np.cumsum(shares)
    record = {
        "n_eff": float(total ** 2 / float(np.dot(w, w))),
        "max_fraction": float(shares[0]),
        "top10_share": float(np.sum(shares[:10])),
        "top100_share": float(np.sum(shares[:100])),
        "hhi": float(np.dot(shares, shares)),
        "n90": int(np.searchsorted(cumulative, 0.90) + 1),
        "n99": int(np.searchsorted(cumulative, 0.99) + 1),
        "nonzero_orbits": int(np.count_nonzero(w > 0)),
    }
    if assignments is not None:
        if k_total is None:
            k_total = int(np.max(assignments)) + 1
        member = np.bincount(assignments, minlength=int(k_total)).astype(float)
        mass = np.zeros(int(k_total))
        np.add.at(mass, assignments, w)
        bundle_shares = np.sort(mass[mass > 0] / total)[::-1]
        bundle_cumulative = np.cumsum(bundle_shares)
        record["bundle"] = {
            "populated_bundles": int(bundle_shares.size),
            "n_eff": float(total ** 2 / float(np.dot(mass, mass))),
            "max_fraction": float(bundle_shares[0]),
            "top10_share": float(np.sum(bundle_shares[:10])),
            "top100_share": float(np.sum(bundle_shares[:100])),
            "hhi": float(np.dot(bundle_shares, bundle_shares)),
            "n90": int(np.searchsorted(bundle_cumulative, 0.90) + 1),
            "n99": int(np.searchsorted(bundle_cumulative, 0.99) + 1),
        }
    return record


def equal_weight_distortion_of(design, assignments):
    """D and populated-bundle count of one equal-weight partition (§11 definition)."""

    sys.path.insert(0, str(REPO / "scripts"))
    from review_fz_energy_basis import equal_weight_distortion

    design_dense = np.asarray(design.todense(), dtype=float)
    record = equal_weight_distortion(design_dense, assignments, int(np.max(assignments)) + 1)
    return {"equal_weight_distortion": record["distortion"], "populated_bundles": record["populated_bundles"]}


def dominant_frequency(series, dt):
    """Hann-windowed FFT peak with log-parabola interpolation; cycles per time unit.

    ``series`` has shape (n_orbits, n_samples), sampled on a per-orbit uniform
    grid with step ``dt`` (scalar or length-n_orbits array; the frozen library
    integrates each orbit over its own 10 periods, so dt differs per orbit).
    Bin 0 carries only leakage and is skipped. Origin: the round-2 orbit
    invariant exploration (scripts/explore_orbit_invariants.py).
    """

    n = series.shape[1]
    window = np.hanning(n)
    centered = series - series.mean(axis=1, keepdims=True)
    spectrum = np.abs(np.fft.rfft(centered * window, axis=1))
    peak = np.argmax(spectrum[:, 1:], axis=1) + 1
    rows = np.arange(series.shape[0])
    alpha = np.log(np.maximum(spectrum[rows, np.maximum(peak - 1, 0)], 1e-300))
    beta = np.log(np.maximum(spectrum[rows, peak], 1e-300))
    gamma = np.log(np.maximum(spectrum[rows, np.minimum(peak + 1, spectrum.shape[1] - 1)], 1e-300))
    denominator = alpha - 2.0 * beta + gamma
    delta = np.where(
        np.abs(denominator) > 1e-12,
        0.5 * (alpha - gamma) / np.where(np.abs(denominator) > 1e-12, denominator, 1.0),
        0.0,
    )
    delta = np.clip(delta, -0.5, 0.5)
    dt_rows = np.broadcast_to(np.asarray(dt, dtype=float), (series.shape[0],))
    return (peak + delta) / (n * dt_rows)


def compute_orbit_variables(names, library, successful, potential):
    """Per-orbit grouping variables on the frozen library; return ({name: array}, seconds).

    "lam_z" and "energy" reproduce the audited per-sample -> orbit-mean math
    verbatim (circularity through the ``_circularity_scale`` table of the same
    potential). "omega_z" is the FFT peak frequency of z(t) over each orbit's
    full sampled span, potential-model-free. "jr_phi"/"jz_phi"/"jphi_phi" are
    orbit means of Staeckel actions evaluated per sample in the azimuthal m=0
    average of the true triaxial potential (CylSpline mmax=0; the round-2
    oracle pinned it to the numerical phi-average at 1.5e-4 with exact
    circular-orbit actions). "jz_over_jtot_phi" is the orbit-mean vertical
    fraction mean(Jz) / mean(Jr + Jz + |Jphi|) in that same basis, identical
    to the round-2 exploration's f_z_phi column. Only requested variables
    are computed.
    """

    started = time.perf_counter()
    is_seed = np.isin(library.seed_index, successful)
    seed_samples = library.phase_space[is_seed]
    seed_owner = library.seed_index[is_seed]
    order = np.argsort(successful)
    column = np.searchsorted(successful[order], seed_owner)
    counts = np.bincount(column, minlength=successful.size)
    values: dict[str, np.ndarray] = {}

    if "lam_z" in names or "energy" in names:
        e_circ, lz_circ = _circularity_scale(potential)
        lz = seed_samples[:, 0] * seed_samples[:, 4] - seed_samples[:, 1] * seed_samples[:, 3]
        energy = 0.5 * np.sum(seed_samples[:, 3:] ** 2, axis=1) + np.asarray(
            potential.potential(seed_samples[:, :3]), dtype=float
        )
        energy = np.clip(energy, e_circ[0], e_circ[-1])
        if "energy" in names:
            orbit_energy = np.zeros(successful.size)
            np.add.at(orbit_energy, column, energy)
            values["energy"] = orbit_energy / counts
        if "lam_z" in names:
            lambda_z = lz / np.maximum(np.interp(energy, e_circ, lz_circ), 1e-12)
            orbit_lambda = np.zeros(successful.size)
            np.add.at(orbit_lambda, column, lambda_z)
            values["lam_z"] = orbit_lambda / counts

    if "omega_z" in names:
        if not np.all(counts == counts[0]):
            raise ValueError("omega_z needs an equal number of samples per orbit")
        if not np.all(np.diff(column) >= 0):
            raise ValueError("omega_z needs library samples grouped by seed in seed order")
        samples_per_orbit = int(counts[0])
        z_series = seed_samples[:, 2].reshape(successful.size, samples_per_orbit)
        times_grid = library.time[is_seed].reshape(successful.size, samples_per_orbit)
        dt_rows = np.median(np.diff(times_grid, axis=1), axis=1)
        values["omega_z"] = dominant_frequency(z_series, dt_rows)

    if any(name in names for name in ("jr_phi", "jz_phi", "jphi_phi", "jz_over_jtot_phi")):
        import agama

        phi_potential = agama.Potential(type="CylSpline", density=potential, mmax=0)
        finder = agama.ActionFinder(phi_potential)
        actions = np.asarray(finder(seed_samples), dtype=float)
        action_sums = np.zeros((successful.size, 4))
        np.add.at(action_sums, column, np.column_stack(
            (actions[:, 0], actions[:, 1], actions[:, 2], np.abs(actions[:, 2]))
        ))
        action_means = action_sums / counts[:, None]
        for index, name in enumerate(("jr_phi", "jz_phi", "jphi_phi")):
            if name in names:
                values[name] = action_means[:, index]
        if "jz_over_jtot_phi" in names:
            values["jz_over_jtot_phi"] = action_means[:, 1] / np.maximum(
                action_means[:, 0] + action_means[:, 1] + action_means[:, 3], 1e-12
            )

    missing = [name for name in names if name not in values]
    if missing:
        raise ValueError(f"grouping variables were not computed: {', '.join(missing)}")
    return values, time.perf_counter() - started


def assemble_bundled_solution(response, problem, bundle_weights, assignments, target_normalized, error_normalized, solve_seconds):
    """Build one production ``WeightSolution`` from a bundled NNLS solution.

    Semantics mirror ``solve_density_weights``: ``inner_objective`` is the
    data term F plus the regularization penalty (reported separately in the
    case JSON), ``kkt_residual`` is the normalized reduced-space KKT, and the
    full-space KKT of the original per-orbit problem is returned alongside
    as a diagnostic only. The same ``assignments`` array must be passed here
    that was passed to ``solve_bundled_weights``.
    """

    from halo_mw_lmc.weights import _primal_kkt_residual

    successful = response.successful_seed_index
    active_columns = np.flatnonzero(problem.active_columns)
    seed_weights = map_bundle_weights_to_seeds(
        bundle_weights, assignments, successful, active_columns, response.seed_count,
    )
    active_weights = seed_weights[successful[active_columns]]
    finite_nonnegative = bool(np.all(np.isfinite(active_weights)) and np.all(active_weights >= 0))
    total = float(np.sum(seed_weights))
    squared = float(np.dot(seed_weights, seed_weights))
    density_residual = problem.design @ active_weights - problem.observed
    data_term = float(np.dot(density_residual, density_residual))
    regularization_penalty = problem.regularization * squared
    reduced_problem = _reduced_augmented_problem(problem, assignments)
    reduced_raw, reduced_normalized = _primal_kkt_residual(reduced_problem, bundle_weights)
    from scipy.sparse import csr_matrix
    from halo_mw_lmc.weights import _problem_fingerprint
    reduced_fingerprint = _problem_fingerprint(
        csr_matrix(reduced_problem.design), reduced_problem.observed, 0.0,
    )
    full_raw, full_normalized = _primal_kkt_residual(problem, active_weights)
    converged = bool(
        finite_nonnegative and total > 0 and reduced_normalized <= REDUCED_KKT_TOLERANCE
    )
    solution_kkt = reduced_normalized if np.isfinite(reduced_normalized) else full_raw
    from halo_mw_lmc.weights import WeightSolution

    return WeightSolution(
        seed_weights=seed_weights,
        model_density=response.model_density(seed_weights),
        target_density=target_normalized,
        target_error=error_normalized,
        inner_objective=data_term + regularization_penalty,
        regularization_penalty=regularization_penalty,
        effective_orbit_count=total**2 / squared if squared > 0 else 0.0,
        maximum_weight_fraction=float(np.max(seed_weights)) / total if total > 0 else 0.0,
        active_orbit_count=int(np.count_nonzero(seed_weights > max(float(np.max(seed_weights)) * 1e-12, 0.0))),
        converged=converged,
        status=0 if converged else 1,
        message="dense NNLS on bundled columns" if converged else "bundled NNLS failed reduced-KKT tolerance",
        iterations=0,
        optimality=reduced_raw if np.isfinite(reduced_raw) else full_raw,
        solver_backend="bundled_dense_nnls",
        kkt_residual=solution_kkt,
        solve_wall_seconds=solve_seconds,
        solver_cost=0.5 * (data_term + regularization_penalty),
        problem_fingerprint=problem.fingerprint,
    ), {"reduced_raw": reduced_raw, "reduced_normalized": reduced_normalized, "full_raw": full_raw, "full_normalized": full_normalized, "reduced_fingerprint": reduced_fingerprint}


def save_attempt(directory: Path, name: str, case: dict, arrays: dict) -> None:
    """Write one case directory: case.json + evaluation.npz."""

    case_dir = directory / name
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "case.json").write_text(json.dumps(case, indent=2, default=float) + "\n")
    np.savez(case_dir / "evaluation.npz", **arrays)


def load_attempt(directory: Path, name: str) -> tuple[dict, dict]:
    """Read back one case directory written by ``save_attempt``."""

    case_dir = directory / name
    case = json.loads((case_dir / "case.json").read_text())
    with np.load(case_dir / "evaluation.npz") as payload:
        arrays = {key: payload[key] for key in payload.files}
    return case, arrays


# -----------------------------------------------------------------------
# Stage-2 driver: full / bundled / random on one frozen library.
# -----------------------------------------------------------------------


def main(config_path: str) -> None:
    global REDUCED_KKT_TOLERANCE
    started_total = time.perf_counter()
    config_file = Path(config_path).resolve()
    with open(config_file, "rb") as handle:
        experiment = tomllib.load(handle)
    run_config_path = REPO / experiment["run_config"]
    frozen_path = REPO / experiment["frozen_cache"]
    frozen_provenance_path = REPO / experiment["frozen_provenance"]
    output_dir = REPO / experiment["output_dir"]
    grouping_spec = resolve_grouping_mode(experiment)
    grouping_mode = grouping_spec["mode"]
    random_seed = int(experiment["random_seed"])
    if int(experiment["repeat_runs"]) != 1:
        raise ValueError("this driver implements the single-measurement round only (repeat_runs = 1)")
    REDUCED_KKT_TOLERANCE = float(experiment.get("reduced_kkt_tolerance", REDUCED_KKT_TOLERANCE))

    from scipy.sparse import csr_matrix

    try:
        import agama  # noqa: F401
    except ImportError:
        # AGAMA is vendored at Agama-master/ (exact casing); the audited
        # prototype scripts put it on sys.path the same way.
        sys.path.insert(0, str(REPO / "Agama-master"))

    from halo_mw_lmc.config import load_run_configuration, resolve_model
    from halo_mw_lmc.density import build_orbit_density_response, density_fit_mask
    from halo_mw_lmc.evaluate import (
        _require_external_response_matches_library,
        evaluate_orbit_library,
        score_orbit_weights,
    )
    from halo_mw_lmc.orbits import OrbitLibrary
    from halo_mw_lmc.prepare import prepare_model_data
    from halo_mw_lmc.potential import (
        ZHU_2026_BEST_FIT,
        ZhuHaloParameters,
        build_potential_from_parameters,
    )
    from halo_mw_lmc.weights import _build_weight_problem, _normalized_target

    # ---- input boundary: run config, prepared data, frozen library ----
    run_configuration = load_run_configuration(run_config_path)
    model = resolve_model(run_configuration["recipe"])
    prepared = prepare_model_data(
        run_configuration["data"]["catalog"],
        run_configuration["data"]["target_density"],
        model,
    )
    config = prepared.config

    frozen = np.load(frozen_path)
    library = OrbitLibrary(
        seed_index=frozen["library_seed_index"],
        time=frozen["library_time"],
        phase_space=frozen["library_phase_space"],
    )
    # The frozen cache stores the 4-sector response matrix; the nphi1 response
    # is rebuilt from the same library on the configured grid and then
    # validated against that library, so all three methods share one response.
    started = time.perf_counter()
    response = build_orbit_density_response(
        library, config["density_grid"], seed_count=prepared.initial_conditions.shape[0],
    )
    response_seconds = time.perf_counter() - started
    _require_external_response_matches_library(response, library, prepared)
    print(f"density grid {config['density_grid'].shape}, response {response.matrix.shape} "
          f"nnz={response.matrix.nnz} built in {response_seconds:.2f}s")

    # ---- full case: production entry, outer timer ----
    started = time.perf_counter()
    evaluation_full = evaluate_orbit_library(library, prepared, response=response)
    full_total = time.perf_counter() - started
    full_solution = evaluation_full.weight_solution

    # ---- grouping: per-seed mean lambda_z and energy in the frozen potential ----
    frozen_provenance = json.loads(frozen_provenance_path.read_text())
    successful = response.successful_seed_index
    if grouping_mode in ("variables", "response_kmeans_warmstart"):
        recorded_parameters = frozen_provenance.get("potential_parameters")
        if recorded_parameters:
            grouping_note = "potential parameters taken from the frozen-run provenance record"
            parameters = ZhuHaloParameters(
                rho0=recorded_parameters["rho0"], log_rs=recorded_parameters["log_rs"],
                phalo=recorded_parameters["phalo"], qhalo=recorded_parameters["qhalo"],
                gamma=recorded_parameters["gamma"],
            )
        else:
            grouping_note = (
                "frozen provenance records no potential parameters; grouping uses "
                "ZHU_2026_BEST_FIT with log_rs=log10(70), which differs from the "
                "production analytic anchor log_rs=1.845; valid only for comparison "
                "within this one frozen library"
            )
            parameters = ZhuHaloParameters(
                rho0=ZHU_2026_BEST_FIT["rho0"], log_rs=ZHU_2026_BEST_FIT["log_rs"],
                phalo=ZHU_2026_BEST_FIT["phalo"], qhalo=ZHU_2026_BEST_FIT["qhalo"],
                gamma=ZHU_2026_BEST_FIT["gamma"],
            )
        potential = build_potential_from_parameters(parameters)
        orbit_variables, grouping_seconds = compute_orbit_variables(
            grouping_spec["variables"], library, successful, potential,
        )
        if grouping_mode == "response_kmeans_warmstart":
            grouping_note = (
                "capped Lloyd refinement on the error-normalized density design "
                "columns, warm-started from physical quantile-grid cell centroids "
                "(potential parameters from the frozen-run provenance record); "
                "the velocity objective never enters the grouping"
            )
        print(f"grouping variables {grouping_spec['variables']} in {grouping_seconds:.2f}s "
              f"(n bins {grouping_spec['n_first']}x{grouping_spec['n_second']})")
    else:
        if grouping_mode == "response_kmeans":
            grouping_note = (
                "response k-means grouping on the error-normalized density design "
                "columns (§12 oracle); no orbit variables and no potential are used, "
                "and the velocity objective never enters the grouping"
            )
        elif grouping_mode == "response_kmeans_minibatch":
            grouping_note = (
                "minibatch k-means grouping on the error-normalized density design "
                "columns (§13 cheap oracle); no orbit variables and no potential "
                "are used, and the velocity objective never enters the grouping"
            )
        else:
            grouping_note = (
                "deterministic PCA quantile grid on the error-normalized density "
                "design columns (§13); no orbit variables, no potential, no "
                "randomness, and the velocity objective never enters the grouping"
            )
        orbit_variables = None

    # ---- shared inner problem: identical normalization path as production ----
    fit_keys = {
        key: config["density_fit"][key]
        for key in ("min_abs_z", "min_spherical_radius", "max_spherical_radius", "require_positive_data")
    }
    mask = density_fit_mask(prepared.target_density, prepared.target_error, config["density_grid"], **fit_keys)
    target_normalized, error_normalized = _normalized_target(
        prepared.target_density, prepared.target_error, mask, response,
        config["weight_model"]["target_normalization"],
    )
    regularization = float(config["weight_model"]["regularization_strength"])
    problem = _build_weight_problem(response, target_normalized, error_normalized, mask, regularization)
    if problem.fingerprint != full_solution.problem_fingerprint:
        raise ValueError("bundled inner problem does not match the production full-solve problem")
    design = problem.design
    active_columns = np.flatnonzero(problem.active_columns)

    def run_bundled_case(name, case_assignments):
        started_case = time.perf_counter()
        started_solve = time.perf_counter()
        u = solve_bundled_weights(design, problem.observed, case_assignments, regularization)
        solve_seconds = time.perf_counter() - started_solve
        started_reconstruct = time.perf_counter()
        solution, kkt = assemble_bundled_solution(
            response, problem, u, case_assignments, target_normalized, error_normalized, solve_seconds,
        )
        reconstruct_seconds = time.perf_counter() - started_reconstruct
        started_score = time.perf_counter()
        evaluation = score_orbit_weights(library, prepared, solution, response=response)
        score_seconds = time.perf_counter() - started_score
        case = {
            "method": name,
            "solver_backend": solution.solver_backend,
            "objective_velocity": evaluation.objective_velocity,
            "selected_objective": evaluation.selected_objective,
            "density_chi2_per_bin": evaluation.density_chi2_per_bin,
            "density_gate_passed": bool(evaluation.density_gate_passed),
            "density_scale": float(evaluation.density.scale),
            "weight_sum": evaluation.weight_sum,
            "effective_orbit_count": solution.effective_orbit_count,
            "maximum_weight_fraction": solution.maximum_weight_fraction,
            "weight_concentration": weight_concentration(
                solution.seed_weights[successful[active_columns]], case_assignments,
            ),
            "active_orbit_count": solution.active_orbit_count,
            "bundle_count": int(np.max(case_assignments)) + 1,
            "active_bundles": int(np.count_nonzero(u > 0)),
            "inner_objective": solution.inner_objective,
            "data_term_F": solution.inner_objective - solution.regularization_penalty,
            "regularization_penalty": solution.regularization_penalty,
            "kkt_reduced_raw": kkt["reduced_raw"],
            "kkt_reduced_normalized": kkt["reduced_normalized"],
            "kkt_full_space_raw": kkt["full_raw"],
            "kkt_full_space_normalized": kkt["full_normalized"],
            "kkt_tolerance_reduced": REDUCED_KKT_TOLERANCE,
            "iterations": None,
            "converged": bool(solution.converged),
            "message": solution.message,
            "problem_fingerprint": problem.fingerprint,
            "reduced_problem_fingerprint": kkt["reduced_fingerprint"],
            "assignments_fingerprint": _assignments_fingerprint(case_assignments),
            "timings_seconds": {
                "grouping": grouping_seconds,
                "grouping_shared_across_cases": True,
                "matrix": None,
                "matrix_note": "reduced/augmented matrix construction happens inside solve_bundled_weights and is not separable without changing the frozen core function",
                "solve": solve_seconds,
                "reconstruct": reconstruct_seconds,
                "score": score_seconds,
                "total_outer": time.perf_counter() - started_case,
                "end_to_end_seconds": None,  # filled below with the shared response build
                "end_to_end_note": "end_to_end_seconds = shared response build + shared grouping + per-method total_outer; artifact writing and plotting are excluded",
            },
        }
        arrays = {
            "seed_weights": solution.seed_weights,
            "orbit_weights": response.sample_weights(solution.seed_weights, library),
            "model_density": evaluation.density.raw_model_density,
            "residual": evaluation.density.residual,
            "fit_mask": evaluation.density.fit_mask,
        }
        return case, arrays, evaluation, u

    if grouping_mode == "variables":
        assignments, k_total = quantile_bundle_grid(
            orbit_variables[grouping_spec["variables"][0]][active_columns],
            orbit_variables[grouping_spec["variables"][1]][active_columns],
            grouping_spec["n_first"], grouping_spec["n_second"],
        )
        distortion_bundled = equal_weight_distortion_of(design, assignments)
    elif grouping_mode == "response_kmeans":
        started = time.perf_counter()
        assignments, distortion_bundled = response_kmeans_assignments(
            design, grouping_spec["k"], seed=grouping_spec["seed"],
        )
        grouping_seconds = time.perf_counter() - started
        k_total = int(np.max(assignments)) + 1
        print(f"response k-means grouping k={grouping_spec['k']} in {grouping_seconds:.2f}s "
              f"(D={distortion_bundled['equal_weight_distortion']:.4f}, "
              f"populated={distortion_bundled['populated_bundles']})")
    elif grouping_mode == "response_kmeans_minibatch":
        started = time.perf_counter()
        assignments, grouping_record = minibatch_kmeans_assignments(
            design, grouping_spec["k"], seed=grouping_spec["seed"],
            batch_size=grouping_spec["minibatch_size"], rounds=grouping_spec["minibatch_rounds"],
        )
        grouping_seconds = time.perf_counter() - started
        distortion_bundled = equal_weight_distortion_of(design, assignments)
        k_total = int(np.max(assignments)) + 1
        print(f"minibatch k-means grouping k={grouping_spec['k']} in {grouping_seconds:.2f}s "
              f"(D={distortion_bundled['equal_weight_distortion']:.4f}, "
              f"populated={distortion_bundled['populated_bundles']}, "
              f"batch={grouping_record['minibatch_batch_size']}x{grouping_record['minibatch_rounds']})")
    elif grouping_mode == "response_pca_grid":
        started = time.perf_counter()
        assignments, grouping_record = response_pca_assignments(
            design, grouping_spec["n_first"], grouping_spec["n_second"], grouping_spec["pca_components"],
        )
        grouping_seconds = time.perf_counter() - started
        distortion_bundled = equal_weight_distortion_of(design, assignments)
        k_total = int(np.max(assignments)) + 1
        explained = "/".join(f"{share:.4f}" for share in grouping_record["pca_explained_variance_share"])
        print(f"response PCA grid {grouping_spec['n_first']}x{grouping_spec['n_second']} "
              f"in {grouping_seconds:.2f}s (D={distortion_bundled['equal_weight_distortion']:.4f}, "
              f"populated={distortion_bundled['populated_bundles']}, explained={explained})")
    else:  # response_kmeans_warmstart
        started = time.perf_counter()
        grid_assignments, _grid_total = quantile_bundle_grid(
            orbit_variables[grouping_spec["variables"][0]][active_columns],
            orbit_variables[grouping_spec["variables"][1]][active_columns],
            grouping_spec["n_first"], grouping_spec["n_second"],
        )
        design_dense = np.asarray(design.todense(), dtype=float)
        grid_member = np.bincount(grid_assignments)
        grid_sums = np.zeros((grid_member.size, design_dense.shape[0]))
        np.add.at(grid_sums, grid_assignments, design_dense.T)
        grid_populated = grid_member > 0
        init_centers = grid_sums[grid_populated] / grid_member[grid_populated][:, None]
        assignments, grouping_record = warmstart_kmeans_assignments(
            design, init_centers, grouping_spec["warmstart_iterations"],
        )
        grouping_seconds += time.perf_counter() - started
        distortion_bundled = equal_weight_distortion_of(design, assignments)
        k_total = int(np.max(assignments)) + 1
        print(f"warm-start k-means from {init_centers.shape[0]} grid centroids, "
              f"{grouping_record['warmstart_iterations_used']} Lloyd iterations in "
              f"{grouping_seconds:.2f}s including variables (D={distortion_bundled['equal_weight_distortion']:.4f}, "
              f"populated={distortion_bundled['populated_bundles']})")
    bundled_case, bundled_arrays, evaluation_bundled, u_bundled = run_bundled_case("bundled", assignments)

    rng = np.random.default_rng(random_seed)
    random_assignments = rng.permutation(assignments)
    # A permutation cannot change the value multiset; this assertion only
    # guards a future switch to non-permutation randomization.
    if not np.array_equal(
        np.sort(np.bincount(assignments, minlength=k_total)),
        np.sort(np.bincount(random_assignments, minlength=k_total)),
    ):
        raise ValueError("random membership does not preserve the physical bundle member counts")
    random_case, random_arrays, evaluation_random, u_random = run_bundled_case("random", random_assignments)
    distortion_random = equal_weight_distortion_of(design, random_assignments)
    bundled_case.update(distortion_bundled)
    random_case.update(distortion_random)
    for case in (bundled_case, random_case):
        case["timings_seconds"]["end_to_end_seconds"] = (
            response_seconds + grouping_seconds + case["timings_seconds"]["total_outer"]
        )

    # ---- full case JSON ----
    full_case = {
        "method": "full",
        "solver_backend": full_solution.solver_backend,
        "objective_velocity": evaluation_full.objective_velocity,
        "selected_objective": evaluation_full.selected_objective,
        "density_chi2_per_bin": evaluation_full.density_chi2_per_bin,
        "density_gate_passed": bool(evaluation_full.density_gate_passed),
        "density_scale": float(evaluation_full.density.scale),
        "weight_sum": evaluation_full.weight_sum,
        "effective_orbit_count": full_solution.effective_orbit_count,
        "maximum_weight_fraction": full_solution.maximum_weight_fraction,
        "weight_concentration": weight_concentration(full_solution.seed_weights),
        "active_orbit_count": full_solution.active_orbit_count,
        "inner_objective": full_solution.inner_objective,
        "data_term_F": full_solution.inner_objective - full_solution.regularization_penalty,
        "regularization_penalty": full_solution.regularization_penalty,
        "kkt_full_space_normalized": full_solution.kkt_residual,
        "iterations": int(full_solution.iterations),
        "converged": bool(full_solution.converged),
        "message": full_solution.message,
        "problem_fingerprint": full_solution.problem_fingerprint,
        "timings_seconds": {
            "input_response_build": response_seconds,
            "grouping": None,
            "matrix": None,
            "solve": full_solution.solve_wall_seconds,
            "reconstruct": None,
            "score": None,
            "total_outer": full_total,
            "end_to_end_seconds": response_seconds + full_total,
            "end_to_end_note": "end_to_end_seconds = shared response build + shared grouping (none for full) + per-method total_outer; artifact writing and plotting are excluded",
            "note": "outer timer covers the whole evaluate_orbit_library call; solve is the production solve_density_weights backend timer; other stages are not separable inside the production entry",
        },
    }
    full_arrays = {
        "seed_weights": full_solution.seed_weights,
        "orbit_weights": response.sample_weights(full_solution.seed_weights, library),
        "model_density": evaluation_full.density.raw_model_density,
        "residual": evaluation_full.density.residual,
        "fit_mask": evaluation_full.density.fit_mask,
    }

    # ---- artifacts ----
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = {
        "experiment": experiment,
        "run_config": str(run_config_path),
        "recipe": model["name"] if "name" in model else run_config_path.name,
        "density_grid_shape": list(config["density_grid"].shape),
        "density_fit": config["density_fit"],
        "weight_model": config["weight_model"],
        "objective": config["objective"],
        "include_velocity": config["include_velocity"],
        "grouping_note": grouping_note,
        "reduced_kkt_tolerance": REDUCED_KKT_TOLERANCE,
    }
    (output_dir / "resolved_config.json").write_text(json.dumps(resolved, indent=2, default=float) + "\n")

    import scipy

    try:
        from threadpoolctl import threadpool_info
        threads = threadpool_info()
    except ImportError:
        threads = "threadpoolctl: unavailable"
    git_head = "unavailable"
    git_dirty = None
    try:
        import subprocess
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True)
        if head.returncode == 0:
            git_head = head.stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True)
        if status.returncode == 0:
            git_dirty = bool(status.stdout.strip())
    except OSError:
        pass
    historical_path = REPO / experiment["historical_reference"] if experiment.get("historical_reference") else None
    provenance = {
        "script_sha256": _sha256(Path(__file__).resolve()),
        "config_sha256": _sha256(config_file),
        "frozen_cache_sha256": _sha256(frozen_path),
        "target_sha256": _sha256(prepared.density_path),
        "git_head": git_head,
        "git_dirty": git_dirty,
        "frozen_provenance": frozen_provenance,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "threadpool_info": threads,
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str) + "\n")

    bundle_payload = {
        "assignments": assignments,
        "random_assignments": random_assignments,
        "member_count": np.bincount(assignments, minlength=k_total),
        "random_member_count": np.bincount(random_assignments, minlength=k_total),
        "u_bundled": u_bundled,
        "u_random": u_random,
        "seed_weights_bundled": bundled_arrays["seed_weights"],
        "seed_weights_random": random_arrays["seed_weights"],
        "successful_seed_index": successful,
        "target_normalized": target_normalized,
        "error_normalized": error_normalized,
        "fit_mask": mask,
    }
    if grouping_mode in ("variables", "response_kmeans_warmstart"):
        bundle_payload["orbit_var_first"] = orbit_variables[grouping_spec["variables"][0]]
        bundle_payload["orbit_var_second"] = orbit_variables[grouping_spec["variables"][1]]
    np.savez(output_dir / "bundles.npz", **bundle_payload)
    save_attempt(output_dir, "full", full_case, full_arrays)
    save_attempt(output_dir, "bundled", bundled_case, bundled_arrays)
    save_attempt(output_dir, "random", random_case, random_arrays)

    historical = None
    if historical_path is not None and historical_path.exists():
        historical = {
            "source": str(historical_path),
            "sha256": _sha256(historical_path),
            "reference_only": True,
            "values": json.loads(historical_path.read_text()),
        }
    open_questions = None
    if historical is not None:
        old_full = historical["values"].get("variants", {}).get("full_lsq_linear", {}).get("objective_velocity")
        production_reference = historical["values"].get("production_reference", {}).get("objective")
        open_questions = {
            "full_reproduction_difference": {
                "new_full_minus_historical_full": None if old_full is None else full_case["objective_velocity"] - old_full,
                "new_full_minus_production_reference": None if production_reference is None else full_case["objective_velocity"] - production_reference,
                "cause": "undetermined (thread nondeterminism vs other sources); deferred to plan stage 3 controlled repeats",
            },
        }
    comparison = {
        "methods": {
            "full": {key: full_case[key] for key in (
                "objective_velocity", "density_chi2_per_bin", "density_gate_passed",
                "weight_sum", "effective_orbit_count", "maximum_weight_fraction",
                "weight_concentration", "active_orbit_count", "timings_seconds",
            )},
            "bundled": {key: bundled_case[key] for key in (
                "objective_velocity", "density_chi2_per_bin", "density_gate_passed",
                "weight_sum", "effective_orbit_count", "maximum_weight_fraction",
                "weight_concentration", "active_orbit_count", "timings_seconds",
            )},
            "random": {key: random_case[key] for key in (
                "objective_velocity", "density_chi2_per_bin", "density_gate_passed",
                "weight_sum", "effective_orbit_count", "maximum_weight_fraction",
                "weight_concentration", "active_orbit_count", "timings_seconds",
            )},
        },
        "notes": {
            "bundled_is_restricted": "bundled/random solve the reduced bundle-coadded problem; their convergence and gates refer to that restricted approximation, not the production per-orbit solver",
            "scoring": "all three methods scored through halo_mw_lmc.evaluate.score_orbit_weights on the same response and target",
            "historical_reference": "values from the pre-repair audit are reference-only and enter no computation",
            "timing_definition": "report end-to-end speedup only from end_to_end_seconds (= shared response build + shared grouping + per-method total_outer); total_outer alone omits grouping and overstates the bundled speedup",
        },
        "open_questions": open_questions,
        "historical_reference": historical,
    }
    (output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2, default=float) + "\n")

    for name, case in (("full", full_case), ("bundled", bundled_case), ("random", random_case)):
        solve = case["timings_seconds"]["solve"]
        print(f"{name:8s}: J={case['objective_velocity']:.3f} "
              f"chi2/bin={case['density_chi2_per_bin']:.4f} gate={case['density_gate_passed']} "
              f"solve={solve if solve is not None else float('nan'):.2f}s "
              f"total={case['timings_seconds']['total_outer']:.2f}s")
    print(f"artifacts: {output_dir} (wall {time.perf_counter() - started_total:.1f}s)")


if __name__ == "__main__":
    default_config = REPO / "configs" / "benchmarks" / "nphi1_bundling.toml"
    main(sys.argv[1] if len(sys.argv) > 1 else str(default_config))
