"""Five-dimensional surrogate-profile objective-difference diagnostics.

The optimizer samples are adaptive design points, not posterior samples.  This
module therefore never turns their projected point density into a confidence
region.  Instead, it fits the saved objective values in the full five-
dimensional search space and profiles nuisance coordinates at each displayed
two-dimensional location.  The drawn contours are GP-surrogate-predicted
profiled objective differences (diagnostic levels), not calibrated confidence
intervals, and the nominal ``Delta Q = 2.30`` level is annotated unresolved
when the surrogate cannot resolve it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Mapping, Protocol

import numpy as np


PARAMETER_NAMES = (
    "qhalo",
    "phalo",
    "rho0",
    "rho0_plus_2logrs",
    "gamma",
)
OBJECTIVE_NAMES = ("total", "velocity", "density")

# Candidate diagnostic objective-difference levels.  These are GP-predicted
# profiled objective differences describing objective variation in the sampled
# region, NOT calibrated confidence boundaries.  A panel draws only the levels
# that cross its reliable-region value range, so not all levels are guaranteed
# to be drawn; when the nominal 2.30 level cannot be resolved the panel is
# annotated "unresolved" rather than presenting the innermost visible contour
# as a constraint range.
DIAGNOSTIC_LEVELS = (2.30, 230.0, 2300.0)

# Line width per diagnostic level (thinner for larger objective differences).
_DIAGNOSTIC_LEVEL_LINEWIDTHS = {2.30: 2.5, 230.0: 2.0, 2300.0: 1.5}

SCATTER_COLOR_MAXIMUM = 3.0


class Surrogate(Protocol):
    """Small prediction interface shared by sklearn and test surrogates."""

    def predict(
        self,
        points: np.ndarray,
        *,
        return_std: bool = False,
    ): ...


@dataclass(frozen=True)
class FittedSurrogate:
    """A trained surrogate paired with the objective scale used to fit it.

    The GP is fit on ``(value - min) / scale`` so that ``return_std=True``
    reports uncertainty in units of the objective's training spread.  This lets
    the predictive-standard-deviation mask compare against
    :attr:`ProfileSettings.maximum_predictive_std` (a multiple of that spread)
    instead of the raw, scale-``1e3``+ chi-square axis, which otherwise fails
    the mask on every real-data panel.

    ``scale`` must be passed to :func:`profile_surrogate_surface` as
    ``objective_scale`` so the profiled delta-chi^2 values are restored to
    physical objective units before contouring.
    """

    model: Surrogate
    scale: float

    def predict(
        self,
        points: np.ndarray,
        *,
        return_std: bool = False,
    ):
        return self.model.predict(points, return_std=return_std)


@dataclass(frozen=True)
class ProfileSettings:
    """Numerical controls for deterministic five-dimensional profiling.

    ``maximum_predictive_std`` is a multiple of the objective's training spread
    (the standard deviation of ``objective - min(objective)`` across the fitted
    samples): the GP is fit on values divided by that spread, so a predictive
    standard deviation below this threshold means the surrogate is trustworthy
    relative to the scale of the target.  It is not an absolute chi-square
    tolerance.
    """

    grid_size: int = 60
    sobol_count: int = 256
    local_starts: int = 8
    sobol_seed: int = 0
    local_maxiter: int = 40
    minimum_samples: int = 50
    maximum_training_samples: int = 600
    retained_best_samples: int = 200
    support_quantile: float = 0.95
    maximum_predictive_std: float = 1.15
    maximum_failed_orbit_fraction: float = 0.05

    def validate(self) -> None:
        if self.grid_size < 2:
            raise ValueError("profile grid_size must be at least two")
        if self.sobol_count < 1 or self.sobol_count & (self.sobol_count - 1):
            raise ValueError("sobol_count must be a positive power of two")
        if not 1 <= self.local_starts <= self.sobol_count:
            raise ValueError("local_starts must lie between one and sobol_count")
        if self.local_maxiter < 1:
            raise ValueError("local_maxiter must be positive")
        if self.minimum_samples < 1:
            raise ValueError("minimum_samples must be positive")
        if self.maximum_training_samples < self.minimum_samples:
            raise ValueError(
                "maximum_training_samples cannot be smaller than minimum_samples"
            )
        if not 0.0 < self.support_quantile < 1.0:
            raise ValueError("support_quantile must lie strictly between zero and one")
        if self.maximum_predictive_std <= 0.0:
            raise ValueError("maximum_predictive_std must be positive")
        if not 0.0 < self.maximum_failed_orbit_fraction < 1.0:
            raise ValueError(
                "maximum_failed_orbit_fraction must lie strictly between zero and one"
            )


@dataclass(frozen=True)
class ConstraintSamples:
    """Validated, de-duplicated samples used by the post-processing GPs."""

    display_coordinates: np.ndarray
    display_normalized_coordinates: np.ndarray
    display_objectives: Mapping[str, np.ndarray]
    coordinates: np.ndarray
    normalized_coordinates: np.ndarray
    objectives: Mapping[str, np.ndarray]
    bounds: np.ndarray


@dataclass(frozen=True)
class PanelSpec:
    """One displayed parameter pair and its nuisance-coordinate definition."""

    name: str
    x_label: str
    y_label: str
    nuisance_indices: tuple[int, int, int]


@dataclass(frozen=True)
class ProfileSurface:
    """One profiled surrogate surface and its interpolation-support audit.

    ``delta_chi2`` holds the raw profiled objective values in physical units,
    minimized over the three nuisance coordinates at each pixel.  It is not
    baseline-subtracted here; callers apply a per-objective shared reference
    (see :func:`_baseline_by_shared_reference`) before comparing across panels.
    """

    x: np.ndarray
    y: np.ndarray
    delta_chi2: np.ndarray
    reliable: np.ndarray
    minimizers: np.ndarray
    predictive_std: np.ndarray
    support_distance: np.ndarray


PANELS = (
    PanelSpec(
        name="gamma_rho0",
        x_label=r"Inner slope $\gamma$",
        y_label=r"$\log_{10}(\rho_0/[M_\odot\,\mathrm{kpc}^{-3}])$",
        nuisance_indices=(0, 1, 3),
    ),
    PanelSpec(
        name="rs_rho0",
        x_label=r"$r_s\ [\mathrm{kpc}]$",
        y_label=r"$\log_{10}(\rho_0/[M_\odot\,\mathrm{kpc}^{-3}])$",
        nuisance_indices=(0, 1, 4),
    ),
    PanelSpec(
        name="qhalo_phalo",
        x_label=r"$q_\mathrm{DM}=Z/X$",
        y_label=r"$p_\mathrm{DM}=Y/X$",
        nuisance_indices=(2, 3, 4),
    ),
)


# Display order for the five-parameter corner figure, matching the paper caption.
DISPLAY_ORDER = ("rho0", "rs", "gamma", "qhalo", "phalo")

PARAM_LABELS = {
    "rho0": r"$\log_{10}(\rho_0/[M_\odot\,\mathrm{kpc}^{-3}])$",
    "rs": r"$r_s\ [\mathrm{kpc}]$",
    "gamma": r"Inner slope $\gamma$",
    "qhalo": r"$q_\mathrm{DM}=Z/X$",
    "phalo": r"$p_\mathrm{DM}=Y/X$",
}

# Internal coordinate index for each direct (non-derived) logical parameter.
_INTERNAL_INDEX = {"qhalo": 0, "phalo": 1, "rho0": 2, "gamma": 4}

# Column and row parameter order are identical for the symmetric corner grid.
_CORNER_NUISANCE_ORDER = DISPLAY_ORDER


@dataclass(frozen=True)
class CornerPanel:
    """One displayed parameter pair of the lower-triangular corner figure."""

    name: str
    x_param: str
    y_param: str
    row_index: int
    col_index: int

    @property
    def x_label(self) -> str:
        return PARAM_LABELS[self.x_param]

    @property
    def y_label(self) -> str:
        return PARAM_LABELS[self.y_param]


def _corner_nuisance_params(panel: CornerPanel) -> tuple[str, str, str]:
    """Return the three logical nuisance parameters for a corner panel."""

    nuisance = tuple(
        param for param in _CORNER_NUISANCE_ORDER if param not in (panel.x_param, panel.y_param)
    )
    if len(nuisance) != 3:
        raise ValueError(f"corner panel {panel.name} must leave exactly three nuisances")
    return nuisance


CORNER_PANELS = tuple(
    CornerPanel(
        name=f"corner_{row}_{col}",
        x_param=DISPLAY_ORDER[col],
        y_param=DISPLAY_ORDER[row],
        row_index=row,
        col_index=col,
    )
    for row in range(len(DISPLAY_ORDER))
    for col in range(row)
)


@dataclass(frozen=True)
class CornerSettings(ProfileSettings):
    """Profiling controls for the lower-cost five-parameter corner figure."""

    grid_size: int = 40


def _logical_to_internal_norm(
    values: Mapping[str, float],
    bounds: np.ndarray,
) -> np.ndarray | None:
    """Map logical ``(rho0, rs, gamma, qhalo, phalo)`` to a normalized 5-D point.

    ``rs`` is derived from the persisted ``rho0_plus_2logrs`` slot, so the
    returned internal coordinate 3 is always ``rho0 + 2 log10(rs)``.  Returns
    ``None`` when the point is non-finite, has a non-positive ``rs``, or falls
    outside the search bounds.
    """

    required = ("rho0", "rs", "gamma", "qhalo", "phalo")
    if any(name not in values for name in required):
        return None
    rho0 = float(values["rho0"])
    rs = float(values["rs"])
    gamma = float(values["gamma"])
    qhalo = float(values["qhalo"])
    phalo = float(values["phalo"])
    if not (rs > 0.0 and np.isfinite([rho0, rs, gamma, qhalo, phalo]).all()):
        return None
    coordinates = np.asarray(
        [qhalo, phalo, rho0, rho0 + 2.0 * np.log10(rs), gamma],
        dtype=float,
    )
    normalized = (coordinates - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    if np.any(normalized < -1e-12) or np.any(normalized > 1.0 + 1e-12):
        return None
    return np.clip(normalized, 0.0, 1.0)


def _corner_rs_log_limits(bounds: np.ndarray) -> np.ndarray:
    """Return ``log10(r_s)`` display limits derived from the search bounds."""

    return np.asarray(
        [
            (bounds[3, 0] - bounds[2, 1]) / 2.0,
            (bounds[3, 1] - bounds[2, 0]) / 2.0,
        ]
    )


def _corner_param_limits(param: str, bounds: np.ndarray) -> np.ndarray:
    """Return the display limits for a logical corner parameter."""

    if param == "rs":
        return 10.0**_corner_rs_log_limits(bounds)
    return bounds[_INTERNAL_INDEX[param]]


def _corner_param_from_normalized(
    param: str,
    value: float,
    bounds: np.ndarray,
) -> float:
    """Convert a normalized nuisance in ``[0, 1]`` to display units."""

    if param == "rs":
        log_lo, log_hi = _corner_rs_log_limits(bounds)
        return 10.0 ** (log_lo + float(value) * (log_hi - log_lo))
    index = _INTERNAL_INDEX[param]
    return bounds[index, 0] + float(value) * (bounds[index, 1] - bounds[index, 0])


def scale_radius_kpc(coordinates: np.ndarray) -> np.ndarray:
    """Return ``r_s`` from the persisted ``rho0 + 2 log10(r_s)`` coordinate."""

    values = np.asarray(coordinates, dtype=float)
    if values.shape[-1] != len(PARAMETER_NAMES):
        raise ValueError("parameter coordinates must have a final dimension of five")
    return 10.0 ** ((values[..., 3] - values[..., 2]) / 2.0)


def _required_sample_columns(samples: np.ndarray) -> None:
    names = set(samples.dtype.names or ())
    required = set(PARAMETER_NAMES) | {
        "objective_velocity",
        "objective_density_velocity",
        "chi2",
    }
    missing = sorted(required - names)
    if missing:
        raise ValueError("sample table is missing columns: " + ", ".join(missing))


def _bounds_array(
    bounds: Mapping[str, tuple[float, float] | list[float]],
) -> np.ndarray:
    try:
        result = np.asarray([bounds[name] for name in PARAMETER_NAMES], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("search bounds do not define all five parameters") from exc
    if result.shape != (len(PARAMETER_NAMES), 2):
        raise ValueError("each search bound must contain exactly two endpoints")
    if not np.all(np.isfinite(result)) or np.any(result[:, 0] >= result[:, 1]):
        raise ValueError("search bounds must be finite and strictly increasing")
    return result


def _group_medians(
    coordinates: np.ndarray,
    objectives: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Collapse repeated evaluated coordinates without favoring run order."""

    unique, inverse = np.unique(coordinates, axis=0, return_inverse=True)
    grouped = {
        name: np.asarray(
            [np.median(values[inverse == index]) for index in range(unique.shape[0])],
            dtype=float,
        )
        for name, values in objectives.items()
    }
    return unique, grouped


def deterministic_maximin_indices(
    normalized_coordinates: np.ndarray,
    ranking_values: np.ndarray,
    *,
    maximum: int,
    retain_best: int,
) -> np.ndarray:
    """Keep low-objective trials, then deterministically fill by maximin distance."""

    points = np.asarray(normalized_coordinates, dtype=float)
    ranking = np.asarray(ranking_values, dtype=float)
    if points.ndim != 2 or ranking.shape != (points.shape[0],):
        raise ValueError("maximin coordinates and ranking have incompatible shapes")
    if maximum < 1:
        raise ValueError("maximum must be positive")
    if points.shape[0] <= maximum:
        return np.arange(points.shape[0], dtype=int)

    best_count = min(max(retain_best, 1), maximum)
    order = np.lexsort((np.arange(ranking.size), ranking))
    selected = list(order[:best_count])
    available = np.ones(points.shape[0], dtype=bool)
    available[selected] = False
    distance = np.full(points.shape[0], np.inf)
    for index in selected:
        distance = np.minimum(
            distance,
            np.linalg.norm(points - points[index], axis=1),
        )
    distance[~available] = -np.inf

    while len(selected) < maximum:
        next_index = int(np.argmax(distance))
        selected.append(next_index)
        available[next_index] = False
        distance = np.minimum(
            distance,
            np.linalg.norm(points - points[next_index], axis=1),
        )
        distance[~available] = -np.inf
    return np.asarray(selected, dtype=int)


def _failed_orbit_admission_mask(
    samples: np.ndarray,
    *,
    maximum_fraction: float,
) -> np.ndarray:
    """Return a mask of trials whose orbit-library loss is within budget.

    Orbits that fail to integrate in a trial potential are excluded from the fit
    design matrix before weights are solved, so a small failed fraction does not
    invalidate the density-constrained solve or the velocity likelihood computed
    on the surviving orbits. This gate caps the orbit-library loss (a profiling
    admission rule), not the solver acceptance criterion.

    When ``successful_orbits`` is present the ratio ``failed / successful`` is
    bounded by ``maximum_fraction``; a non-positive denominator leaves the ratio
    undefined, so those rows fall back to the strict zero-failed-orbits rule.
    Old tables without ``successful_orbits`` cannot define a ratio and keep the
    conservative ``failed == 0`` rule.
    """

    names = set(samples.dtype.names or ())
    if "failed_orbits" not in names:
        return np.ones(samples.shape[0], dtype=bool)
    failed = np.asarray(samples["failed_orbits"], dtype=float)
    if "successful_orbits" not in names:
        return failed == 0.0
    successful = np.asarray(samples["successful_orbits"], dtype=float)
    usable = successful > 0.0
    fraction_rule = failed <= maximum_fraction * successful
    return np.where(usable, fraction_rule, failed == 0.0)


def prepare_constraint_samples(
    samples: np.ndarray,
    bounds: Mapping[str, tuple[float, float] | list[float]],
    *,
    settings: ProfileSettings = ProfileSettings(),
) -> ConstraintSamples:
    """Build clean five-dimensional training data from persisted trial rows."""

    settings.validate()
    _required_sample_columns(samples)
    bound_array = _bounds_array(bounds)
    coordinates = np.column_stack(
        [np.asarray(samples[name], dtype=float) for name in PARAMETER_NAMES]
    )
    objectives = {
        "total": 2.0 * np.asarray(samples["objective_density_velocity"], dtype=float),
        "velocity": 2.0 * np.asarray(samples["objective_velocity"], dtype=float),
        "density": np.asarray(samples["chi2"], dtype=float),
    }
    finite = np.all(np.isfinite(coordinates), axis=1)
    for values in objectives.values():
        finite &= np.isfinite(values)
    names = set(samples.dtype.names or ())
    if "weight_solver_converged" in names:
        finite &= np.asarray(samples["weight_solver_converged"], dtype=float) > 0.5
    finite &= _failed_orbit_admission_mask(
        samples,
        maximum_fraction=settings.maximum_failed_orbit_fraction,
    )
    finite &= np.all(coordinates >= bound_array[:, 0], axis=1)
    finite &= np.all(coordinates <= bound_array[:, 1], axis=1)

    coordinates = coordinates[finite]
    objectives = {name: values[finite] for name, values in objectives.items()}
    if coordinates.size == 0:
        raise ValueError("sample table contains no valid five-dimensional trials")
    coordinates, objectives = _group_medians(coordinates, objectives)
    span = bound_array[:, 1] - bound_array[:, 0]
    normalized = (coordinates - bound_array[:, 0]) / span
    display_coordinates = coordinates.copy()
    display_normalized_coordinates = normalized.copy()
    display_objectives = {name: values.copy() for name, values in objectives.items()}
    selected = deterministic_maximin_indices(
        normalized,
        objectives["total"],
        maximum=settings.maximum_training_samples,
        retain_best=settings.retained_best_samples,
    )
    coordinates = coordinates[selected]
    normalized = normalized[selected]
    objectives = {name: values[selected] for name, values in objectives.items()}
    return ConstraintSamples(
        display_coordinates=display_coordinates,
        display_normalized_coordinates=display_normalized_coordinates,
        display_objectives=display_objectives,
        coordinates=coordinates,
        normalized_coordinates=normalized,
        objectives=objectives,
        bounds=bound_array,
    )


def shared_sobol_points(settings: ProfileSettings) -> np.ndarray:
    """Return the one reusable nuisance design used at every displayed pixel."""

    settings.validate()
    try:
        from scipy.stats import qmc
    except ImportError as exc:
        raise RuntimeError("SciPy is required for surrogate profiling") from exc
    sequence = qmc.Sobol(d=3, scramble=True, seed=settings.sobol_seed)
    exponent = int(math.log2(settings.sobol_count))
    return sequence.random_base2(m=exponent)


def _fit_surrogates(data: ConstraintSamples) -> dict[str, FittedSurrogate]:
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is required for five-dimensional GP profiling"
        ) from exc

    fitted: dict[str, FittedSurrogate] = {}
    for name, values in data.objectives.items():
        shifted = values - np.min(values)
        # Fit on ``shifted / scale`` so ``return_std=True`` is expressed in
        # units of the objective training spread.  A constant or non-finite
        # target cannot define a spread; fall back to scale 1.0 and fit the
        # min-shifted target as-is (``normalize_y`` handles the offset).
        scale = float(np.std(shifted))
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        kernel = ConstantKernel(
            constant_value=1.0,
            constant_value_bounds=(1e-3, 1e3),
        ) * Matern(
            length_scale=np.ones(len(PARAMETER_NAMES)),
            length_scale_bounds=(0.03, 10.0),
            nu=2.5,
        )
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=0,
            random_state=0,
        )
        model.fit(data.normalized_coordinates, shifted / scale)
        fitted[name] = FittedSurrogate(model=model, scale=scale)
    return fitted


def _panel_axes(
    panel: PanelSpec | CornerPanel,
    bounds: np.ndarray,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(panel, CornerPanel):
        x_limits = _corner_param_limits(panel.x_param, bounds)
        y_limits = _corner_param_limits(panel.y_param, bounds)
    elif panel.name == "gamma_rho0":
        x_limits = bounds[4]
        y_limits = bounds[2]
    elif panel.name == "rs_rho0":
        log_rs_limits = np.asarray(
            [
                (bounds[3, 0] - bounds[2, 1]) / 2.0,
                (bounds[3, 1] - bounds[2, 0]) / 2.0,
            ]
        )
        x_limits = 10.0**log_rs_limits
        y_limits = bounds[2]
    elif panel.name == "qhalo_phalo":
        x_limits = bounds[0]
        y_limits = bounds[1]
    else:
        raise ValueError(f"unknown parameter-constraint panel: {panel.name}")
    return (
        np.linspace(x_limits[0], x_limits[1], grid_size),
        np.linspace(y_limits[0], y_limits[1], grid_size),
    )


def _normalize_fixed(value: float, index: int, bounds: np.ndarray) -> float:
    return (value - bounds[index, 0]) / (bounds[index, 1] - bounds[index, 0])


def _embed_corner_panel_points(
    panel: CornerPanel,
    x_value: float,
    y_value: float,
    nuisance: np.ndarray,
    bounds: np.ndarray,
) -> np.ndarray | None:
    """Embed one corner panel location into normalized 5-D coordinates.

    The two displayed parameters are fixed; the remaining three logical
    parameters come from the nuisance columns in ``_corner_nuisance_params``
    order.  ``rs`` always contributes through ``rho0_plus_2logrs``, so any
    displayed ``rs`` (or a nuisance ``rs``) is coupled to the chosen ``rho0``.
    """

    nuisance_values = np.atleast_2d(np.asarray(nuisance, dtype=float))
    if nuisance_values.shape[1] != 3:
        raise ValueError("each nuisance point must contain three coordinates")
    nuisance_params = _corner_nuisance_params(panel)
    embedded = np.full(
        (nuisance_values.shape[0], len(PARAMETER_NAMES)),
        np.nan,
        dtype=float,
    )
    for row, nuisance_point in enumerate(nuisance_values):
        values: dict[str, float] = {panel.x_param: x_value, panel.y_param: y_value}
        for param, value in zip(nuisance_params, nuisance_point):
            # Nuisance columns are normalized [0, 1] design points (the Sobol
            # sequence), so convert each to display units before embedding.
            values[param] = _corner_param_from_normalized(param, value, bounds)
        normalized = _logical_to_internal_norm(values, bounds)
        if normalized is not None:
            embedded[row] = normalized
    return embedded


def _embed_panel_points(
    panel: PanelSpec | CornerPanel,
    x_value: float,
    y_value: float,
    nuisance: np.ndarray,
    bounds: np.ndarray,
) -> np.ndarray | None:
    if isinstance(panel, CornerPanel):
        return _embed_corner_panel_points(panel, x_value, y_value, nuisance, bounds)
    nuisance_values = np.atleast_2d(np.asarray(nuisance, dtype=float))
    if nuisance_values.shape[1] != 3:
        raise ValueError("each nuisance point must contain three coordinates")
    points = np.zeros((nuisance_values.shape[0], len(PARAMETER_NAMES)), dtype=float)
    points[:, panel.nuisance_indices] = nuisance_values

    if panel.name == "gamma_rho0":
        points[:, 4] = _normalize_fixed(x_value, 4, bounds)
        points[:, 2] = _normalize_fixed(y_value, 2, bounds)
    elif panel.name == "rs_rho0":
        if x_value <= 0.0:
            return None
        combined = y_value + 2.0 * np.log10(x_value)
        points[:, 2] = _normalize_fixed(y_value, 2, bounds)
        points[:, 3] = _normalize_fixed(combined, 3, bounds)
    elif panel.name == "qhalo_phalo":
        points[:, 0] = _normalize_fixed(x_value, 0, bounds)
        points[:, 1] = _normalize_fixed(y_value, 1, bounds)
    else:
        raise ValueError(f"unknown parameter-constraint panel: {panel.name}")

    if np.any(points < -1e-12) or np.any(points > 1.0 + 1e-12):
        return None
    return np.clip(points, 0.0, 1.0)


def _predict_mean(surrogate: Surrogate, points: np.ndarray) -> np.ndarray:
    return np.asarray(surrogate.predict(points), dtype=float).reshape(-1)


def _bounded_local_minimum(
    surrogate: Surrogate,
    panel: PanelSpec | CornerPanel,
    x_value: float,
    y_value: float,
    starts: np.ndarray,
    bounds: np.ndarray,
    *,
    maxiter: int,
) -> tuple[np.ndarray, float]:
    try:
        from scipy.optimize import minimize
    except ImportError as exc:
        raise RuntimeError("SciPy is required for bounded GP profiling") from exc

    best_nuisance = np.asarray(starts[0], dtype=float)
    initial = _embed_panel_points(panel, x_value, y_value, best_nuisance, bounds)
    if initial is None or not np.all(np.isfinite(initial)):
        raise ValueError("cannot optimize an invalid displayed parameter point")
    best_value = float(_predict_mean(surrogate, initial)[0])

    def objective(nuisance: np.ndarray) -> float:
        embedded = _embed_panel_points(
            panel,
            x_value,
            y_value,
            nuisance,
            bounds,
        )
        if embedded is None or not np.all(np.isfinite(embedded)):
            return np.inf
        return float(_predict_mean(surrogate, embedded)[0])

    for start in starts:
        result = minimize(
            objective,
            np.asarray(start, dtype=float),
            method="L-BFGS-B",
            bounds=[(0.0, 1.0)] * 3,
            options={"maxiter": maxiter, "ftol": 1e-9},
        )
        candidate = np.clip(np.asarray(result.x, dtype=float), 0.0, 1.0)
        value = objective(candidate)
        if np.isfinite(value) and value < best_value:
            best_nuisance = candidate
            best_value = value
    return best_nuisance, best_value


def _support_radius(sample_points: np.ndarray, quantile: float) -> float:
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError("SciPy is required for GP support diagnostics") from exc
    if sample_points.shape[0] < 2:
        return math.inf
    tree = cKDTree(sample_points)
    distances, _ = tree.query(sample_points, k=2)
    return float(np.quantile(distances[:, 1], quantile))


def profile_surrogate_surface(
    surrogate: Surrogate,
    support_points: np.ndarray,
    bounds: np.ndarray,
    panel: PanelSpec | CornerPanel,
    sobol_points: np.ndarray,
    *,
    settings: ProfileSettings,
    objective_scale: float = 1.0,
) -> ProfileSurface:
    """Profile one full-dimensional surrogate over one displayed parameter pair.

    ``objective_scale`` is the spread by which the surrogate target was divided
    before fitting (see :func:`_fit_surrogates`).  The profiled values are
    restored to physical objective units using this factor.

    The returned :class:`ProfileSurface.delta_chi2` holds the **raw profiled
    objective values in physical units** (minimized over the three nuisance
    coordinates at each pixel).  No baseline is subtracted here: a per-objective
    reference minimum is applied by the build callers across the panels of that
    objective (see :func:`_baseline_by_shared_reference`) so that surfaces from
    different panels share one zero point.  The predictive-standard-deviation
    mask stays in the scaled units so it compares against
    :attr:`ProfileSettings.maximum_predictive_std` as a multiple of the target
    spread.  Callers that pass a surrogate returning values in physical units
    (tests) leave ``objective_scale`` at its default of ``1.0``.
    """

    if not (np.isfinite(objective_scale) and objective_scale > 0.0):
        raise ValueError("objective_scale must be finite and positive")
    settings.validate()
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError("SciPy is required for GP support diagnostics") from exc
    support = np.asarray(support_points, dtype=float)
    if support.ndim != 2 or support.shape[1] != len(PARAMETER_NAMES):
        raise ValueError("support points must have shape (n, 5)")
    sobol = np.asarray(sobol_points, dtype=float)
    if sobol.shape != (settings.sobol_count, 3):
        raise ValueError("Sobol nuisance design has an unexpected shape")

    x, y = _panel_axes(panel, bounds, settings.grid_size)
    values = np.full((y.size, x.size), np.nan)
    standard_deviation = np.full_like(values, np.nan)
    minimizers = np.full((y.size, x.size, len(PARAMETER_NAMES)), np.nan)

    for y_index, y_value in enumerate(y):
        for x_index, x_value in enumerate(x):
            candidates = _embed_panel_points(
                panel,
                x_value,
                y_value,
                sobol,
                bounds,
            )
            if candidates is None:
                continue
            # ``rs``-derived corner panels can produce out-of-domain nuisance
            # points; drop them while preserving the sobol row correspondence
            # so the local-minimum starts map back to real nuisance columns.
            valid_rows = np.flatnonzero(np.all(np.isfinite(candidates), axis=1))
            if valid_rows.size == 0:
                continue
            candidate_values = _predict_mean(surrogate, candidates[valid_rows])
            finite = np.flatnonzero(np.isfinite(candidate_values))
            if finite.size == 0:
                continue
            count = min(settings.local_starts, finite.size)
            start_rows = valid_rows[finite]
            ordering = start_rows[
                np.argsort(candidate_values[finite], kind="stable")[:count]
            ]
            nuisance, value = _bounded_local_minimum(
                surrogate,
                panel,
                x_value,
                y_value,
                sobol[ordering],
                bounds,
                maxiter=settings.local_maxiter,
            )
            best_point = _embed_panel_points(
                panel,
                x_value,
                y_value,
                nuisance,
                bounds,
            )
            if best_point is None or not np.all(np.isfinite(best_point)):
                continue
            prediction, std = surrogate.predict(best_point, return_std=True)
            values[y_index, x_index] = min(value, float(np.asarray(prediction)[0]))
            standard_deviation[y_index, x_index] = float(np.asarray(std)[0])
            minimizers[y_index, x_index] = best_point[0]

    finite_values = np.isfinite(values)
    tree = cKDTree(support)
    support_distance = np.full_like(values, np.nan)
    valid_minimizers = np.all(np.isfinite(minimizers), axis=2)
    if np.any(valid_minimizers):
        distance, _ = tree.query(minimizers[valid_minimizers], k=1)
        support_distance[valid_minimizers] = distance
    radius = _support_radius(support, settings.support_quantile)
    reliable = (
        finite_values
        & (support_distance <= radius)
        & (standard_deviation <= settings.maximum_predictive_std)
    )
    # Restore physical objective units for the raw profiled surface; the
    # predictive-standard-deviation mask above stays in scaled units.
    if objective_scale != 1.0:
        values = values * objective_scale
    return ProfileSurface(
        x=x,
        y=y,
        delta_chi2=values,
        reliable=reliable,
        minimizers=minimizers,
        predictive_std=standard_deviation,
        support_distance=support_distance,
    )


def _logical_from_internal(coordinates: np.ndarray) -> dict[str, np.ndarray]:
    """Expand internal 5-D coordinates into logical parameter arrays."""

    values = np.asarray(coordinates, dtype=float)
    if values.ndim < 2 or values.shape[-1] != len(PARAMETER_NAMES):
        raise ValueError("internal coordinates must have a final dimension of five")
    return {
        "rho0": values[..., 2],
        "rs": scale_radius_kpc(values),
        "gamma": values[..., 4],
        "qhalo": values[..., 0],
        "phalo": values[..., 1],
    }


def _panel_sample_coordinates(
    panel: PanelSpec | CornerPanel,
    coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(panel, CornerPanel):
        logical = _logical_from_internal(coordinates)
        return logical[panel.x_param], logical[panel.y_param]
    if panel.name == "gamma_rho0":
        return coordinates[:, 4], coordinates[:, 2]
    if panel.name == "rs_rho0":
        return scale_radius_kpc(coordinates), coordinates[:, 2]
    if panel.name == "qhalo_phalo":
        return coordinates[:, 0], coordinates[:, 1]
    raise ValueError(f"unknown parameter-constraint panel: {panel.name}")


def _baseline_by_shared_reference(
    surfaces: Mapping[str, Mapping[str, ProfileSurface]],
) -> Mapping[str, Mapping[str, ProfileSurface]]:
    """Apply one per-objective, support-checked reference minimum across panels.

    Raw profiled values in each :class:`ProfileSurface.delta_chi2` are in
    physical objective units, expressed relative to each objective's training
    minimum; the profile routine subtracts no baseline of its own.  To keep the
    displayed objective differences comparable across panels, this subtracts a
    single reference per objective: the minimum of the raw profiled values over
    every reliable pixel of every panel that reports that objective.  If an
    objective has no reliable pixel in any panel, its surfaces are left
    unchanged.

    The returned mapping holds new :class:`ProfileSurface` instances (the
    frozen dataclass is not mutated); the input mapping is unchanged.
    """

    references: dict[str, float] = {}
    for by_objective in surfaces.values():
        for objective, surface in by_objective.items():
            reliable_values = surface.delta_chi2[surface.reliable]
            if reliable_values.size:
                current = references.get(objective)
                candidate = float(np.nanmin(reliable_values))
                if current is None or candidate < current:
                    references[objective] = candidate

    baselined: dict[str, dict[str, ProfileSurface]] = {}
    for panel_name, by_objective in surfaces.items():
        baselined[panel_name] = {}
        for objective, surface in by_objective.items():
            reference = references.get(objective)
            if reference is None:
                baselined[panel_name][objective] = surface
                continue
            delta = surface.delta_chi2 - reference
            baselined[panel_name][objective] = replace(
                surface,
                delta_chi2=delta,
            )
    return baselined


def _cleaned_reliable(reliable: np.ndarray) -> np.ndarray:
    """Drop isolated reliable pixels via an 8-neighbour count filter.

    A reliable pixel is kept only when at least five of its eight neighbours are
    also reliable.  This removes isolated line/pixel fragments from the contours
    so the diagnostic levels read more clearly.  It is purely a presentation
    cleanup: it does **not** validate the surrogate or change which pixels are
    considered statistically supported.
    """

    reliable = np.asarray(reliable, dtype=bool)
    if reliable.ndim != 2:
        raise ValueError("reliable mask must be two-dimensional")
    try:
        from scipy.ndimage import convolve
    except ImportError:
        convolve = None
    neighbors = np.zeros(reliable.shape, dtype=float)
    if convolve is not None:
        kernel = np.ones((3, 3), dtype=float)
        kernel[1, 1] = 0.0
        neighbors = convolve(
            reliable.astype(float),
            kernel,
            mode="constant",
            cval=0.0,
        )
    else:
        padded = np.pad(reliable.astype(float), 1, mode="constant", constant_values=0.0)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                neighbors += padded[1 + di : 1 + di + reliable.shape[0], 1 + dj : 1 + dj + reliable.shape[1]]
    return reliable & (neighbors >= 5.0)


def _draw_profile_contour(
    axis,
    surface: ProfileSurface,
    *,
    color: str,
    linestyle: str,
) -> dict[float, bool]:
    """Draw the diagnostic-level contours for one reliable panel surface.

    ``surface.delta_chi2`` is assumed to already be baseline-subtracted against
    the per-objective shared reference (see :func:`_baseline_by_shared_reference`)
    so the value range is comparable across panels.  For each level in
    :data:`DIAGNOSTIC_LEVELS`, the contour is drawn only when the level lies
    strictly inside the cleaned reliable-region value range; otherwise it is not
    drawn.  Returns a dict mapping each candidate level to whether it was drawn
    in this panel.  Contours are drawn on the neighbour-cleaned reliable mask.
    """

    clean = _cleaned_reliable(surface.reliable)
    masked = np.ma.masked_where(~clean, surface.delta_chi2)
    finite = masked.compressed()
    drawn: dict[float, bool] = {}
    if finite.size == 0:
        return {level: False for level in DIAGNOSTIC_LEVELS}
    low = float(np.min(finite))
    high = float(np.max(finite))
    for level in DIAGNOSTIC_LEVELS:
        if low < level < high:
            axis.contour(
                surface.x,
                surface.y,
                masked,
                levels=[level],
                colors=[color],
                linestyles=[linestyle],
                linewidths=_DIAGNOSTIC_LEVEL_LINEWIDTHS.get(
                    level, _DIAGNOSTIC_LEVEL_LINEWIDTHS[DIAGNOSTIC_LEVELS[0]]
                ),
            )
            drawn[level] = True
        else:
            drawn[level] = False
    return drawn


def _contour_touches_boundary(surface: ProfileSurface) -> bool:
    """Return whether the cleaned reliable region touches the panel boundary.

    Contours drawn on a reliable region that reaches the panel edge are cut off
    by the support boundary and must be labelled truncated rather than closed.
    """

    clean = _cleaned_reliable(surface.reliable)
    if clean.shape[0] < 3 or clean.shape[1] < 3:
        return bool(np.any(clean))
    return bool(
        np.any(clean[0, :])
        or np.any(clean[-1, :])
        or np.any(clean[:, 0])
        or np.any(clean[:, -1])
    )


def _format_level(level: float) -> str:
    """Render a diagnostic level for a legend/label without trailing zeros."""

    if level == int(level):
        return str(int(level))
    return f"{level:.2f}"


def _draw_panel_annotations(
    axis,
    drawn: dict[float, bool],
    *,
    truncated: bool,
) -> None:
    """Annotate unresolved 2.30 and support-boundary truncation for a panel."""

    if not drawn.get(2.30, False):
        axis.text(
            0.03,
            0.97,
            "2.30 unresolved",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            color="0.45",
        )
    if truncated:
        axis.text(
            0.97,
            0.03,
            "truncated",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color="0.55",
        )


def _fallback_reason(data: ConstraintSamples, settings: ProfileSettings) -> str | None:
    if data.coordinates.shape[0] < settings.minimum_samples:
        return f"GP profile unavailable: {data.coordinates.shape[0]} valid unique trials; at least {settings.minimum_samples} required"
    variation = np.ptp(data.normalized_coordinates, axis=0)
    constant = [name for name, span in zip(PARAMETER_NAMES, variation) if span < 1e-8]
    if constant:
        return "GP profile unavailable: insufficient variation in " + ", ".join(constant)
    return None


def build_parameter_constraints_figure(
    samples: np.ndarray,
    bounds: Mapping[str, tuple[float, float] | list[float]],
    *,
    settings: ProfileSettings = ProfileSettings(),
):
    """Return the three-panel constraint figure from saved optimizer samples.

    GP fitting/profiling failures deliberately degrade to a labelled scatter-only
    figure.  Static report generation must remain usable for short engineering
    benchmarks and older runs.
    """

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    settings.validate()
    try:
        data = prepare_constraint_samples(samples, bounds, settings=settings)
    except Exception as exc:
        figure, axes = plt.subplots(
            1,
            3,
            figsize=(14.0, 4.8),
            constrained_layout=True,
        )
        for axis, panel in zip(axes, PANELS):
            axis.set_xlabel(panel.x_label)
            axis.set_ylabel(panel.y_label)
            axis.grid(alpha=0.15)
        figure.text(
            0.5,
            0.99,
            f"parameter constraints unavailable: {type(exc).__name__}: {exc}",
            ha="center",
            va="top",
            fontsize=9,
            color="0.3",
        )
        figure.suptitle(
            "Five-dimensional GP profile constraints",
            y=1.08,
        )
        return figure
    failure = _fallback_reason(data, settings)
    surfaces: dict[str, dict[str, ProfileSurface]] = {}
    if failure is None:
        try:
            surrogates = _fit_surrogates(data)
            sobol = shared_sobol_points(settings)
            for panel in PANELS:
                surfaces[panel.name] = {
                    objective: profile_surrogate_surface(
                        surrogates[objective],
                        data.display_normalized_coordinates,
                        data.bounds,
                        panel,
                        sobol,
                        settings=settings,
                        objective_scale=surrogates[objective].scale,
                    )
                    for objective in OBJECTIVE_NAMES
                }
            # Apply one per-objective shared, support-checked reference minimum
            # so the diagnostic deltas are comparable across panels.
            surfaces = _baseline_by_shared_reference(surfaces)
        except Exception as exc:  # reporting fallback is intentional
            failure = f"GP profile unavailable: {type(exc).__name__}: {exc}"
            surfaces = {}

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(14.0, 4.8),
        constrained_layout=True,
    )
    total_delta = data.display_objectives["total"] - np.min(
        data.display_objectives["total"]
    )
    _objective_colors = {"total": "#9b0000", "velocity": "black", "density": "#5573b7"}
    _objective_styles = {"total": "solid", "velocity": "dashed", "density": "dashed"}
    _drawn_levels: dict[str, set[float]] = {objective: set() for objective in OBJECTIVE_NAMES}
    scatter = None
    for axis, panel in zip(axes, PANELS):
        sample_x, sample_y = _panel_sample_coordinates(panel, data.display_coordinates)
        scatter = axis.scatter(
            sample_x,
            sample_y,
            c=np.clip(total_delta, 0.0, SCATTER_COLOR_MAXIMUM),
            vmin=0.0,
            vmax=SCATTER_COLOR_MAXIMUM,
            cmap="Spectral",
            s=16,
            alpha=0.72,
            linewidths=0.0,
            rasterized=True,
        )
        truncated = False
        panel_drawn: set[float] = set()
        for objective in OBJECTIVE_NAMES:
            if panel.name not in surfaces or objective not in surfaces[panel.name]:
                continue
            drawn = _draw_profile_contour(
                axis,
                surfaces[panel.name][objective],
                color=_objective_colors[objective],
                linestyle=_objective_styles[objective],
            )
            for level, is_drawn in drawn.items():
                if is_drawn:
                    _drawn_levels[objective].add(level)
                    panel_drawn.add(level)
            truncated |= _contour_touches_boundary(surfaces[panel.name][objective])
        if panel.name in surfaces:
            _draw_panel_annotations(
                axis,
                {level: level in panel_drawn for level in DIAGNOSTIC_LEVELS},
                truncated=truncated,
            )
        if panel.name == "qhalo_phalo":
            lower = max(data.bounds[0, 0], data.bounds[1, 0])
            upper = min(data.bounds[0, 1], data.bounds[1, 1])
            axis.plot([lower, upper], [lower, upper], "k:", linewidth=1.0)
        axis.set_xlabel(panel.x_label)
        axis.set_ylabel(panel.y_label)
        axis.grid(alpha=0.15)

    if scatter is not None:
        colorbar = figure.colorbar(
            scatter,
            ax=axes,
            location="top",
            shrink=0.26,
            pad=0.02,
            extend="max",
        )
        colorbar.set_label(r"actual trial $\Delta\chi^2_\mathrm{tot}$")
    legend_handles: list[Line2D] = []
    if any(_drawn_levels.values()):
        for objective in OBJECTIVE_NAMES:
            for level in sorted(_drawn_levels[objective]):
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=_objective_colors[objective],
                        linewidth=_DIAGNOSTIC_LEVEL_LINEWIDTHS.get(level, 2.0),
                        linestyle=_objective_styles[objective],
                        label=(
                            f"{objective} $\\Delta Q$ = {_format_level(level)}"
                        ),
                    )
                )
    if failure is None:
        if legend_handles:
            figure.legend(
                handles=legend_handles,
                loc="upper center",
                ncol=min(3, len(legend_handles)),
                bbox_to_anchor=(0.67, 1.01),
            )
        else:
            figure.text(
                0.67,
                0.99,
                "no level resolved",
                ha="center",
                va="top",
                fontsize=8,
                color="0.45",
            )
        figure.text(
            0.5,
            0.005,
            "Diagnostic GP-surrogate objective differences with diagnostic levels; "
            "contours clip at the support boundary and are truncated there. "
            "Not calibrated confidence intervals.",
            ha="center",
            va="bottom",
            fontsize=8,
            color="0.35",
        )
    else:
        figure.text(
            0.5,
            0.99,
            failure,
            ha="center",
            va="top",
            fontsize=9,
            color="0.3",
        )
    figure.suptitle(
        "Five-dimensional GP profiled objective-difference diagnostics "
        "(adaptive trials shown as points)",
        y=1.08,
    )
    return figure


def _empty_corner_figure(message: str):
    """Return a labelled 5x5 scatter-only corner figure for degraded runs."""

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        5,
        5,
        figsize=(16.0, 15.0),
        constrained_layout=True,
    )
    for row in range(len(DISPLAY_ORDER)):
        for column in range(len(DISPLAY_ORDER)):
            axis = axes[row, column]
            if column > row:
                axis.set_axis_off()
            elif column == row:
                axis.text(
                    0.5,
                    0.5,
                    PARAM_LABELS[DISPLAY_ORDER[row]],
                    ha="center",
                    va="center",
                    fontsize=12,
                )
                axis.set_axis_off()
    for panel in CORNER_PANELS:
        axis = axes[panel.row_index, panel.col_index]
        if panel.col_index == 0:
            axis.set_ylabel(panel.y_label)
        if panel.row_index == len(DISPLAY_ORDER) - 1:
            axis.set_xlabel(panel.x_label)
        axis.grid(alpha=0.15)
    figure.text(
        0.5,
        1.02,
        f"parameter constraints unavailable: {message}",
        ha="center",
        va="top",
        fontsize=9,
        color="0.3",
    )
    figure.suptitle("Five-dimensional GP profiled objective-difference diagnostics", y=1.08)
    return figure


def persist_corner_surfaces(
    surfaces: Mapping[str, Mapping[str, ProfileSurface]],
    path,
) -> None:
    """Write profiled corner surfaces to a ``.npz`` for report re-use.

    ``surfaces`` maps panel name to a mapping of objective name to
    :class:`ProfileSurface`.  Stored arrays are ``x``, ``y``, ``delta_chi2``,
    ``reliable``, ``predictive_std`` (units of the objective training spread,
    see :func:`profile_surrogate_surface`), and ``support_distance``
    (normalized search space) for each panel and objective, keyed
    ``<panel>__<objective>__<field>``, together with the panel names and
    objective names used.

    ``delta_chi2`` is stored in physical objective units.  When the surfaces
    come from ``build_parameter_constraints_corner_figure`` (the production
    path) the per-objective shared reference has already been subtracted, so
    the reliable global minimum is zero and re-applying
    :func:`_baseline_by_shared_reference` is a no-op; callers may also persist
    raw pre-baseline surfaces, in which case the cross-panel shared reference
    must be applied before treating differences across panels as comparable.
    """

    import numpy as np

    panel_names: list[str] = []
    objective_names: list[str] = []
    arrays: dict[str, np.ndarray] = {}
    for panel_name, by_objective in surfaces.items():
        for objective, surface in by_objective.items():
            panel_names.append(panel_name)
            objective_names.append(objective)
            base = f"{panel_name}__{objective}"
            arrays[f"{base}__x"] = surface.x
            arrays[f"{base}__y"] = surface.y
            arrays[f"{base}__delta_chi2"] = surface.delta_chi2
            arrays[f"{base}__reliable"] = surface.reliable
            arrays[f"{base}__predictive_std"] = surface.predictive_std
            arrays[f"{base}__support_distance"] = surface.support_distance
    arrays.setdefault("panel_names", np.asarray(panel_names, dtype=str))
    arrays.setdefault("objective_names", np.asarray(objective_names, dtype=str))
    np.savez(path, **arrays)


def build_parameter_constraints_corner_figure(
    samples: np.ndarray,
    bounds: Mapping[str, tuple[float, float] | list[float]],
    *,
    settings: ProfileSettings | None = None,
    return_artifacts: bool = False,
):
    """Return the lower-triangular five-parameter profiled objective diagnostics.

    Each off-diagonal panel shows the diagnostic ``Delta Q`` contours for the
    total objective (solid) and the density objective (dashed) on the GP
    surrogate, clipped to the neighbour-cleaned trial-support and GP-uncertainty
    mask, with the actual adaptive trials overlaid as small points.  A degraded,
    labelled scatter-only figure is returned instead of raising.

    The contours are GP-surrogate-predicted profiled objective differences, not
    calibrated confidence intervals; each panel draws only the candidate levels
    that cross its reliable-region value range, and an unresolved 2.30 (or a
    support-boundary-truncated contour) is annotated rather than presented as a
    constraint range.

    When ``return_artifacts`` is true, return ``(figure, surfaces)`` where
    ``surfaces`` maps panel name to objective name to a
    :class:`ProfileSurface`, allowing the caller to persist them without
    recomputing the surrogate profiling.
    """

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    corner_settings = settings if settings is not None else CornerSettings()
    corner_settings.validate()
    try:
        data = prepare_constraint_samples(samples, bounds, settings=corner_settings)
    except Exception as exc:
        figure = _empty_corner_figure(f"{type(exc).__name__}: {exc}")
        if return_artifacts:
            return figure, {}
        return figure

    failure = _fallback_reason(data, corner_settings)
    surfaces: dict[str, dict[str, ProfileSurface]] = {}
    if failure is None:
        try:
            surrogates = _fit_surrogates(data)
            sobol = shared_sobol_points(corner_settings)
            for panel in CORNER_PANELS:
                surfaces[panel.name] = {
                    objective: profile_surrogate_surface(
                        surrogates[objective],
                        data.display_normalized_coordinates,
                        data.bounds,
                        panel,
                        sobol,
                        settings=corner_settings,
                        objective_scale=surrogates[objective].scale,
                    )
                    for objective in ("total", "density")
                }
            # One per-objective shared, support-checked reference minimum so the
            # diagnostic deltas are comparable across the off-diagonal panels.
            surfaces = _baseline_by_shared_reference(surfaces)
        except Exception as exc:
            failure = f"GP profile unavailable: {type(exc).__name__}: {exc}"
            surfaces = {}

    figure, axes = plt.subplots(
        5,
        5,
        figsize=(16.0, 15.0),
        sharex="col",
        sharey="row",
        constrained_layout=True,
    )
    total_delta = data.display_objectives["total"] - np.min(
        data.display_objectives["total"]
    )
    _objective_colors = {"total": "#9b0000", "density": "#5573b7"}
    _objective_styles = {"total": "solid", "density": "dashed"}
    _drawn_levels: dict[str, set[float]] = {"total": set(), "density": set()}
    scatter = None
    for row in range(len(DISPLAY_ORDER)):
        for column in range(row + 1, len(DISPLAY_ORDER)):
            axes[row, column].set_axis_off()
    for panel in CORNER_PANELS:
        axis = axes[panel.row_index, panel.col_index]
        sample_x, sample_y = _panel_sample_coordinates(
            panel,
            data.display_coordinates,
        )
        scatter = axis.scatter(
            sample_x,
            sample_y,
            c=np.clip(total_delta, 0.0, SCATTER_COLOR_MAXIMUM),
            vmin=0.0,
            vmax=SCATTER_COLOR_MAXIMUM,
            cmap="Spectral",
            s=10,
            alpha=0.3,
            linewidths=0.0,
            rasterized=True,
        )
        truncated = False
        panel_drawn: set[float] = set()
        for objective in ("total", "density"):
            if panel.name not in surfaces or objective not in surfaces[panel.name]:
                continue
            drawn = _draw_profile_contour(
                axis,
                surfaces[panel.name][objective],
                color=_objective_colors[objective],
                linestyle=_objective_styles[objective],
            )
            for level, is_drawn in drawn.items():
                if is_drawn:
                    _drawn_levels[objective].add(level)
                    panel_drawn.add(level)
            truncated |= _contour_touches_boundary(surfaces[panel.name][objective])
        if panel.name in surfaces:
            _draw_panel_annotations(
                axis,
                {level: level in panel_drawn for level in DIAGNOSTIC_LEVELS},
                truncated=truncated,
            )
        if {panel.x_param, panel.y_param} == {"qhalo", "phalo"}:
            lower = max(data.bounds[0, 0], data.bounds[1, 0])
            upper = min(data.bounds[0, 1], data.bounds[1, 1])
            axis.plot([lower, upper], [lower, upper], "k:", linewidth=1.0)
        if panel.col_index == 0:
            axis.set_ylabel(panel.y_label)
        if panel.row_index == len(DISPLAY_ORDER) - 1:
            axis.set_xlabel(panel.x_label)
        axis.grid(alpha=0.15)

    for index, axis in enumerate(np.diag(axes)):
        axis.text(
            0.5,
            0.5,
            PARAM_LABELS[DISPLAY_ORDER[index]],
            ha="center",
            va="center",
            fontsize=12,
        )
        axis.set_axis_off()

    if scatter is not None:
        colorbar = figure.colorbar(
            scatter,
            ax=axes,
            location="top",
            shrink=0.22,
            pad=0.02,
            extend="max",
        )
        colorbar.set_label(r"actual trial $\Delta\chi^2_\mathrm{tot}$")
    legend_handles: list[Line2D] = []
    if any(_drawn_levels.values()):
        for objective in ("total", "density"):
            for level in sorted(_drawn_levels[objective]):
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=_objective_colors[objective],
                        linewidth=_DIAGNOSTIC_LEVEL_LINEWIDTHS.get(level, 2.0),
                        linestyle=_objective_styles[objective],
                        label=(
                            f"{objective} $\\Delta Q$ = {_format_level(level)}"
                        ),
                    )
                )
    if failure is None:
        if legend_handles:
            figure.legend(
                handles=legend_handles,
                loc="upper left",
                ncol=1,
                bbox_to_anchor=(0.01, 1.01),
                frameon=True,
            )
        else:
            figure.text(
                0.01,
                0.99,
                "no level resolved",
                ha="left",
                va="top",
                fontsize=8,
                color="0.45",
            )
        figure.suptitle(
            "Five-dimensional GP profiled objective-difference diagnostics "
            "(corner view; adaptive trials shown as points)",
            y=1.08,
        )
        figure.text(
            0.5,
            1.035,
            "Contours show GP-surrogate-predicted profiled objective differences "
            "within the sampled region, describing objective variation and possible "
            "parameter-degeneracy directions. The current evidence does not reliably "
            "resolve the $\\Delta Q = 2.30$ level; the shown contours are not "
            "calibrated confidence intervals, and no inference is made beyond the "
            "support boundary.",
            ha="center",
            va="bottom",
            fontsize=8,
            color="0.35",
        )
    else:
        figure.text(
            0.5,
            1.02,
            failure,
            ha="center",
            va="top",
            fontsize=9,
            color="0.3",
        )
        figure.suptitle(
            "Five-dimensional GP profiled objective-difference diagnostics "
            "(corner view; adaptive trials shown as points)",
            y=1.08,
        )
    if return_artifacts:
        return figure, surfaces
    return figure


def search_bounds_from_resolved_config(
    document: Mapping[str, object],
) -> dict[str, tuple[float, float]]:
    """Read search bounds from the persisted resolved-run configuration."""

    try:
        optimizer = document["optimizer"]
        if not isinstance(optimizer, Mapping):
            raise TypeError
        raw_bounds = optimizer["bounds"]
        if not isinstance(raw_bounds, Mapping):
            raise TypeError
        bounds = _bounds_array(raw_bounds)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("resolved configuration has no valid optimizer bounds") from exc
    return {
        name: (float(bounds[index, 0]), float(bounds[index, 1]))
        for index, name in enumerate(PARAMETER_NAMES)
    }
