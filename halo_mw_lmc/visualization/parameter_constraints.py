"""Five-dimensional surrogate-profile parameter-constraint figures.

The optimizer samples are adaptive design points, not posterior samples.  This
module therefore never turns their projected point density into a confidence
region.  Instead, it fits the saved objective values in the full five-
dimensional search space and profiles nuisance coordinates at each displayed
two-dimensional location.
"""

from __future__ import annotations

from dataclasses import dataclass
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
CONTOUR_LEVEL = 2.30
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
class ProfileSettings:
    """Numerical controls for deterministic five-dimensional profiling."""

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
    """One profiled surrogate surface and its interpolation-support audit."""

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
    if "failed_orbits" in names:
        finite &= np.asarray(samples["failed_orbits"], dtype=float) == 0.0
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


def _fit_surrogates(data: ConstraintSamples) -> dict[str, Surrogate]:
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is required for five-dimensional GP profiling"
        ) from exc

    fitted: dict[str, Surrogate] = {}
    for name, values in data.objectives.items():
        shifted = values - np.min(values)
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
        model.fit(data.normalized_coordinates, shifted)
        fitted[name] = model
    return fitted


def _panel_axes(
    panel: PanelSpec,
    bounds: np.ndarray,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if panel.name == "gamma_rho0":
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


def _embed_panel_points(
    panel: PanelSpec,
    x_value: float,
    y_value: float,
    nuisance: np.ndarray,
    bounds: np.ndarray,
) -> np.ndarray | None:
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
    panel: PanelSpec,
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
    if initial is None:
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
        if embedded is None:
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
    panel: PanelSpec,
    sobol_points: np.ndarray,
    *,
    settings: ProfileSettings,
) -> ProfileSurface:
    """Profile one full-dimensional surrogate over one displayed parameter pair."""

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
            candidate_values = _predict_mean(surrogate, candidates)
            finite = np.flatnonzero(np.isfinite(candidate_values))
            if finite.size == 0:
                continue
            count = min(settings.local_starts, finite.size)
            ordering = finite[np.argsort(candidate_values[finite], kind="stable")[:count]]
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
            if best_point is None:
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
    delta = np.full_like(values, np.nan)
    baseline_mask = reliable if np.any(reliable) else finite_values
    if np.any(baseline_mask):
        delta[finite_values] = values[finite_values] - np.min(values[baseline_mask])
    return ProfileSurface(
        x=x,
        y=y,
        delta_chi2=delta,
        reliable=reliable,
        minimizers=minimizers,
        predictive_std=standard_deviation,
        support_distance=support_distance,
    )


def _panel_sample_coordinates(
    panel: PanelSpec,
    coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if panel.name == "gamma_rho0":
        return coordinates[:, 4], coordinates[:, 2]
    if panel.name == "rs_rho0":
        return scale_radius_kpc(coordinates), coordinates[:, 2]
    if panel.name == "qhalo_phalo":
        return coordinates[:, 0], coordinates[:, 1]
    raise ValueError(f"unknown parameter-constraint panel: {panel.name}")


def _draw_profile_contour(
    axis,
    surface: ProfileSurface,
    *,
    color: str,
    linestyle: str,
) -> None:
    masked = np.ma.masked_where(~surface.reliable, surface.delta_chi2)
    finite = masked.compressed()
    if finite.size == 0 or np.min(finite) > CONTOUR_LEVEL or np.max(finite) < CONTOUR_LEVEL:
        return
    axis.contour(
        surface.x,
        surface.y,
        masked,
        levels=[CONTOUR_LEVEL],
        colors=[color],
        linestyles=[linestyle],
        linewidths=2.0,
    )


def _fallback_reason(data: ConstraintSamples, settings: ProfileSettings) -> str | None:
    if data.coordinates.shape[0] < settings.minimum_samples:
        return (
            f"GP profile unavailable: {data.coordinates.shape[0]} valid unique trials; "
            f"at least {settings.minimum_samples} required"
        )
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
                    )
                    for objective in OBJECTIVE_NAMES
                }
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
    scatter = None
    for axis, panel in zip(axes, PANELS):
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
            s=16,
            alpha=0.72,
            linewidths=0.0,
            rasterized=True,
        )
        if panel.name in surfaces:
            _draw_profile_contour(
                axis,
                surfaces[panel.name]["total"],
                color="#9b0000",
                linestyle="solid",
            )
            _draw_profile_contour(
                axis,
                surfaces[panel.name]["velocity"],
                color="black",
                linestyle="dashed",
            )
            _draw_profile_contour(
                axis,
                surfaces[panel.name]["density"],
                color="#5573b7",
                linestyle="dashed",
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
    legend_handles = [
        Line2D(
            [0],
            [0],
            color="#9b0000",
            linewidth=2.0,
            label=r"total $\Delta\chi^2=2.30$",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=2.0,
            linestyle="--",
            label=r"velocity $\Delta\chi^2=2.30$",
        ),
        Line2D(
            [0],
            [0],
            color="#5573b7",
            linewidth=2.0,
            linestyle="--",
            label=r"density $\Delta\chi^2=2.30$",
        ),
    ]
    if failure is None:
        figure.legend(
            handles=legend_handles,
            loc="upper center",
            ncol=3,
            bbox_to_anchor=(0.67, 1.01),
        )
        figure.text(
            0.5,
            0.005,
            "Contours are shown only where five-dimensional trial support and "
            "GP-uncertainty checks pass.",
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
        "Five-dimensional GP profile constraints (adaptive trials shown as points)",
        y=1.08,
    )
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
