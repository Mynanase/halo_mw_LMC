"""Paper-style diagnostics for the azimuth-resolved Zhu comparison."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from .density import DensityComparison
from .velocity import (
    VelocityDistributionComparison,
    multinomial_histogram_uncertainty,
)


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class IsodensityShapeProfile:
    """Major-axis radius and axis ratio inferred from density intercepts."""

    radius: FloatArray
    axis_ratio: FloatArray
    density_level: FloatArray


def _positive_log_limits(*arrays: np.ndarray) -> tuple[float, float]:
    parts = [
        np.asarray(values)[np.isfinite(values) & (values > 0)] for values in arrays
    ]
    positive = np.concatenate(parts) if parts else np.array([], dtype=float)
    if positive.size == 0:
        return -1.0, 1.0
    lower, upper = np.nanpercentile(np.log10(positive), [2, 98])
    if not np.isfinite(lower) or not np.isfinite(upper) or lower == upper:
        return float(lower - 0.5), float(upper + 0.5)
    return float(lower), float(upper)


def _log_density(values: np.ndarray) -> np.ndarray:
    result = np.full_like(np.asarray(values, dtype=float), np.nan)
    np.log10(values, out=result, where=np.asarray(values) > 0)
    return result


def _monotonic_intercept(
    coordinates: np.ndarray,
    density: np.ndarray,
    level: float,
) -> float:
    valid = np.isfinite(coordinates) & np.isfinite(density) & (density > 0)
    coordinates = np.asarray(coordinates[valid], dtype=float)
    density = np.asarray(density[valid], dtype=float)
    if coordinates.size < 2:
        return np.nan

    order = np.argsort(coordinates)
    coordinates = coordinates[order]
    density = np.minimum.accumulate(density[order])
    if level > density[0] or level < density[-1]:
        return np.nan

    reversed_density = density[::-1]
    reversed_coordinates = coordinates[::-1]
    unique_density, unique_index = np.unique(reversed_density, return_index=True)
    if unique_density.size < 2:
        return np.nan
    return float(
        np.interp(level, unique_density, reversed_coordinates[unique_index])
    )


def isodensity_shape_profile(
    density: np.ndarray,
    comparison: DensityComparison,
    phi_index: int,
    *,
    n_levels: int = 7,
) -> IsodensityShapeProfile:
    """Estimate ``q=z_iso/R_iso`` from major/minor-axis density intercepts.

    The profiles on the lowest-z and lowest-R grid lines are monotonized
    outwards before interpolation.  This makes the diagnostic stable for
    sparse empirical maps while retaining the contour-intercept definition
    used by Zhu et al.
    """

    grid = comparison.grid
    values = np.asarray(density, dtype=float)
    if values.shape != grid.shape:
        raise ValueError(f"density has shape {values.shape}; expected {grid.shape}")
    if not 0 <= phi_index < grid.shape[-1]:
        raise IndexError("phi_index is outside the density grid")
    if n_levels < 2:
        raise ValueError("n_levels must be at least two")

    r_centers, z_centers, _ = grid.centers
    radial_profile = values[:, 0, phi_index]
    vertical_profile = values[0, :, phi_index]
    positive = np.concatenate(
        (
            radial_profile[np.isfinite(radial_profile) & (radial_profile > 0)],
            vertical_profile[np.isfinite(vertical_profile) & (vertical_profile > 0)],
        )
    )
    if positive.size < 4:
        empty = np.array([], dtype=float)
        return IsodensityShapeProfile(empty, empty, empty)

    low, high = np.nanpercentile(np.log10(positive), [15, 85])
    levels = np.logspace(low, high, n_levels)
    radius: list[float] = []
    axis_ratio: list[float] = []
    used_levels: list[float] = []
    for level in levels:
        r_iso = _monotonic_intercept(r_centers, radial_profile, level)
        z_iso = _monotonic_intercept(z_centers, vertical_profile, level)
        if np.isfinite(r_iso) and np.isfinite(z_iso) and r_iso > 0 and z_iso > 0:
            radius.append(r_iso)
            axis_ratio.append(z_iso / r_iso)
            used_levels.append(level)

    order = np.argsort(radius)
    return IsodensityShapeProfile(
        radius=np.asarray(radius, dtype=float)[order],
        axis_ratio=np.asarray(axis_ratio, dtype=float)[order],
        density_level=np.asarray(used_levels, dtype=float)[order],
    )


def _draw_contours(axis, density: np.ndarray, grid) -> None:
    positive = density[np.isfinite(density) & (density > 0)]
    if positive.size < 5:
        return
    levels = np.unique(np.nanpercentile(np.log10(positive), [20, 40, 60, 80]))
    if levels.size:
        r_centers, z_centers, _ = grid.centers
        axis.contour(
            r_centers,
            z_centers,
            _log_density(density).T,
            levels=levels,
            colors="black",
            linewidths=0.55,
            linestyles="--",
            alpha=0.75,
        )


def _azimuthal_density_average(
    values: np.ndarray,
    comparison: DensityComparison,
) -> np.ndarray:
    weights = np.diff(comparison.grid.phi_edges) / (2 * np.pi)
    return np.sum(np.asarray(values) * weights[None, None, :], axis=2)


def _azimuthal_density_error(comparison: DensityComparison) -> np.ndarray:
    weights = np.diff(comparison.grid.phi_edges) / (2 * np.pi)
    return np.sqrt(
        np.sum(
            (comparison.data_error * weights[None, None, :]) ** 2,
            axis=2,
        )
    )


def plot_density_comparison(
    comparison: DensityComparison,
    output: str | Path,
) -> None:
    """Write a four-row overview with an average plus every azimuth bin."""

    import matplotlib.pyplot as plt

    grid = comparison.grid
    nphi = grid.shape[-1]
    ncolumns = nphi + 1
    figure, axes = plt.subplots(
        4,
        ncolumns,
        figsize=(max(3.0 * ncolumns, 8.0), 11.0),
        squeeze=False,
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    extent = [grid.r_edges[0], grid.r_edges[-1], grid.z_edges[0], grid.z_edges[-1]]
    log_min, log_max = _positive_log_limits(
        comparison.data_density,
        comparison.model_density,
    )
    relative_error = np.divide(
        comparison.data_error,
        comparison.data_density,
        out=np.full_like(comparison.data_error, np.nan),
        where=comparison.data_density > 0,
    )
    finite_relative_error = relative_error[np.isfinite(relative_error)]
    relative_max = (
        float(np.nanpercentile(finite_relative_error, 95))
        if finite_relative_error.size
        else 1.0
    )
    relative_max = max(relative_max, np.finfo(float).eps)

    average_data = _azimuthal_density_average(
        comparison.data_density,
        comparison,
    )
    average_model = _azimuthal_density_average(
        comparison.model_density,
        comparison,
    )
    average_error = _azimuthal_density_error(comparison)
    average_relative_error = np.divide(
        average_error,
        average_data,
        out=np.full_like(average_error, np.nan),
        where=average_data > 0,
    )
    average_residual = np.divide(
        average_data - average_model,
        average_error,
        out=np.full_like(average_error, np.nan),
        where=average_error > 0,
    )
    average_residual[~np.any(comparison.fit_mask, axis=2)] = np.nan
    panels = [
        (
            "φ averaged",
            average_data,
            average_model,
            average_relative_error,
            average_residual,
        )
    ]
    panels.extend(
        (
            (
                f"{phi_lo:.0f}° ≤ φ < {phi_hi:.0f}°",
                comparison.data_density[:, :, iphi],
                comparison.model_density[:, :, iphi],
                relative_error[:, :, iphi],
                comparison.residual[:, :, iphi],
            )
        )
        for iphi, (phi_lo, phi_hi) in enumerate(
            np.rad2deg(
                np.column_stack((grid.phi_edges[:-1], grid.phi_edges[1:]))
            )
        )
    )

    density_image = relative_image = residual_image = None
    for column, (
        title,
        data_slice,
        model_slice,
        relative_slice,
        residual_slice,
    ) in enumerate(panels):
        axes[0, column].set_title(title)
        for row, density_slice in enumerate((data_slice, model_slice)):
            density_image = axes[row, column].imshow(
                _log_density(density_slice).T,
                origin="lower",
                extent=extent,
                aspect="auto",
                cmap="viridis",
                vmin=log_min,
                vmax=log_max,
            )
            _draw_contours(axes[row, column], density_slice, grid)
        relative_image = axes[2, column].imshow(
            relative_slice.T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="magma",
            vmin=0,
            vmax=relative_max,
        )
        residual_image = axes[3, column].imshow(
            residual_slice.T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="coolwarm",
            vmin=-3,
            vmax=3,
        )
        axes[3, column].set_xlabel("R [kpc]")

    labels = (
        "target log₁₀ ν",
        "model log₁₀ ν",
        "target σ/ν",
        "(target-model)/σ",
    )
    for row, label in enumerate(labels):
        axes[row, 0].set_ylabel(f"{label}\nz [kpc]")
    if density_image is not None:
        figure.colorbar(density_image, ax=axes[:2, :], label="log₁₀ ν")
    if relative_image is not None:
        figure.colorbar(relative_image, ax=axes[2, :], label="relative error")
    if residual_image is not None:
        figure.colorbar(
            residual_image,
            ax=axes[3, :],
            label="standardized residual",
        )
    figure.suptitle(
        f"φ-resolved density: global scale={comparison.scale:.5g}, "
        f"χ²={comparison.chi2:.3f}"
    )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def plot_density_phi_pages(
    comparison: DensityComparison,
    output_directory: str | Path,
) -> list[Path]:
    """Write one Zhu-Fig.-6-style density page for each azimuth bin."""

    import matplotlib.pyplot as plt

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    grid = comparison.grid
    extent = [grid.r_edges[0], grid.r_edges[-1], grid.z_edges[0], grid.z_edges[-1]]
    log_min, log_max = _positive_log_limits(
        comparison.data_density,
        comparison.model_density,
    )
    relative_error = np.divide(
        comparison.data_error,
        comparison.data_density,
        out=np.full_like(comparison.data_error, np.nan),
        where=comparison.data_density > 0,
    )
    finite_relative_error = relative_error[np.isfinite(relative_error)]
    relative_max = (
        float(np.nanpercentile(finite_relative_error, 95))
        if finite_relative_error.size
        else 1.0
    )
    relative_max = max(relative_max, np.finfo(float).eps)
    written: list[Path] = []
    for iphi in range(grid.shape[-1]):
        figure, axes = plt.subplots(
            2,
            2,
            figsize=(8.2, 7.2),
            constrained_layout=True,
            sharex=True,
            sharey=True,
        )
        density_image = None
        for axis, title, density in (
            (axes[0, 0], "Target density", comparison.data_density[:, :, iphi]),
            (axes[0, 1], "Model density", comparison.model_density[:, :, iphi]),
        ):
            density_image = axis.imshow(
                _log_density(density).T,
                origin="lower",
                extent=extent,
                aspect="auto",
                cmap="viridis",
                vmin=log_min,
                vmax=log_max,
            )
            _draw_contours(axis, density, grid)
            axis.set_title(title)
        error_image = axes[1, 0].imshow(
            relative_error[:, :, iphi].T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="magma",
            vmin=0,
            vmax=relative_max,
        )
        residual_image = axes[1, 1].imshow(
            comparison.residual[:, :, iphi].T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="coolwarm",
            vmin=-3,
            vmax=3,
        )
        axes[1, 0].set_title("Target relative error")
        axes[1, 1].set_title("Standardized residual")
        for axis in axes[:, 0]:
            axis.set_ylabel("z [kpc]")
        for axis in axes[1, :]:
            axis.set_xlabel("R [kpc]")
        figure.colorbar(density_image, ax=axes[0, :], label="log₁₀ ν")
        figure.colorbar(error_image, ax=axes[1, 0], label="σ/ν")
        figure.colorbar(residual_image, ax=axes[1, 1], label="(target-model)/σ")
        phi_lo, phi_hi = np.rad2deg(grid.phi_edges[iphi : iphi + 2])
        figure.suptitle(
            f"{phi_lo:.0f}° ≤ φ < {phi_hi:.0f}°; "
            f"χ²φ={comparison.chi2_by_phi[iphi]:.2f}"
        )
        path = output_directory / f"density_phi{iphi:02d}.pdf"
        figure.savefig(path, bbox_inches="tight")
        plt.close(figure)
        written.append(path)
    return written


def plot_density_shape(
    comparison: DensityComparison,
    output: str | Path,
) -> None:
    """Plot target/model isodensity axis ratios separately in every phi bin."""

    import matplotlib.pyplot as plt

    grid = comparison.grid
    nphi = grid.shape[-1]
    figure, axes = plt.subplots(
        1,
        nphi,
        figsize=(max(3.2 * nphi, 7.0), 3.6),
        squeeze=False,
        constrained_layout=True,
        sharey=True,
    )
    for iphi, axis in enumerate(axes[0]):
        target = isodensity_shape_profile(
            comparison.data_density,
            comparison,
            iphi,
        )
        model = isodensity_shape_profile(
            comparison.model_density,
            comparison,
            iphi,
        )
        axis.plot(
            target.radius,
            target.axis_ratio,
            marker="*",
            color="0.25",
            label="Target",
        )
        axis.plot(model.radius, model.axis_ratio, color="red", label="Model")
        phi_lo, phi_hi = np.rad2deg(grid.phi_edges[iphi : iphi + 2])
        axis.set_title(f"{phi_lo:.0f}° ≤ φ < {phi_hi:.0f}°")
        axis.set_xlabel("R of isodensity contour [kpc]")
        axis.grid(alpha=0.2)
    axes[0, 0].set_ylabel("q★ = ziso / Riso")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="outside upper right", ncol=2)
    figure.suptitle(
        "Azimuth-resolved stellar-halo flattening",
        x=0.01,
        ha="left",
    )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def _selected_indices(size: int, preferred: tuple[int, ...]) -> list[int]:
    selected = [index for index in preferred if index < size]
    if selected:
        return selected
    return list(range(size))


def _velocity_panel_values(
    comparison: VelocityDistributionComparison,
    radius_index: int,
    theta_index: int,
    phi_index: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if phi_index is not None:
        return (
            comparison.data_probability[radius_index, theta_index, phi_index],
            comparison.data_uncertainty[radius_index, theta_index, phi_index],
            comparison.model_probability[radius_index, theta_index, phi_index],
            float(comparison.data_occupancy[radius_index, theta_index, phi_index]),
        )

    data_occupancy = np.sum(
        comparison.data_occupancy[radius_index, theta_index, :]
    )
    data_counts = np.sum(
        comparison.data_probability[radius_index, theta_index, :, :]
        * comparison.data_occupancy[radius_index, theta_index, :, None],
        axis=0,
    )
    data_probability = np.divide(
        data_counts,
        data_occupancy,
        out=np.zeros_like(data_counts),
        where=data_occupancy > 0,
    )
    data_uncertainty = multinomial_histogram_uncertainty(
        data_probability[None, :],
        np.asarray([data_occupancy]),
    )[0]

    model_occupancy = np.sum(
        comparison.model_occupancy[radius_index, theta_index, :]
    )
    model_counts = np.sum(
        comparison.model_probability[radius_index, theta_index, :, :]
        * comparison.model_occupancy[radius_index, theta_index, :, None],
        axis=0,
    )
    model_probability = np.divide(
        model_counts,
        model_occupancy,
        out=np.zeros_like(model_counts),
        where=model_occupancy > 0,
    )
    return data_probability, data_uncertainty, model_probability, float(data_occupancy)


def _coarsen_velocity_panel(
    velocity_edges: np.ndarray,
    data_probability: np.ndarray,
    model_probability: np.ndarray,
    data_occupancy: float,
    bin_factor: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate adjacent likelihood bins for plotting without changing the fit."""

    if isinstance(bin_factor, bool) or not isinstance(bin_factor, (int, np.integer)):
        raise ValueError("velocity plotting bin factor must be an integer")
    if bin_factor < 1:
        raise ValueError("velocity plotting bin factor must be positive")

    edges = np.asarray(velocity_edges, dtype=float)
    data = np.asarray(data_probability, dtype=float)
    model = np.asarray(model_probability, dtype=float)
    fine_bin_count = edges.size - 1
    if data.shape != (fine_bin_count,) or model.shape != (fine_bin_count,):
        raise ValueError("velocity panel values do not match the velocity grid")
    coarse_starts = np.arange(0, fine_bin_count, bin_factor)
    coarse_edges = np.append(edges[coarse_starts], edges[-1])
    coarse_centers = 0.5 * (coarse_edges[:-1] + coarse_edges[1:])
    coarse_data = np.add.reduceat(data, coarse_starts)
    coarse_model = np.add.reduceat(model, coarse_starts)
    coarse_uncertainty = multinomial_histogram_uncertainty(
        coarse_data[None, :],
        np.asarray([data_occupancy], dtype=float),
    )[0]
    return coarse_centers, coarse_data, coarse_uncertainty, coarse_model


def plot_velocity_distributions(
    comparisons: Mapping[str, VelocityDistributionComparison],
    output_directory: str | Path,
    *,
    minimum_radius: float = 8.0,
    theta_indices: tuple[int, ...] = (0, 2, 4),
    velocity_bin_factor: int = 3,
) -> list[Path]:
    """Write velocity grids, coarsening only their visual representation."""

    import matplotlib.pyplot as plt

    if not comparisons:
        return []
    required = ("vr", "vphi", "vtheta")
    missing = [name for name in required if name not in comparisons]
    if missing:
        raise ValueError(f"missing velocity components: {', '.join(missing)}")
    grid = comparisons["vr"].grid
    if any(comparisons[name].grid.shape != grid.shape for name in required):
        raise ValueError("velocity comparisons do not share one grid")

    radial_indices = [
        index
        for index, lower in enumerate(grid.radius_edges[:-1])
        if lower >= minimum_radius
    ]
    if not radial_indices:
        radial_indices = list(range(grid.shape[0]))
    selected_theta = _selected_indices(grid.shape[1], theta_indices)
    nrows = len(radial_indices)
    ntheta = len(selected_theta)
    ncolumns = len(required) * ntheta
    labels = {"vr": "vᵣ", "vphi": "vφ", "vtheta": "vθ"}
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for iphi in [None, *range(grid.shape[2])]:
        figure, axes = plt.subplots(
            nrows,
            ncolumns,
            figsize=(2.05 * ncolumns, 1.75 * nrows + 1.0),
            squeeze=False,
            constrained_layout=True,
            sharex=True,
        )
        legend_handles = None
        for row, iradius in enumerate(radial_indices):
            for component_index, component in enumerate(required):
                comparison = comparisons[component]
                for theta_column, itheta in enumerate(selected_theta):
                    column = component_index * ntheta + theta_column
                    axis = axes[row, column]
                    data, uncertainty, model, data_occupancy = (
                        _velocity_panel_values(
                            comparison,
                            iradius,
                            itheta,
                            iphi,
                        )
                    )
                    centers, data, uncertainty, model = _coarsen_velocity_panel(
                        grid.velocity_edges,
                        data,
                        model,
                        data_occupancy,
                        velocity_bin_factor,
                    )
                    if data_occupancy > 0:
                        data_line = axis.plot(
                            centers,
                            data,
                            color="0.45",
                            linewidth=1.5,
                            label="Data",
                        )[0]
                        axis.plot(
                            centers,
                            np.clip(data - uncertainty, 0, None),
                            color="0.6",
                            linewidth=0.8,
                            linestyle="--",
                        )
                        axis.plot(
                            centers,
                            data + uncertainty,
                            color="0.6",
                            linewidth=0.8,
                            linestyle="--",
                            label="Data ±1σ",
                        )
                    else:
                        data_line = None
                    model_line = axis.plot(
                        centers,
                        model,
                        color="red",
                        linewidth=1.5,
                        label="Model",
                    )[0]
                    if legend_handles is None and data_line is not None:
                        legend_handles = (data_line, model_line)
                    axis.set_yticks([])
                    axis.grid(alpha=0.12)
                    if row == 0:
                        theta_lo, theta_hi = np.rad2deg(
                            grid.theta_edges[itheta : itheta + 2]
                        )
                        axis.set_title(f"θ={theta_lo:.0f}–{theta_hi:.0f}°")
                    if row == nrows - 1:
                        axis.set_xlabel(f"{labels[component]} [km s⁻¹]")
                    if column == 0:
                        r_lo, r_hi = grid.radius_edges[iradius : iradius + 2]
                        axis.set_ylabel(f"{r_lo:g}–{r_hi:g} kpc")
                    occupancy = int(data_occupancy)
                    axis.text(
                        0.97,
                        0.92,
                        f"N={occupancy}",
                        transform=axis.transAxes,
                        ha="right",
                        va="top",
                        fontsize=6,
                        color="0.35",
                    )
        if iphi is None:
            title = "Velocity distributions, φ averaged"
            filename = "velocity_phi_average.pdf"
        else:
            phi_lo, phi_hi = np.rad2deg(grid.phi_edges[iphi : iphi + 2])
            title = (
                f"Velocity distributions, {phi_lo:.0f}° ≤ φ < {phi_hi:.0f}°"
            )
            filename = f"velocity_phi{iphi:02d}.pdf"
        figure.suptitle(title, x=0.01, ha="left")
        if legend_handles is not None:
            figure.legend(
                legend_handles,
                ("Data", "Model"),
                loc="outside upper right",
                ncol=2,
            )
        path = output_directory / filename
        figure.savefig(path, bbox_inches="tight")
        plt.close(figure)
        written.append(path)
    return written


def plot_model_diagnostics(
    density: DensityComparison,
    velocities: Mapping[str, VelocityDistributionComparison],
    output_directory: str | Path,
    *,
    velocity_bin_factor: int = 3,
) -> list[Path]:
    """Create the complete phi-resolved density and velocity plot set."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    overview = output_directory / "density_overview.pdf"
    shape = output_directory / "density_flattening.pdf"
    plot_density_comparison(density, overview)
    plot_density_shape(density, shape)
    written = [overview, shape]
    written.extend(plot_density_phi_pages(density, output_directory))
    written.extend(
        plot_velocity_distributions(
            velocities,
            output_directory,
            velocity_bin_factor=velocity_bin_factor,
        )
    )
    return written
