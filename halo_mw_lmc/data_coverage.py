"""Coverage diagnostics for incomplete six-dimensional stellar catalogues."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .grids import CylindricalGrid
from .phase_space import SphericalPhaseSpace, cartesian_to_spherical_phase_space


FloatArray = NDArray[np.float64]

PHASE_SPACE_COLUMNS = ("x_gc", "y_gc", "z_gc", "vx_gc", "vy_gc", "vz_gc")
DEFAULT_SPHERICAL_RADIUS_EDGES = np.array(
    [4, 6, 8, 10, 12, 15, 20, 30, 50],
    dtype=float,
)
DEFAULT_THETA_EDGES = np.deg2rad(
    np.array([0, 15, 30, 45, 60, 90], dtype=float)
)


def _validated_edges(values: ArrayLike, name: str) -> FloatArray:
    edges = np.asarray(values, dtype=float)
    if (
        edges.ndim != 1
        or edges.size < 2
        or not np.all(np.isfinite(edges))
        or np.any(np.diff(edges) <= 0)
    ):
        raise ValueError(f"{name} must be finite and strictly increasing")
    return edges.copy()


def _occupancy_statistics(counts: np.ndarray) -> dict[str, Any]:
    flat = np.asarray(counts, dtype=float).ravel()
    occupied = flat[flat > 0]
    total_cells = int(flat.size)
    empty_cells = int(np.count_nonzero(flat == 0))
    return {
        "total_cells": total_cells,
        "occupied_cells": int(occupied.size),
        "empty_cells": empty_cells,
        "empty_fraction": empty_cells / total_cells if total_cells else 0.0,
        "cells_with_1_to_5_stars": int(
            np.count_nonzero((flat >= 1) & (flat <= 5))
        ),
        "cells_with_1_to_10_stars": int(
            np.count_nonzero((flat >= 1) & (flat <= 10))
        ),
        "minimum_nonempty_count": int(np.min(occupied)) if occupied.size else 0,
        "median_nonempty_count": (
            float(np.median(occupied)) if occupied.size else 0.0
        ),
        "maximum_count": int(np.max(occupied)) if occupied.size else 0,
    }


@dataclass(frozen=True)
class DataCoverage:
    """Finite catalogue rows and their occupancy on the two fitting grids."""

    input_rows: int
    column_finite_rows: tuple[int, ...]
    position_finite_rows: int
    positions: FloatArray
    initial_conditions: FloatArray
    phase_space: SphericalPhaseSpace
    cylindrical_radius: FloatArray
    rzphi_grid: CylindricalGrid
    rzphi_counts: FloatArray
    spherical_radius_edges: FloatArray
    theta_edges: FloatArray
    phi_edges: FloatArray
    rtheta_phi_counts: FloatArray

    @property
    def complete_phase_space_rows(self) -> int:
        return int(self.initial_conditions.shape[0])

    @property
    def spherical_cell_volumes(self) -> FloatArray:
        radial = np.diff(self.spherical_radius_edges**3) / 3.0
        latitude = np.diff(np.sin(self.theta_edges))
        azimuth = np.diff(self.phi_edges)
        return (
            radial[:, None, None]
            * latitude[None, :, None]
            * azimuth[None, None, :]
        )

    @property
    def rzphi_sampling_density(self) -> FloatArray:
        return np.divide(
            self.rzphi_counts,
            self.rzphi_grid.volumes,
            out=np.zeros_like(self.rzphi_counts),
            where=self.rzphi_grid.volumes > 0,
        )

    @property
    def rtheta_phi_sampling_density(self) -> FloatArray:
        volumes = self.spherical_cell_volumes
        return np.divide(
            self.rtheta_phi_counts,
            volumes,
            out=np.zeros_like(self.rtheta_phi_counts),
            where=volumes > 0,
        )

    def summary(self) -> dict[str, Any]:
        complete = self.complete_phase_space_rows
        return {
            "input_rows": self.input_rows,
            "finite_rows_by_column": {
                name: count
                for name, count in zip(PHASE_SPACE_COLUMNS, self.column_finite_rows)
            },
            "finite_position_rows": self.position_finite_rows,
            "complete_6d_rows": complete,
            "complete_6d_fraction": (
                complete / self.input_rows if self.input_rows else 0.0
            ),
            "rzphi": {
                "shape": list(self.rzphi_counts.shape),
                "in_grid_rows": int(np.sum(self.rzphi_counts)),
                "outside_grid_rows": complete - int(np.sum(self.rzphi_counts)),
                "in_grid_fraction": (
                    float(np.sum(self.rzphi_counts)) / complete if complete else 0.0
                ),
                "rows_by_phi": [
                    int(value) for value in np.sum(self.rzphi_counts, axis=(0, 1))
                ],
                **_occupancy_statistics(self.rzphi_counts),
            },
            "rtheta_phi": {
                "shape": list(self.rtheta_phi_counts.shape),
                "in_grid_rows": int(np.sum(self.rtheta_phi_counts)),
                "outside_grid_rows": complete
                - int(np.sum(self.rtheta_phi_counts)),
                "in_grid_fraction": (
                    float(np.sum(self.rtheta_phi_counts)) / complete
                    if complete
                    else 0.0
                ),
                "rows_by_phi": [
                    int(value)
                    for value in np.sum(self.rtheta_phi_counts, axis=(0, 1))
                ],
                **_occupancy_statistics(self.rtheta_phi_counts),
            },
        }


def build_data_coverage(
    initial_conditions: ArrayLike,
    *,
    rzphi_grid: CylindricalGrid | None = None,
    spherical_radius_edges: ArrayLike = DEFAULT_SPHERICAL_RADIUS_EDGES,
    theta_edges: ArrayLike = DEFAULT_THETA_EDGES,
) -> DataCoverage:
    """Measure raw catalogue coverage without correcting the selection function."""

    initial = np.asarray(initial_conditions, dtype=float)
    if initial.ndim != 2 or initial.shape[1] != 6:
        raise ValueError("initial_conditions must have shape (N, 6)")

    grid = rzphi_grid or CylindricalGrid.uniform()
    radius_edges = _validated_edges(
        spherical_radius_edges,
        "spherical_radius_edges",
    )
    latitude_edges = _validated_edges(theta_edges, "theta_edges")
    if latitude_edges[0] < -np.pi / 2 or latitude_edges[-1] > np.pi / 2:
        raise ValueError("theta_edges must lie within [-π/2, π/2]")

    finite_by_column = tuple(
        int(np.count_nonzero(np.isfinite(initial[:, index])))
        for index in range(initial.shape[1])
    )
    position_mask = np.all(np.isfinite(initial[:, :3]), axis=1)
    complete_mask = np.all(np.isfinite(initial), axis=1)
    positions = initial[position_mask, :3]
    complete = initial[complete_mask]
    if complete.shape[0] == 0:
        raise ValueError("catalogue contains no complete finite 6D rows")

    phase = cartesian_to_spherical_phase_space(
        *[complete[:, index] for index in range(6)]
    )
    cylindrical_radius = np.hypot(complete[:, 0], complete[:, 1])
    rzphi_counts = grid.histogram(
        cylindrical_radius,
        complete[:, 2],
        phase.phi,
    )

    wrapped_phi = grid.wrap_phi(phase.phi)
    rtheta_phi_counts, _ = np.histogramdd(
        np.column_stack((phase.radius, phase.theta, wrapped_phi)),
        bins=(radius_edges, latitude_edges, grid.phi_edges),
    )
    return DataCoverage(
        input_rows=int(initial.shape[0]),
        column_finite_rows=finite_by_column,
        position_finite_rows=int(np.count_nonzero(position_mask)),
        positions=positions,
        initial_conditions=complete,
        phase_space=phase,
        cylindrical_radius=cylindrical_radius,
        rzphi_grid=grid,
        rzphi_counts=np.asarray(rzphi_counts, dtype=float),
        spherical_radius_edges=radius_edges,
        theta_edges=latitude_edges,
        phi_edges=grid.phi_edges.copy(),
        rtheta_phi_counts=np.asarray(rtheta_phi_counts, dtype=float),
    )


def _sample_indices(size: int, maximum: int, seed: int) -> NDArray[np.int64]:
    if maximum < 1:
        raise ValueError("maximum plotted points must be positive")
    if size <= maximum:
        return np.arange(size, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(size, size=maximum, replace=False))


def _log_norm(values: np.ndarray):
    from matplotlib.colors import LogNorm

    positive = np.asarray(values, dtype=float)
    positive = positive[np.isfinite(positive) & (positive > 0)]
    if positive.size == 0:
        return None
    lower = float(np.min(positive))
    upper = float(np.max(positive))
    if lower == upper:
        upper = lower * 10.0
    return LogNorm(vmin=lower, vmax=upper)


def _hexbin_panel(
    axis,
    x: np.ndarray,
    y: np.ndarray,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xlabel: str,
    ylabel: str,
    maximum_points: int,
    seed: int,
):
    from matplotlib.colors import LogNorm

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= xlim[0])
        & (x <= xlim[1])
        & (y >= ylim[0])
        & (y <= ylim[1])
    )
    x_valid = np.asarray(x[valid], dtype=float)
    y_valid = np.asarray(y[valid], dtype=float)
    if x_valid.size == 0:
        axis.text(0.5, 0.5, "No finite points", transform=axis.transAxes, ha="center")
        axis.set_xlim(*xlim)
        axis.set_ylim(*ylim)
        return None

    image = axis.hexbin(
        x_valid,
        y_valid,
        gridsize=48,
        extent=(*xlim, *ylim),
        mincnt=1,
        norm=LogNorm(),
        cmap="viridis",
        linewidths=0,
    )
    selected = _sample_indices(x_valid.size, maximum_points, seed)
    axis.scatter(
        x_valid[selected],
        y_valid[selected],
        s=1.0,
        color="black",
        alpha=0.12,
        linewidths=0,
        rasterized=True,
    )
    axis.set_xlim(*xlim)
    axis.set_ylim(*ylim)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.12)
    return image


def plot_position_projections(
    coverage: DataCoverage,
    output: str | Path,
    *,
    spatial_limit: float = 50.0,
    maximum_points: int = 20_000,
    random_state: int = 0,
) -> Path:
    """Plot Cartesian, cylindrical, and angular projections of catalogue points."""

    import matplotlib.pyplot as plt

    position = coverage.positions
    x, y, z = position.T
    cylindrical_radius = np.hypot(x, y)
    phi = np.rad2deg(np.arctan2(y, x))
    projections = (
        (x, y, (-spatial_limit, spatial_limit), (-spatial_limit, spatial_limit), "x [kpc]", "y [kpc]"),
        (x, z, (-spatial_limit, spatial_limit), (-spatial_limit, spatial_limit), "x [kpc]", "z [kpc]"),
        (y, z, (-spatial_limit, spatial_limit), (-spatial_limit, spatial_limit), "y [kpc]", "z [kpc]"),
        (cylindrical_radius, z, (0.0, spatial_limit), (-spatial_limit, spatial_limit), "R [kpc]", "z [kpc]"),
        (phi, cylindrical_radius, (-180.0, 180.0), (0.0, spatial_limit), "φ [deg]", "R [kpc]"),
        (phi, z, (-180.0, 180.0), (-spatial_limit, spatial_limit), "φ [deg]", "z [kpc]"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for index, (axis, panel) in enumerate(zip(axes.flat, projections)):
        image = _hexbin_panel(
            axis,
            panel[0],
            panel[1],
            xlim=panel[2],
            ylim=panel[3],
            xlabel=panel[4],
            ylabel=panel[5],
            maximum_points=maximum_points,
            seed=random_state + index,
        )
        if image is not None:
            figure.colorbar(image, ax=axis, label="stars per hexagon")
    figure.suptitle(
        f"Observed position coverage ({coverage.position_finite_rows:,} finite positions)",
        x=0.01,
        ha="left",
    )
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_velocity_projections(
    coverage: DataCoverage,
    output: str | Path,
    *,
    velocity_limit: float = 600.0,
    maximum_points: int = 20_000,
    random_state: int = 0,
) -> Path:
    """Plot Cartesian and spherical velocity projections for complete 6D rows."""

    import matplotlib.pyplot as plt

    velocity = coverage.initial_conditions[:, 3:]
    vx, vy, vz = velocity.T
    phase = coverage.phase_space
    limit = (-velocity_limit, velocity_limit)
    projections = (
        (vx, vy, "vₓ [km s⁻¹]", "vᵧ [km s⁻¹]"),
        (vx, vz, "vₓ [km s⁻¹]", "v_z [km s⁻¹]"),
        (vy, vz, "vᵧ [km s⁻¹]", "v_z [km s⁻¹]"),
        (phase.radial_velocity, phase.azimuthal_velocity, "vᵣ [km s⁻¹]", "vφ [km s⁻¹]"),
        (phase.radial_velocity, phase.polar_velocity, "vᵣ [km s⁻¹]", "vθ [km s⁻¹]"),
        (phase.azimuthal_velocity, phase.polar_velocity, "vφ [km s⁻¹]", "vθ [km s⁻¹]"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for index, (axis, (x, y, xlabel, ylabel)) in enumerate(
        zip(axes.flat, projections)
    ):
        image = _hexbin_panel(
            axis,
            x,
            y,
            xlim=limit,
            ylim=limit,
            xlabel=xlabel,
            ylabel=ylabel,
            maximum_points=maximum_points,
            seed=random_state + index,
        )
        if image is not None:
            figure.colorbar(image, ax=axis, label="stars per hexagon")
    figure.suptitle(
        f"Observed velocity coverage ({coverage.complete_phase_space_rows:,} complete 6D rows)",
        x=0.01,
        ha="left",
    )
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def _phi_titles(phi_edges: np.ndarray) -> list[str]:
    return [
        "All φ",
        *[
            f"{lower:.0f}° ≤ φ < {upper:.0f}°"
            for lower, upper in zip(
                np.rad2deg(phi_edges[:-1]),
                np.rad2deg(phi_edges[1:]),
            )
        ],
    ]


def _plot_spatial_occupancy(
    *,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    phi_edges: np.ndarray,
    counts: np.ndarray,
    volumes: np.ndarray,
    point_x: np.ndarray,
    point_y: np.ndarray,
    point_phi: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    output: str | Path,
    maximum_points: int,
    random_state: int,
) -> Path:
    import matplotlib.pyplot as plt

    count_panels = [np.sum(counts, axis=2), *[counts[:, :, i] for i in range(counts.shape[2])]]
    density_panels = [
        np.divide(
            np.sum(counts, axis=2),
            np.sum(volumes, axis=2),
            out=np.zeros_like(np.sum(counts, axis=2)),
            where=np.sum(volumes, axis=2) > 0,
        ),
        *[
            np.divide(
                counts[:, :, i],
                volumes[:, :, i],
                out=np.zeros_like(counts[:, :, i]),
                where=volumes[:, :, i] > 0,
            )
            for i in range(counts.shape[2])
        ],
    ]
    count_norm = _log_norm(np.concatenate([panel.ravel() for panel in count_panels]))
    density_norm = _log_norm(
        np.concatenate([panel.ravel() for panel in density_panels])
    )
    ncolumns = len(count_panels)
    figure, axes = plt.subplots(
        2,
        ncolumns,
        figsize=(3.2 * ncolumns, 7.0),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    wrapped_phi = (point_phi - phi_edges[0]) % (2 * np.pi) + phi_edges[0]
    count_image = None
    density_image = None
    for column, panel_title in enumerate(_phi_titles(phi_edges)):
        phi_mask = np.ones(point_phi.shape, dtype=bool)
        if column > 0:
            lower, upper = phi_edges[column - 1 : column + 1]
            phi_mask = (wrapped_phi >= lower) & (wrapped_phi < upper)
        candidates = np.flatnonzero(
            phi_mask
            & np.isfinite(point_x)
            & np.isfinite(point_y)
            & (point_x >= x_edges[0])
            & (point_x <= x_edges[-1])
            & (point_y >= y_edges[0])
            & (point_y <= y_edges[-1])
        )
        selected_local = _sample_indices(
            candidates.size,
            maximum_points,
            random_state + column,
        )
        selected = candidates[selected_local]
        for row, (panels, norm, row_label) in enumerate(
            (
                (count_panels, count_norm, "Raw occupancy N"),
                (density_panels, density_norm, "Sampling density [stars kpc⁻³]"),
            )
        ):
            axis = axes[row, column]
            masked = np.ma.masked_less_equal(panels[column], 0)
            image = axis.pcolormesh(
                x_edges,
                y_edges,
                masked.T,
                shading="auto",
                cmap="viridis",
                norm=norm,
                rasterized=True,
            )
            if row == 0:
                count_image = image
            else:
                density_image = image
            axis.scatter(
                point_x[selected],
                point_y[selected],
                s=1.0,
                color="black",
                alpha=0.22,
                linewidths=0,
                rasterized=True,
            )
            axis.grid(alpha=0.12)
            if row == 0:
                axis.set_title(panel_title)
            if column == 0:
                axis.set_ylabel(f"{ylabel}\n{row_label}")
            if row == 1:
                axis.set_xlabel(xlabel)
    if count_image is not None:
        figure.colorbar(count_image, ax=axes[0, :], label="stars per cell")
    if density_image is not None:
        figure.colorbar(
            density_image,
            ax=axes[1, :],
            label="stars kpc⁻³",
        )
    figure.suptitle(title, x=0.01, ha="left")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_rzphi_coverage(
    coverage: DataCoverage,
    output: str | Path,
    *,
    maximum_points: int = 10_000,
    random_state: int = 0,
) -> Path:
    """Plot raw occupancy and sampling density on the active R-z-phi grid."""

    phase = coverage.phase_space
    return _plot_spatial_occupancy(
        x_edges=coverage.rzphi_grid.r_edges,
        y_edges=coverage.rzphi_grid.z_edges,
        phi_edges=coverage.rzphi_grid.phi_edges,
        counts=coverage.rzphi_counts,
        volumes=coverage.rzphi_grid.volumes,
        point_x=coverage.cylindrical_radius,
        point_y=coverage.initial_conditions[:, 2],
        point_phi=phase.phi,
        xlabel="R [kpc]",
        ylabel="z [kpc]",
        title="Observed coverage in the R-z-φ fitting grid",
        output=output,
        maximum_points=maximum_points,
        random_state=random_state,
    )


def plot_rtheta_phi_coverage(
    coverage: DataCoverage,
    output: str | Path,
    *,
    maximum_points: int = 10_000,
    random_state: int = 0,
) -> Path:
    """Plot raw occupancy and sampling density on the velocity spatial grid."""

    phase = coverage.phase_space
    return _plot_spatial_occupancy(
        x_edges=coverage.spherical_radius_edges,
        y_edges=np.rad2deg(coverage.theta_edges),
        phi_edges=coverage.phi_edges,
        counts=coverage.rtheta_phi_counts,
        volumes=coverage.spherical_cell_volumes,
        point_x=phase.radius,
        point_y=np.rad2deg(phase.theta),
        point_phi=phase.phi,
        xlabel="r [kpc]",
        ylabel="θ [deg]",
        title="Observed coverage in the r-θ-φ velocity grid",
        output=output,
        maximum_points=maximum_points,
        random_state=random_state,
    )


def _plot_profile_family(
    axis,
    centers: np.ndarray,
    counts: np.ndarray,
    volumes: np.ndarray,
    phi_edges: np.ndarray,
    *,
    density: bool,
) -> None:
    import matplotlib.pyplot as plt

    total_counts = np.sum(counts, axis=1)
    total_volumes = np.sum(volumes, axis=1)
    total = (
        np.divide(
            total_counts,
            total_volumes,
            out=np.zeros_like(total_counts),
            where=total_volumes > 0,
        )
        if density
        else total_counts
    )
    axis.step(centers, total, where="mid", color="black", linewidth=2, label="All φ")
    colors = plt.colormaps["tab10"]
    for iphi in range(counts.shape[1]):
        values = (
            np.divide(
                counts[:, iphi],
                volumes[:, iphi],
                out=np.zeros_like(counts[:, iphi]),
                where=volumes[:, iphi] > 0,
            )
            if density
            else counts[:, iphi]
        )
        lower, upper = np.rad2deg(phi_edges[iphi : iphi + 2])
        axis.step(
            centers,
            values,
            where="mid",
            color=colors(iphi % 10),
            label=f"{lower:.0f}°–{upper:.0f}°",
        )
    positive = total[total > 0]
    linthresh = max(float(np.min(positive)) / 2.0, np.finfo(float).tiny) if positive.size else 1.0
    axis.set_yscale("symlog", linthresh=linthresh)
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.2)


def plot_sampling_profiles(
    coverage: DataCoverage,
    output: str | Path,
) -> Path:
    """Plot marginal raw counts and volume-normalized sampling-density profiles."""

    import matplotlib.pyplot as plt

    rz_counts = coverage.rzphi_counts
    rz_volumes = coverage.rzphi_grid.volumes
    rt_counts = coverage.rtheta_phi_counts
    rt_volumes = coverage.spherical_cell_volumes
    r_centers, z_centers, _ = coverage.rzphi_grid.centers
    spherical_centers = 0.5 * (
        coverage.spherical_radius_edges[:-1]
        + coverage.spherical_radius_edges[1:]
    )
    theta_centers = np.rad2deg(
        0.5 * (coverage.theta_edges[:-1] + coverage.theta_edges[1:])
    )
    profiles = (
        (
            r_centers,
            np.sum(rz_counts, axis=1),
            np.sum(rz_volumes, axis=1),
            "R [kpc]",
        ),
        (
            z_centers,
            np.sum(rz_counts, axis=0),
            np.sum(rz_volumes, axis=0),
            "z [kpc]",
        ),
        (
            spherical_centers,
            np.sum(rt_counts, axis=1),
            np.sum(rt_volumes, axis=1),
            "r [kpc]",
        ),
        (
            theta_centers,
            np.sum(rt_counts, axis=0),
            np.sum(rt_volumes, axis=0),
            "θ [deg]",
        ),
    )
    figure, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    for column, (centers, counts, volumes, xlabel) in enumerate(profiles):
        _plot_profile_family(
            axes[0, column],
            centers,
            counts,
            volumes,
            coverage.phi_edges,
            density=False,
        )
        _plot_profile_family(
            axes[1, column],
            centers,
            counts,
            volumes,
            coverage.phi_edges,
            density=True,
        )
        axes[0, column].set_xlabel(xlabel)
        axes[1, column].set_xlabel(xlabel)
        if column == 0:
            axes[0, column].set_ylabel("Raw star count")
            axes[1, column].set_ylabel("Sampling density [stars kpc⁻³]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper right", ncol=5)
    figure.suptitle("Observed marginal coverage profiles", x=0.01, ha="left")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_coverage_summary(
    coverage: DataCoverage,
    output: str | Path,
) -> Path:
    """Create a compact brief of incomplete rows, empty cells, and phi counts."""

    import matplotlib.pyplot as plt

    summary = coverage.summary()
    rz = summary["rzphi"]
    rt = summary["rtheta_phi"]
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes[0, 0].axis("off")
    axes[0, 0].text(
        0.02,
        0.98,
        "\n".join(
            (
                f"Input rows: {summary['input_rows']:,}",
                f"Finite positions: {summary['finite_position_rows']:,}",
                f"Complete 6D rows: {summary['complete_6d_rows']:,}",
                f"Complete fraction: {summary['complete_6d_fraction']:.1%}",
                "",
                f"R-z-φ in-grid rows: {rz['in_grid_rows']:,}",
                f"R-z-φ empty cells: {rz['empty_fraction']:.1%}",
                f"r-θ-φ in-grid rows: {rt['in_grid_rows']:,}",
                f"r-θ-φ empty cells: {rt['empty_fraction']:.1%}",
            )
        ),
        transform=axes[0, 0].transAxes,
        va="top",
        family="monospace",
        fontsize=11,
    )

    phi_centers = np.rad2deg(
        0.5 * (coverage.phi_edges[:-1] + coverage.phi_edges[1:])
    )
    width = 0.38 * np.rad2deg(np.diff(coverage.phi_edges))
    axes[0, 1].bar(
        phi_centers - width / 2,
        rz["rows_by_phi"],
        width=width,
        label="R-z-φ grid",
    )
    axes[0, 1].bar(
        phi_centers + width / 2,
        rt["rows_by_phi"],
        width=width,
        label="r-θ-φ grid",
    )
    axes[0, 1].set_xlabel("φ-bin center [deg]")
    axes[0, 1].set_ylabel("Stars in grid")
    axes[0, 1].legend()
    axes[0, 1].grid(axis="y", alpha=0.2)

    for axis, counts, label in (
        (axes[1, 0], coverage.rzphi_counts, "R-z-φ"),
        (axes[1, 1], coverage.rtheta_phi_counts, "r-θ-φ"),
    ):
        positive = counts[counts > 0]
        if positive.size:
            upper = max(int(np.max(positive)), 1)
            bins = np.unique(
                np.geomspace(1, upper + 1, min(24, upper + 1)).astype(int)
            )
            if bins.size < 2:
                bins = np.array([0.5, 1.5])
            axis.hist(positive, bins=bins, color="0.35")
            axis.set_xscale("log")
            axis.set_yscale("log")
        empty = np.count_nonzero(counts == 0)
        axis.set_title(f"{label}: {empty}/{counts.size} cells empty")
        axis.set_xlabel("Stars per non-empty cell")
        axis.set_ylabel("Number of cells")
        axis.grid(alpha=0.2)
    figure.suptitle("Observed six-dimensional coverage brief", x=0.01, ha="left")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_all_data_coverage(
    coverage: DataCoverage,
    output_directory: str | Path,
    *,
    spatial_limit: float = 50.0,
    velocity_limit: float = 600.0,
    maximum_points: int = 20_000,
    random_state: int = 0,
) -> list[Path]:
    """Write the complete data-only coverage diagnostic set."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    return [
        plot_coverage_summary(coverage, output / "coverage_summary.pdf"),
        plot_position_projections(
            coverage,
            output / "position_projections.pdf",
            spatial_limit=spatial_limit,
            maximum_points=maximum_points,
            random_state=random_state,
        ),
        plot_velocity_projections(
            coverage,
            output / "velocity_projections.pdf",
            velocity_limit=velocity_limit,
            maximum_points=maximum_points,
            random_state=random_state,
        ),
        plot_rzphi_coverage(
            coverage,
            output / "rzphi_coverage.pdf",
            maximum_points=maximum_points,
            random_state=random_state,
        ),
        plot_rtheta_phi_coverage(
            coverage,
            output / "rtheta_phi_coverage.pdf",
            maximum_points=maximum_points,
            random_state=random_state,
        ),
        plot_sampling_profiles(
            coverage,
            output / "sampling_profiles.pdf",
        ),
    ]
