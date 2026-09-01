"""Static diagnostics for trial-specific orbit-weight solutions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class OrbitWeightSummary:
    """Display quantities derived without changing the stored solved weights."""

    normalized_weights: FloatArray
    active_mask: NDArray[np.bool_]
    log10_active_fraction: FloatArray
    total_weight: float
    effective_orbit_count: float
    maximum_weight_fraction: float

    @property
    def orbit_count(self) -> int:
        return int(self.normalized_weights.size)

    @property
    def active_orbit_count(self) -> int:
        return int(np.count_nonzero(self.active_mask))

    @property
    def inactive_orbit_count(self) -> int:
        return self.orbit_count - self.active_orbit_count


def summarize_orbit_weights(
    weights: ArrayLike,
    *,
    relative_active_threshold: float = 1e-12,
) -> OrbitWeightSummary:
    """Compute scale-free display diagnostics for one solved weight vector.

    The relative activity threshold matches the density-solved optimizer.  The
    division by the total weight is only for comparing distributions between
    runs; it does not alter the unconstrained scale of the stored solution.
    """

    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("orbit weights must be a finite one-dimensional array")
    if np.any(values < 0):
        raise ValueError("orbit weights must be non-negative")
    if not np.isfinite(relative_active_threshold) or relative_active_threshold < 0:
        raise ValueError("relative_active_threshold must be finite and non-negative")

    total = float(np.sum(values))
    if total <= 0:
        raise ValueError("orbit weights must have a positive total")
    normalized = values / total
    maximum = float(np.max(values))
    active_mask = values > maximum * relative_active_threshold
    active_fraction = normalized[active_mask]
    if active_fraction.size == 0:
        raise ValueError("orbit weights contain no active values")
    squared = float(np.dot(normalized, normalized))
    return OrbitWeightSummary(
        normalized_weights=normalized,
        active_mask=active_mask,
        log10_active_fraction=np.log10(active_fraction),
        total_weight=total,
        effective_orbit_count=1.0 / squared,
        maximum_weight_fraction=float(np.max(normalized)),
    )


def shared_log_weight_edges(
    summaries: Mapping[str, OrbitWeightSummary],
    *,
    bins: int = 32,
) -> FloatArray:
    """Return common log-weight bin edges for comparable small multiples."""

    if not summaries:
        raise ValueError("at least one orbit-weight summary is required")
    if bins < 2:
        raise ValueError("bins must be at least two")
    values = np.concatenate(
        [summary.log10_active_fraction for summary in summaries.values()]
    )
    lower = float(np.min(values))
    upper = float(np.max(values))
    if lower == upper:
        lower -= 0.5
        upper += 0.5
    return np.linspace(lower, upper, bins + 1)


def orbit_weight_histograms(
    summary: OrbitWeightSummary,
    edges: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """Return active-orbit counts and percent of total weight in each bin."""

    bin_edges = np.asarray(edges, dtype=float)
    if (
        bin_edges.ndim != 1
        or bin_edges.size < 3
        or not np.all(np.isfinite(bin_edges))
        or not np.all(np.diff(bin_edges) > 0)
    ):
        raise ValueError("histogram edges must be finite and strictly increasing")
    count, _ = np.histogram(summary.log10_active_fraction, bins=bin_edges)
    weight_share, _ = np.histogram(
        summary.log10_active_fraction,
        bins=bin_edges,
        weights=summary.normalized_weights[summary.active_mask] * 100.0,
    )
    return count.astype(float), weight_share.astype(float)


def plot_orbit_weight_histograms(
    weights_by_case: Mapping[str, ArrayLike],
    output: str | Path,
    *,
    bins: int = 32,
    title: str = "Density-solved orbit-weight distributions",
) -> None:
    """Write shared-scale count and weight-share histograms for several runs."""

    if not weights_by_case:
        raise ValueError("at least one orbit-weight vector is required")
    summaries = {
        label: summarize_orbit_weights(weights)
        for label, weights in weights_by_case.items()
    }
    edges = shared_log_weight_edges(summaries, bins=bins)

    import matplotlib.pyplot as plt

    rows = len(summaries)
    figure, axes = plt.subplots(
        rows,
        2,
        figsize=(11.0, 2.15 * rows + 2.4),
        sharex=True,
        sharey="col",
        squeeze=False,
    )
    for row, (label, summary) in enumerate(summaries.items()):
        count, weight_share = orbit_weight_histograms(summary, edges)
        count_axis, weight_axis = axes[row]
        count_axis.stairs(count, edges, fill=True, alpha=0.75, color="C0")
        weight_axis.stairs(
            np.where(weight_share > 0, weight_share, np.nan),
            edges,
            fill=True,
            alpha=0.75,
            color="C1",
        )
        for axis in (count_axis, weight_axis):
            axis.set_yscale("log")
            axis.grid(axis="y", which="both", linewidth=0.45, alpha=0.35)
            axis.set_xlim(edges[0], edges[-1])
        count_axis.set_ylabel("Orbit count")
        weight_axis.set_ylabel("Weight share [%]")
        count_axis.text(
            0.02,
            0.92,
            label,
            transform=count_axis.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )
        weight_axis.text(
            0.02,
            0.92,
            (
                f"active {summary.active_orbit_count:,}/{summary.orbit_count:,}; "
                f"N_eff={summary.effective_orbit_count:.1f}; "
                f"max={100.0 * summary.maximum_weight_fraction:.2f}%"
            ),
            transform=weight_axis.transAxes,
            ha="left",
            va="top",
            fontsize="small",
        )

    axes[0, 0].set_title("Active orbit count per bin")
    axes[0, 1].set_title("Total solved weight carried per bin")
    for axis in axes[-1]:
        axis.set_xlabel(r"Display-normalized weight  $\log_{10}(w_i / \sum_j w_j)$")
    figure.suptitle(
        title
        + "\n"
        + (
            r"Active means $w_i > 10^{-12}\max(w)$; normalization is for "
            "the plotted axis only"
        ),
        y=0.995,
    )
    figure.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.075,
        top=0.78,
        hspace=0.12,
        wspace=0.16,
    )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
