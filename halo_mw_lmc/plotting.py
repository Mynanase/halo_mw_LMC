"""Diagnostic plots for the azimuth-resolved density comparison."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .density import DensityComparison


def plot_density_comparison(
    comparison: DensityComparison,
    output: str | Path,
) -> None:
    """Write data/model/residual panels with one column per phi bin."""

    import matplotlib.pyplot as plt

    grid = comparison.grid
    nphi = grid.shape[-1]
    width = max(3.0 * nphi, 8.0)
    figure, axes = plt.subplots(
        3,
        nphi,
        figsize=(width, 9.0),
        squeeze=False,
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    extent = [
        grid.r_edges[0],
        grid.r_edges[-1],
        grid.z_edges[0],
        grid.z_edges[-1],
    ]
    positive = np.concatenate(
        (
            comparison.data_density[comparison.data_density > 0],
            comparison.model_density[comparison.model_density > 0],
        )
    )
    if positive.size:
        log_min, log_max = np.nanpercentile(np.log10(positive), [2, 98])
    else:
        log_min, log_max = -1.0, 1.0

    density_images = []
    residual_image = None
    for iphi in range(nphi):
        phi_lo, phi_hi = np.rad2deg(grid.phi_edges[iphi : iphi + 2])
        axes[0, iphi].set_title(f"{phi_lo:.0f}° ≤ φ < {phi_hi:.0f}°")
        for row, density in enumerate(
            (comparison.data_density, comparison.model_density)
        ):
            display = np.where(density[:, :, iphi] > 0, np.log10(density[:, :, iphi]), np.nan)
            image = axes[row, iphi].imshow(
                display.T,
                origin="lower",
                extent=extent,
                aspect="auto",
                cmap="viridis",
                vmin=log_min,
                vmax=log_max,
            )
            density_images.append(image)
        residual_image = axes[2, iphi].imshow(
            comparison.residual[:, :, iphi].T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="coolwarm",
            vmin=-3,
            vmax=3,
        )
        axes[2, iphi].set_xlabel("R [kpc]")

    for row, label in enumerate(("data log₁₀ ν", "model log₁₀ ν", "(data-model)/σ")):
        axes[row, 0].set_ylabel(f"{label}\nz [kpc]")
    figure.colorbar(density_images[0], ax=axes[:2, :], label="log₁₀ ν")
    if residual_image is not None:
        figure.colorbar(residual_image, ax=axes[2, :], label="standardized residual")
    figure.suptitle(
        f"global scale={comparison.scale:.5g}, χ²={comparison.chi2:.3f}"
    )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)
