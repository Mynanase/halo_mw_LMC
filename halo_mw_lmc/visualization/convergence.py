"""Convergence figures built from persisted scalar optimizer samples."""

from __future__ import annotations

import numpy as np


def build_convergence_figure(samples: np.ndarray):
    """Return a Matplotlib figure without reading data or writing a file."""

    import matplotlib.pyplot as plt

    names = set(samples.dtype.names or ())
    missing = sorted({"iteration", "objective"} - names)
    if missing:
        raise ValueError("sample table is missing columns: " + ", ".join(missing))
    iteration = np.asarray(samples["iteration"], dtype=float)
    objective = np.asarray(samples["objective"], dtype=float)
    finite = np.isfinite(iteration) & np.isfinite(objective)
    iteration = iteration[finite]
    objective = objective[finite]
    if objective.size == 0:
        raise ValueError("sample table contains no finite objective values")
    best_so_far = np.minimum.accumulate(objective)

    figure, axis = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    axis.scatter(iteration, objective, s=6, alpha=0.35, label="trial objective")
    axis.plot(
        iteration,
        best_so_far,
        color="red",
        linewidth=1.5,
        label="best-so-far",
    )
    best_iteration = int(iteration[np.argmin(objective)])
    axis.axvline(best_iteration, color="0.5", linestyle="--", alpha=0.7)
    axis.set_xlabel("iteration")
    axis.set_ylabel("selected objective")
    axis.set_yscale("log" if np.all(objective > 0) else "symlog")
    axis.grid(alpha=0.2)
    axis.legend(loc="upper right")
    axis.set_title(
        f"Optimization trajectory ({objective.size} trials; "
        f"best at iter {best_iteration})"
    )
    return figure
