#!/usr/bin/env python3
"""Real orbit-weight vectors from the exact graph-smoothing scan (run 7ba8c44d).

(a) sorted per-orbit weights at three settings -- the anchor and both ends of
the concentration transition; (b) Lorenz curves of the same solutions with
the measured $N_{\\mathrm{eff}}$ marked where cumulative share crosses the
diagonal-based inverse. Weights are the archived seed_weights.npz of the run
(survived on the shared .agent-local path; superseded runs' weight files were
overwritten before this report).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import orx_figstyle as style

style.use_style()

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"

archive = np.load(DATA / "graph_seed_weights.npz")
rows = json.loads((DATA / "graph_scan.json").read_text())["rows"]
by_strength = {float(r["strength"]): r for r in rows}

PICKS = [("graph_0", "$\\rho=0$ (anchor)"), ("graph_0.0001", "$\\rho=10^{-4}$"), ("graph_0.01", "$\\rho=10^{-2}$")]
COLORS = [style.PALETTE["black"], style.PALETTE["cyan"], style.PALETTE["orange"]]

fig, axes = style.figure_grid(1, 2, width=style.TEXT, ratio=0.42, sharey=False)
ax_a, ax_b = axes

for (key, label), color in zip(PICKS, COLORS):
    weights = np.sort(archive[key])[::-1]
    weights = weights[weights > 0]
    rank = np.arange(1, weights.size + 1)
    ax_a.plot(rank, weights, lw=1.3, color=color,
              label=f"{label}: {weights.size} orbits, $N_{{\\mathrm{{eff}}}}$={by_strength[{'graph_0': 0.0, 'graph_0.0001': 1e-4, 'graph_0.01': 1e-2}[key]]['weight_n_eff']:.1f}")

ax_a.set_yscale("log")
ax_a.set_xlabel("orbit rank (heaviest first)")
ax_a.set_ylabel("orbit weight $w_j$")
ax_a.legend(fontsize=6.5, frameon=False)

for (key, label), color in zip(PICKS, COLORS):
    weights = np.sort(archive[key])[::-1]
    total = weights.sum()
    cumulative = np.cumsum(weights) / total
    orbits = np.arange(1, weights.size + 1) / weights.size
    ax_b.plot(orbits, cumulative, lw=1.3, color=color, label=label)
    neff = by_strength[{"graph_0": 0.0, "graph_0.0001": 1e-4, "graph_0.01": 1e-2}[key]]["weight_n_eff"]
    active_fraction = np.searchsorted(cumulative, 1.0 - 1.0 / neff) / cumulative.size
    ax_b.plot([active_fraction], [1.0 - 1.0 / neff], marker="o", ms=4,
              color=color, linestyle="none")

ax_b.plot([0, 1], [0, 1], color=style.BASELINE, lw=0.8, linestyle=":")
ax_b.text(0.32, 0.24, "uniform ($N_{\\mathrm{eff}}=n$)", fontsize=6.5, color=style.BASELINE, rotation=38)
ax_b.set_xlabel("fraction of orbits (heaviest first)")
ax_b.set_ylabel("cumulative weight share")
ax_b.set_xlim(0, 1)
ax_b.set_ylim(0, 1)

style.panel_labels([ax_a, ax_b])
style.save(fig, str(HERE / "weight-profiles"))
