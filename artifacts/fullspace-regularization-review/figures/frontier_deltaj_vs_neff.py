#!/usr/bin/env python3
"""Velocity cost / gain against achieved weight concentration, per regularizer.

Every number comes from a run artifact: graph points from run 7ba8c44d's
scan.json, L2 from run a3d9dc34's log CSV, entropy (converged) from run
144acf4a's scan.json read before the shared path was overwritten, entropy
(SLSQP, non-converged) from run d8e30cd1's log. Single seed per point;
frontier lines are guides through measured settings, not interpolations.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import orx_figstyle as style

style.use_style()

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
ANCHOR_J = 139620.6  # mu=rho=lambda=0 anchor, identical in all four scans

rows = json.loads((DATA / "graph_scan.json").read_text())["rows"]
graph = {
    "neff": np.array([r["weight_n_eff"] for r in rows]),
    "dj": np.array([r["objective_velocity"] - ANCHOR_J for r in rows]),
}


def read_csv(name):
    with open(DATA / name, newline="") as handle:
        lines = [line for line in handle if not line.startswith("#")]
    return list(csv.DictReader(lines))


l2_rows = read_csv("l2_scan.csv")
l2 = {
    "neff": np.array([float(r["N_eff"]) for r in l2_rows]),
    "dj": np.array([float(r["delta_J"]) for r in l2_rows]),
}
ent_rows = read_csv("entropy_newton_scan.csv")
entropy = {
    "neff": np.array([float(r["N_eff"]) for r in ent_rows]),
    "dj": np.array([float(r["velocity_J"]) - ANCHOR_J for r in ent_rows]),
}
slsqp_rows = read_csv("entropy_slsqp_scan.csv")
slsqp = {
    "neff": np.array([float(r["N_eff"]) for r in slsqp_rows]),
    "dj": np.array([float(r["velocity_J"]) - ANCHOR_J for r in slsqp_rows]),
}

fig, ax = style.figure(width=style.TEXT, ratio=0.62)
ax.axhline(0.0, color=style.BASELINE, lw=style.BASELINE and 0.8, zorder=1)

series = [
    ("L2 ridge (NNLS-exact, 9 settings)", l2, style.PALETTE["blue"], "o", "-"),
    ("Graph smoothing (NNLS-exact, 10 settings)", graph, style.PALETTE["orange"], "s", "-"),
    ("Entropy, converged Newton (9 settings)", entropy, style.PALETTE["red"], "^", "-"),
    ("Entropy, SLSQP parent run (non-converged)", slsqp, style.PALETTE["black"], "v", (0, (4, 3))),
]
for label, data, color, marker, dashes in series:
    ax.plot(
        data["neff"], data["dj"], marker=marker, ms=4.5, lw=1.4, color=color,
        linestyle=dashes, markerfacecolor="none" if dashes != "-" else color,
        markeredgewidth=1.2, label=label, zorder=3,
    )

ax.plot([24.8], [0.0], marker="*", ms=11, color=style.PALETTE["black"],
        linestyle="none", label="shared anchor ($\\mu=\\rho=\\lambda=0$)", zorder=4)
ax.annotate("anchor: $N_{\\mathrm{eff}}=24.8$, max share 17.9%",
            xy=(24.8, 0.0), xytext=(30, 520), fontsize=7.5,
            arrowprops=dict(arrowstyle="-", color=style.BASELINE, lw=0.7))

ax.annotate("density gate\nfails at every point\n(shells not shown)",
            xy=(70, -1500), xytext=(78, -450), fontsize=7.5, color=style.BASELINE,
            arrowprops=dict(arrowstyle="-", color=style.BASELINE, lw=0.7))

ax.set_xlabel("effective orbit count $N_{\\mathrm{eff}}$ (higher = less concentrated)")
ax.set_ylabel("$\\Delta J$ vs anchor (lower is better)")
ax.set_xlim(10, 100)
ax.set_ylim(-2100, 4000)
ax.legend(fontsize=7.5, loc="upper right", frameon=False)

style.save(fig, str(HERE / "frontier-deltaj-neff"))
