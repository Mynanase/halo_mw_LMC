#!/usr/bin/env python3
"""Regularizer response panels: what each knob moves, and what it cannot.

(a) concentration response, (b) total weight mass (the entropy degeneracy),
(c) velocity-objective change, (d) the scored gate metric versus its limit.
Weight_sum exists for graph (scan.json) and converged entropy (scan.json read
before overwrite); the L2 log did not record it, so it is absent in (b).
Source runs: graph 7ba8c44d, L2 a3d9dc34, entropy 144acf4a (converged) and
d8e30cd1 (SLSQP, non-converged). Single seed per point.
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
ANCHOR_J = 139620.6
GATE_LIMIT = 2.0  # configs/recipes/zhu_2026_density_solved.toml
R840_SCORED = 0.4217  # run a7092dee, r8-40 pipeline, same synthetic target

rows = json.loads((DATA / "graph_scan.json").read_text())["rows"]
graph = {
    "strength": np.array([r["strength"] for r in rows]),
    "neff": np.array([r["weight_n_eff"] for r in rows]),
    "wsum": np.array([r["weight_sum"] for r in rows]),
    "dj": np.array([r["objective_velocity"] - ANCHOR_J for r in rows]),
    "scored": np.array([r["density_chi2_per_bin_scored"] for r in rows]),
}


def read_csv(name):
    with open(DATA / name, newline="") as handle:
        lines = [line for line in handle if not line.startswith("#")]
    return list(csv.DictReader(lines))


l2_rows = read_csv("l2_scan.csv")
l2 = {
    "strength": np.array([float(r["lambda"]) for r in l2_rows]),
    "neff": np.array([float(r["N_eff"]) for r in l2_rows]),
    "dj": np.array([float(r["delta_J"]) for r in l2_rows]),
    "scored": np.array([float(r["chi2_per_bin_scored"]) for r in l2_rows]),
}
ent_rows = read_csv("entropy_newton_scan.csv")
entropy = {
    "strength": np.array([float(r["mu"]) for r in ent_rows]),
    "neff": np.array([float(r["N_eff"]) for r in ent_rows]),
    "wsum": np.array([float(r["weight_sum"]) for r in ent_rows]),
    "dj": np.array([float(r["velocity_J"]) - ANCHOR_J for r in ent_rows]),
    "scored": np.full(len(ent_rows), 99.9799),
}
slsqp_rows = read_csv("entropy_slsqp_scan.csv")
slsqp_neff = {
    "strength": np.array([float(r["mu"]) for r in slsqp_rows]),
    "neff": np.array([float(r["N_eff"]) for r in slsqp_rows]),
}

fig, axes = style.figure_grid(2, 2, width=style.TEXT, ratio=0.55)
(ax_a, ax_b), (ax_c, ax_d) = axes

ax_a.plot(l2["strength"][1:], l2["neff"][1:], marker="o", ms=3.5, lw=1.3,
          color=style.PALETTE["blue"], label="L2 ridge")
ax_a.plot(graph["strength"][1:], graph["neff"][1:], marker="s", ms=3.5, lw=1.3,
          color=style.PALETTE["orange"], label="graph smoothing")
ax_a.plot(entropy["strength"][1:], entropy["neff"][1:], marker="^", ms=3.5, lw=1.3,
          color=style.PALETTE["red"], label="entropy (converged)")
ax_a.plot(slsqp_neff["strength"][1:], slsqp_neff["neff"][1:], marker="v", ms=3.5, lw=1.1,
          color=style.PALETTE["black"], linestyle=(0, (4, 3)),
          markerfacecolor="none", label="entropy (SLSQP, non-conv.)")
ax_a.set_xscale("log")
ax_a.set_xlim(7e-7, 5e2)
ax_a.set_ylabel("$N_{\\mathrm{eff}}$")
ax_a.set_xlabel("regularizer strength ($\\lambda$, $\\rho$, or $\\mu$)")

ax_b.plot(graph["strength"][1:], graph["wsum"][1:], marker="s", ms=3.5, lw=1.3,
          color=style.PALETTE["orange"], label="graph smoothing")
ax_b.plot(entropy["strength"][1:], entropy["wsum"][1:], marker="^", ms=3.5, lw=1.3,
          color=style.PALETTE["red"], label="entropy (converged)")
ax_b.axhline(7.575, color=style.BASELINE, lw=0.8, linestyle=":")
ax_b.text(2e-6, 7.75, "anchor mass 7.575", fontsize=6.5, color=style.BASELINE)
ax_b.set_xscale("log")
ax_b.set_xlim(7e-7, 5e2)
ax_b.set_ylabel("total weight")
ax_b.yaxis.set_major_locator(plt.MaxNLocator(4))
ax_b.set_xlabel("regularizer strength ($\\lambda$, $\\rho$, or $\\mu$)")

ax_c.plot(l2["strength"][1:], l2["dj"][1:], marker="o", ms=3.5, lw=1.3,
          color=style.PALETTE["blue"], label="L2 ridge")
ax_c.plot(graph["strength"][1:], graph["dj"][1:], marker="s", ms=3.5, lw=1.3,
          color=style.PALETTE["orange"], label="graph smoothing")
ax_c.plot(entropy["strength"][1:], entropy["dj"][1:], marker="^", ms=3.5, lw=1.3,
          color=style.PALETTE["red"], label="entropy (converged)")
ax_c.axhline(0.0, color=style.BASELINE, lw=0.8)
handles, labels = ax_c.get_legend_handles_labels()
ax_c.set_xscale("log")
ax_c.set_xlim(7e-7, 5e2)
ax_c.set_ylabel("$\\Delta J$ vs anchor")
ax_c.set_xlabel("regularizer strength ($\\lambda$, $\\rho$, or $\\mu$)")

for data, color, marker in (
    (l2, style.PALETTE["blue"], "o"),
    (graph, style.PALETTE["orange"], "s"),
    (entropy, style.PALETTE["red"], "^"),
):
    ax_d.plot(data["strength"][1:], data["scored"][1:], marker=marker, ms=3.5, lw=1.3,
              color=color, label=None)
ax_d.axhline(GATE_LIMIT, color=style.PALETTE["black"], lw=1.0)
ax_d.text(2e-6, GATE_LIMIT * 1.35, "gate limit 2.0", fontsize=6.5)
ax_d.plot([1.0], [R840_SCORED], marker="*", ms=10, color=style.PALETTE["green"],
          linestyle="none")
ax_d.text(1.35, R840_SCORED, "r8-40 pipeline\n(same target, 0.42)", fontsize=6.5,
          va="center")
ax_d.set_xscale("log")
ax_d.set_xlim(7e-7, 5e2)
ax_d.set_yscale("log")
ax_d.set_ylim(0.3, 400)
ax_d.yaxis.set_minor_locator(plt.NullLocator())
ax_d.set_xlabel("regularizer strength ($\\lambda$, $\\rho$, or $\\mu$)")
ax_d.set_ylabel("scored $\\chi^2$/bin (gate metric)")

style.panel_labels([ax_a, ax_b, ax_c, ax_d])
fig.legend(handles, labels, loc="outside lower center", ncols=4, fontsize=6.5, frameon=False)
fig.get_layout_engine().set(w_pad=0.12, h_pad=0.12)
style.save(fig, str(HERE / "response-panels"))
