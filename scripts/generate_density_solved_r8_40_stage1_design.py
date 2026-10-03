#!/usr/bin/env python3
"""Generate the stage-1 fixed-point screening design (r8-40, joint objective).

Emits 48 five-dimensional points: 45 scrambled Sobol points (scipy qmc,
seed 0) in the screening box plus 3 anchor points (paper best, the 5-point
ranking best "more_extended", and the wide-scan best). The Sobol generator is
deterministic for a given scipy version; the committed shard configs are the
authoritative record of the design, so regeneration is convenience only.

Screening box (covers the convex hull of the three anchors with margin, and
is strictly inside the wide-scan bounds):

    qhalo              [0.80, 1.28]
    phalo              [0.70, 0.96]
    rho0               [5.55, 6.55]
    rho0_plus_2logrs   [9.25, 10.20]
    gamma              [0.70, 1.45]

Sharding: 12 shards x 4 points, round-robin so every shard spans the box.

Usage:
    python scripts/generate_density_solved_r8_40_stage1_design.py \
        [--points-per-shard 4] [--shards 12]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import qmc

REPOSITORY = Path(__file__).resolve().parent.parent

BOX = {
    "qhalo": (0.80, 1.28),
    "phalo": (0.70, 0.96),
    "rho0": (5.55, 6.55),
    "rho0_plus_2logrs": (9.25, 10.20),
    "gamma": (0.70, 1.45),
}
ANCHORS = {
    "paper_best": [0.920, 0.800, 6.200, 9.890, 1.000],
    "more_extended": [0.920, 0.800, 5.900, 10.050, 0.800],
    "wide_scan_best": [1.222, 0.895, 5.616, 9.354, 1.330],
}


def build_points(total_sobol: int) -> tuple[list[list[float]], list[str]]:
    # Generate the next power of two and slice: keeps Sobol balance properties
    # for the leading subset and avoids the scipy imbalance warning.
    power = 1 << int(np.ceil(np.log2(total_sobol)))
    sampler = qmc.Sobol(d=len(BOX), scramble=True, seed=0)
    unit = sampler.random(power)[:total_sobol]
    lower = np.array([value[0] for value in BOX.values()])
    upper = np.array([value[1] for value in BOX.values()])
    points = lower + unit * (upper - lower)
    points = np.round(points, 3)
    labels = [f"sobol_{index:03d}" for index in range(total_sobol)]
    all_points = [ANCHORS[name] for name in ANCHORS] + points.tolist()
    all_labels = list(ANCHORS) + labels
    return all_points, all_labels


def check_box(points: list[list[float]]) -> None:
    array = np.asarray(points)
    lower = np.array([value[0] for value in BOX.values()])
    upper = np.array([value[1] for value in BOX.values()])
    if not (np.all(array >= lower - 1e-9) and np.all(array <= upper + 1e-9)):
        raise SystemExit("design point outside the screening box")


def shard(points: list[list[float]], shards: int, per_shard: int) -> list[list[int]]:
    """Round-robin assignment so each shard spans the box."""
    order = []
    columns: list[list[int]] = [[] for _ in range(shards)]
    for index in range(len(points)):
        columns[index % shards].append(index)
    flat = [index for column in columns for index in column]
    for start in range(0, len(flat), per_shard):
        order.append(flat[start : start + per_shard])
    if any(len(group) != per_shard for group in order):
        raise SystemExit("point count is not divisible by shards x per-shard")
    return order


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=int, default=12)
    parser.add_argument("--points-per-shard", type=int, default=4)
    args = parser.parse_args()
    total = args.shards * args.points_per_shard
    if total <= len(ANCHORS):
        raise SystemExit("need more points than anchors")

    points, labels = build_points(total - len(ANCHORS))
    check_box(points)
    groups = shard(points, args.shards, args.points_per_shard)
    unique = {tuple(point) for point in points}
    if len(unique) != len(points):
        raise SystemExit("duplicate points in the generated design")

    for number, group in enumerate(groups, start=1):
        shard_id = f"{number:02d}"
        rows = []
        for position, index in enumerate(group):
            tail = "," if position < len(group) - 1 else ""
            rows.append(
                f"[{points[index][0]:.3f}, {points[index][1]:.3f}, "
                f"{points[index][2]:.3f}, {points[index][3]:.3f}, "
                f"{points[index][4]:.3f}],  # {labels[index]}{tail}"
            )
        print(f"# shard {shard_id}")
        print(f"iterations = {len(group)}")
        print("fixed_points = [\n  " + "\n  ".join(rows) + "\n]")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
