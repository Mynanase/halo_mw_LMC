#!/usr/bin/env python3
"""Compare the paired fixed-point 8--40 kpc tolerance-ranking runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from halo_mw_lmc.artifacts import load_sample_table


COORDINATE_COLUMNS = (
    "qhalo",
    "phalo",
    "rho0",
    "rho0_plus_2logrs",
    "gamma",
)
POINT_LABELS = (
    "paper_best",
    "flatter_more_triaxial",
    "rounder",
    "more_concentrated",
    "more_extended",
)
RUN_NAMES = {
    "tol1e7": "density-solved-r8-40-potential-ranking-tol1e7",
    "tol1e8": "density-solved-r8-40-potential-ranking-tol1e8",
}


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(order.size, dtype=int)
    ranks[order] = np.arange(order.size)
    return ranks


def _pairwise_order_agreement(left: np.ndarray, right: np.ndarray) -> float:
    agreements = []
    for first in range(left.size):
        for second in range(first + 1, left.size):
            left_sign = np.sign(left[first] - left[second])
            right_sign = np.sign(right[first] - right[second])
            agreements.append(left_sign == right_sign)
    return float(np.mean(agreements)) if agreements else 1.0


def _load_run(run: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    required = (
        "iteration",
        *COORDINATE_COLUMNS,
        "objective",
        "objective_velocity",
        "density_shell_phi_gate_passed",
        "weight_solver_converged",
        "failed_orbits",
    )
    samples = load_sample_table(run / "sample.dat", required_columns=required)
    if samples.size != len(POINT_LABELS):
        raise ValueError(
            f"expected {len(POINT_LABELS)} fixed points in {run}, got {samples.size}"
        )
    iterations = np.asarray(samples["iteration"], dtype=int)
    if not np.array_equal(iterations, np.arange(len(POINT_LABELS))):
        raise ValueError(f"unexpected iteration order in {run}")
    coordinates = np.column_stack(
        [np.asarray(samples[name], dtype=float) for name in COORDINATE_COLUMNS]
    )
    resolved = json.loads((run / "resolved_config.json").read_text())
    optimizer = resolved.get("optimizer", {})
    configured_points = np.asarray(optimizer.get("fixed_points"), dtype=float)
    if optimizer.get("schedule") != "fixed_points":
        raise ValueError(f"run does not record a fixed-point schedule: {run}")
    if configured_points.shape != coordinates.shape or not np.array_equal(
        configured_points,
        coordinates,
    ):
        raise ValueError(f"sample coordinates do not match resolved fixed points: {run}")
    objective = np.asarray(samples["objective_velocity"], dtype=float)
    finite = np.isfinite(objective) & (objective < 1e30)
    gates = np.asarray(samples["density_shell_phi_gate_passed"], dtype=int) == 1
    converged = np.asarray(samples["weight_solver_converged"], dtype=int) == 1
    no_failed_orbits = np.asarray(samples["failed_orbits"], dtype=int) == 0
    valid = finite & gates & converged & no_failed_orbits
    metadata = {
        "run_directory": str(run),
        "git_commit": resolved.get("git_commit"),
        "git_dirty": resolved.get("git_dirty"),
        "lsmr_tol": resolved.get("weight_model", {}).get("lsmr_tol"),
        "input_sha256": (
            run / "benchmark_metadata/input-sha256.txt"
        ).read_text(),
        "all_points_valid": bool(np.all(valid)),
        "valid_by_point": valid.tolist(),
    }
    return coordinates, objective, metadata


def compare_runs(runs_root: str | Path) -> dict[str, object]:
    root = Path(runs_root).expanduser().resolve()
    left_coordinates, left, left_metadata = _load_run(root / RUN_NAMES["tol1e7"])
    right_coordinates, right, right_metadata = _load_run(root / RUN_NAMES["tol1e8"])
    if left_metadata["lsmr_tol"] != 1e-7 or right_metadata["lsmr_tol"] != 1e-8:
        raise ValueError("paired runs do not record the expected LSMR tolerances")
    if not np.array_equal(left_coordinates, right_coordinates):
        raise ValueError("paired runs did not evaluate identical potential coordinates")

    left_rank = _rank(left)
    right_rank = _rank(right)
    spearman = float(np.corrcoef(left_rank, right_rank)[0, 1])
    pairwise = _pairwise_order_agreement(left, right)
    absolute_offset = right - left
    differential_shift = absolute_offset - absolute_offset[0]
    left_span = float(np.ptp(left))
    right_span = float(np.ptp(right))
    comparison_span = max(left_span, right_span)
    max_differential_shift = float(np.max(np.abs(differential_shift)))
    shift_fraction = (
        max_differential_shift / comparison_span
        if comparison_span > 0
        else float("inf")
    )
    same_best = int(np.argmin(left)) == int(np.argmin(right))
    all_valid = bool(
        left_metadata["all_points_valid"] and right_metadata["all_points_valid"]
    )
    criteria = {
        "all_points_valid": all_valid,
        "same_best_point": same_best,
        "spearman_rank_correlation": spearman,
        "spearman_minimum": 0.9,
        "pairwise_order_agreement": pairwise,
        "pairwise_minimum": 0.9,
        "max_differential_shift_fraction_of_span": shift_fraction,
        "shift_fraction_maximum": 0.1,
    }
    criteria["ranking_stable"] = bool(
        all_valid
        and same_best
        and spearman >= 0.9
        and pairwise >= 0.9
        and shift_fraction <= 0.1
    )

    points = []
    for index, label in enumerate(POINT_LABELS):
        points.append(
            {
                "index": index,
                "label": label,
                "coordinates": {
                    name: float(value)
                    for name, value in zip(
                        COORDINATE_COLUMNS,
                        left_coordinates[index],
                    )
                },
                "objective_tol1e7": float(left[index]),
                "objective_tol1e8": float(right[index]),
                "absolute_tolerance_offset": float(absolute_offset[index]),
                "differential_shift_from_paper_best": float(
                    differential_shift[index]
                ),
                "rank_tol1e7": int(left_rank[index]),
                "rank_tol1e8": int(right_rank[index]),
            }
        )
    return {
        "schema_version": 1,
        "runs": {"tol1e7": left_metadata, "tol1e8": right_metadata},
        "points": points,
        "objective_span": {"tol1e7": left_span, "tol1e8": right_span},
        "criteria": criteria,
        "provenance": {
            "same_git_head": (
                left_metadata["git_commit"] == right_metadata["git_commit"]
            ),
            "same_input_hashes": (
                left_metadata["input_sha256"]
                == right_metadata["input_sha256"]
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs_root", nargs="?", default="runs")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    document = compare_runs(arguments.runs_root)
    rendered = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(rendered, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered)
        print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
