#!/usr/bin/env python3
"""Compare the five saved 8--40 kpc one-factor benchmark cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np

from halo_mw_lmc.artifacts import best_sample, load_best_evaluation, load_run_summary


RUN_NAMES = {
    "baseline": "density-solved-r8-40-paper-best-benchmark",
    "tol-1e-7": "density-solved-r8-40-tol1e7-paper-best-benchmark",
    "tol-1e-8": "density-solved-r8-40-tol1e8-paper-best-benchmark",
    "reg-1e-5": "density-solved-r8-40-reg1e5-paper-best-benchmark",
    "reg-1e-4": "density-solved-r8-40-reg1e4-paper-best-benchmark",
}


def _normalized_weights(weights: np.ndarray) -> np.ndarray:
    values = np.asarray(weights, dtype=float)
    total = float(np.sum(values))
    return values / total if total > 0 else np.zeros_like(values)


def _relative_difference(left: float, right: float) -> float:
    return abs(left - right) / max(1.0, abs(left), abs(right))


def _time_metrics(run: Path) -> dict[str, object]:
    path = run / "benchmark_metadata" / "time-v.txt"
    if not path.is_file():
        return {"available": False, "wall_clock": None, "maximum_rss_kbytes": None}
    text = path.read_text()
    wall = re.search(
        r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(\S+)",
        text,
    )
    rss = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    return {
        "available": True,
        "wall_clock": wall.group(1) if wall else None,
        "maximum_rss_kbytes": int(rss.group(1)) if rss else None,
    }


def compare_runs(runs_root: str | Path) -> dict[str, object]:
    root = Path(runs_root).expanduser().resolve()
    results: dict[str, dict[str, object]] = {}
    weights: dict[str, np.ndarray] = {}
    shell_phi: dict[str, np.ndarray] = {}
    for case, name in RUN_NAMES.items():
        run = root / name
        summary = load_run_summary(run)
        stored = load_best_evaluation(run)
        row = best_sample(summary.samples)
        if stored.density_shells is None:
            raise ValueError(f"run lacks shell diagnostics: {run}")
        objective = float(row["objective"])
        weights[case] = _normalized_weights(stored.weight_solution.seed_weights)
        shell_phi[case] = stored.density_shells.chi2_per_bin_by_shell_phi
        weight_config = summary.config.get("weight_model", {})
        results[case] = {
            "run_directory": str(run),
            "git_commit": summary.config.get("git_commit"),
            "git_dirty": summary.config.get("git_dirty"),
            "objective": objective,
            "objective_velocity": float(row["objective_velocity"]),
            "density_chi2_per_bin": float(row["density_chi2_per_bin"]),
            "density_shell_phi_gate_passed": bool(
                stored.metadata.get("density_shell_phi_gate_passed")
            ),
            "weight_solver_converged": bool(stored.weight_solution.converged),
            "effective_orbit_count": float(
                stored.weight_solution.effective_orbit_count
            ),
            "maximum_weight_fraction": float(
                stored.weight_solution.maximum_weight_fraction
            ),
            "active_orbit_count": int(stored.weight_solution.active_orbit_count),
            "regularization_strength": weight_config.get("regularization_strength"),
            "lsmr_tol": weight_config.get("lsmr_tol"),
            "performance": _time_metrics(run),
        }

    baseline = weights["baseline"]
    for case in results:
        if weights[case].shape != baseline.shape:
            raise ValueError(f"weight vector shape mismatch for {case}")
        results[case]["normalized_weight_l1_from_baseline"] = float(
            np.sum(np.abs(weights[case] - baseline))
        )

    left = "tol-1e-7"
    right = "tol-1e-8"
    objective_delta = _relative_difference(
        float(results[left]["objective"]),
        float(results[right]["objective"]),
    )
    shell_delta = float(np.max(np.abs(shell_phi[left] - shell_phi[right])))
    weight_delta = float(np.sum(np.abs(weights[left] - weights[right])))
    stability = {
        "objective_relative_difference": objective_delta,
        "objective_threshold": 1e-5,
        "shell_phi_max_absolute_difference": shell_delta,
        "shell_phi_threshold": 1e-3,
        "normalized_weight_l1_difference": weight_delta,
        "normalized_weight_l1_threshold": 1e-2,
        "passed": bool(
            objective_delta <= 1e-5
            and shell_delta <= 1e-3
            and weight_delta <= 1e-2
        ),
    }
    commits = {result["git_commit"] for result in results.values()}
    reproducible = len(commits) == 1 and all(
        result["git_dirty"] is False for result in results.values()
    )
    return {
        "cases": results,
        "tight_tolerance_stability": stability,
        "same_clean_commit": reproducible,
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
