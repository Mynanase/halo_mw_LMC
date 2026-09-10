#!/usr/bin/env python3
"""Thin entry point for the density-solved solver-budget experiment phases.

Usage:
    python scripts/benchmark_density_solved_r8_40_solver_budget.py CONFIG \
        --phase PHASE [--baseline-ref GIT_REF]

Only ``--phase preflight`` runs today; the numerical phases refuse to start
until their contract step in docs/solver_budget_experiment.md is implemented.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from halo_mw_lmc.workflows.solver_budget import (  # noqa: E402
    SOLVER_BUDGET_PHASES,
    run_solver_budget_phase,
)


def _print_preflight(summary: dict) -> None:
    print(f"phase: {summary['phase']}")
    print(f"benchmark: {summary['benchmark_id']}")
    print(f"config: {summary['config_path']}")
    print(f"recipe: {summary['recipe_path']}")
    print(f"output root: {summary['output_root']}")
    for point in summary["points"]:
        coordinates = " ".join(f"{value:g}" for value in point["coordinates"])
        print(f"point: {point['name']} = [{coordinates}] ({point['source']})")
    for method in summary["methods"]:
        tolerance = (
            f"lsmr_tol={method['lsmr_tol']:g}"
            if method["lsmr_tol"] is not None
            else f"solver_tolerance={method['solver_tolerance']:g}"
        )
        print(
            f"method: {method['name']} solver={method['solver']} "
            f"max_iter={method['max_iter']} {tolerance}"
        )
    levels = ", ".join(str(level) for level in summary["budget_max_iter"])
    print(f"budget curve max_iter: {levels}")
    print(f"timeout: {summary['timeout_seconds']:g} s, repeats: {summary['repeats']}")
    threads = " ".join(
        f"{entry['variable']}={entry['value']}" for entry in summary["threads"]
    )
    print(f"threads: {threads}")
    for entry in summary["inputs"]:
        print(
            f"input: {entry['label']} {entry['path']} "
            f"({entry['bytes']} bytes, sha256={entry['sha256']})"
        )
    print(f"gnu time: {summary['gnu_time']}")
    provenance = summary["provenance"]
    state = "dirty" if provenance["dirty"] else "clean"
    print(f"git: {provenance['head']} ({state})")
    for warning in summary["warnings"]:
        print(f"warning: {warning}")
    print("preflight: PASS")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run one phase of the density-solved solver-budget experiment "
            "(docs/solver_budget_experiment.md)"
        )
    )
    parser.add_argument("config", help="solver-budget benchmark TOML")
    parser.add_argument("--phase", required=True, choices=SOLVER_BUDGET_PHASES)
    parser.add_argument(
        "--baseline-ref",
        default=None,
        help="git ref of the pre-merge implementation (parity phase only)",
    )
    arguments = parser.parse_args(argv)
    try:
        summary = run_solver_budget_phase(
            arguments.config,
            arguments.phase,
            baseline_ref=arguments.baseline_ref,
        )
    except NotImplementedError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (RuntimeError, ValueError) as exc:
        print(f"solver-budget {arguments.phase} failed: {exc}", file=sys.stderr)
        return 1
    _print_preflight(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
