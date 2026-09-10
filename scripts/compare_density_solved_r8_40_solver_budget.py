#!/usr/bin/env python3
"""Read-only comparator for saved density-solved solver-budget artifacts.

Usage:
    python scripts/compare_density_solved_r8_40_solver_budget.py CONFIG --phase PHASE

This script only reads files written by the benchmark runner.  It never
imports or calls AGAMA and never integrates orbits or solves weights: missing
evidence is reported as an error instead of being recomputed.  Step 1 verifies
that each case of the requested phase carries its evidence manifest; the
numeric gate evaluation of docs/solver_budget_experiment.md ("预先固定的数值
门槛" and the step-5 selection rules) is added in the artifact step.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from halo_mw_lmc.workflows.solver_budget import (  # noqa: E402
    SOLVER_BUDGET_PHASES,
    load_solver_budget_plan,
)


CASE_MANIFEST_NAME = "case.json"
CASE_ARTIFACT_NAMES = (
    "case.json",
    "resolved_config.json",
    "best/metadata.json",
    "best/evaluation.npz",
    "stdout.log",
    "stderr.log",
    "time-v.txt",
)


def compare_solver_budget_phase(config_path: str | Path, phase: str) -> dict[str, object]:
    """List saved artifacts for one phase and refuse to proceed without them."""

    if phase not in SOLVER_BUDGET_PHASES or phase == "preflight":
        raise ValueError(
            f"phase {phase!r} has no comparison evidence; expected one of "
            + ", ".join(
                name for name in SOLVER_BUDGET_PHASES if name != "preflight"
            )
        )
    plan = load_solver_budget_plan(config_path)
    phase_root = plan.output_root / phase
    if not phase_root.is_dir():
        raise RuntimeError(f"no saved artifacts for phase {phase!r}: {phase_root}")
    cases = sorted(path for path in phase_root.iterdir() if path.is_dir())
    if not cases:
        raise RuntimeError(f"phase {phase!r} has no case directories: {phase_root}")
    missing_manifests = [
        str(case / CASE_MANIFEST_NAME)
        for case in cases
        if not (case / CASE_MANIFEST_NAME).is_file()
    ]
    if missing_manifests:
        raise RuntimeError(
            "missing case evidence: " + ", ".join(missing_manifests)
        )
    return {
        "phase": phase,
        "config_path": str(plan.source_path),
        "output_root": str(plan.output_root),
        "phase_root": str(phase_root),
        "cases": [
            {
                "case": case.name,
                "path": str(case),
                "artifacts": [
                    name
                    for name in CASE_ARTIFACT_NAMES
                    if (case / name).is_file()
                ],
            }
            for case in cases
        ],
        # Numeric gate evaluation (reference qualification, objective and
        # prediction errors, ranking stability, repeat tolerance) is implemented
        # in the artifact step of docs/solver_budget_experiment.md.
        "gates_evaluated": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare saved solver-budget artifacts (read-only)"
    )
    parser.add_argument("config", help="solver-budget benchmark TOML")
    parser.add_argument(
        "--phase",
        required=True,
        choices=[name for name in SOLVER_BUDGET_PHASES if name != "preflight"],
    )
    arguments = parser.parse_args(argv)
    try:
        summary = compare_solver_budget_phase(arguments.config, arguments.phase)
    except (RuntimeError, ValueError) as exc:
        print(f"solver-budget comparison failed: {exc}", file=sys.stderr)
        return 1
    print(f"phase: {summary['phase']}")
    print(f"phase root: {summary['phase_root']}")
    for case in summary["cases"]:
        artifacts = ", ".join(case["artifacts"]) or "no artifacts"
        print(f"case: {case['case']} -> {artifacts}")
    print(f"gates evaluated: {summary['gates_evaluated']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
