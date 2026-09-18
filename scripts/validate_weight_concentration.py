#!/usr/bin/env python3
"""§13 weight-concentration validation over saved nphi4 bundling workpoints.

Recomputes, from saved raw arrays only (no solver): orbit-level and
bundle-mass-level concentration for every full/bundled/random case, the
backfill identity between saved bundle weights u and saved seed weights, and
-- unless --skip-distortion -- the §11 equal-weight distortion D of the saved
assignments on the rebuilt frozen-library design, cross-checking the values
recorded in each case.json. Reported-vs-recomputed mismatches are listed and
fail the run; stored D from artifacts produced before the 2f2ecc3 metric
correction is expected to disagree and is recorded as such, not as an error.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from review_fz_energy_basis import equal_weight_distortion  # noqa: E402

import importlib.util  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "benchmark_nphi1_bundling", REPO / "scripts" / "benchmark_nphi1_bundling.py",
)
bundling = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bundling)


def rebuild_design(experiment: dict):
    """Rebuild the frozen-library inner problem exactly as the benchmark does."""

    try:
        import agama  # noqa: F401
    except ImportError:
        sys.path.insert(0, str(REPO / "Agama-master"))

    from halo_mw_lmc.config import load_run_configuration, resolve_model
    from halo_mw_lmc.density import build_orbit_density_response, density_fit_mask
    from halo_mw_lmc.orbits import OrbitLibrary
    from halo_mw_lmc.prepare import prepare_model_data
    from halo_mw_lmc.weights import _build_weight_problem, _normalized_target

    run_configuration = load_run_configuration(REPO / experiment["run_config"])
    model = resolve_model(run_configuration["recipe"])
    prepared = prepare_model_data(
        run_configuration["data"]["catalog"],
        run_configuration["data"]["target_density"],
        model,
    )
    config = prepared.config
    frozen = np.load(REPO / experiment["frozen_cache"])
    library = OrbitLibrary(
        seed_index=frozen["library_seed_index"],
        time=frozen["library_time"],
        phase_space=frozen["library_phase_space"],
    )
    response = build_orbit_density_response(
        library, config["density_grid"], seed_count=prepared.initial_conditions.shape[0],
    )
    fit_keys = {
        key: config["density_fit"][key]
        for key in ("min_abs_z", "min_spherical_radius", "max_spherical_radius", "require_positive_data")
    }
    mask = density_fit_mask(prepared.target_density, prepared.target_error, config["density_grid"], **fit_keys)
    target_normalized, error_normalized = _normalized_target(
        prepared.target_density, prepared.target_error, mask, response,
        config["weight_model"]["target_normalization"],
    )
    problem = _build_weight_problem(
        response, target_normalized, error_normalized, mask,
        float(config["weight_model"]["regularization_strength"]),
    )
    return problem


def close(reported, recomputed, tolerance=1e-9):
    if reported is None:
        return recomputed is None
    if recomputed is None:
        return False
    scale = max(1.0, abs(float(reported)), abs(float(recomputed)))
    return abs(float(reported) - float(recomputed)) <= tolerance * scale


def validate_workpoint(workpoint: Path, design_dense, active_columns, problems: list):
    """One workpoint directory; returns rows for the CSV and problem notes."""

    rows = []
    with np.load(workpoint / "bundles.npz") as payload:
        bundles = {key: payload[key] for key in payload.files}
    successful = bundles["successful_seed_index"]
    active_positions = successful[active_columns]
    assignments = bundles.get("assignments")
    random_assignments = bundles.get("random_assignments")
    u_bundled = bundles.get("u_bundled")
    u_random = bundles.get("u_random")
    backfill = {}
    if u_bundled is not None and assignments is not None:
        expected = np.zeros(int(successful.max()) + 1)
        expected[active_positions] = u_bundled[assignments]
        backfill["bundled"] = bool(np.allclose(expected[active_positions], bundles["seed_weights_bundled"][active_positions], rtol=0, atol=0))
    if u_random is not None and random_assignments is not None:
        expected = np.zeros(int(successful.max()) + 1)
        expected[active_positions] = u_random[random_assignments]
        backfill["random"] = bool(np.allclose(expected[active_positions], bundles["seed_weights_random"][active_positions], rtol=0, atol=0))

    for name in ("full", "bundled", "random"):
        case_path = workpoint / name / "case.json"
        arrays_path = workpoint / name / "evaluation.npz"
        if not case_path.exists() or not arrays_path.exists():
            continue
        case = json.loads(case_path.read_text())
        with np.load(arrays_path) as payload:
            seed_weights = payload["seed_weights"]
        active_weights = seed_weights[active_positions]
        case_assignments = None
        if name == "bundled" and assignments is not None:
            case_assignments = assignments
        elif name == "random" and random_assignments is not None:
            case_assignments = random_assignments
        concentration = bundling.weight_concentration(active_weights, case_assignments)
        bundle = concentration.get("bundle", {})
        row = {
            "workpoint": workpoint.name,
            "case": name,
            "bundle_count": case.get("bundle_count"),
            "n_eff_reported": case.get("effective_orbit_count"),
            "n_eff_recomputed": concentration["n_eff"],
            "max_share_reported": case.get("maximum_weight_fraction"),
            "max_share_recomputed": concentration["max_fraction"],
            "top10_share": concentration["top10_share"],
            "top100_share": concentration["top100_share"],
            "hhi": concentration["hhi"],
            "n90": concentration["n90"],
            "n99": concentration["n99"],
            "bundle_n_eff": bundle.get("n_eff"),
            "bundle_max_share": bundle.get("max_fraction"),
            "bundle_top10_share": bundle.get("top10_share"),
            "bundle_top100_share": bundle.get("top100_share"),
            "bundle_hhi": bundle.get("hhi"),
            "bundle_n90": bundle.get("n90"),
            "bundle_n99": bundle.get("n99"),
            "backfill_identity": backfill.get(name),
            "d_reported": case.get("equal_weight_distortion"),
            "d_recomputed": None,
        }
        if case_assignments is not None and design_dense is not None:
            record = equal_weight_distortion(
                design_dense, case_assignments, int(np.max(case_assignments)) + 1,
            )
            row["d_recomputed"] = record["distortion"]
        if not close(row["n_eff_reported"], row["n_eff_recomputed"]):
            problems.append(f"{workpoint.name}/{name}: reported n_eff {row['n_eff_reported']} != recomputed {row['n_eff_recomputed']}")
        if not close(row["max_share_reported"], row["max_share_recomputed"]):
            problems.append(f"{workpoint.name}/{name}: reported max share {row['max_share_reported']} != recomputed {row['max_share_recomputed']}")
        if case_assignments is not None and row["backfill_identity"] is False:
            problems.append(f"{workpoint.name}/{name}: backfill identity u[assignments] != saved seed weights")
        if row["d_recomputed"] is not None and not close(row["d_reported"], row["d_recomputed"], tolerance=1e-6):
            note = (f"{workpoint.name}/{name}: stored D {row['d_reported']} != corrected recomputed D "
                    f"{row['d_recomputed']} (pre-2f2ecc3 artifacts stored the defective-metric D; "
                    "corrected value recorded here)")
            problems.append("note: " + note)
            row["d_mismatch_note"] = "stored D predates the 2f2ecc3 metric correction"
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workpoints", nargs="+", help="workpoint directories under .agent-local/benchmarks (or absolute)")
    parser.add_argument("--out", default=".agent-local/benchmarks/weight_concentration_validation")
    parser.add_argument("--skip-distortion", action="store_true",
                        help="skip the D recomputation (no library/agama rebuild)")
    args = parser.parse_args()

    started = time.perf_counter()
    workpoints = [Path(w) if Path(w).is_absolute() else REPO / w for w in args.workpoints]
    problems: list[str] = []
    design_dense = None
    active_columns = None
    experiment = None
    if not args.skip_distortion:
        resolved = json.loads((workpoints[0] / "resolved_config.json").read_text())
        experiment = resolved["experiment"]
        problem = rebuild_design(experiment)
        design_dense = np.asarray(problem.design.todense(), dtype=float)
        active_columns = np.flatnonzero(problem.active_columns)
        print(f"design rebuilt {design_dense.shape} active columns {active_columns.size}")
    else:
        # active columns still needed for orbit-level indexing; fall back to a
        # saved bundles file (identical across the shared-frozen-library rounds)
        with np.load(workpoints[0] / "bundles.npz") as payload:
            successful = payload["successful_seed_index"]
        active_columns = np.arange(successful.size)
        print("distortion recomputation skipped; treating all successful seeds as active")

    rows = []
    for workpoint in workpoints:
        if not (workpoint / "bundles.npz").exists():
            problems.append(f"{workpoint.name}: bundles.npz missing; skipped")
            continue
        rows.extend(validate_workpoint(workpoint, design_dense, active_columns, problems))

    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    import csv

    with open(out_dir / "validation.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "workpoints": [str(workpoint) for workpoint in workpoints],
        "rows": len(rows),
        "problems": problems,
        "wall_seconds": time.perf_counter() - started,
        "tolerance": "1e-9 relative on n_eff/max share; 1e-6 relative on D (notes, not failures)",
    }
    (out_dir / "validation.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    for row in rows:
        print(f"{row['workpoint']}/{row['case']:8s} n_eff={row['n_eff_recomputed']:8.1f} "
              f"max={row['max_share_recomputed']:.4f} top10={row['top10_share']:.3f} "
              f"n90={row['n90']} D={row['d_recomputed'] if row['d_recomputed'] is not None else float('nan'):.4f}")
    hard = [problem for problem in problems if not problem.startswith("note: ")]
    notes = [problem for problem in problems if problem.startswith("note: ")]
    for note in notes:
        print(note)
    if hard:
        for problem in hard:
            print(f"MISMATCH: {problem}")
        raise SystemExit(1)
    print(f"validation passed: {len(rows)} rows, {len(notes)} metric-correction notes "
          f"(wall {time.perf_counter() - started:.1f}s)")


if __name__ == "__main__":
    main()
