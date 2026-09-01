#!/usr/bin/env python3
"""Plot orbit-weight histograms for the five saved 8--40 kpc cases."""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
import tarfile

import numpy as np

from halo_mw_lmc.visualization.weights import plot_orbit_weight_histograms


REPOSITORY = Path(__file__).resolve().parents[1]
RUN_NAMES = {
    "baseline": "density-solved-r8-40-paper-best-benchmark",
    "tol = 1e-7": "density-solved-r8-40-tol1e7-paper-best-benchmark",
    "tol = 1e-8": "density-solved-r8-40-tol1e8-paper-best-benchmark",
    "reg = 1e-5": "density-solved-r8-40-reg1e5-paper-best-benchmark",
    "reg = 1e-4": "density-solved-r8-40-reg1e4-paper-best-benchmark",
}
DEFAULT_ARCHIVES = (
    REPOSITORY / "density-solved-r8-40-paper-best-benchmark.tar.gz",
    REPOSITORY / "density-solved-r8-40-sensitivity-benchmarks.tar.gz",
)


def _weights_from_npz(source) -> np.ndarray:
    with np.load(source, allow_pickle=False) as archive:
        if "weight_seed_weights" not in archive:
            raise ValueError("evaluation archive lacks weight_seed_weights")
        return np.asarray(archive["weight_seed_weights"], dtype=float).copy()


def _weights_from_tar(archive_path: Path, member_name: str) -> np.ndarray | None:
    with tarfile.open(archive_path, mode="r:gz") as archive:
        try:
            member = archive.getmember(member_name)
        except KeyError:
            return None
        stream = archive.extractfile(member)
        if stream is None:
            raise ValueError(f"cannot read {member_name} from {archive_path}")
        return _weights_from_npz(BytesIO(stream.read()))


def load_case_weights(
    runs_root: Path,
    archive_paths: tuple[Path, ...],
) -> dict[str, np.ndarray]:
    """Read each saved vector from an unpacked run or a result archive."""

    result: dict[str, np.ndarray] = {}
    for label, run_name in RUN_NAMES.items():
        evaluation = runs_root / run_name / "best" / "evaluation.npz"
        if evaluation.is_file():
            result[label] = _weights_from_npz(evaluation)
            continue
        member = f"runs/{run_name}/best/evaluation.npz"
        for archive_path in archive_paths:
            if not archive_path.is_file():
                continue
            weights = _weights_from_tar(archive_path, member)
            if weights is not None:
                result[label] = weights
                break
        else:
            checked = ", ".join(str(path) for path in archive_paths)
            raise FileNotFoundError(
                f"cannot find {evaluation} or {member} in archives: {checked}"
            )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=REPOSITORY / "runs",
        help="root containing unpacked run directories",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        action="append",
        help="result tar.gz to search; repeat for multiple archives",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPOSITORY
            / ".agent-local/figures/density_solved_r8_40_orbit_weights.png"
        ),
        help="PNG or PDF destination",
    )
    parser.add_argument("--bins", type=int, default=32)
    arguments = parser.parse_args()
    archives = tuple(arguments.archive) if arguments.archive else DEFAULT_ARCHIVES
    weights_by_case = load_case_weights(arguments.runs_root, archives)
    plot_orbit_weight_histograms(
        weights_by_case,
        arguments.output,
        bins=arguments.bins,
        title="Density-solved orbit-weight distributions: 8--40 kpc tests",
    )
    print(arguments.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
