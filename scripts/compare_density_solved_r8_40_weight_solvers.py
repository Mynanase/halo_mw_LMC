#!/usr/bin/env python3
"""Compare repeated paper-best weight-solver benchmark artifacts."""

from __future__ import annotations

import argparse

from halo_mw_lmc.weight_solver_benchmark import write_solver_comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directories", nargs="+")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    result = write_solver_comparison(
        arguments.run_directories,
        arguments.output,
    )
    print(f"winner={result['winner']}")
    print(f"production_ready={result['production_ready']}")
    print(f"output={arguments.output}")


if __name__ == "__main__":
    main()
