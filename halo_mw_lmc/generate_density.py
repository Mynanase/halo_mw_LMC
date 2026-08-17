"""One-config command for generating analytic target-density artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from .configuration import (
    ConfigurationError,
    load_synthetic_density_configuration,
)
from .workflows.synthetic_density import generate_synthetic_density


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="halo-mw-lmc-density",
        description="Generate one gridded target-density NPZ from a strict TOML config.",
    )
    parser.add_argument("config", type=Path, help="synthetic-density TOML file")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        configuration = load_synthetic_density_configuration(args.config)
        result = generate_synthetic_density(configuration)
    except (
        ConfigurationError,
        FileExistsError,
        FileNotFoundError,
        OSError,
        ValueError,
    ) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"wrote {result.output_path}")
    print(f"grid shape: {result.grid_shape}")
    print(
        "quadrature relative difference: "
        f"median={result.median_quadrature_relative_difference:.3e}, "
        f"max={result.maximum_quadrature_relative_difference:.3e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
