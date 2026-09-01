#!/usr/bin/env python3
"""Low-frequency synthetic-target entry point kept outside the main CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from halo_mw_lmc.configuration import load_synthetic_density_configuration
from halo_mw_lmc.workflows.synthetic_density import generate_synthetic_density


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate one synthetic target from a strict TOML configuration."
    )
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    configuration = load_synthetic_density_configuration(args.config)
    result = generate_synthetic_density(configuration)
    print(f"wrote {result.output_path}")
    print(f"grid shape: {result.grid_shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
