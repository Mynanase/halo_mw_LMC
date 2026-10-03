"""Allow ``python -m halo_mw_lmc`` to mirror the installed command."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
