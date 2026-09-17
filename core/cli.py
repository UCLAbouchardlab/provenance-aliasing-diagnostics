from __future__ import annotations

import argparse
from collections.abc import Sequence

from . import __version__


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="provenance-aliasing",
        description="Metadata-based provenance-aliasing diagnostics.",
        epilog=(
            "This development scaffold provides help and version information. "
            "The validate and diagnose commands are planned after the shared "
            "input and reporting contracts are implemented."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.parse_args(argv)
    parser.print_help()
    return 0

