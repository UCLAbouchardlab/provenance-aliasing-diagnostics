from __future__ import annotations

from collections.abc import Sequence

def main(argv: Sequence[str] | None = None) -> int:
    """Delegate to the operational CLI without duplicating its implementation."""
    from .api.cli import main as operational_main

    return operational_main(argv)

