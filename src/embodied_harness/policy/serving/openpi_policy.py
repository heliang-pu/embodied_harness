"""Backward-compatible import for :mod:`embodied_harness.policy.vla.openpi`.

Use ``python -m embodied_harness.policy.vla.openpi`` for the canonical entry point.
This module remains executable so existing launch scripts keep working.
"""

from __future__ import annotations

from ..vla.openpi import OpenPIPolicy, main

__all__ = ["OpenPIPolicy", "main"]


if __name__ == "__main__":
    main()
