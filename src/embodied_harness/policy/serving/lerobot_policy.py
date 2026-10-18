"""Backward-compatible import for :mod:`embodied_harness.policy.vla.lerobot`.

Use ``python -m embodied_harness.policy.vla.lerobot`` for the canonical entry point.
This module remains executable so existing launch scripts keep working.
"""

from __future__ import annotations

from ..vla.lerobot import PROMPT_KEY, LeRobotPolicy, main

__all__ = ["PROMPT_KEY", "LeRobotPolicy", "main"]


if __name__ == "__main__":
    main()
