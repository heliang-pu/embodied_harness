"""The subtask active at a frame, from the `language_persistent` rows of a LeRobot dataset.

Plain Python, so the harness can test it without lerobot; `integrations.lerobot` wraps it in a
LeRobot processor step.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

SUBTASK_STYLE = "subtask"
_EPSILON_S = 1e-4  # rows and frames carry float32 timestamps


def active_text(rows: Iterable[Mapping[str, Any]], t: float, style: str = SUBTASK_STYLE) -> str | None:
    """The content of the latest `style` row at or before `t` (the last one listed on a tie), if any."""
    best: tuple[float, str] | None = None
    for row in rows:
        if row.get("style") != style or row.get("timestamp") is None:
            continue
        stamp = float(row["timestamp"])
        content = (row.get("content") or "").strip()
        if content and stamp <= t + _EPSILON_S and (best is None or stamp >= best[0]):
            best = (stamp, content)
    return best[1] if best is not None else None


def subtask_prompts(
    tasks: Sequence[str], rows: Sequence[Sequence[Mapping[str, Any]]], stamps: Sequence[float], style: str
) -> list[str]:
    """Per sample: the active subtask text, or the sample's own task where no subtask row applies."""
    return [
        active_text(sample_rows or (), float(t), style) or task
        for task, sample_rows, t in zip(tasks, rows, stamps, strict=True)
    ]
