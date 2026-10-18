"""Per-episode identifiers such as sc_0001 (skill call) and pc_0001 (planner call)."""

from __future__ import annotations


class IdSequence:
    def __init__(self, prefix: str, start: int = 0) -> None:
        self._prefix = prefix
        self._count = start

    @property
    def count(self) -> int:
        return self._count

    def next(self) -> str:
        self._count += 1
        return f"{self._prefix}_{self._count:04d}"
