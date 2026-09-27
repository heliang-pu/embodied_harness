"""Action chunk queue with instruction versions and handoff modes (design doc §6)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from ..core.types import ActionChunk

HANDOFF_MODES = ("finish_chunk", "hard")


@dataclass(frozen=True)
class QueuedAction:
    version: int
    action: Any


class ChunkQueue:
    """Holds action chunks and decides what happens to old actions when the instruction changes.

    finish_chunk: the chunk being executed runs to its end, later old chunks are dropped.
    hard: every action of an older instruction is dropped at once.
    """

    def __init__(self, handoff: str = "finish_chunk") -> None:
        if handoff not in HANDOFF_MODES:
            raise ValueError(f"handoff must be one of {HANDOFF_MODES}, got {handoff!r}")
        self._handoff = handoff
        self._chunks: deque[tuple[int, deque[Any]]] = deque()
        self._version = 0

    @property
    def version(self) -> int:
        return self._version

    def switch(self, version: int) -> None:
        self._version = version
        head = self._chunks[0] if self._chunks else None
        self._chunks.clear()
        if self._handoff == "finish_chunk" and head is not None and head[0] < version and head[1]:
            self._chunks.append(head)

    def push(self, chunk: ActionChunk) -> bool:
        """Queue a chunk; chunks computed for an older instruction are dropped."""
        if chunk.version != self._version:
            return False
        if chunk.actions:
            self._chunks.append((chunk.version, deque(chunk.actions)))
        return True

    def pop(self) -> QueuedAction | None:
        while self._chunks:
            version, actions = self._chunks[0]
            if actions:
                return QueuedAction(version, actions.popleft())
            self._chunks.popleft()
        return None

    def pending(self, version: int | None = None) -> int:
        """Actions still queued for one instruction version (the current one by default)."""
        target = self._version if version is None else version
        return sum(len(actions) for v, actions in self._chunks if v == target)

    def clear(self) -> None:
        self._chunks.clear()

    def __len__(self) -> int:
        return sum(len(actions) for _, actions in self._chunks)
