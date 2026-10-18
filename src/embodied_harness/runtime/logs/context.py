"""Log context: fields every log line and trace event carries without the caller passing them."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

CONTEXT_FIELDS = ("episode_id", "t", "state", "skill_call_id", "planner_call_id")

_context: ContextVar[dict[str, Any] | None] = ContextVar("embodied_harness_log_context", default=None)


def current_context() -> dict[str, Any]:
    return dict(_context.get() or {})


def update_context(**fields: Any) -> None:
    """Set fields for the rest of the current context, e.g. the episode id or the current state."""
    merged = current_context()
    merged.update(fields)
    _context.set(merged)


def clear_context() -> None:
    _context.set(None)


@contextmanager
def scoped_context(**fields: Any) -> Iterator[None]:
    token = _context.set({**current_context(), **fields})
    try:
        yield
    finally:
        _context.reset(token)
