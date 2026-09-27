"""Clocks. Components read time only through a clock, so episodes can also run in simulated time."""

from __future__ import annotations

import time
from typing import Protocol


class Clock(Protocol):
    def now(self) -> float: ...

    def sleep_until(self, deadline: float) -> None: ...


class RealClock:
    """Monotonic wall time; used on the robot."""

    def now(self) -> float:
        return time.monotonic()

    def sleep_until(self, deadline: float) -> None:
        delay = deadline - time.monotonic()
        if delay > 0:
            time.sleep(delay)


class SimClock:
    """Simulated time. Sleeping moves the clock forward instantly, so runs are fast and repeatable."""

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def now(self) -> float:
        return self._now

    def sleep_until(self, deadline: float) -> None:
        self._now = max(self._now, deadline)

    def advance(self, seconds: float) -> None:
        self._now += seconds
