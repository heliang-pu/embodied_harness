"""Health summary: one INFO line every few seconds and totals at the end (design doc §12)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("fmc3_harness.health")


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def _fmt(value: float | None, unit: str, digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}f}{unit}"


@dataclass
class _Window:
    start: float
    ticks: int = 0
    tick_ms: list[float] = field(default_factory=list)
    overruns: int = 0
    max_overrun_ms: float = 0.0
    planner_s: list[float] = field(default_factory=list)
    resamples: int = 0


class HealthMonitor:
    """Tracks loop timing, queue depth, planner latency and resamples."""

    def __init__(self, every_s: float, period_s: float) -> None:
        self._every_s = every_s
        self._period_ms = period_s * 1000.0
        self._window: _Window | None = None
        self._queue_depth = 0
        self._ticks = 0
        self._overruns = 0
        self._resamples = 0
        self._tick_ms: list[float] = []
        self._planner_s: list[float] = []

    def record_tick(self, now: float, cost_s: float, queue_depth: int) -> None:
        """cost_s is the compute time of one tick; more than one control period counts as an overrun."""
        window = self._ensure(now)
        cost_ms = cost_s * 1000.0
        window.ticks += 1
        window.tick_ms.append(cost_ms)
        self._ticks += 1
        self._tick_ms.append(cost_ms)
        self._queue_depth = queue_depth
        if cost_ms > self._period_ms:
            window.overruns += 1
            window.max_overrun_ms = max(window.max_overrun_ms, cost_ms - self._period_ms)
            self._overruns += 1

    def record_planner_latency(self, seconds: float) -> None:
        if self._window is not None:
            self._window.planner_s.append(seconds)
        self._planner_s.append(seconds)

    def record_resample(self) -> None:
        if self._window is not None:
            self._window.resamples += 1
        self._resamples += 1

    def maybe_log(self, now: float) -> None:
        window = self._window
        if window is None or now - window.start < self._every_s:
            return
        elapsed = max(now - window.start, 1e-9)
        logger.info(
            "ctrl %.1f Hz, tick p95 %s, overruns %d (max %s), queue %d, planner p50/p95 %s/%s, resamples %d",
            window.ticks / elapsed,
            _fmt(_percentile(window.tick_ms, 0.95), " ms", 2),
            window.overruns,
            _fmt(window.max_overrun_ms, " ms"),
            self._queue_depth,
            _fmt(_percentile(window.planner_s, 0.5), " s", 3),
            _fmt(_percentile(window.planner_s, 0.95), " s", 3),
            window.resamples,
        )
        self._window = _Window(start=now)

    def summary(self) -> dict[str, Any]:
        return {
            "ticks": self._ticks,
            "overruns": self._overruns,
            "resamples": self._resamples,
            "tick_ms_p95": _percentile(self._tick_ms, 0.95),
            "planner_s_p50": _percentile(self._planner_s, 0.5),
            "planner_s_p95": _percentile(self._planner_s, 0.95),
        }

    def _ensure(self, now: float) -> _Window:
        if self._window is None:
            self._window = _Window(start=now)
        return self._window
