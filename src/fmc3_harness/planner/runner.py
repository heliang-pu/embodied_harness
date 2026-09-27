"""Runs planner calls so a slow model never blocks the control loop (design doc §3).

The supervisor submits a context and polls every tick; timeouts are measured on the
supervisor's clock.
"""

from __future__ import annotations

import contextvars
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from typing import Protocol

from .base import Planner, PlannerContext, PlannerOutput


class PlannerRunner(Protocol):
    def submit(self, context: PlannerContext) -> None: ...

    def poll(self) -> PlannerOutput | BaseException | None:
        """The output, the exception the planner raised, or None while it is still running."""
        ...

    def cancel(self) -> None: ...

    def close(self) -> None: ...


def _timed_plan(planner: Planner, context: PlannerContext) -> PlannerOutput:
    start = time.perf_counter()
    output = planner.plan(context)
    return replace(output, latency_s=time.perf_counter() - start)


class InlinePlannerRunner:
    """Calls the planner inside submit(). Deterministic, so tests and the mock oracle use it."""

    def __init__(self, planner: Planner) -> None:
        self._planner = planner
        self._result: PlannerOutput | BaseException | None = None

    def submit(self, context: PlannerContext) -> None:
        try:
            self._result = _timed_plan(self._planner, context)
        except Exception as exc:
            self._result = exc

    def poll(self) -> PlannerOutput | BaseException | None:
        result, self._result = self._result, None
        return result

    def cancel(self) -> None:
        self._result = None

    def close(self) -> None:
        pass


class ThreadPlannerRunner:
    """Calls the planner on a worker thread; used for real models."""

    def __init__(self, planner: Planner) -> None:
        self._planner = planner
        self._pool = self._new_pool()
        self._future: Future[PlannerOutput] | None = None

    @staticmethod
    def _new_pool() -> ThreadPoolExecutor:
        return ThreadPoolExecutor(max_workers=1, thread_name_prefix="planner")

    def submit(self, context: PlannerContext) -> None:
        # Run in a copy of the current context so the planner thread logs with the same fields.
        ctx = contextvars.copy_context()
        self._future = self._pool.submit(ctx.run, _timed_plan, self._planner, context)

    def poll(self) -> PlannerOutput | BaseException | None:
        future = self._future
        if future is None or not future.done():
            return None
        self._future = None
        error = future.exception()
        return error if error is not None else future.result()

    def cancel(self) -> None:
        future, self._future = self._future, None
        if future is not None and not future.cancel() and not future.done():
            # A call already running cannot be interrupted. Leave it on the old pool and start a
            # fresh one, so the next call does not queue behind a hung request.
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = self._new_pool()

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
