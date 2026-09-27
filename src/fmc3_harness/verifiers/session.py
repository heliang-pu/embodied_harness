"""Judges one running skill call from proprio-based rules, a stall check and a timeout (design doc §8)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..core.types import Outcome, OutcomeStatus
from ..skills.registry import CompiledCondition, CompiledSkill

_EPSILON = 1e-9
_STALL_TOLERANCE = 1e-4


class VerifierSession:
    """Verdict tracking for one skill call.

    Failure conditions beat success conditions, which beat the timeout. A condition only
    counts after it has held for its hold_s, so a single noisy tick cannot switch skills.
    """

    def __init__(self, skill: CompiledSkill, *, started_at: float, timeout_s: float) -> None:
        self._skill = skill
        self._started_at = started_at
        self._timeout_s = timeout_s
        self._true_since: dict[tuple[str, int], float] = {}
        self._last_proprio: dict[str, Any] | None = None
        self._last_change = started_at

    @property
    def started_at(self) -> float:
        return self._started_at

    @property
    def timeout_s(self) -> float:
        return self._timeout_s

    def evaluate(
        self, now: float, namespace: Mapping[str, Any], proprio: Mapping[str, Any]
    ) -> Outcome | None:
        """Returns the verdict, or None while the call is still running."""
        elapsed = now - self._started_at
        failed = self._first_held("failure", self._skill.failure, now, namespace)
        if failed is not None:
            condition, held = failed
            return self._verdict(
                OutcomeStatus.FAILURE, condition.reason, elapsed, condition=condition.expr.source, held_s=held
            )

        stall_s = self._skill.spec.stall_s
        if stall_s is not None:
            still_for = self._still_for(now, proprio)
            if still_for >= stall_s - _EPSILON:
                return self._verdict(OutcomeStatus.FAILURE, "stall", elapsed, still_s=still_for)

        succeeded = self._first_held("success", self._skill.success, now, namespace)
        if elapsed >= self._skill.spec.min_duration_s - _EPSILON:
            if succeeded is not None:
                condition, held = succeeded
                return self._verdict(
                    OutcomeStatus.SUCCESS, None, elapsed, condition=condition.expr.source, held_s=held
                )
            if not self._skill.success and namespace.get("executor_done"):
                return self._verdict(OutcomeStatus.SUCCESS, None, elapsed, condition="executor_done")

        if elapsed >= self._timeout_s - _EPSILON:
            return self._verdict(OutcomeStatus.FAILURE, "timeout", elapsed, timeout_s=self._timeout_s)
        return None

    def _first_held(
        self,
        group: str,
        conditions: tuple[CompiledCondition, ...],
        now: float,
        namespace: Mapping[str, Any],
    ) -> tuple[CompiledCondition, float] | None:
        # Every condition is evaluated on every tick so its "true since" timer stays accurate.
        found = None
        for index, condition in enumerate(conditions):
            key = (group, index)
            if bool(condition.expr(namespace)):
                held = now - self._true_since.setdefault(key, now)
                if found is None and held >= condition.hold_s - _EPSILON:
                    found = (condition, held)
            else:
                self._true_since.pop(key, None)
        return found

    def _still_for(self, now: float, proprio: Mapping[str, Any]) -> float:
        current = dict(proprio)
        previous = self._last_proprio
        if previous is None or any(_changed(previous.get(k), v) for k, v in current.items()):
            self._last_change = now
        self._last_proprio = current
        return now - self._last_change

    @staticmethod
    def _verdict(status: OutcomeStatus, reason: str | None, elapsed: float, **evidence: Any) -> Outcome:
        details = {k: round(v, 3) if isinstance(v, float) else v for k, v in evidence.items()}
        details["elapsed_s"] = round(elapsed, 3)
        return Outcome(status, reason, details)


def _changed(old: Any, new: Any) -> bool:
    if isinstance(old, (int, float)) and isinstance(new, (int, float)) and not isinstance(new, bool):
        return abs(float(new) - float(old)) > _STALL_TOLERANCE
    return old != new
