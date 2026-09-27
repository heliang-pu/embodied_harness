"""Plays a fixed list of skill calls.

On the robot this is the oracle that measures the executor's upper bound: the order is right
by construction, so every failure belongs to the executor or to the transitions.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from ..core.types import DecisionReason, SkillCall
from .base import PlannerContext, PlannerOutput

_REPEAT_REASONS = (DecisionReason.SKILL_FAILED, DecisionReason.HUMAN_RESUME, DecisionReason.RESUME)


class SequencePlanner:
    name = "sequence"

    def __init__(self, calls: Sequence[Mapping[str, Any]]) -> None:
        self._calls = [SkillCall(str(c["skill"]), dict(c.get("args", {}))) for c in calls]
        self._next = 0

    def plan(self, context: PlannerContext) -> PlannerOutput:
        # After a failure, an interruption or a rejected output the same step is issued again.
        repeat = context.reason in _REPEAT_REASONS or context.resample > 0
        if repeat and self._next > 0:
            self._next -= 1
        if self._next >= len(self._calls):
            decision: dict[str, Any] = {"done": True}
        else:
            call = self._calls[self._next]
            self._next += 1
            decision = call.to_dict()
        decision["memory"] = f"第 {self._next}/{len(self._calls)} 步"
        return PlannerOutput(raw=json.dumps(decision, ensure_ascii=False), meta={"planner": self.name})
