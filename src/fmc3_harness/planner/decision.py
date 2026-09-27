"""Parses the planner's JSON output (design doc §5)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..core.types import SkillCall

_FIELDS = {"skill", "args", "done", "memory"}


class DecisionError(ValueError):
    pass


@dataclass(frozen=True)
class PlannerDecision:
    call: SkillCall | None  # None means the planner declared the task done
    memory: str

    @property
    def done(self) -> bool:
        return self.call is None

    def to_dict(self) -> dict[str, Any]:
        if self.call is None:
            return {"done": True, "memory": self.memory}
        return {"skill": self.call.skill, "args": dict(self.call.args), "memory": self.memory}


def parse_decision(raw: str) -> PlannerDecision:
    """Accepts `{"skill": ..., "args": {...}, "memory": ...}` or `{"done": true, "memory": ...}`."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DecisionError(f"output is not JSON: {exc.msg}") from None
    if not isinstance(data, dict):
        raise DecisionError("output must be a JSON object")
    unknown = set(data) - _FIELDS
    if unknown:
        raise DecisionError(f"unexpected field(s): {', '.join(sorted(unknown))}")
    memory = data.get("memory")
    if not isinstance(memory, str):
        raise DecisionError("output needs a string field 'memory'")
    done = data.get("done", False)
    if not isinstance(done, bool):
        raise DecisionError("'done' must be true or false")
    if done:
        if "skill" in data:
            raise DecisionError("output cannot have both 'done' and 'skill'")
        return PlannerDecision(None, memory)
    skill = data.get("skill")
    if not isinstance(skill, str) or not skill:
        raise DecisionError("output needs 'skill' (a string) or 'done': true")
    args = data.get("args", {})
    if not isinstance(args, dict):
        raise DecisionError("'args' must be an object")
    return PlannerDecision(SkillCall(skill, dict(args)), memory)
