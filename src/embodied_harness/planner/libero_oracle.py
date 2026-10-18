"""Oracle planner for LIBERO long tasks.

It reads which goals hold from the simulator (privileged, `context.info["goals"]` of the libero env) and
asks for the first goal, in the episode's order, whose condition does not hold; once all hold, it is done.
A goal that stops holding (the bowl knocked off the plate) is asked for again.

With `planner.call: periodic` it replaces the running goal as soon as the simulator says it holds, the way
CALVIN and RoboCerebra switch subtasks with their success checks. That measures the policy and the
transitions on their own: an upper bound for the rest of the loop, never a planner for a real robot.
Skills are named after the goals' LIBERO tasks unless `skills` maps a task to another skill name.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from .base import PlannerContext, PlannerOutput

if TYPE_CHECKING:
    from ..config.loader import Bundle


class LiberoOracle:
    name = "libero_oracle"

    def __init__(self, skills: Mapping[str, str] | None = None) -> None:
        self._skills = dict(skills or {})

    @classmethod
    def check_config(cls, bundle: Bundle) -> list[str]:
        """Validate-time checks (factory.check_bundle): the env is libero and every goal is a task skill."""
        env = bundle.run.env
        if env.type != "libero":
            return [f"planner libero_oracle reads the libero env's goals; env.type is {env.type!r}"]
        goals = env.params.get("goals") or []
        goals = [goals] if isinstance(goals, str) else list(goals)
        skills = bundle.run.planner.params.get("skills") or {}
        return [
            f"planner libero_oracle: goal {goal!r} is the skill {skills.get(goal, goal)!r}, "
            "which task.skills does not list"
            for goal in goals
            if skills.get(goal, goal) not in bundle.task.skills
        ]

    def plan(self, context: PlannerContext) -> PlannerOutput:
        goals = context.info.get("goals")
        if not isinstance(goals, list):
            raise ValueError("the LIBERO oracle needs the libero env's privileged state in context.info")
        done = [goal["goal"] for goal in goals if goal["done"]]
        pending = [goal["goal"] for goal in goals if not goal["done"]]
        decision: dict[str, Any]
        if pending:
            decision = {"skill": self._skills.get(pending[0], pending[0])}
            decision["memory"] = f"已完成 {len(done)}/{len(goals)}；下一步 {pending[0]}"
        else:
            decision = {"done": True, "memory": f"已完成 {len(done)}/{len(goals)}"}
        meta = {"planner": self.name, "goals_done": len(done), "goals": len(goals)}
        return PlannerOutput(raw=json.dumps(decision, ensure_ascii=False), meta=meta)
