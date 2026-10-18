"""Recorded dynamic task goals and dispatch history survive offline replay."""

from __future__ import annotations

from embodied_harness.config.schema import TaskSpec
from embodied_harness.core.types import DecisionReason
from embodied_harness.planner.base import PlannerContext
from embodied_harness.runtime.trace.replay import context_from_record


def test_replay_preserves_episode_goal_and_dispatch_history(tmp_path):
    configured = TaskSpec(id="pickx", goal="placeholder", skills=["pick", "place"])
    history = [{"skill": "pick", "args": {"color": "red"}}]
    original = PlannerContext(
        task=configured.model_copy(update={"goal": "place red twice"}),
        images={},
        proprio={"gripper_width": 0.02},
        memory="first pick",
        reason=DecisionReason.SKILL_DONE,
        skills=list(configured.skills),
        history=history,
    )
    restored = context_from_record({"input": original.summary()}, tmp_path, configured)
    assert restored.task.goal == original.task.goal
    assert restored.history == history
    assert restored.memory == original.memory
    assert configured.goal == "placeholder"


def test_legacy_replay_without_goal_or_history_uses_configured_task(tmp_path):
    task = TaskSpec(id="fixed", goal="sort all", skills=["pick"])
    record = {"input": {"proprio": {}, "memory": "", "reason": "start", "skills": ["pick"]}}
    restored = context_from_record(record, tmp_path, task)
    assert restored.task.goal == "sort all"
    assert restored.history == []
