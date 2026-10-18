"""The planner contract (design doc §5).

Input: task, camera images, proprio and the planner's own compressed memory from last time.
Output: the next subtask (or done) plus the rewritten memory, as JSON.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..config.schema import TaskSpec
from ..core.types import DecisionReason

# Settings that only newer exports record, so they are compared only when the export has them: the
# fingerprints of the configured goal and of the rest of the system prompt (what each one covers).
RECORDED_ONLY = {"goal": "goal", "system": "rule, lesson, skill or done sentence"}
# Settings every exported sample records for itself, so `data eval`, which renders each sample with its
# own value, does not compare them: the goal.
PER_SAMPLE = ("goal",)


@dataclass
class PlannerContext:
    task: TaskSpec
    images: dict[str, Any]
    proprio: dict[str, Any]
    memory: str
    reason: DecisionReason
    skills: list[str]
    resample: int = 0
    # Privileged simulator state. Only the mock oracle reads it.
    info: dict[str, Any] = field(default_factory=dict)
    # Calls dispatched so far this episode, oldest first, without consecutive repeats. Planners that
    # condition on their past subgoals (RoboMME's symbolic memory) render this instead of `memory`.
    history: list[dict[str, Any]] = field(default_factory=list)
    # The ledger (task progress) at this call; None when unknown. Rendered with show_progress.
    ledger: dict[str, Any] | None = None
    # The last main call and how it ended, {"call": key, "status": success|failure|preempted|skipped,
    # "reason": ...}; None before the first. Rendered with show_state.
    last: dict[str, Any] | None = None
    # Why the supervisor refused the previous answer of this round (only when resample > 0), e.g.
    # "pick: precondition not met: not holding". Rendered with show_rejections. Samples exported from
    # demonstrations never have one; samples exported from recorded runs keep the recorded ones.
    rejected: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        """What the trace records. Images are referenced by camera name, not stored inline."""
        return {
            "task": self.task.id,
            "goal": self.task.goal,
            "reason": self.reason.value,
            "resample": self.resample,
            "cameras": sorted(self.images),
            "proprio": dict(self.proprio),
            "memory": self.memory,
            "history": list(self.history),
            "ledger": self.ledger,
            "last": self.last,
            "rejected": list(self.rejected),
            "skills": list(self.skills),
        }


@dataclass(frozen=True)
class PlannerOutput:
    raw: str
    latency_s: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)
    # Files to keep with the call, e.g. {"head.jpg": b"..."}: the images exactly as the model saw them.
    attachments: dict[str, bytes] = field(default_factory=dict)


class Planner(Protocol):
    name: str

    def plan(self, context: PlannerContext) -> PlannerOutput: ...


def settings_differences(
    current: Mapping[str, Any], trained: Mapping[str, Any], *, skip: Collection[str] = ()
) -> list[str]:
    """The planner settings (a planner's settings()) that differ between a live planner and the export
    its model was trained on, by name, leaving out `skip`. An option that is off reads the same as one an
    older export did not record, and a RECORDED_ONLY setting is compared only when the export has it."""

    def value(settings: Mapping[str, Any], key: str) -> Any:
        found = settings.get(key)
        return None if found is False or found == [] or found == {} else found

    return sorted(
        key
        for key in set(current) | set(trained)
        if key not in skip
        and not (key in RECORDED_ONLY and key not in trained)
        and value(current, key) != value(trained, key)
    )


def unrecorded_warning(
    current: Mapping[str, Any], trained: Mapping[str, Any], *, skip: Collection[str] = ()
) -> str | None:
    """A warning when the export lacks RECORDED_ONLY settings the live planner has (it was made before
    they were recorded), so settings_differences cannot tell whether what they fingerprint changed."""
    missing = [key for key in RECORDED_ONLY if key not in skip and key not in trained and current.get(key)]
    if not missing:
        return None
    changed = ", ".join(RECORDED_ONLY[key] for key in missing)
    return (
        f"the training export has no {' or '.join(missing)} fingerprint (it was made before they were "
        f"recorded): a changed {changed} goes unnoticed; export again to compare them"
    )
