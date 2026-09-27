"""The planner contract (design doc §5).

Input: task, camera images, proprio and the planner's own compressed memory from last time.
Output: the next subtask (or done) plus the rewritten memory, as JSON.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..config.schema import TaskSpec
from ..core.types import DecisionReason


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

    def summary(self) -> dict[str, Any]:
        """What the trace records. Images are referenced by camera name, not stored inline."""
        return {
            "task": self.task.id,
            "reason": self.reason.value,
            "resample": self.resample,
            "cameras": sorted(self.images),
            "proprio": dict(self.proprio),
            "memory": self.memory,
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
