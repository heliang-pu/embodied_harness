"""Plain data types shared by every component."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DecisionReason(str, Enum):
    """Why the supervisor asks the planner for a decision."""

    START = "start"
    SKILL_DONE = "skill_done"
    SKILL_FAILED = "skill_failed"
    CHECK_FAILED = "check_failed"
    HUMAN_RESUME = "human_resume"
    RESUME = "resume"
    PERIODIC = "periodic"  # a check while a skill runs (planner.call: periodic)


class OutcomeStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"


@dataclass(frozen=True)
class SkillCall:
    """A skill from the vocabulary with its arguments filled in, i.e. one subtask."""

    skill: str
    args: dict[str, Any] = field(default_factory=dict)

    def key(self) -> str:
        """Stable text form used in logs and for counting repeated failures."""
        if not self.args:
            return self.skill
        inner = ", ".join(f"{name}={value}" for name, value in sorted(self.args.items()))
        return f"{self.skill}({inner})"

    def to_dict(self) -> dict[str, Any]:
        return {"skill": self.skill, "args": dict(self.args)}


@dataclass(frozen=True)
class Prompt:
    """A prompt library entry resolved for one call: the exact text the executor was trained on."""

    id: str
    text: str


@dataclass
class Observation:
    """Everything the harness reads from the environment on one control tick."""

    t: float = 0.0
    images: dict[str, Any] = field(default_factory=dict)
    proprio: dict[str, Any] = field(default_factory=dict)
    # Privileged ground truth. Only simulated environments fill it, and nothing on the
    # real-robot path may depend on it. The harness reads just `info["success"]`, once, when
    # the env ends the episode, to record the env's own verdict for scoring.
    info: dict[str, Any] = field(default_factory=dict)
    # The environment ended the episode: its task finished or failed by its own rules, or it hit
    # its step limit. The harness stops; it never plans around this.
    terminated: bool = False


@dataclass(frozen=True)
class ActionChunk:
    """Actions from one executor inference, tagged with the instruction version they belong to."""

    version: int
    actions: list[Any]


@dataclass(frozen=True)
class Outcome:
    """The verdict on one skill call."""

    status: OutcomeStatus
    reason: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status is OutcomeStatus.SUCCESS

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status.value, "reason": self.reason, "evidence": dict(self.evidence)}
