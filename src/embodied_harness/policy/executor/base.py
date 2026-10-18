"""The executor contract (design doc §6)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ...config.schema import PromptLibrary
from ...core.types import ActionChunk, Observation, Prompt, SkillCall


@dataclass(frozen=True)
class ExecutorCaps:
    chunk_size: int
    rtc: bool = False
    target_slots: tuple[str, ...] = ()


class Executor(Protocol):
    def prompt_library(self) -> PromptLibrary | None:
        """The subtask prompts this model was trained on; None for scripted primitives."""
        ...

    def capabilities(self) -> ExecutorCaps: ...

    def reset(self) -> None:
        """Clear internal state at the start of an episode, including any short-term memory."""
        ...

    def set_instruction(self, call: SkillCall, prompt: Prompt | None, version: int) -> None: ...

    def infer_chunk(self, obs: Observation) -> ActionChunk:
        """Actions for the current instruction, tagged with its version."""
        ...

    def is_done(self) -> bool:
        """True once a finite executor (a scripted primitive) has produced all its actions."""
        ...
