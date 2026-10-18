"""Runs harness-side primitives such as `retract`: a fixed number of steps the env knows how to execute.

Each step is `{"primitive": name, "version": v}`, plus `"args"` when the call has arguments (e.g. a
transition step that moves to the pre-grasp pose of `color`).
"""

from __future__ import annotations

from collections.abc import Mapping

from ...core.types import ActionChunk, Observation, Prompt, SkillCall
from .base import ExecutorCaps


class ScriptedExecutor:
    def __init__(self, steps: Mapping[str, int] | None = None, default_steps: int = 5) -> None:
        self._steps = dict(steps or {})
        self._default_steps = default_steps
        self._call: SkillCall | None = None
        self._remaining = 0
        self._version = 0

    def prompt_library(self) -> None:
        return None

    def capabilities(self) -> ExecutorCaps:
        return ExecutorCaps(chunk_size=self._default_steps)

    def reset(self) -> None:
        self._call = None
        self._remaining = 0
        self._version = 0

    def set_instruction(self, call: SkillCall, prompt: Prompt | None, version: int) -> None:
        self._call = call
        self._version = version
        self._remaining = self._steps.get(call.skill, self._default_steps)

    def infer_chunk(self, obs: Observation) -> ActionChunk:
        if self._call is None:
            raise RuntimeError("infer_chunk called before set_instruction")
        count, self._remaining = self._remaining, 0
        action: dict[str, object] = {"primitive": self._call.skill, "version": self._version}
        if self._call.args:
            action["args"] = dict(self._call.args)
        return ActionChunk(self._version, [dict(action) for _ in range(count)])

    def is_done(self) -> bool:
        return self._call is not None and self._remaining == 0
