"""Stand-in VLA for the mock world. Its "action" is the prompt id it was told to run."""

from __future__ import annotations

from ..config.schema import PromptLibrary
from ..core.types import ActionChunk, Observation, Prompt, SkillCall
from .base import ExecutorCaps


class MockVLAExecutor:
    def __init__(self, library: PromptLibrary, chunk_size: int = 5) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        self._library = library
        self._chunk_size = chunk_size
        self._prompt: Prompt | None = None
        self._version = 0

    def prompt_library(self) -> PromptLibrary:
        return self._library

    def capabilities(self) -> ExecutorCaps:
        return ExecutorCaps(chunk_size=self._chunk_size)

    def reset(self) -> None:
        self._prompt = None
        self._version = 0

    def set_instruction(self, call: SkillCall, prompt: Prompt | None, version: int) -> None:
        if prompt is None:
            raise ValueError(f"{call.skill} has no prompt; the VLA only runs prompts from its library")
        if self._library.get(prompt.id) is None:
            raise ValueError(f"prompt {prompt.id!r} is not in library {self._library.library!r}")
        self._prompt = prompt
        self._version = version

    def infer_chunk(self, obs: Observation) -> ActionChunk:
        if self._prompt is None:
            raise RuntimeError("infer_chunk called before set_instruction")
        action = {"prompt": self._prompt.id, "version": self._version}
        return ActionChunk(self._version, [dict(action) for _ in range(self._chunk_size)])

    def is_done(self) -> bool:
        return False
