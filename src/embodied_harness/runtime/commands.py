"""Where human commands come from while the supervisor holds (design doc §13)."""

from __future__ import annotations

import sys
from collections.abc import Iterable
from typing import Protocol

from .logs.codes import ErrorCode
from .supervisor import HUMAN_COMMANDS


class CommandSource(Protocol):
    def poll(self, code: ErrorCode | None, message: str | None, failed_call: str | None) -> str | None:
        """A command for the current hold, or None to keep holding."""
        ...


class AutoCommands:
    """Answers a hold without a human: always `abort`, or `resume` a few times and then abort."""

    def __init__(self, policy: str = "abort", max_resumes: int = 3) -> None:
        if policy not in ("abort", "resume"):
            raise ValueError(f"policy must be 'abort' or 'resume', got {policy!r}")
        self._policy = policy
        self._max_resumes = max_resumes
        self._resumes = 0

    def poll(self, code: ErrorCode | None, message: str | None, failed_call: str | None) -> str | None:
        if self._policy == "resume" and self._resumes < self._max_resumes:
            self._resumes += 1
            return "resume"
        return "abort"


class ScriptedCommands:
    """Plays a fixed list of commands, then aborts. Used by tests."""

    def __init__(self, commands: Iterable[str]) -> None:
        self._commands = list(commands)

    def poll(self, code: ErrorCode | None, message: str | None, failed_call: str | None) -> str | None:
        return self._commands.pop(0) if self._commands else "abort"


class TerminalCommands:
    """Asks the operator. Blocking is fine here: the robot is already holding still."""

    def poll(self, code: ErrorCode | None, message: str | None, failed_call: str | None) -> str | None:
        label = f" [{code.value}]" if code else ""
        print(f"\nHOLD{label}: {message}", file=sys.stderr)
        if failed_call:
            print(f"last failed call: {failed_call}", file=sys.stderr)
        while True:
            try:
                answer = input(f"{' / '.join(HUMAN_COMMANDS)} > ").strip().lower()
            except EOFError:
                return "abort"
            if answer in HUMAN_COMMANDS:
                return answer
