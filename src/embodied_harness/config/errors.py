"""Config errors carry every problem found, so one `validate` run reports them all."""

from __future__ import annotations

from ..runtime.logs.codes import ErrorCode


class ConfigError(Exception):
    def __init__(self, problems: list[str], code: ErrorCode = ErrorCode.E_CONFIG) -> None:
        self.problems = list(problems)
        self.code = code
        super().__init__("\n".join(f"- {problem}" for problem in self.problems))
