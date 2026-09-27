"""Error codes shared by logs, trace events and outcome.json (design doc §12)."""

from __future__ import annotations

from enum import Enum


class ErrorCode(str, Enum):
    E_CONFIG = "E_CONFIG"
    E_PLANNER_INVALID = "E_PLANNER_INVALID"
    E_PLANNER_TIMEOUT = "E_PLANNER_TIMEOUT"
    E_EXEC_UNDERRUN = "E_EXEC_UNDERRUN"
    E_CTRL_OVERRUN = "E_CTRL_OVERRUN"
    E_ENV_DISCONNECT = "E_ENV_DISCONNECT"
    E_LIBRARY_MISMATCH = "E_LIBRARY_MISMATCH"
    E_BUDGET = "E_BUDGET"
    E_INTERNAL = "E_INTERNAL"
