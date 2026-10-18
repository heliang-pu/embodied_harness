"""How far along a fixed plan an episode got, from its trace alone (long-horizon metrics).

The reference plan is the task's `plan` in config.snapshot.yaml, else the sequence planner's params.calls,
so old sequence traces are measurable too. Every main skill call is placed on a plan step: exactly, through
the sequence planner's `meta.sequence_index` (searching forward when a human retried a later step), or else
by its call key (the step at the position, else the nearest earlier step with that call: a redo; else
off-plan). The position is the number of steps done in order: a success at step s moves it to s + 1, a
dispatch before it (a rollback) moves it back.

- completed: the final position; the whole plan when the episode succeeded (`episode_succeeded`: the env's
  verdict when it gave one, else done).
- furthest: the highest position reached; a later rollback does not lower it.
- first_failure: the position at the first main-call failure, local retries included. This is the CALVIN
  chain length, without any recovery; completed minus first_failure is what recovery bought.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ...core.types import SkillCall
from .reader import read_events

SNAPSHOT_FILE = "config.snapshot.yaml"


def episode_succeeded(status: str | None, env_success: bool | None) -> bool:
    """Whether an episode succeeded: by the env's own verdict when it gave one (a simulator, also for
    an episode the harness finished as done), else by finishing as done (a real robot, or a trace from
    before done episodes recorded the verdict). The one rule of `run`, `report`, `suite`, plan progress
    and `data from-runs --success-only`."""
    if env_success is not None:
        return env_success is True
    return status == "done"


@dataclass
class PlanProgress:
    plan: list[str]  # the reference plan, as call keys
    completed: int
    furthest: int
    first_failure: int
    rollbacks: int = 0  # times the position went back
    off_plan: int = 0  # main calls that match no step
    exact: bool = False  # positions from the sequence planner's meta, not call keys
    stopped_at: int | None = None  # the step (0-based) an unfinished episode stopped at
    attempts: list[int] = field(default_factory=list)  # per step: main dispatches
    first_try: list[bool | None] = field(default_factory=list)  # per step: its first dispatch succeeded
    failures: list[dict[str, int]] = field(default_factory=list)  # per step: failure reasons
    rollbacks_from: list[int] = field(default_factory=list)  # per step: rollbacks that left it
    skipped: list[int] = field(default_factory=list)  # per step: human skips that completed it

    @property
    def steps(self) -> int:
        return len(self.plan)


def _key(data: Any) -> str | None:
    if isinstance(data, str):
        return SkillCall(data).key()
    if isinstance(data, Mapping) and isinstance(data.get("skill"), str):
        return SkillCall(data["skill"], dict(data.get("args") or {})).key()
    return None


def read_snapshot(directory: Path) -> dict[str, Any] | None:
    path = directory / SNAPSHOT_FILE
    if not path.is_file():
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return None
    return data if isinstance(data, dict) else None


def reference_plan(snapshot: Mapping[str, Any]) -> list[str] | None:
    """task.plan, else a sequence planner's params.calls, as call keys; None without a fixed order."""
    steps = (snapshot.get("task") or {}).get("plan")
    if steps is None:
        planner = (snapshot.get("run") or {}).get("planner") or {}
        if planner.get("type") == "sequence":
            steps = (planner.get("params") or {}).get("calls")
    if not isinstance(steps, list) or not steps:
        return None
    keys = [_key(step) for step in steps]
    return None if any(key is None for key in keys) else [str(key) for key in keys]


def plan_progress(
    plan: Sequence[str], events: Sequence[Mapping[str, Any]], outcome: Mapping[str, Any]
) -> PlanProgress:
    n = len(plan)
    exact = any(
        e.get("type") == "planner_response" and "sequence_index" in (e.get("meta") or {}) for e in events
    )
    progress = PlanProgress(
        list(plan),
        completed=0,
        furthest=0,
        first_failure=0,
        exact=exact,
        attempts=[0] * n,
        first_try=[None] * n,
        failures=[{} for _ in range(n)],
        rollbacks_from=[0] * n,
        skipped=[0] * n,
    )
    failures = [Counter() for _ in range(n)]
    proposals: dict[str, int] = {}  # planner call id -> accepted sequence index
    latest: int | None = None
    started: dict[str, int | None] = {}
    open_calls: dict[str, int | None] = {}
    cursor = 0
    first_failure: int | None = None
    last_failed: int | None = None

    def find_forward(start: int, key: str) -> int | None:
        return next((j for j in range(start, n) if plan[j] == key), None)

    def place(event: Mapping[str, Any], key: str) -> int | None:
        if exact:
            index = proposals.get(str(event.get("planner_call_id")), latest)
            if index is None:
                return None
            return index if plan[index] == key else find_forward(index, key)
        if cursor < n and plan[cursor] == key:
            return cursor
        return next((j for j in range(min(cursor, n) - 1, -1, -1) if plan[j] == key), None)

    def succeed(step: int) -> None:
        nonlocal cursor
        cursor = step + 1
        progress.furthest = max(progress.furthest, cursor)

    for event in events:
        kind = event.get("type")
        if kind == "planner_response":
            meta = event.get("meta") or {}
            if "sequence_index" in meta and not event.get("errors"):
                index = int(meta["sequence_index"])
                if 0 <= index < n:
                    proposals[str(event.get("call_id"))] = latest = index
        elif kind == "skill_start" and event.get("purpose", "main") == "main":
            key = _key(event.get("call"))
            step = None if key is None else place(event, key)
            started[str(event.get("call_id"))] = step
            open_calls[str(event.get("call_id"))] = step
            if step is None:
                progress.off_plan += 1
                continue
            if step < cursor:
                progress.rollbacks += 1
                progress.rollbacks_from[min(cursor, n - 1)] += 1
                cursor = step
            progress.attempts[step] += 1
        elif kind == "skill_end" and event.get("purpose", "main") == "main":
            call_id = str(event.get("call_id"))
            open_calls.pop(call_id, None)
            step = started.get(call_id)
            outcome_data = event.get("outcome") or {}
            status = outcome_data.get("status")
            if status == "failure" and first_failure is None:
                first_failure = cursor
            if step is None:
                continue
            first = progress.attempts[step] == 1 and progress.first_try[step] is None
            if status == "success":
                if first:
                    progress.first_try[step] = True
                last_failed = None
                succeed(step)
            elif status == "failure":
                if first:
                    progress.first_try[step] = False
                failures[step][str(outcome_data.get("reason") or "failure")] += 1
                last_failed = step
        elif kind == "human_command" and event.get("command") == "skip":
            key = _key(event.get("failed_call"))
            step = last_failed if last_failed is not None and key == plan[last_failed] else None
            if step is None and key is not None:
                step = find_forward(cursor, key)
            if step is not None:
                progress.skipped[step] += 1
                last_failed = None
                succeed(step)
        elif kind == "env_end":
            success = event.get("success")
            for step in open_calls.values():
                if step is None:
                    continue
                if success is True:
                    if progress.first_try[step] is None and progress.attempts[step] == 1:
                        progress.first_try[step] = True
                    succeed(step)
                elif first_failure is None:
                    first_failure = cursor
            open_calls.clear()
    succeeded = episode_succeeded(outcome.get("status"), outcome.get("env_success"))
    progress.completed = n if succeeded else cursor
    progress.furthest = n if succeeded else max(progress.furthest, cursor)
    progress.first_failure = progress.completed if first_failure is None else first_failure
    progress.stopped_at = None if progress.completed >= n else progress.completed
    progress.failures = [dict(counter) for counter in failures]
    return progress


def episode_progress(directory: Path) -> PlanProgress | None:
    """The progress of one episode directory, or None when its config has no fixed plan."""
    snapshot = read_snapshot(directory)
    plan = reference_plan(snapshot) if snapshot else None
    if plan is None:
        return None
    outcome_path = directory / "outcome.json"
    outcome = json.loads(outcome_path.read_text(encoding="utf-8")) if outcome_path.is_file() else {}
    return plan_progress(plan, read_events(directory), outcome)
