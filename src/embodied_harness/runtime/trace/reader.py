"""Reads a trace back and rebuilds what an interrupted episode needs to resume (design doc §13).

The ledger and the planner's memory are event-sourced: replaying `ledger_update` and
`memory_update` events gives exactly the state the supervisor had. Simulated envs also leave
their latest state in env_snapshot.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...core.ledger import LedgerOp
from ...core.types import SkillCall
from .writer import ENV_SNAPSHOT_FILE


@dataclass
class RestoredState:
    ledger_ops: list[LedgerOp]
    memory: str
    skill_calls: int
    planner_calls: int
    env_snapshot: dict[str, Any] | None
    last_seq: int
    last_t: float
    finished: bool
    history: list[SkillCall] = field(default_factory=list)  # main calls, without consecutive repeats
    last: dict[str, Any] | None = None  # how the last main call ended
    instruction: str | None = None  # episode instruction captured before an interruption
    # Raw events are retained for planner-specific state restoration (for example a fixed
    # sequence cursor, which cannot be reconstructed from the de-duplicated history alone).
    events: list[dict[str, Any]] = field(default_factory=list)


def read_events(episode_dir: Path) -> list[dict[str, Any]]:
    path = episode_dir / "events.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"no trace at {path}")
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                break  # a crash can leave the last line half written; everything before it is intact
    return events


def read_env_snapshot(episode_dir: Path) -> dict[str, Any] | None:
    path = episode_dir / ENV_SNAPSHOT_FILE
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def restore_state(events: list[dict[str, Any]], env_snapshot: dict[str, Any] | None = None) -> RestoredState:
    ops: list[LedgerOp] = []
    memory = ""
    skill_calls = planner_calls = 0
    finished = False
    history: list[SkillCall] = []
    last_result: dict[str, Any] | None = None
    instruction: str | None = None

    def call_of(data: dict[str, Any]) -> SkillCall:
        return SkillCall(str(data["skill"]), dict(data.get("args") or {}))

    for event in events:
        kind = event.get("type")
        if kind == "ledger_update":
            ops.extend(LedgerOp.from_dict(op) for op in event["ops"])
        elif kind == "instruction":
            instruction = str(event.get("text") or "") or None
        elif kind == "memory_update":
            memory = event["memory"]
        elif kind == "skill_start":
            skill_calls += 1
            call = call_of(event["call"])
            if event.get("purpose", "main") == "main" and (not history or history[-1].key() != call.key()):
                history.append(call)
        elif kind == "skill_end" and event.get("purpose", "main") == "main":
            outcome = event.get("outcome") or {}
            last_result = {
                "call": call_of(event["call"]).key(),
                "status": outcome.get("status"),
                "reason": outcome.get("reason"),
            }
        elif kind == "human_command" and event.get("command") == "skip" and event.get("failed_call"):
            last_result = {"call": call_of(event["failed_call"]).key(), "status": "skipped", "reason": None}
        elif kind == "planner_request":
            planner_calls += 1
        elif kind == "episode_end":
            finished = True
    last = events[-1] if events else {}
    return RestoredState(
        ledger_ops=ops,
        memory=memory,
        skill_calls=skill_calls,
        planner_calls=planner_calls,
        env_snapshot=env_snapshot,
        last_seq=int(last.get("seq", 0)),
        last_t=float(last.get("t", 0.0)),
        finished=finished,
        history=history,
        last=last_result,
        instruction=instruction,
        events=events,
    )


def load_restored(episode_dir: Path) -> RestoredState:
    return restore_state(read_events(episode_dir), read_env_snapshot(episode_dir))
