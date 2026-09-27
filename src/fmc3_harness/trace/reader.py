"""Reads a trace back and rebuilds what an interrupted episode needs to resume (design doc §13).

The ledger and the planner's memory are event-sourced: replaying `ledger_update` and
`memory_update` events gives exactly the state the supervisor had. Simulated envs also leave
their latest state in env_snapshot.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..memory.ledger import LedgerOp
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
    for event in events:
        kind = event.get("type")
        if kind == "ledger_update":
            ops.extend(LedgerOp.from_dict(op) for op in event["ops"])
        elif kind == "memory_update":
            memory = event["memory"]
        elif kind == "skill_start":
            skill_calls += 1
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
    )


def load_restored(episode_dir: Path) -> RestoredState:
    return restore_state(read_events(episode_dir), read_env_snapshot(episode_dir))
