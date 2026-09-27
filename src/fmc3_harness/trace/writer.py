"""Writes one episode's trace: events, planner calls, actions and the outcome (design doc §11)."""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from ..core.clock import Clock
from ..logs.context import current_context

EVENT_CONTEXT = ("episode_id", "state", "skill_call_id", "planner_call_id")
ENV_SNAPSHOT_FILE = "env_snapshot.json"


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(dataclasses.asdict(value))
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _dumps(value: Any) -> str:
    return json.dumps(to_jsonable(value), ensure_ascii=False, default=str)


class TraceWriter:
    """Append-only writer. Events are flushed one by one so a crash loses at most the last line."""

    def __init__(
        self, episode_dir: Path, clock: Clock, *, t0: float | None = None, start_seq: int = 0
    ) -> None:
        self.dir = episode_dir
        (self.dir / "planner").mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._t0 = clock.now() if t0 is None else t0
        self._seq = start_seq
        self._events = (self.dir / "events.jsonl").open("a", encoding="utf-8")
        self._actions = (self.dir / "actions.jsonl").open("a", encoding="utf-8")

    @property
    def elapsed(self) -> float:
        """Seconds since the episode started; resumed episodes continue the original timeline."""
        return self._clock.now() - self._t0

    def event(self, kind: str, **payload: Any) -> dict[str, Any]:
        self._seq += 1
        entry: dict[str, Any] = {
            "seq": self._seq,
            "t": round(self.elapsed, 4),
            "wall": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "type": kind,
        }
        context = current_context()
        for name in EVENT_CONTEXT:
            if context.get(name) is not None and name not in payload:
                entry[name] = context[name]
        entry.update(to_jsonable(payload))
        self._events.write(_dumps(entry) + "\n")
        self._events.flush()
        return entry

    def action(self, version: int, action: Any) -> None:
        self._actions.write(
            _dumps({"t": round(self.elapsed, 4), "version": version, "action": action}) + "\n"
        )

    def planner_call(self, call_id: str, record: dict[str, Any]) -> None:
        self.write_json(Path("planner") / f"{call_id}.json", record)

    def write_json(self, name: str | Path, data: Any, *, atomic: bool = False) -> None:
        text = json.dumps(to_jsonable(data), ensure_ascii=False, indent=2, default=str) + "\n"
        path = self.dir / name
        if not atomic:
            path.write_text(text, encoding="utf-8")
            return
        # Write then rename, so a crash mid-write never leaves a half-written file behind.
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)

    def write_bytes(self, name: str | Path, data: bytes) -> None:
        path = self.dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def write_yaml(self, name: str | Path, data: Any) -> None:
        text = yaml.safe_dump(to_jsonable(data), allow_unicode=True, sort_keys=False)
        (self.dir / name).write_text(text, encoding="utf-8")

    def close(self) -> None:
        self._events.close()
        self._actions.close()
