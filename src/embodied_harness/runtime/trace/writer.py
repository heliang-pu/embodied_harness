"""Writes one episode's trace: events, planner calls, actions and the outcome (design doc §11)."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from ...core.clock import Clock
from ..logs.context import current_context

EVENT_CONTEXT = ("episode_id", "state", "skill_call_id", "planner_call_id")
ENV_SNAPSHOT_FILE = "env_snapshot.json"
SNAPSHOTS_DIR = "snapshots"  # snapshots/<call_id>.json: the world when that call started (trace.snapshots)


def _numpy_like(value: Any) -> bool:
    """A numpy array or scalar, recognized without importing numpy: it has tolist() and is not a JSON value
    already (np.float64 is a float), nor a class (np.ndarray's tolist() needs an instance)."""
    return not isinstance(value, (str, bool, int, float, type)) and callable(getattr(value, "tolist", None))


def to_jsonable(value: Any, *, default: Callable[[Any], Any] | None = None) -> Any:
    """The one JSON normalizer of traces and reports: enums by value, dataclasses and models as dicts, tuples
    and sets as lists, paths as text, and numpy arrays and scalars as their exact numbers (tolist()), never
    their printed text. Any other value, a longdouble or a class among them, goes through `default` when
    given, else is left to the dumper."""
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(dataclasses.asdict(value), default=default)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item, default=default) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item, default=default) for item in value]
    if isinstance(value, Path):
        return str(value)
    if _numpy_like(value):
        converted = value.tolist()
        if not _numpy_like(converted):  # a longdouble's tolist() is a longdouble again
            return to_jsonable(converted, default=default)
    return value if default is None else default(value)


def _dumps(value: Any) -> str:
    return json.dumps(to_jsonable(value), ensure_ascii=False, default=str)


def _as_prompted(record: dict[str, Any]) -> dict[str, Any]:
    """A planner record whose input proprio keeps each numpy value as the text the prompt showed.

    `replay` and `data from-runs` render the recorded input again, and planner/render.format_proprio prints
    such a value with str(), in numpy's own layout, which a list of its numbers would not reproduce."""
    summary = record.get("input")
    proprio = summary.get("proprio") if isinstance(summary, Mapping) else None
    if not isinstance(proprio, Mapping):
        return record
    shown = {key: str(value) if _numpy_like(value) else value for key, value in proprio.items()}
    return {**record, "input": {**summary, "proprio": shown}}


def last_record(path: Path) -> dict[str, Any] | None:
    """The last JSON line of a JSONL file, skipping a torn last line; None when there is none."""
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        position = handle.seek(0, 2)
        data = b""
        while position > 0:
            size = min(position, 1 << 16)
            position -= size
            handle.seek(position)
            data = handle.read(size) + data
            lines = data.split(b"\n")
            for line in reversed(lines if position == 0 else lines[1:]):  # lines[0] may be cut
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except (ValueError, UnicodeError):
                    continue  # the torn tail a crash can leave
                if isinstance(record, dict):
                    return record
    return None


def repair_tail(path: Path) -> None:
    """Before appending: drop a torn last line, or end a complete last JSON value with its newline."""
    if not path.is_file() or path.stat().st_size == 0:
        return
    with path.open("r+b") as handle:
        handle.seek(-1, 2)
        if handle.read(1) == b"\n":
            return
        end = handle.seek(0, 2)
        position, tail, boundary = end, b"", 0
        while position:
            size = min(position, 1 << 16)
            position -= size
            handle.seek(position)
            chunk = handle.read(size)
            newline = chunk.rfind(b"\n")
            if newline >= 0:
                boundary = position + newline + 1
                tail = chunk[newline + 1 :] + tail
                break
            tail = chunk + tail
        try:
            json.loads(tail)
        except (ValueError, UnicodeError):
            handle.truncate(boundary)
        else:
            handle.seek(0, 2)
            handle.write(b"\n")


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
        repair_tail(self.dir / "events.jsonl")
        repair_tail(self.dir / "actions.jsonl")
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
        entry = to_jsonable({**entry, **payload})  # one pass over the whole event
        self._events.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        self._events.flush()
        return entry

    def action(self, version: int, action: Any) -> None:
        self._actions.write(
            _dumps({"t": round(self.elapsed, 4), "version": version, "action": action}) + "\n"
        )

    def planner_call(self, call_id: str, record: dict[str, Any]) -> None:
        self.write_json(Path("planner") / f"{call_id}.json", _as_prompted(record))

    def write_json(self, name: str | Path, data: Any, *, atomic: bool = False) -> None:
        text = json.dumps(to_jsonable(data), ensure_ascii=False, indent=2, default=str) + "\n"
        path = self.dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
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
