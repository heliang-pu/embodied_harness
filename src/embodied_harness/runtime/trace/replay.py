"""Planner replay: run a planner on recorded planner calls and compare decisions (design doc §11).

Each recorded call keeps its input summary and the images exactly as the model saw them, so a
newly trained planner can be checked offline against earlier runs before it goes on the robot.
Replay is open loop: it answers "would the new planner choose differently in the same
situation", not "what would have happened next".
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...config.schema import TaskSpec
from ...core.types import DecisionReason
from ...planner.base import Planner, PlannerContext
from ...planner.decision import DecisionError, parse_decision


@dataclass
class ReplayRow:
    episode: str
    call_id: str
    recorded: dict[str, Any] | None
    replayed: dict[str, Any] | None
    error: str | None
    same_skill: bool
    same_call: bool
    latency_s: float | None


def recorded_calls(episode_dir: Path) -> list[dict[str, Any]]:
    paths = sorted((episode_dir / "planner").glob("pc_*.json"))
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def context_from_record(record: dict[str, Any], episode_dir: Path, task: TaskSpec) -> PlannerContext:
    summary = record["input"]
    if summary.get("goal") and summary["goal"] != task.goal:
        task = task.model_copy(update={"goal": summary["goal"]})
    images = {
        Path(relative).stem: (episode_dir / relative).read_bytes()
        for relative in record.get("attachments", [])
    }
    return PlannerContext(
        task=task,
        images=images,
        proprio=dict(summary["proprio"]),
        memory=summary["memory"],
        reason=DecisionReason(summary["reason"]),
        skills=list(summary["skills"]),
        resample=int(summary.get("resample", 0)),
        history=list(summary.get("history") or []),  # planners with show_history render it
        ledger=summary.get("ledger"),  # with show_progress; None in traces from before it was recorded
        last=summary.get("last"),  # with show_state
        rejected=list(summary.get("rejected") or []),  # with show_rejections
    )


def compare_decisions(recorded: dict[str, Any] | None, replayed: dict[str, Any] | None) -> tuple[bool, bool]:
    if recorded is None or replayed is None:
        return False, False
    if recorded.get("done") or replayed.get("done"):
        both_done = bool(recorded.get("done")) and bool(replayed.get("done"))
        return both_done, both_done
    same_skill = recorded.get("skill") == replayed.get("skill")
    same_call = same_skill and (recorded.get("args") or {}) == (replayed.get("args") or {})
    return same_skill, same_call


def replay(planner: Planner, episode_dirs: list[Path], task: TaskSpec) -> list[ReplayRow]:
    rows = []
    for episode_dir in episode_dirs:
        for record in recorded_calls(episode_dir):
            context = context_from_record(record, episode_dir, task)
            replayed: dict[str, Any] | None = None
            error: str | None = None
            latency: float | None = None
            started = time.perf_counter()
            try:
                output = planner.plan(context)
                latency = time.perf_counter() - started
                replayed = parse_decision(output.raw).to_dict()
            except DecisionError as exc:
                error = f"unparseable output: {exc}"
            except Exception as exc:
                error = repr(exc)
            recorded = record.get("decision")
            same_skill, same_call = compare_decisions(recorded, replayed)
            rows.append(
                ReplayRow(
                    episode_dir.name,
                    record["call_id"],
                    recorded,
                    replayed,
                    error,
                    same_skill,
                    same_call,
                    latency,
                )
            )
    return rows


def summarize(rows: list[ReplayRow]) -> dict[str, Any]:
    compared = [row for row in rows if row.recorded is not None]
    count = len(compared)
    same_skill = sum(row.same_skill for row in compared)
    same_call = sum(row.same_call for row in compared)
    return {
        "calls": len(rows),
        "compared": count,
        "errors": sum(row.error is not None for row in rows),
        "same_skill": same_skill,
        "same_call": same_call,
        "same_skill_rate": same_skill / count if count else None,
        "same_call_rate": same_call / count if count else None,
    }
