"""`embodied-harness probe`: rerun skill calls from recorded start states.

A run with `trace.snapshots: true` keeps the world as it was when each skill call started
(snapshots/<call_id>.json). The probe restores one of those states and runs a single call there until its
verifier decides: no planner, no recovery table, no transition rules. Repeating it gives the call's success
rate from exactly that start. Two uses:
- readiness: the same call from a clean start (e.g. the first call of an episode) and from the state a
  previous skill left behind. A skill that succeeds from the first and fails from the second cannot start
  from where the previous one stops, which a transition rule or more data from such states can fix;
- instruction following: different calls from one start (e.g. pick(color=red) and pick(color=green)). The
  verdict comes from each skill's verifiers, so this only separates targets a verifier can tell apart.

Simulated envs only: the env needs snapshot() and restore(). Repeated trials differ only where the env or
the policy is random (a flow-matching policy samples its noise each time).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config.loader import Bundle
from ..config.schema import TaskSpec
from ..config.skills import task_namespace
from ..core.clock import SimClock
from ..core.ledger import Ledger, LedgerOp
from ..core.types import SkillCall
from ..data.labels import parse_call
from ..policy.executor.chunk_queue import ChunkQueue
from .factory import build_env, build_policy, build_scripted
from .trace.reader import read_events
from .trace.writer import SNAPSHOTS_DIR
from .verifier import VerifierSession

START = "START"
_CALL_ID = re.compile(r"^sc_\d+$")


@dataclass(frozen=True)
class StartState:
    """The world when one recorded call started, and what led there."""

    episode_dir: Path
    call_id: str
    call: SkillCall  # the call that was run from here
    previous: str  # the main call started before it, or START
    snapshot: dict[str, Any]
    ledger: dict[str, Any]  # the ledger at that moment (verifier conditions may read it)
    goal: str | None  # the episode's own instruction, when the task takes it from the episode

    @property
    def label(self) -> str:
        return f"{self.episode_dir.name}:{self.call_id}"


@dataclass
class Trial:
    start: str
    after: str
    call: str
    trial: int
    status: str  # success, failure, not_ready (a precondition was false) or ended (the env ended it)
    reason: str | None
    elapsed_s: float
    env_success: bool | None = None


def load_start_states(spec: str, bundle: Bundle) -> list[StartState]:
    """`EPISODE_DIR:CALL_ID` for one call's start, or `EPISODE_DIR` for every main call with a snapshot."""
    head, _, tail = spec.rpartition(":")
    directory, wanted = (Path(head), tail) if head and _CALL_ID.match(tail) else (Path(spec), None)
    events = read_events(directory)
    goal = next((str(e["text"]) for e in events if e.get("type") == "instruction"), None)
    ledger = Ledger(bundle.task.ledger)
    previous = START
    starts: list[StartState] = []
    for event in events:
        kind = event.get("type")
        if kind == "ledger_update":
            for op in event["ops"]:
                ledger.apply(LedgerOp.from_dict(op))
        if kind != "skill_start":
            continue
        call = SkillCall(str(event["call"]["skill"]), dict(event["call"].get("args") or {}))
        call_id = str(event["call_id"])
        main = event.get("purpose", "main") == "main"
        if wanted == call_id or (wanted is None and main):
            path = directory / SNAPSHOTS_DIR / f"{call_id}.json"
            if not path.is_file():
                raise FileNotFoundError(
                    f"{path} not found; record the run with `trace.snapshots: true` (a simulated env)"
                )
            snapshot = json.loads(path.read_text(encoding="utf-8"))
            starts.append(StartState(directory, call_id, call, previous, snapshot, ledger.view(), goal))
        if main:
            previous = call.key()
    if wanted is not None and not starts:
        raise ValueError(f"{directory} has no skill call {wanted}")
    return starts


def run_call(
    bundle: Bundle,
    env: Any,
    policy: Any,
    scripted: Any,
    call: SkillCall,
    ledger: dict[str, Any],
    *,
    task: TaskSpec | None = None,
) -> tuple[str, str | None, float, bool | None]:
    """Run one call from the env's current state until its verifier decides.

    Returns (status, reason, elapsed seconds, the env's own verdict if it ended the episode).
    """
    registry = bundle.registry
    skill = registry.skill(call.skill)
    executor = scripted if skill.scripted else policy
    clock = SimClock()
    queue = ChunkQueue(bundle.run.executor.handoff)
    version = 1
    executor.set_instruction(call, registry.resolve_prompt(call), version)
    queue.switch(version)
    session = VerifierSession(skill, started_at=clock.now(), timeout_s=registry.timeout_for(call.skill))
    period = 1.0 / bundle.run.runtime.control_hz
    task_ns = task_namespace(task if task is not None else bundle.task)
    first = True
    while True:
        obs = env.observe()
        elapsed = clock.now() - session.started_at
        if obs.terminated:
            success = obs.info.get("success")
            return "ended", None, elapsed, None if success is None else bool(success)
        namespace: dict[str, Any] = dict(obs.proprio)
        namespace.update(registry.predicates(obs.proprio))
        namespace.update(ledger=ledger, task=task_ns, args=dict(call.args))
        namespace["executor_done"] = executor.is_done() and queue.pending(version) == 0
        if first:
            unmet = [expr.source for expr in skill.preconditions if not expr(namespace)]
            if unmet:
                return "not_ready", ", ".join(unmet), 0.0, None
            first = False
        outcome = session.evaluate(clock.now(), namespace, obs.proprio)
        if outcome is not None:
            return outcome.status.value, outcome.reason, elapsed, None
        if queue.pending(version) == 0 and not executor.is_done():
            queue.push(executor.infer_chunk(obs))
        item = queue.pop()
        if item is not None:
            env.act(item.action)
        else:
            env.hold()
        clock.sleep_until(clock.now() + period)


def probe(
    bundle: Bundle,
    starts: Sequence[StartState],
    calls: Sequence[SkillCall] = (),
    *,
    trials: int = 5,
    seed: int = 0,
    env: Any = None,
    policy: Any = None,
) -> list[Trial]:
    """Run each call (the recorded one when `calls` is empty) `trials` times from each start state."""
    for call in calls:
        problems = bundle.registry.check_call(call)
        if problems:
            raise ValueError("; ".join(problems))
    env = env if env is not None else build_env(bundle)
    if not callable(getattr(env, "restore", None)):
        raise ValueError(f"env {bundle.run.env.type!r} cannot restore a snapshot; probing needs a simulator")
    policy = policy if policy is not None else build_policy(bundle)
    scripted = build_scripted(bundle)
    results: list[Trial] = []
    try:
        for start in starts:
            task = bundle.task if start.goal is None else bundle.task.model_copy(update={"goal": start.goal})
            for executor in (policy, scripted):
                begin = getattr(executor, "begin_episode", None)
                if callable(begin):
                    begin(task)
            for call in calls or [start.call]:
                for trial in range(trials):
                    env.reset(seed + trial)
                    env.restore(start.snapshot)
                    policy.reset()
                    scripted.reset()
                    status, reason, elapsed, env_success = run_call(
                        bundle, env, policy, scripted, call, start.ledger, task=task
                    )
                    results.append(
                        Trial(
                            start.label,
                            start.previous,
                            call.key(),
                            trial,
                            status,
                            reason,
                            round(elapsed, 3),
                            env_success,
                        )
                    )
    finally:
        close = getattr(policy, "close", None)
        if callable(close):
            close()
        env.close()
    return results


@dataclass
class ProbeRow:
    start: str
    after: str
    call: str
    trials: int = 0
    success: int = 0
    failure: int = 0
    not_ready: int = 0
    reasons: Counter[str] = field(default_factory=Counter)
    durations: list[float] = field(default_factory=list)

    @property
    def rate(self) -> float | None:
        ran = self.trials - self.not_ready
        return self.success / ran if ran else None


def summarize(trials: Sequence[Trial]) -> list[ProbeRow]:
    rows: dict[tuple[str, str], ProbeRow] = {}
    for trial in trials:
        key = (trial.start, trial.call)
        row = rows.get(key)
        if row is None:
            row = rows[key] = ProbeRow(trial.start, trial.after, trial.call)
        row.trials += 1
        succeeded = trial.status == "success" or (trial.status == "ended" and trial.env_success is True)
        if succeeded:
            row.success += 1
            row.durations.append(trial.elapsed_s)
        elif trial.status == "not_ready":
            row.not_ready += 1
            row.reasons[f"not ready: {trial.reason}"] += 1
        else:
            row.failure += 1
            row.reasons[trial.reason or ("env_ended" if trial.status == "ended" else trial.status)] += 1
    return list(rows.values())


def format_probe(rows: Sequence[ProbeRow]) -> str:
    lines = [
        "| start | after | call | trials | success | failed | rate | mean s (success) | failure reasons |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        rate = "-" if row.rate is None else f"{row.rate:.0%}"
        mean = f"{sum(row.durations) / len(row.durations):.1f}" if row.durations else "-"
        reasons = ", ".join(f"{reason} x{count}" for reason, count in row.reasons.most_common(3)) or "-"
        lines.append(
            f"| {row.start} | {row.after} | {row.call} | {row.trials} | {row.success} | "
            f"{row.failure + row.not_ready} | {rate} | {mean} | {reasons} |"
        )
    return "\n".join(lines)


def parse_calls(texts: Sequence[str]) -> list[SkillCall]:
    return [parse_call(text) for text in texts]
