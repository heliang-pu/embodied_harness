"""Plays a fixed list of skill calls: the task's plan (task.plan) or its own params.calls.

A known fixed order provides a baseline for the executor and transitions, and runs long tasks whose order
never changes. Verified outcomes move it: a success advances, a failure repeats the step, and a failure
whose on_failure entry names a `rollback` skill resumes from the nearest earlier step calling that skill
(a drop undid the pick), when that step can start. After a HOLD a human resume repeats the latest proposal,
unless a main call finished since it was made (a budget, an e-stop or a harness error between a verified
success and the next planner call): then the plan goes on after it. It is only as appropriate as the
supplied steps; visual contingencies can still require a different order.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from ..config.expressions import ExpressionError
from ..config.skills import task_namespace
from ..core.types import DecisionReason, SkillCall
from .base import PlannerContext, PlannerOutput

if TYPE_CHECKING:
    from ..config.loader import Bundle
    from ..config.schema import PlannerConfig, TaskSpec
    from ..config.skills import SkillRegistry

_REPEAT_REASONS = (DecisionReason.SKILL_FAILED, DecisionReason.HUMAN_RESUME, DecisionReason.RESUME)
_DONE_STATUSES = ("success", "skipped")


def _parse_calls(raw: Any) -> tuple[list[SkillCall] | None, list[str]]:
    """params.calls as calls, or the problems that stop it from being read."""
    if not isinstance(raw, list):
        return None, ["planner.params.calls: expected a list of {skill, args}"]
    calls: list[SkillCall] = []
    problems: list[str] = []
    for index, item in enumerate(raw):
        if (
            not isinstance(item, Mapping)
            or not isinstance(item.get("skill"), str)
            or not isinstance(item.get("args") or {}, Mapping)
        ):
            problems.append(f"planner.params.calls[{index}]: expected {{skill, args}}, got {item!r}")
            continue
        calls.append(SkillCall(item["skill"], dict(item.get("args") or {})))
    return (None if problems else calls), problems


class SequencePlanner:
    name = "sequence"

    def __init__(self, calls: Sequence[Mapping[str, Any]] | None = None) -> None:
        """Without calls, bind() takes the task's plan."""
        self._given = calls is not None
        self._calls = [SkillCall(str(c["skill"]), dict(c.get("args", {}))) for c in calls or ()]
        self._next = 0
        # The first RESUME request uses the recovered cursor without repeating its previous step.
        self._resume_cursor_ready = False
        # The latest proposal (len(calls) for done; None before the first), which a resample issues again, and
        # the last main call and ledger it was made with: a HUMAN_RESUME compares them to tell whether a call
        # finished since.
        self._proposed: int | None = None
        self._seen: tuple[dict[str, Any] | None, dict[str, Any] | None] | None = None
        # Set by bind(); an unbound planner never rolls back.
        self._registry: SkillRegistry | None = None
        self._rollbacks: list[dict[str, int]] = []
        self._done_when: str | None = None

    @classmethod
    def check_config(cls, bundle: Bundle) -> list[str]:
        """Validate-time checks (factory.check_bundle) of what bind() would refuse. Nothing to check when
        neither params.calls nor task.plan is set."""
        planner = bundle.run.planner
        raw = planner.params.get("calls")
        plan = bundle.registry.plan
        if raw is None and plan is None:
            return []
        problems: list[str] = []
        if planner.call != "boundary":
            problems.append(
                "planner.call: the sequence planner proposes a step when the previous one ends; "
                "use call: boundary"
            )
        if raw is not None:
            calls, parse_problems = _parse_calls(raw)
            problems += parse_problems
            if calls is not None:
                problems += bundle.registry.check_plan(calls, "planner.params.calls")
                if plan is not None and [c.key() for c in calls] != [c.key() for c in plan]:
                    problems.append(
                        "planner.params.calls differs from task.plan; leave it out to play task.plan"
                    )
        return problems

    def bind(self, *, registry: SkillRegistry, task: TaskSpec, planner_config: PlannerConfig) -> None:
        """Called by factory.bind_planner: takes task.plan when no calls were given, and the skills'
        rollback rules."""
        plan = registry.plan
        if planner_config.call != "boundary":
            raise ValueError("the sequence planner needs planner.call: boundary")
        if not self._given:
            if plan is None:
                raise ValueError("the sequence planner has no steps: set task.plan or planner.params.calls")
            self._calls = list(plan)
        elif plan is not None and [c.key() for c in self._calls] != [c.key() for c in plan]:
            raise ValueError("planner.params.calls differs from task.plan")
        self._registry = registry
        self._rollbacks = registry.plan_rollbacks(self._calls)
        self._done_when = task.done_when

    def restore_trace(self, events: Sequence[Mapping[str, Any]]) -> None:
        """Rebuild the cursor from responses and main-call outcomes, including legacy traces.

        Dispatch history merges adjacent equal calls and omits accepted calls still waiting for
        transitions, so it cannot identify a sequence position. New responses carry the exact
        index; older ones are replayed using their request reason and resample count. A finished
        main call advances the cursor, while an unfinished or rejected proposal is repeated. A main call
        that differs from its proposal (a human retry of the step a rollback left) is the first step with
        that call at or after the proposal, and its success or skip moves the cursor past it, as plan() does.
        """
        self._next = 0
        self._resume_cursor_ready = False
        self._proposed, self._seen = None, None
        requests: dict[str, Mapping[str, Any]] = {}
        accepted: dict[str, int] = {}
        started: dict[str, int] = {}
        pending: int | None = None
        latest_accepted: int | None = None

        def call_key(data: Mapping[str, Any]) -> str:
            return SkillCall(str(data["skill"]), dict(data.get("args") or {})).key()

        def completed(index: int) -> None:
            nonlocal pending
            if index == pending:
                pending = None
            elif pending is not None and index > pending:
                pending = None
                self._next = index + 1

        for event in events:
            kind = event.get("type")
            if kind == "episode_resume":
                if pending is not None:
                    self._next = pending
                self._resume_cursor_ready = True
            elif kind == "planner_request":
                requests[str(event["call_id"])] = event
            elif kind == "planner_response":
                request = requests.get(str(event["call_id"]))
                if request is None:
                    raise ValueError("sequence resume requires the request for every planner response")
                meta = event.get("meta") or {}
                if "sequence_index" in meta:
                    index = int(meta["sequence_index"])
                else:
                    index = self._proposal_index(
                        DecisionReason(request["reason"]), int(request.get("resample", 0))
                    )
                decision = event.get("decision") or {}
                if event.get("errors") and not decision:
                    # An unparsable answer has no sequence position. The next valid resample
                    # still carries enough information to restore the cursor.
                    continue
                valid = 0 <= index <= len(self._calls)
                if index == len(self._calls):
                    valid = valid and decision.get("done") is True
                else:
                    valid = valid and "skill" in decision and call_key(decision) == self._calls[index].key()
                if not valid:
                    raise ValueError(
                        f"sequence resume: {event['call_id']} does not match configured step {index + 1}"
                    )
                self._next = min(index + 1, len(self._calls))
                self._resume_cursor_ready = False
                pending = index if index < len(self._calls) else None
                if not event.get("errors") and index < len(self._calls):
                    accepted[str(event["call_id"])] = latest_accepted = index
            elif kind == "skill_start" and event.get("purpose", "main") == "main":
                index = accepted.get(str(event.get("planner_call_id")), latest_accepted)
                key = call_key(event["call"])
                if index is not None and key != self._calls[index].key():
                    index = self._find(index, key)
                if index is None:
                    raise ValueError("sequence resume cannot match a main skill to a recorded proposal")
                started[str(event["call_id"])] = index
            elif kind == "skill_end" and event.get("purpose", "main") == "main":
                index = started.get(str(event["call_id"]))
                if index is not None and event.get("outcome", {}).get("status") in _DONE_STATUSES:
                    completed(index)
            elif kind == "human_command" and event.get("command") == "skip":
                call = event.get("failed_call")
                if pending is not None and call:
                    index = self._find(pending, call_key(call))
                    if index is not None:
                        completed(index)
        if pending is not None:
            self._next = pending
        self._resume_cursor_ready = True

    def _proposal_index(self, reason: DecisionReason, resample: int) -> int:
        restored_resume = self._resume_cursor_ready and reason is DecisionReason.RESUME
        repeat = resample > 0 or (reason in _REPEAT_REASONS and not restored_resume)
        return self._next - 1 if repeat and self._next > 0 else self._next

    def _find(self, start: int, key: str) -> int | None:
        """The first step at or after `start` that makes the call `key`."""
        return next((j for j in range(start, len(self._calls)) if self._calls[j].key() == key), None)

    def _follow(self, index: int, last: Mapping[str, Any] | None) -> int:
        """After a main call finished (success or skip): the step after it. That is the latest proposal, or
        a later step when a human retried or skipped the step a rollback left instead. `index` when `last`
        did not finish a step at or after the latest proposal."""
        proposed = self._next - 1
        if proposed < 0 or not last or last.get("status") not in _DONE_STATUSES:
            return index
        found = self._find(proposed, str(last.get("call")))
        return index if found is None else min(found + 1, len(self._calls))

    def _after_hold(self, context: PlannerContext) -> int:
        """HUMAN_RESUME: the latest proposal again, unless a main call finished since it was made (a HOLD
        between a verified success and the next planner call); then the step after that call. `last` alone
        cannot tell, since it still describes the call before the proposal when the proposal never ended, so
        it is compared (with the ledger) to what the proposal was made with."""
        if self._resume_cursor_ready or self._proposed is None:
            return self._next  # the cursor restored from the trace, or nothing proposed yet
        if (context.last, context.ledger) != self._seen:
            return self._follow(self._proposed, context.last)
        return self._proposed

    def _rollback(self, index: int, context: PlannerContext) -> tuple[int, dict[str, Any]]:
        """The step to propose instead of repeating step `index` after it failed, and the meta saying why."""
        last = context.last
        if self._registry is None or not last or last.get("status") != "failure":
            return index, {}
        if last.get("call") != self._calls[index].key():
            return index, {}  # the last failure is not this step's (e.g. a rollback target's resample)
        reason = last.get("reason")
        target = self._rollbacks[index].get(str(reason))
        if target is None:
            return index, {}
        unmet = self._unmet(self._calls[target], context)
        if unmet:
            return index, {"rollback_blocked": {"to": target, "reason": reason, "unmet": unmet}}
        return target, {"rollback": {"from": index, "reason": reason}}

    def _unmet(self, call: SkillCall, context: PlannerContext) -> list[str]:
        """The call's preconditions that do not hold now, evaluated the way the supervisor checks them."""
        registry = self._registry
        assert registry is not None
        try:
            namespace: dict[str, Any] = dict(context.proprio)
            namespace.update(registry.predicates(context.proprio))
            namespace["ledger"] = context.ledger if context.ledger is not None else {}
            namespace["task"] = task_namespace(context.task)
            namespace["args"] = dict(call.args)
            namespace["executor_done"] = False
            return [expr.source for expr in registry.skill(call.skill).preconditions if not expr(namespace)]
        except ExpressionError as exc:
            return [str(exc)]

    def plan(self, context: PlannerContext) -> PlannerOutput:
        # After a failure, an interruption or a rejected output the same step is issued again.
        if context.resample > 0 and self._proposed is not None:
            index = self._proposed  # a refused answer is issued again, a refused done included
        elif context.reason is DecisionReason.HUMAN_RESUME and context.resample == 0:
            index = self._after_hold(context)
        else:
            index = self._proposal_index(context.reason, context.resample)
            if context.reason is DecisionReason.SKILL_DONE and context.resample == 0:
                index = self._follow(index, context.last)
        self._resume_cursor_ready = False
        self._seen = copy.deepcopy((context.last, context.ledger))
        extra: dict[str, Any] = {}
        if index >= len(self._calls):
            if context.reason is DecisionReason.CHECK_FAILED:
                done_when = f" ({self._done_when})" if self._done_when else ""
                raise RuntimeError(f"the plan has ended but done_when{done_when} is false")
            decision: dict[str, Any] = {"done": True}
        else:
            index, extra = self._rollback(index, context)
            decision = self._calls[index].to_dict()
        self._proposed = index
        self._next = min(index + 1, len(self._calls))
        decision["memory"] = f"第 {self._next}/{len(self._calls)} 步"
        return PlannerOutput(
            raw=json.dumps(decision, ensure_ascii=False),
            meta={"planner": self.name, "sequence_index": index, **extra},
        )
