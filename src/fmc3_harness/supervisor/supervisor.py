"""The supervisor: a deterministic state machine that owns the control flow.

The planner only proposes. The supervisor decides when to call it, checks what comes back,
dispatches skill calls, judges them through verifiers, updates the ledger and recovers from
failures (design doc §7 and §13). Given the same inputs it always takes the same branch,
which is what makes traces replayable.
"""

from __future__ import annotations

import dataclasses
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..config.schema import PlannerConfig, TaskSpec
from ..core.clock import Clock
from ..core.ids import IdSequence
from ..core.types import DecisionReason, Observation, Outcome, Prompt, SkillCall
from ..env.base import Env
from ..executor.base import Executor
from ..executor.chunk_queue import ChunkQueue
from ..logs.codes import ErrorCode
from ..logs.context import update_context
from ..logs.health import HealthMonitor
from ..memory.ledger import Ledger, LedgerError
from ..planner.base import PlannerContext, PlannerOutput
from ..planner.decision import DecisionError, PlannerDecision, parse_decision
from ..planner.runner import PlannerRunner
from ..skills.expressions import ExpressionError
from ..skills.registry import CompiledSkill, SkillRegistry
from ..trace.writer import ENV_SNAPSHOT_FILE, TraceWriter
from ..verifiers.session import VerifierSession

logger = logging.getLogger("fmc3_harness.supervisor")

HUMAN_COMMANDS = ("resume", "retry", "skip", "abort")


class State(str, Enum):
    PLAN = "PLAN"
    EXECUTE = "EXECUTE"
    RETRY = "RETRY"
    HOLD = "HOLD"
    DONE = "DONE"
    ABORT = "ABORT"


@dataclass
class ActiveCall:
    """The skill call currently running."""

    call: SkillCall
    skill: CompiledSkill
    prompt: Prompt | None
    version: int
    call_id: str
    executor: Executor
    session: VerifierSession
    purpose: str  # "main", or "before_retry" for a recovery primitive
    retries: dict[str, int] = field(default_factory=dict)


@dataclass
class PlanJob:
    call_id: str
    context: PlannerContext
    submitted_at: float


@dataclass
class EpisodeResult:
    status: str  # "done" or "aborted"
    error_code: str | None
    message: str | None
    ledger: dict[str, Any]
    memory: str
    skill_calls: int
    planner_calls: int
    resamples: int
    human_commands: list[str]
    duration_s: float

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class Supervisor:
    def __init__(
        self,
        *,
        registry: SkillRegistry,
        task: TaskSpec,
        planner_config: PlannerConfig,
        runner: PlannerRunner,
        vla: Executor,
        scripted: Executor,
        queue: ChunkQueue,
        env: Env,
        ledger: Ledger,
        trace: TraceWriter,
        clock: Clock,
        health: HealthMonitor | None = None,
        memory: str = "",
        skill_calls: int = 0,
        planner_calls: int = 0,
    ) -> None:
        self._registry = registry
        self._task = task
        self._planner_cfg = planner_config
        self._runner = runner
        self._vla = vla
        self._scripted = scripted
        self._queue = queue
        self._env = env
        self._ledger = ledger
        self._trace = trace
        self._clock = clock
        self._health = health
        self._task_ns = {
            "id": task.id,
            "goal": task.goal,
            "rules": list(task.rules),
            "expected_total": task.expected_total,
        }

        self.state = State.PLAN
        self.memory = memory
        self.resamples = 0
        self.human_commands: list[str] = []
        self.result: EpisodeResult | None = None

        self._skill_ids = IdSequence("sc", skill_calls)
        self._planner_ids = IdSequence("pc", planner_calls)
        self._version = 0
        self._active: ActiveCall | None = None
        self._plan: PlanJob | None = None
        self._plan_reason = DecisionReason.START
        self._plan_failures = 0
        self._resample = 0
        self._next_plan_at: float | None = None
        self._retry_queue: list[str] = []
        self._retry_target: tuple[SkillCall, dict[str, int]] | None = None
        self._failure_streak: dict[str, int] = defaultdict(int)
        self._last_failed: ActiveCall | None = None
        self._last_call_key: str | None = None
        self._last_switch_at: float | None = None
        self._hold_code: ErrorCode | None = None
        self._hold_message: str | None = None
        self._time_budget_hit = False
        self._started_at = clock.now()
        self._last_obs: Observation | None = None
        update_context(state=self.state.value)

    # --- public ----------------------------------------------------------------------------------

    @property
    def finished(self) -> bool:
        return self.state in (State.DONE, State.ABORT)

    @property
    def executing(self) -> bool:
        return self.state in (State.EXECUTE, State.RETRY)

    @property
    def active(self) -> ActiveCall | None:
        return self._active

    @property
    def skill_calls(self) -> int:
        return self._skill_ids.count

    @property
    def planner_calls(self) -> int:
        return self._planner_ids.count

    def hold_info(self) -> tuple[ErrorCode | None, str | None, str | None]:
        failed = self._last_failed.call.key() if self._last_failed else None
        return self._hold_code, self._hold_message, failed

    def start(self, reason: DecisionReason = DecisionReason.START) -> None:
        self._started_at = self._clock.now()
        self._request_plan(reason)

    def tick(self, obs: Observation) -> None:
        self._last_obs = obs
        if self.finished or self.state is State.HOLD:
            return
        try:
            if self._time_budget_exceeded():
                return
            if self.state is State.PLAN:
                self._tick_plan(obs)
            else:
                self._tick_execute(obs)
        except (ExpressionError, LedgerError) as exc:
            logger.error("config expression failed at runtime: %s", exc, extra={"code": ErrorCode.E_CONFIG})
            self._trace.event("error", code=ErrorCode.E_CONFIG, error=str(exc))
            self.hold(f"config error at runtime: {exc}", ErrorCode.E_CONFIG)

    def executor_done(self) -> bool:
        active = self._active
        return active is not None and active.executor.is_done() and self._queue.pending(active.version) == 0

    def hold(self, message: str, code: ErrorCode | None = None) -> None:
        """Stop in place and wait for a human (design doc §13)."""
        if self.finished:
            return
        self._runner.cancel()
        self._plan = None
        self._queue.clear()
        self._env.hold()
        self._hold_code, self._hold_message = code, message
        self._set_state(State.HOLD)
        self._trace.event("hold", code=code, message=message)
        if code is not None:
            logger.error("HOLD: %s", message, extra={"code": code})
        else:
            logger.warning("HOLD: %s", message)

    def emergency_stop(self) -> None:
        self.hold("emergency stop")

    def command(self, command: str) -> None:
        """A human decision while holding: resume, retry, skip or abort."""
        if self.state is not State.HOLD:
            logger.warning("ignoring %r: not holding", command)
            return
        if command not in HUMAN_COMMANDS:
            logger.warning("unknown command %r; expected one of %s", command, ", ".join(HUMAN_COMMANDS))
            return
        failed = self._last_failed
        obs = self._last_obs or Observation()
        if command == "skip" and failed is None:
            logger.warning("nothing to skip: the hold was not caused by a failed skill")
            return
        self.human_commands.append(command)
        self._trace.event(
            "human_command", command=command, failed_call=failed.call.to_dict() if failed else None
        )
        logger.info("human command: %s", command)
        if command == "abort":
            self._finish(State.ABORT)
        elif command == "resume":
            if failed is not None:
                self._failure_streak.pop(failed.call.key(), None)
            self._request_plan(DecisionReason.HUMAN_RESUME)
        elif command == "retry" and failed is not None:
            self._failure_streak.pop(failed.call.key(), None)
            self._dispatch(failed.call, obs, "main")
        elif command == "retry":
            self._request_plan(DecisionReason.HUMAN_RESUME)
        elif failed is not None:  # skip: the human finished this step by hand
            self._commit(failed.call, obs, {"skipped_by": "human"})

    # --- planning --------------------------------------------------------------------------------

    def _request_plan(self, reason: DecisionReason) -> None:
        self._plan_reason = reason
        self._plan = None
        self._plan_failures = 0
        self._resample = 0
        self._next_plan_at = None
        self._active = None
        update_context(skill_call_id=None)
        self._set_state(State.PLAN)

    def _tick_plan(self, obs: Observation) -> None:
        now = self._clock.now()
        if self._plan is None:
            if self._next_plan_at is not None and now < self._next_plan_at:
                return
            if not self._submit_plan(obs):
                return
        assert self._plan is not None
        result = self._runner.poll()
        if result is None:
            if now - self._plan.submitted_at > self._planner_cfg.timeout_s:
                self._runner.cancel()
                self._planner_failed(
                    f"planner call {self._plan.call_id} timed out after {self._planner_cfg.timeout_s:g}s"
                )
            return
        job, self._plan = self._plan, None
        if isinstance(result, BaseException):
            logger.error("planner call %s raised %r", job.call_id, result, exc_info=result)
            self._trace.event("planner_error", call_id=job.call_id, error=repr(result))
            self._planner_failed(f"planner call {job.call_id} failed: {result!r}")
            return
        self._handle_output(job, result, obs)

    def _submit_plan(self, obs: Observation) -> bool:
        budget = self._task.budget.max_planner_calls
        if budget is not None and self.planner_calls >= budget:
            self.hold(f"planner call budget used up ({budget})", ErrorCode.E_BUDGET)
            return False
        call_id = self._planner_ids.next()
        update_context(planner_call_id=call_id)
        context = PlannerContext(
            task=self._task,
            images=dict(obs.images),
            proprio=dict(obs.proprio),
            memory=self.memory,
            reason=self._plan_reason,
            skills=list(self._task.skills),
            resample=self._resample,
            info=obs.info,
        )
        self._plan = PlanJob(call_id, context, self._clock.now())
        self._trace.event(
            "planner_request", call_id=call_id, reason=self._plan_reason, resample=self._resample
        )
        self._runner.submit(context)
        return True

    def _planner_failed(self, message: str) -> None:
        """Timeouts and errors: retry with growing backoff, then hold."""
        self._plan = None
        self._plan_failures += 1
        retries = self._planner_cfg.retries
        if self._plan_failures > retries:
            self.hold(f"{message}; gave up after {retries} retries", ErrorCode.E_PLANNER_TIMEOUT)
            return
        delay = self._planner_cfg.retry_backoff_s * 2 ** (self._plan_failures - 1)
        self._next_plan_at = self._clock.now() + delay
        logger.warning(
            "%s; retry %d/%d in %.2fs",
            message,
            self._plan_failures,
            retries,
            delay,
            extra={"code": ErrorCode.E_PLANNER_TIMEOUT},
        )

    def _handle_output(self, job: PlanJob, output: PlannerOutput, obs: Observation) -> None:
        if self._health is not None:
            self._health.record_planner_latency(output.latency_s)
        errors: list[str] = []
        decision: PlannerDecision | None = None
        try:
            decision = parse_decision(output.raw)
        except DecisionError as exc:
            errors.append(str(exc))
        if decision is not None:
            errors += self._check_decision(decision, obs)
        attachments = []
        for name, data in output.attachments.items():
            relative = f"planner/{job.call_id}/{name}"
            self._trace.write_bytes(relative, data)
            attachments.append(relative)
        self._trace.planner_call(
            job.call_id,
            {
                "call_id": job.call_id,
                "input": job.context.summary(),
                "attachments": attachments,
                "raw": output.raw,
                "latency_s": round(output.latency_s, 4),
                "meta": output.meta,
                "decision": decision.to_dict() if decision else None,
                "errors": errors,
            },
        )
        self._trace.event(
            "planner_response",
            call_id=job.call_id,
            latency_s=round(output.latency_s, 4),
            decision=decision.to_dict() if decision else None,
            errors=errors,
        )
        if errors:
            self._reject(job, errors)
            return
        assert decision is not None
        self._update_memory(decision.memory, job.call_id)
        if decision.call is None:
            logger.info("planner %s declared the task done", job.call_id)
            self._check_done(claimed_by_planner=True)
            return
        logger.info("planner %s chose %s (%.3fs)", job.call_id, decision.call.key(), output.latency_s)
        self._dispatch(decision.call, obs, "main")

    def _reject(self, job: PlanJob, errors: list[str]) -> None:
        self.resamples += 1
        if self._health is not None:
            self._health.record_resample()
        self._trace.event("decision_rejected", call_id=job.call_id, errors=errors, resample=self._resample)
        limit = self._planner_cfg.max_resamples
        if self._resample >= limit:
            self.hold(
                f"planner output still invalid after {self._resample} resample(s): {'; '.join(errors)}",
                ErrorCode.E_PLANNER_INVALID,
            )
            return
        self._resample += 1
        logger.warning(
            "planner call %s rejected (%s); resampling %d/%d",
            job.call_id,
            "; ".join(errors),
            self._resample,
            limit,
            extra={"code": ErrorCode.E_PLANNER_INVALID},
        )

    def _check_decision(self, decision: PlannerDecision, obs: Observation) -> list[str]:
        call = decision.call
        if call is None:
            return []
        if call.skill not in self._task.skills:
            return [f"{call.skill!r} is not one of the task's skills ({', '.join(self._task.skills)})"]
        problems = self._registry.check_call(call)
        if problems:
            return problems
        namespace = self._namespace(obs, call)
        unmet = [
            expr.source for expr in self._registry.skill(call.skill).preconditions if not expr(namespace)
        ]
        if unmet:
            return [f"{call.key()}: precondition not met: {', '.join(unmet)}"]
        return []

    def _update_memory(self, memory: str, call_id: str) -> None:
        limit = self._planner_cfg.memory_max_chars
        if len(memory) > limit:
            logger.warning("planner memory has %d characters; truncated to %d", len(memory), limit)
            memory = memory[:limit]
        self.memory = memory
        self._trace.event("memory_update", call_id=call_id, memory=memory)

    def _check_done(self, *, claimed_by_planner: bool) -> None:
        done = self._registry.is_done(self._ledger.view(), self._task_ns)
        passed = done if done is not None else claimed_by_planner
        self._trace.event(
            "check",
            claimed_by_planner=claimed_by_planner,
            done_when=self._task.done_when,
            passed=passed,
            ledger=self._ledger.view(),
        )
        if passed:
            self._finish(State.DONE)
        elif claimed_by_planner:
            logger.warning("planner declared done but done_when is false: %s", self._task.done_when)
            self._request_plan(DecisionReason.CHECK_FAILED)
        else:
            self._request_plan(DecisionReason.SKILL_DONE)

    # --- execution ------------------------------------------------------------------------------

    def _dispatch(
        self, call: SkillCall, obs: Observation, purpose: str, retries: dict[str, int] | None = None
    ) -> None:
        budget = self._task.budget.max_skill_calls
        if budget is not None and self.skill_calls >= budget:
            self.hold(f"skill call budget used up ({budget})", ErrorCode.E_BUDGET)
            return
        skill = self._registry.skill(call.skill)
        prompt = self._registry.resolve_prompt(call)
        executor = self._scripted if skill.scripted else self._vla
        self._version += 1
        call_id = self._skill_ids.next()
        update_context(skill_call_id=call_id)
        executor.set_instruction(call, prompt, self._version)
        self._queue.switch(self._version)
        now = self._clock.now()
        session = VerifierSession(skill, started_at=now, timeout_s=self._registry.timeout_for(call.skill))
        self._active = ActiveCall(
            call, skill, prompt, self._version, call_id, executor, session, purpose, dict(retries or {})
        )
        self._set_state(State.RETRY if purpose == "before_retry" else State.EXECUTE)
        self._trace.event(
            "boundary",
            call_id=call_id,
            previous=self._last_call_key,
            current=call.key(),
            since_last_switch_s=None
            if self._last_switch_at is None
            else round(now - self._last_switch_at, 3),
            proprio=dict(obs.proprio),
        )
        self._trace.event(
            "skill_start",
            call_id=call_id,
            call=call.to_dict(),
            prompt={"id": prompt.id, "text": prompt.text} if prompt else None,
            version=self._version,
            purpose=purpose,
            retries=dict(retries or {}),
            timeout_s=round(session.timeout_s, 3),
        )
        detail = f"'{prompt.text}'" if prompt else "scripted"
        logger.info(
            "start %s -> %s%s", call.key(), detail, " (recovery)" if purpose == "before_retry" else ""
        )
        self._last_call_key = call.key()
        self._last_switch_at = now

    def _tick_execute(self, obs: Observation) -> None:
        active = self._active
        assert active is not None
        outcome = active.session.evaluate(self._clock.now(), self._namespace(obs, active.call), obs.proprio)
        if outcome is None:
            return
        self._trace.event(
            "skill_end",
            call_id=active.call_id,
            call=active.call.to_dict(),
            purpose=active.purpose,
            outcome=outcome.to_dict(),
        )
        self._snapshot_env()
        if outcome.ok:
            self._on_success(active, obs, outcome)
        else:
            self._on_failure(active, obs, outcome)

    def _on_success(self, active: ActiveCall, obs: Observation, outcome: Outcome) -> None:
        if active.purpose == "before_retry":
            logger.info("recovery step %s done", active.call.key())
            self._continue_retry(obs)
            return
        logger.info("%s succeeded (%s)", active.call.key(), outcome.evidence.get("condition"))
        self._commit(active.call, obs, outcome.evidence)

    def _commit(self, call: SkillCall, obs: Observation, evidence: dict[str, Any]) -> None:
        skill = self._registry.skill(call.skill)
        namespace = self._namespace(obs, call)
        ops = [effect.evaluate(namespace) for effect in skill.effects]
        for op in ops:
            self._ledger.apply(op)
        if ops:
            self._trace.event(
                "ledger_update",
                call=call.to_dict(),
                ops=[op.to_dict() for op in ops],
                ledger=self._ledger.view(),
                evidence=evidence,
            )
        self._failure_streak.pop(call.key(), None)
        self._last_failed = None
        self._active = None
        if self._registry.is_done(self._ledger.view(), self._task_ns):
            self._check_done(claimed_by_planner=False)
        else:
            self._request_plan(DecisionReason.SKILL_DONE)

    def _on_failure(self, active: ActiveCall, obs: Observation, outcome: Outcome) -> None:
        reason = outcome.reason or "failure"
        if active.purpose == "before_retry":
            self.hold(f"recovery step {active.call.key()} failed ({reason})")
            return
        key = active.call.key()
        self._failure_streak[key] += 1
        streak = self._failure_streak[key]
        self._last_failed = active
        logger.warning("%s failed: %s (%d in a row)", key, reason, streak)
        if streak >= self._registry.max_consecutive_failures:
            self.hold(f"{key} failed {streak} times in a row (last: {reason})")
            return

        action = self._registry.recovery_for(active.call.skill, reason)
        used = active.retries.get(reason, 0)
        if used < action.retry:
            retries = {**active.retries, reason: used + 1}
            self._trace.event(
                "recovery",
                call=active.call.to_dict(),
                reason=reason,
                action="retry",
                attempt=used + 1,
                max_retries=action.retry,
                before_retry=list(action.before_retry),
            )
            logger.info("retrying %s after %s (%d/%d)", key, reason, used + 1, action.retry)
            if action.before_retry:
                self._retry_target = (active.call, retries)
                self._retry_queue = list(action.before_retry)
                self._continue_retry(obs)
            else:
                self._dispatch(active.call, obs, "main", retries)
            return
        self._trace.event("recovery", call=active.call.to_dict(), reason=reason, action=action.escalate)
        if action.escalate == "hold":
            self.hold(f"{key} failed ({reason}); the recovery table says hold")
            return
        logger.info("handing %s failure (%s) back to the planner", key, reason)
        self._request_plan(DecisionReason.SKILL_FAILED)

    def _continue_retry(self, obs: Observation) -> None:
        if self._retry_queue:
            self._dispatch(SkillCall(self._retry_queue.pop(0)), obs, "before_retry")
            return
        assert self._retry_target is not None
        call, retries = self._retry_target
        self._retry_target = None
        self._dispatch(call, obs, "main", retries)

    # --- helpers ---------------------------------------------------------------------------------

    def _namespace(self, obs: Observation, call: SkillCall | None) -> dict[str, Any]:
        namespace: dict[str, Any] = dict(obs.proprio)
        namespace.update(self._registry.predicates(obs.proprio))
        namespace["ledger"] = self._ledger.view()
        namespace["task"] = self._task_ns
        namespace["args"] = dict(call.args) if call is not None else {}
        namespace["executor_done"] = self.executor_done()
        return namespace

    def _time_budget_exceeded(self) -> bool:
        limit = self._task.budget.max_time_s
        if limit is None or self._time_budget_hit or self._clock.now() - self._started_at <= limit:
            return False
        # Report once; if a human resumes, the episode is allowed to go on.
        self._time_budget_hit = True
        self.hold(f"time budget used up ({limit:g}s)", ErrorCode.E_BUDGET)
        return True

    def _snapshot_env(self) -> None:
        """Simulated envs can save their state at every skill boundary so a resume restores the world too."""
        snapshot = getattr(self._env, "snapshot", None)
        state = snapshot() if callable(snapshot) else None
        if state is not None:
            self._trace.write_json(ENV_SNAPSHOT_FILE, state, atomic=True)

    def _set_state(self, state: State) -> None:
        previous = self.state
        self.state = state
        update_context(state=state.value)
        if previous is not state:
            self._trace.event("state_change", previous=previous, current=state)

    def _finish(self, state: State) -> None:
        self._runner.cancel()
        self._queue.clear()
        self._active = None
        self._set_state(state)
        aborted = state is State.ABORT
        self.result = EpisodeResult(
            status="aborted" if aborted else "done",
            error_code=self._hold_code.value if aborted and self._hold_code else None,
            message=self._hold_message if aborted else None,
            ledger=self._ledger.view(),
            memory=self.memory,
            skill_calls=self.skill_calls,
            planner_calls=self.planner_calls,
            resamples=self.resamples,
            human_commands=list(self.human_commands),
            duration_s=round(self._clock.now() - self._started_at, 3),
        )
        self._trace.event("episode_end", **self.result.to_dict())
        if aborted:
            logger.warning("episode aborted: %s", self._hold_message)
        else:
            logger.info("episode done: ledger %s", self.result.ledger)
