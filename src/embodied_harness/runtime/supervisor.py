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
from pathlib import Path
from typing import Any

from ..config.expressions import ExpressionError
from ..config.schema import PlannerConfig, TaskSpec
from ..config.skills import CompiledSkill, CompiledTransition, SkillRegistry, task_namespace
from ..core.clock import Clock
from ..core.ids import IdSequence
from ..core.ledger import Ledger, LedgerError
from ..core.types import DecisionReason, Observation, Outcome, Prompt, SkillCall
from ..env.base import Env
from ..planner.base import PlannerContext, PlannerOutput
from ..planner.decision import DecisionError, PlannerDecision, parse_decision
from ..planner.runner import PlannerRunner
from ..policy.executor.base import Executor
from ..policy.executor.chunk_queue import ChunkQueue
from .logs.codes import ErrorCode
from .logs.context import update_context
from .logs.health import HealthMonitor
from .trace.writer import ENV_SNAPSHOT_FILE, SNAPSHOTS_DIR, TraceWriter
from .verifier import VerifierSession

logger = logging.getLogger("embodied_harness.supervisor")

HUMAN_COMMANDS = ("resume", "retry", "skip", "abort")
PRELUDES = ("before_retry", "transition")  # primitives run before a call, then the call itself


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
    purpose: str  # "main"; "before_retry" (recovery) or "transition" (a transition rule) for a primitive
    retries: dict[str, int] = field(default_factory=dict)
    open: bool = True  # its skill_start has no skill_end yet
    # handoff finish_chunk: the previous call's chunk is still running, so this call has not moved yet.
    # Its verifier starts (and its snapshot for `probe` is retaken) when its own first action goes out.
    waiting: bool = False


@dataclass
class PlanJob:
    call_id: str
    context: PlannerContext
    submitted_at: float


@dataclass
class EpisodeResult:
    status: str  # "done", "aborted", or "ended" when the env ended the episode itself
    error_code: str | None
    message: str | None
    ledger: dict[str, Any]
    memory: str
    skill_calls: int
    planner_calls: int
    resamples: int
    human_commands: list[str]
    duration_s: float
    # The env's own verdict (its info["success"], read once): when it ended the episode, or when the
    # harness finished it as done and the env reports one. Scoring only; nothing decides on it.
    env_success: bool | None = None
    # The env's own details of that verdict (its info["score"], read once when the episode ends however it
    # ends), e.g. how many goals of a long task were reached. Scoring only.
    env_score: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _env_verdict(obs: Observation | None) -> bool | None:
    """The env's own `info["success"]`, or None when it reports none. Read for scoring only."""
    success = obs.info.get("success") if obs is not None else None
    return None if success is None else bool(success)


def _env_score(obs: Observation | None) -> dict[str, Any] | None:
    """The env's own `info["score"]`, or None when it reports none. Read for scoring only."""
    score = obs.info.get("score") if obs is not None else None
    return dict(score) if isinstance(score, dict) else None


class Supervisor:
    def __init__(
        self,
        *,
        registry: SkillRegistry,
        task: TaskSpec,
        planner_config: PlannerConfig,
        runner: PlannerRunner,
        policy: Executor,
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
        history: list[SkillCall] | None = None,
        last: dict[str, Any] | None = None,
        keep_snapshots: bool = False,
    ) -> None:
        """`memory`, the counters, `history` and `last` continue a resumed episode. `keep_snapshots` saves
        the world at the start of every call (trace.snapshots)."""
        self._keep_snapshots = keep_snapshots
        self._registry = registry
        self._task = task
        self._planner_cfg = planner_config
        self._runner = runner
        self._policy = policy
        self._scripted = scripted
        self._queue = queue
        self._env = env
        self._ledger = ledger
        self._trace = trace
        self._clock = clock
        self._health = health
        self._task_ns = task_namespace(task)

        self.state = State.PLAN
        self.memory = memory
        self.resamples = 0
        self.human_commands: list[str] = []
        self.result: EpisodeResult | None = None

        self._skill_ids = IdSequence("sc", skill_calls)
        self._planner_ids = IdSequence("pc", planner_calls)
        self._version = 0
        self._calls_by_version: dict[int, ActiveCall] = {}
        self._active: ActiveCall | None = None
        self._plan: PlanJob | None = None
        self._plan_reason = DecisionReason.START
        self._plan_failures = 0
        self._resample = 0
        self._rejections: list[str] = []  # why the last answer of this round was refused
        self._next_plan_at: float | None = None
        self._retry_queue: list[SkillCall] = []
        self._retry_target: tuple[SkillCall, dict[str, int]] | None = None
        self._prelude = "before_retry"  # the purpose of the primitives in _retry_queue
        self._failure_streak: dict[str, int] = defaultdict(int)
        # The main call the episode is working on: the last one dispatched (or being prepared by
        # primitives), until it succeeds or a human skips it. A HOLD is about this call: `retry` runs
        # it again and `skip` commits it; without one (e.g. a hold while planning after a success)
        # both are refused.
        self._subject: SkillCall | None = None
        self._last_call_key: str | None = None
        self._last_switch_at: float | None = None
        self._hold_code: ErrorCode | None = None
        self._hold_message: str | None = None
        self._time_budget_hit = False
        self._started_at = clock.now()
        self._last_obs: Observation | None = None
        # Dispatched main calls, without consecutive repeats.
        self._history: list[SkillCall] = list(history or [])
        # How the last main call ended, for planners that show it: {call, status, reason}.
        self._last_result: dict[str, Any] | None = last
        self._ticks_since_plan = 0
        # Why the running call's last periodic answer, a done that done_policy: continue ignored, was
        # refused: the call's next check asks again with it. A new round or a new main call drops it.
        self._periodic_refusal: list[str] | None = None
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

    def call_for_version(self, version: int) -> ActiveCall | None:
        """Return the call that produced an action version, including a finishing old chunk."""
        return self._calls_by_version.get(version)

    @property
    def skill_calls(self) -> int:
        return self._skill_ids.count

    @property
    def planner_calls(self) -> int:
        return self._planner_ids.count

    def hold_info(self) -> tuple[ErrorCode | None, str | None, str | None]:
        """The hold's error code and message, and the call `retry` and `skip` would act on (None when
        the hold is not about a call)."""
        held = self._subject.key() if self._subject is not None else None
        return self._hold_code, self._hold_message, held

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
        """Stop in place and wait for a human (design doc §13).

        A call still running (a runner error, an executor underrun, a budget, an emergency stop) ends
        here with a skill_end `preempted` / `hold`. The hold is about the main call in hand (see
        `_subject`); for an interrupted primitive, that is the call it was preparing.
        """
        if self.finished:
            return
        self._runner.cancel()
        self._plan = None
        self._queue.clear()
        self._env.hold()
        self._close_interrupted(code, message)
        self._retry_queue, self._retry_target = [], None  # a human decides what runs next
        self._hold_code, self._hold_message = code, message
        self._set_state(State.HOLD)
        self._trace.event("hold", code=code, message=message)
        if code is not None:
            logger.error("HOLD: %s", message, extra={"code": code})
        else:
            logger.warning("HOLD: %s", message)

    def end_by_env(self, obs: Observation) -> None:
        """The env ended the episode (e.g. the task's button was pressed, or its step limit hit)."""
        if self.finished:
            return
        success = _env_verdict(obs)
        active = self._active.call.key() if self._active is not None else None
        self._trace.event("env_end", success=success, status=obs.info.get("status"), active=active)
        logger.info("env ended the episode (success=%s) during %s", success, active or self.state.value)
        self._last_obs = obs  # its score is the one the episode ended with
        self._finish(State.DONE, ended_by_env=True, env_success=success)

    def emergency_stop(self) -> None:
        self.hold("emergency stop")

    def command(self, command: str) -> bool:
        """A human decision while holding: resume, retry, skip or abort. False when it was refused and
        the supervisor keeps holding.

        `retry` and `skip` act on the call the hold is about (`hold_info`). `retry` runs it again from
        the start, so its preconditions must hold now, as for a planner's choice: after a hold that
        interrupted a pick which had already grasped, repeating the pick would start the policy from a
        state it was never meant for. Then `skip` (the human finished it), `resume` (the planner looks
        at the world again) or `abort` remain.
        """
        if self.state is not State.HOLD:
            logger.warning("ignoring %r: not holding", command)
            return False
        if command not in HUMAN_COMMANDS:
            logger.warning("unknown command %r; expected one of %s", command, ", ".join(HUMAN_COMMANDS))
            return False
        held = self._subject
        obs = self._last_obs or Observation()
        if command in ("retry", "skip") and held is None:
            logger.warning(
                "nothing to %s: this hold (%s) is not about a skill call; resume or abort",
                command,
                self._hold_message,
            )
            return False
        if command == "retry":
            assert held is not None
            unmet = self._unmet_preconditions(held, obs)
            if unmet:
                logger.warning(
                    "cannot retry %s: precondition not met: %s; skip, resume or abort",
                    held.key(),
                    ", ".join(unmet),
                )
                return False
        self.human_commands.append(command)
        self._trace.event("human_command", command=command, failed_call=held.to_dict() if held else None)
        logger.info("human command: %s%s", command, f" ({held.key()})" if held else "")
        if command == "abort":
            self._finish(State.ABORT)
        elif command == "resume":
            if held is not None:
                self._failure_streak.pop(held.key(), None)
            self._request_plan(DecisionReason.HUMAN_RESUME)
        elif command == "retry":
            assert held is not None
            self._failure_streak.pop(held.key(), None)
            self._dispatch(held, obs, "main")
        else:  # skip: the human finished this step by hand
            assert held is not None
            self._commit(held, obs, {"skipped_by": "human"})
        return True

    # --- planning --------------------------------------------------------------------------------

    def _request_plan(self, reason: DecisionReason) -> None:
        if self._plan is not None:  # a periodic check still in flight is superseded
            self._runner.cancel()
        self._plan_reason = reason
        self._plan = None
        self._plan_failures = 0
        self._resample = 0
        self._rejections = []
        self._periodic_refusal = None
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
            history=[call.to_dict() for call in self._history],
            ledger=self._ledger.view(),
            last=dict(self._last_result) if self._last_result is not None else None,
            rejected=list(self._rejections) if self._resample > 0 else [],
        )
        self._ticks_since_plan = 0
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
        decision, errors = self._evaluate_output(job, output, obs)
        if errors:
            self._reject(job, errors)
            return
        assert decision is not None
        if decision.call is None:
            logger.info("planner %s declared the task done", job.call_id)
            if self._done_refusal() is None:  # a refused done is a refused answer: its memory is dropped
                self._update_memory(decision.memory, job.call_id)
            self._check_done(claimed_by_planner=True)
            return
        self._update_memory(decision.memory, job.call_id)
        logger.info("planner %s chose %s (%.3fs)", job.call_id, decision.call.key(), output.latency_s)
        self._dispatch(decision.call, obs, "main")

    def _evaluate_output(
        self, job: PlanJob, output: PlannerOutput, obs: Observation, running: SkillCall | None = None
    ) -> tuple[PlannerDecision | None, list[str]]:
        """Parse and check one planner output and record it in the trace. `running`: the call a
        periodic check was asked about."""
        if self._health is not None:
            self._health.record_planner_latency(output.latency_s)
        errors: list[str] = []
        decision: PlannerDecision | None = None
        try:
            decision = parse_decision(output.raw)
        except DecisionError as exc:
            errors.append(str(exc))
        if decision is not None:
            errors += self._check_decision(decision, obs, running)
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
            meta=output.meta,
            decision=decision.to_dict() if decision else None,
            errors=errors,
        )
        return decision, errors

    def _reject(self, job: PlanJob, errors: list[str]) -> None:
        self.resamples += 1
        if self._health is not None:
            self._health.record_resample()
        self._trace.event("decision_rejected", call_id=job.call_id, errors=errors, resample=self._resample)
        if any("ledger progress is complete" in error for error in errors):
            self.hold(
                "the ledger shows all bottles released, but RoboTwin has not emitted a terminal success; "
                "no further skill is dispatched",
                ErrorCode.E_PLANNER_INVALID,
            )
            return
        # A stale VLM can keep proposing a recovery that the ledger has already completed.  Do
        # not turn that single invalid proposal into an episode abort; start a fresh planning round
        # with the rejection visible so it can choose the pending placement/object instead.
        if self._ledger.view().get("recovered") and any(
            "recover_dropped_bottle" in error for error in errors
        ):
            self._trace.event("planner_recovery_duplicate_ignored", call_id=job.call_id, errors=errors)
            logger.warning(
                "planner proposed recover_dropped_bottle after recovery succeeded; requesting a fresh plan"
            )
            self._request_plan(DecisionReason.CHECK_FAILED)
            self._rejections = list(errors)
            return
        limit = self._planner_cfg.max_resamples
        if self._resample >= limit:
            self.hold(
                f"planner output still invalid after {self._resample} resample(s): {'; '.join(errors)}",
                ErrorCode.E_PLANNER_INVALID,
            )
            return
        self._resample += 1
        self._rejections = list(errors)
        logger.warning(
            "planner call %s rejected (%s); resampling %d/%d",
            job.call_id,
            "; ".join(errors),
            self._resample,
            limit,
            extra={"code": ErrorCode.E_PLANNER_INVALID},
        )

    def _check_decision(
        self, decision: PlannerDecision, obs: Observation, running: SkillCall | None = None
    ) -> list[str]:
        """A done passes here (_done_refusal judges it). A call must be a task skill with valid arguments,
        and its preconditions must hold, unless it confirms `running`: that is no dispatch, and the running
        call may itself have made its start conditions false (a pick makes `not holding` false)."""
        call = decision.call
        if call is None:
            return []
        disposed = self._ledger.view().get("disposed")
        if self._task.expected_total is not None and isinstance(disposed, dict):
            try:
                if sum(int(value) for value in disposed.values()) >= self._task.expected_total:
                    return ["ledger progress is complete; wait for the environment terminal verdict"]
            except (TypeError, ValueError):
                pass
        if call.skill not in self._task.skills:
            return [f"{call.skill!r} is not one of the task's skills ({', '.join(self._task.skills)})"]
        problems = self._registry.check_call(call)
        if problems:
            return problems
        if running is not None and call.key() == running.key():
            return []
        unmet = self._unmet_preconditions(call, obs)
        if unmet:
            return [f"{call.key()}: precondition not met: {', '.join(unmet)}"]
        return []

    def _unmet_preconditions(self, call: SkillCall, obs: Observation) -> list[str]:
        namespace = self._namespace(obs, call)
        return [expr.source for expr in self._registry.skill(call.skill).preconditions if not expr(namespace)]

    def _update_memory(self, memory: str, call_id: str) -> None:
        limit = self._planner_cfg.memory_max_chars
        if len(memory) > limit:
            logger.warning("planner memory has %d characters; truncated to %d", len(memory), limit)
            memory = memory[:limit]
        self.memory = memory
        self._trace.event("memory_update", call_id=call_id, memory=memory)

    def _done_refusal(self) -> tuple[str, str] | None:
        """Why a planner's `done` does not end the episode, as (reason, explanation), or None when it
        does. The task's done_when decides, as it does after a commit. Without one the planner is
        trusted, unless planner.allow_done is false: then only the env's terminal verdict ends it."""
        done = self._registry.is_done(self._ledger.view(), self._task_ns)
        if done is False:
            return "done_when_false", f"done_when is not met: {self._task.done_when}"
        if done is None and not self._planner_cfg.allow_done:
            return "done_not_allowed", "planner.allow_done is false; the env ends this episode"
        return None

    def _check_done(self, *, claimed_by_planner: bool) -> None:
        if claimed_by_planner:
            refusal = self._done_refusal()
            passed = refusal is None
        else:  # after a commit: done_when alone decides
            refusal, passed = None, bool(self._registry.is_done(self._ledger.view(), self._task_ns))
        self._trace.event(
            "check",
            claimed_by_planner=claimed_by_planner,
            done_when=self._task.done_when,
            passed=passed,
            ledger=self._ledger.view(),
        )
        if passed:
            self._finish(State.DONE)
        elif refusal is not None:
            self._refuse_done(refusal[1])
        else:
            self._request_plan(DecisionReason.SKILL_DONE)

    def _refuse_done(self, why: str) -> None:
        """A refused `done` (see _done_refusal) counts as a refused answer of the planning round: ask
        again (reason check_failed) at the resample temperature and with the refusal for
        show_rejections, and hold after max_resamples, so a planner that keeps claiming done cannot
        loop forever."""
        errors = [f"done: {why}"]
        self.resamples += 1
        if self._health is not None:
            self._health.record_resample()
        limit = self._planner_cfg.max_resamples
        if self._resample >= limit:
            self.hold(
                f"planner declared done after {self._resample} resample(s) but {why}",
                ErrorCode.E_PLANNER_INVALID,
            )
            return
        resample = self._resample + 1
        logger.warning(
            "planner declared done but %s; resampling %d/%d",
            why,
            resample,
            limit,
            extra={"code": ErrorCode.E_PLANNER_INVALID},
        )
        self._request_plan(DecisionReason.CHECK_FAILED)
        self._resample, self._rejections = resample, errors

    # --- execution ------------------------------------------------------------------------------

    def _dispatch(
        self,
        call: SkillCall,
        obs: Observation,
        purpose: str,
        retries: dict[str, int] | None = None,
        *,
        transitions: bool = True,
    ) -> None:
        if self._plan is not None:
            # A periodic check still in flight was asked about the call that ran before (e.g. the
            # attempt that just failed); its answer must not preempt this one.
            self._runner.cancel()
            self._plan = None
        if purpose == "main":
            self._subject = call
            self._periodic_refusal = None  # it was about the call that ran before, or an earlier attempt
        if purpose == "main" and transitions:
            found = self._transition_for(call, obs)
            if found is not None:
                rule, namespace = found
                steps = [step.call(namespace) for step in rule.run]
                problems = [problem for step in steps for problem in self._registry.check_call(step)]
                if problems:
                    self.hold(f"transition rule {rule.name}: {'; '.join(problems)}", ErrorCode.E_CONFIG)
                    return
                previous = self._history[-1].to_dict() if self._history else None
                keys = [step.key() for step in steps]
                self._trace.event(
                    "transition",
                    rule=rule.name,
                    when=rule.when.source,
                    run=keys,
                    previous=previous,
                    next=call.to_dict(),
                )
                logger.info("transition rule %s: %s before %s", rule.name, ", ".join(keys), call.key())
                self._retry_target = (call, dict(retries or {}))
                self._retry_queue = steps
                self._prelude = "transition"
                self._continue_retry(obs)
                return
        budget = self._task.budget.max_skill_calls
        if budget is not None and self.skill_calls >= budget:
            self.hold(f"skill call budget used up ({budget})", ErrorCode.E_BUDGET)
            return
        skill = self._registry.skill(call.skill)
        prompt = self._registry.resolve_prompt(call)
        executor = self._scripted if skill.scripted else self._policy
        self._version += 1
        call_id = self._skill_ids.next()
        update_context(skill_call_id=call_id)
        if self._keep_snapshots:  # rewritten when the call starts moving, if it waits (finish_chunk)
            self._snapshot_env(Path(SNAPSHOTS_DIR) / f"{call_id}.json")
        executor.set_instruction(call, prompt, self._version)
        self._queue.switch(self._version)
        # finish_chunk: the rest of the previous chunk runs first; this call starts after it.
        waiting = self._queue.finishing() > 0
        now = self._clock.now()
        session = VerifierSession(skill, started_at=now, timeout_s=self._registry.timeout_for(call.skill))
        self._active = ActiveCall(
            call,
            skill,
            prompt,
            self._version,
            call_id,
            executor,
            session,
            purpose,
            dict(retries or {}),
            waiting=waiting,
        )
        self._calls_by_version[self._version] = self._active
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
        logger.info("start %s -> %s%s", call.key(), detail, f" ({purpose})" if purpose in PRELUDES else "")
        self._last_call_key = call.key()
        self._last_switch_at = now
        if purpose == "main" and (not self._history or self._history[-1].key() != call.key()):
            self._history.append(call)

    def _tick_execute(self, obs: Observation) -> None:
        active = self._active
        assert active is not None
        if active.waiting:
            if self._queue.finishing():
                return  # the previous call's chunk is still running: this call has not moved yet
            self._start_moving(active)
            return
        periodic = self._planner_cfg.call == "periodic" and active.purpose == "main"
        if periodic:
            self._ticks_since_plan += 1
        # The verifier goes first: a verdict on this tick beats a periodic answer, so a verified success
        # always commits its effects (ending the call cancels a check still in flight).
        outcome = active.session.evaluate(self._clock.now(), self._namespace(obs, active.call), obs.proprio)
        if outcome is None:
            if periodic:
                self._tick_periodic(obs, active)
            return
        self._end_call(active, outcome.to_dict())
        self._snapshot_env()
        if outcome.ok:
            self._on_success(active, obs, outcome)
        else:
            self._on_failure(active, obs, outcome)

    def _tick_periodic(self, obs: Observation, active: ActiveCall) -> None:
        """Ask the planner again while a skill runs, on the ticks where the verifier has no verdict.

        A check is due every `every_steps` control steps, once the executor has used up its chunk,
        so the next chunk is inferred for whatever the planner says now. It is held back while the
        call's success is pending, since no answer may leave the call before that verdict.
        """
        if self._plan is None:
            due = self._ticks_since_plan >= self._planner_cfg.every_steps
            if not due or self._queue.pending(active.version) > 0 or active.session.success_pending():
                return
            refusal, self._periodic_refusal = self._periodic_refusal, None
            if refusal is None:
                self._plan_reason, self._resample, self._rejections = DecisionReason.PERIODIC, 0, []
            else:  # asked again after an ignored done, as a resample of that answer
                self._plan_reason, self._resample, self._rejections = DecisionReason.CHECK_FAILED, 1, refusal
            if not self._submit_plan(obs):
                return
        assert self._plan is not None
        result = self._runner.poll()
        if result is None:
            if self._clock.now() - self._plan.submitted_at > self._planner_cfg.timeout_s:
                self._runner.cancel()
                logger.warning(
                    "periodic planner check %s timed out; keeping %s", self._plan.call_id, active.call.key()
                )
                self._plan = None
            return
        job, self._plan = self._plan, None
        if isinstance(result, BaseException):
            logger.error(
                "periodic planner check %s raised %r; keeping %s", job.call_id, result, active.call.key()
            )
            self._trace.event("planner_error", call_id=job.call_id, error=repr(result))
            return
        decision, errors = self._evaluate_output(job, result, obs, running=active.call)
        if errors:
            self.resamples += 1
            self._trace.event("decision_rejected", call_id=job.call_id, errors=errors, periodic=True)
            logger.warning(
                "periodic planner check %s rejected (%s); keeping %s",
                job.call_id,
                "; ".join(errors),
                active.call.key(),
            )
            return
        assert decision is not None
        leaving = decision.call is None or decision.call.key() != active.call.key()
        blocked = None
        if leaving and not self._may_leave(active, obs):
            blocked = "switch_when"
        elif leaving and active.session.success_pending():
            # A check asked before the call's success became pending (a threaded planner answers
            # later): let the verdict land and commit its effects. The planner is asked again once the
            # call has ended.
            blocked = "success_pending"
        if blocked is not None:
            proposed = decision.call.key() if decision.call else "done"
            self._trace.event(
                "switch_blocked",
                call_id=job.call_id,
                current=active.call.key(),
                proposed=proposed,
                proprio=dict(obs.proprio),
                reason=blocked,
            )
            why = (
                f"none of {[expr.source for expr in active.skill.switch_when]} holds"
                if blocked == "switch_when"
                else "its success is pending"
            )
            logger.info("switch from %s to %s blocked: %s", active.call.key(), proposed, why)
            return
        if decision.call is None:
            refusal = self._done_refusal()
            if refusal is not None and self._planner_cfg.done_policy == "continue":
                # Keep the running call: the env's terminal verdict still decides success. Record the
                # check as the strict path does, so reports count every refused done. Like any refused
                # answer's, its memory is dropped; the call did not end, so no `last` result is written.
                # The call's next check is due at once and asks again with the refusal.
                self._trace.event(
                    "check",
                    claimed_by_planner=True,
                    done_when=self._task.done_when,
                    passed=False,
                    ledger=self._ledger.view(),
                )
                self._trace.event(
                    "planner_done_ignored",
                    call_id=job.call_id,
                    current=active.call.key(),
                    reason=refusal[0],
                )
                logger.warning(
                    "planner %s claimed done while %s is still active (%s); keeping it and asking again",
                    job.call_id,
                    active.call.key(),
                    refusal[1],
                    extra={"code": ErrorCode.E_PLANNER_INVALID},
                )
                self._periodic_refusal = [f"done: {refusal[1]}"]
                self._ticks_since_plan = self._planner_cfg.every_steps
                return
            if refusal is None:  # a refused done is a refused answer: its memory is dropped
                self._update_memory(decision.memory, job.call_id)
            self._last_result = {
                "call": active.call.key(),
                "status": "preempted",
                "reason": "planner_done",
            }
            logger.info("planner %s declared the task done during %s", job.call_id, active.call.key())
            self._end_call(
                active,
                {
                    "status": "preempted",
                    "reason": "planner_done",
                    "evidence": {"planner_call_id": job.call_id},
                },
            )
            self._active = None
            self._check_done(claimed_by_planner=True)
            return
        self._update_memory(decision.memory, job.call_id)
        if decision.call.key() != active.call.key():
            self._last_result = {"call": active.call.key(), "status": "preempted", "reason": "planner"}
        if decision.call.key() == active.call.key():
            self._trace.event("planner_confirm", call_id=job.call_id, current=active.call.key())
            return
        self._end_call(
            active,
            {
                "status": "preempted",
                "reason": "planner",
                "evidence": {"planner_call_id": job.call_id, "next": decision.call.key()},
            },
        )
        logger.info("planner %s switches %s -> %s", job.call_id, active.call.key(), decision.call.key())
        self._dispatch(decision.call, obs, "main")

    def _end_call(self, active: ActiveCall, outcome: dict[str, Any]) -> None:
        """Record how a started call ended; every skill_start gets exactly one skill_end."""
        active.open = False
        self._trace.event(
            "skill_end",
            call_id=active.call_id,
            call=active.call.to_dict(),
            purpose=active.purpose,
            outcome=outcome,
        )

    def _close_interrupted(self, code: ErrorCode | None, message: str) -> None:
        """A hold stops the call still running: it ends `preempted` with reason `hold`, the way a
        periodic switch ends one, so resume, report and the data tools see a closed call."""
        active = self._active
        if active is None or not active.open:
            return
        self._end_call(
            active,
            {"status": "preempted", "reason": "hold", "evidence": {"code": code, "message": message}},
        )
        if active.purpose == "main":
            self._last_result = {"call": active.call.key(), "status": "preempted", "reason": "hold"}

    def _start_moving(self, active: ActiveCall) -> None:
        """The previous call's chunk has run out: this call's first chunk is inferred on this tick and
        its first action goes out now, so its verifier (min_duration_s, timeout, stall, hold_s timers)
        starts here instead of judging the previous call's leftover motion. Its trace.snapshots start
        state, saved at dispatch, is replaced by the world as it is now, where `probe` must restart it."""
        active.waiting = False
        active.session = VerifierSession(
            active.skill, started_at=self._clock.now(), timeout_s=active.session.timeout_s
        )
        if self._keep_snapshots:
            self._snapshot_env(Path(SNAPSHOTS_DIR) / f"{active.call_id}.json")

    def _may_leave(self, active: ActiveCall, obs: Observation) -> bool:
        """A planner may switch away from a skill only while one of its switch_when conditions holds."""
        if not active.skill.switch_when:
            return True
        namespace = self._namespace(obs, active.call)
        return any(bool(expr(namespace)) for expr in active.skill.switch_when)

    def _on_success(self, active: ActiveCall, obs: Observation, outcome: Outcome) -> None:
        if active.purpose in PRELUDES:
            logger.info("%s step %s done", active.purpose, active.call.key())
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
        self._subject = None
        self._active = None
        status = "skipped" if evidence.get("skipped_by") else "success"
        self._last_result = {"call": call.key(), "status": status, "reason": None}
        if self._registry.is_done(self._ledger.view(), self._task_ns):
            self._check_done(claimed_by_planner=False)
        else:
            self._request_plan(DecisionReason.SKILL_DONE)

    def _on_failure(self, active: ActiveCall, obs: Observation, outcome: Outcome) -> None:
        reason = outcome.reason or "failure"
        if active.purpose in PRELUDES:
            kind = "recovery" if active.purpose == "before_retry" else "transition"
            prepared = f" before {self._subject.key()}" if self._subject is not None else ""
            self.hold(f"{kind} step {active.call.key()} failed ({reason}){prepared}")
            return
        key = active.call.key()
        self._failure_streak[key] += 1
        streak = self._failure_streak[key]
        self._subject = active.call
        self._last_result = {"call": key, "status": "failure", "reason": reason}
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
                self._retry_queue = [SkillCall(name) for name in action.before_retry]
                self._prelude = "before_retry"
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
        """Run the next queued primitive, then the call they prepared (without checking transitions again)."""
        if self._retry_queue:
            self._dispatch(self._retry_queue.pop(0), obs, self._prelude)
            return
        assert self._retry_target is not None
        call, retries = self._retry_target
        self._retry_target = None
        self._dispatch(call, obs, "main", retries, transitions=False)

    def _transition_for(
        self, call: SkillCall, obs: Observation
    ) -> tuple[CompiledTransition, dict[str, Any]] | None:
        """The first transition rule that applies before `call`, if any, with the namespace it read
        (its steps compute their arguments from it)."""
        rules = self._registry.transitions
        if not rules:
            return None
        previous = self._history[-1] if self._history else None
        namespace = self._namespace(obs, call)
        namespace["previous"] = (
            {"skill": previous.skill, "args": dict(previous.args)}
            if previous
            else {"skill": None, "args": {}}
        )
        namespace["next"] = {"skill": call.skill, "args": dict(call.args)}
        rule = next((rule for rule in rules if bool(rule.when(namespace))), None)
        return None if rule is None else (rule, namespace)

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

    def _snapshot_env(self, name: str | Path = ENV_SNAPSHOT_FILE) -> None:
        """Simulated envs can save their state at every skill boundary so a resume restores the world too
        (and, with trace.snapshots, at every call's start so `probe` can rerun it)."""
        snapshot = getattr(self._env, "snapshot", None)
        state = snapshot() if callable(snapshot) else None
        if state is not None:
            self._trace.write_json(name, state, atomic=True)

    def _set_state(self, state: State) -> None:
        previous = self.state
        self.state = state
        update_context(state=state.value)
        if previous is not state:
            self._trace.event("state_change", previous=previous, current=state)

    def _finish(self, state: State, *, ended_by_env: bool = False, env_success: bool | None = None) -> None:
        self._runner.cancel()
        self._queue.clear()
        self._active = None
        self._set_state(state)
        aborted = state is State.ABORT
        if state is State.DONE and not ended_by_env:
            # Scoring only, as in end_by_env: a simulator's own verdict on the episode the harness
            # (done_when, or a planner's done without one) finished. A real robot reports none.
            env_success = _env_verdict(self._last_obs)
            if env_success is not None:
                logger.info("the env's own verdict on the finished episode: success=%s", env_success)
        env_score = _env_score(self._last_obs)
        if env_score is not None:
            logger.info("the env's own score: %s", env_score)
        self.result = EpisodeResult(
            status="aborted" if aborted else ("ended" if ended_by_env else "done"),
            error_code=self._hold_code.value if aborted and self._hold_code else None,
            message=self._hold_message if aborted else None,
            ledger=self._ledger.view(),
            memory=self.memory,
            skill_calls=self.skill_calls,
            planner_calls=self.planner_calls,
            resamples=self.resamples,
            human_commands=list(self.human_commands),
            duration_s=round(self._clock.now() - self._started_at, 3),
            env_success=env_success,
            env_score=env_score,
        )
        self._trace.event("episode_end", **self.result.to_dict())
        if aborted:
            logger.warning("episode aborted: %s", self._hold_message)
        else:
            logger.info("episode done: ledger %s", self.result.ledger)
