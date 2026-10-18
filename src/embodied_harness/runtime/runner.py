"""Runs one episode: the control loop that ties env, executors, planner and supervisor together.

M0 runs the loop on one thread, one tick per control period. With the simulated clock an
episode runs as fast as the CPU allows and is fully repeatable; with the real clock it paces
itself to control_hz. Planner calls can run on a worker thread so a slow model never blocks
the loop (design doc §3).
"""

from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import __version__
from ..config.loader import Bundle
from ..config.schema import TaskSpec
from ..core.clock import Clock, RealClock, SimClock, next_deadline
from ..core.ledger import Ledger
from ..core.types import DecisionReason, Observation
from ..planner.runner import InlinePlannerRunner, PlannerRunner, ThreadPlannerRunner
from ..policy.executor.chunk_queue import ChunkQueue
from .commands import AutoCommands, CommandSource
from .factory import (
    bind_planner,
    build_env,
    build_planner,
    build_policy,
    build_scripted,
    trained_settings_warnings,
)
from .logs.codes import ErrorCode
from .logs.context import clear_context, update_context
from .logs.health import HealthMonitor
from .logs.setup import close_file_log, open_file_log
from .supervisor import ActiveCall, EpisodeResult, State, Supervisor
from .trace.frames import FRAMES_FILE, EpisodeFrames, last_record
from .trace.reader import RestoredState, load_restored
from .trace.video import EpisodeVideo, episode_video_path
from .trace.writer import TraceWriter

logger = logging.getLogger("embodied_harness.runtime")

MAX_CONSECUTIVE_ERRORS = 3


class ResumeError(RuntimeError):
    """An episode that cannot be resumed (finished, or its world cannot be brought back)."""


@dataclass
class EpisodeRun:
    episode_dir: Path
    result: EpisodeResult | None  # None when the loop stopped early (max_ticks), e.g. a simulated crash


class EpisodeRunner:
    def __init__(
        self,
        bundle: Bundle,
        *,
        run_dir: Path,
        episode_id: str = "ep_0001",
        seed: int | None = None,
        commands: CommandSource | None = None,
        clock: Clock | None = None,
        env: Any = None,
        planner: Any = None,
        policy: Any = None,
        resume: bool = False,
        max_ticks: int | None = None,
    ) -> None:
        self.bundle = bundle
        self.episode_dir = run_dir / episode_id
        self.episode_id = episode_id
        self.seed = seed
        self.commands = commands or AutoCommands("abort")
        runtime = bundle.run.runtime
        self.clock = clock or (SimClock() if runtime.clock == "sim" else RealClock())
        self.env = env
        self.planner = planner
        self.policy = policy
        self.resume = resume
        self.max_ticks = max_ticks

    def run(self) -> EpisodeRun:
        bundle = self.bundle
        if self.resume:
            restored: RestoredState | None = self._restored()
        else:
            restored = None
            if (self.episode_dir / "events.jsonl").exists():
                raise FileExistsError(
                    f"{self.episode_dir} already has a trace; resume it or use a new run id"
                )
        clear_context()
        update_context(episode_id=self.episode_id, t=0.0)
        file_log = open_file_log(self.episode_dir / "episode.log", bundle.run.logging.file_level)
        trace: TraceWriter | None = None
        env: Any = None
        planner_runner: PlannerRunner | None = None
        try:
            env = self.env if self.env is not None else build_env(bundle)
            if restored is not None:
                self._check_resumable(env)  # before anything is appended to the trace
            trace = TraceWriter(
                self.episode_dir,
                self.clock,
                t0=None if restored is None else self.clock.now() - self._resume_t(restored),
                start_seq=0 if restored is None else restored.last_seq,
            )
            planner = self.planner if self.planner is not None else build_planner(bundle)
            bind_planner(planner, bundle)
            for warning in trained_settings_warnings(planner, bundle):
                logger.warning(warning)
            if restored is not None:
                restore_trace = getattr(planner, "restore_trace", None)
                if callable(restore_trace):
                    restore_trace(restored.events)
            planner_runner = (
                ThreadPlannerRunner(planner) if bundle.run.planner.threaded else InlinePlannerRunner(planner)
            )
            return self._run(trace, env, planner_runner, restored)
        finally:
            if planner_runner is not None:
                planner_runner.close()
            if trace is not None:
                trace.close()
            if env is not None:
                env.close()
            close_file_log(file_log)

    def _restored(self) -> RestoredState:
        restored = load_restored(self.episode_dir)
        if restored.finished:
            raise ResumeError(f"{self.episode_dir} already finished; there is nothing to resume")
        return restored

    def _check_resumable(self, env: Any) -> None:
        """A simulator (`simulated = True`) resumes only if it can restore its world: reset alone would
        start a new scene under the ledger, memory and history restored from the trace."""
        if getattr(env, "simulated", False) and not callable(getattr(env, "restore", None)):
            raise ResumeError(
                f"{self.episode_dir}: {type(env).__name__} is a simulator without restore(), so a resume "
                "would continue the recorded ledger, memory and history in a freshly reset scene; "
                "run the episode again instead"
            )

    def _resume_t(self, restored: RestoredState) -> float:
        """Where the resumed timeline continues: after the last event, action or frame recorded (actions
        and frames are written every tick, events only when something happens)."""
        times = [restored.last_t]
        for name in ("actions.jsonl", FRAMES_FILE):
            record = last_record(self.episode_dir / name)
            if record is not None and isinstance(record.get("t"), (int, float)):
                times.append(float(record["t"]))
        return max(times)

    def _reset_seed(self, restored: RestoredState) -> int | None:
        """A resume resets the env with the seed the episode started with (the scene it recorded), not
        with the seed of the resuming command."""
        start = next((e for e in restored.events if e.get("type") == "episode_start"), None)
        if start is None:
            logger.warning("the trace has no episode_start; resetting the env with seed %s", self.seed)
            return self.seed
        recorded = start.get("seed")
        if self.seed is not None and self.seed != recorded:
            logger.warning("resuming with the recorded seed %s, not %s", recorded, self.seed)
        return recorded

    def _run(
        self, trace: TraceWriter, env: Any, planner_runner: PlannerRunner, restored: RestoredState | None
    ) -> EpisodeRun:
        bundle = self.bundle
        runtime = bundle.run.runtime
        if restored is None:
            seed, segment = self.seed, 0
            trace.write_yaml("config.snapshot.yaml", {"harness_version": __version__, **bundle.snapshot()})
            trace.event("episode_start", seed=seed, task=bundle.task.id, run=bundle.run.name)
            logger.info("episode %s started (task %s, seed %s)", self.episode_id, bundle.task.id, seed)
        else:
            seed = self._reset_seed(restored)
            # The k-th resume records its frames and videos as segment k, next to the earlier ones.
            segment = 1 + sum(event.get("type") == "episode_resume" for event in restored.events)
            trace.event(
                "episode_resume", ledger_ops=len(restored.ledger_ops), memory=restored.memory, seed=seed
            )
            logger.info("episode %s resumed from %s (seed %s)", self.episode_id, self.episode_dir, seed)

        policy = self.policy if self.policy is not None else build_policy(bundle)
        scripted = build_scripted(bundle)
        env.reset(seed)
        if restored is not None and restored.env_snapshot is not None and hasattr(env, "restore"):
            env.restore(restored.env_snapshot)
        task = self._episode_task(env, trace, instruction=None if restored is None else restored.instruction)
        for executor in (policy, scripted):
            begin = getattr(executor, "begin_episode", None)
            if callable(begin):
                begin(task)
        policy.reset()
        scripted.reset()

        queue = ChunkQueue(bundle.run.executor.handoff)
        ledger = (
            Ledger(bundle.task.ledger)
            if restored is None
            else Ledger.replay(bundle.task.ledger, restored.ledger_ops)
        )
        health = HealthMonitor(every_s=runtime.health_every_s, period_s=1.0 / runtime.control_hz)
        supervisor = Supervisor(
            registry=bundle.registry,
            task=task,
            planner_config=bundle.run.planner,
            runner=planner_runner,
            policy=policy,
            scripted=scripted,
            queue=queue,
            env=env,
            ledger=ledger,
            trace=trace,
            clock=self.clock,
            health=health,
            memory="" if restored is None else restored.memory,
            skill_calls=0 if restored is None else restored.skill_calls,
            planner_calls=0 if restored is None else restored.planner_calls,
            history=None if restored is None else restored.history,
            last=None if restored is None else restored.last,
            keep_snapshots=bundle.run.trace.snapshots,
        )
        supervisor.start(DecisionReason.START if restored is None else DecisionReason.RESUME)
        try:
            self._loop(supervisor, queue, env, trace, health, segment)
        finally:
            close = getattr(policy, "close", None)
            if callable(close):
                close()

        result = supervisor.result
        if result is not None:
            trace.write_json(
                "outcome.json",
                {**result.to_dict(), "attribution": None, "health": health.summary()},
            )
            logger.info("health totals: %s", health.summary())
        return EpisodeRun(self.episode_dir, result)

    def _episode_task(self, env: Any, trace: TraceWriter, instruction: str | None = None) -> TaskSpec:
        """The task with this episode's instruction as its goal, when the task says so."""
        task = self.bundle.task
        if task.goal_source != "episode":
            return task
        instruction = instruction or (
            env.instruction() if callable(getattr(env, "instruction", None)) else None
        )
        if not instruction:
            raise RuntimeError("task.goal_source is 'episode' but the env gave no instruction after reset")
        trace.event("instruction", text=instruction)
        logger.info("episode instruction: %s", instruction)
        return task.model_copy(update={"goal": instruction})

    def _loop(
        self,
        supervisor: Supervisor,
        queue: ChunkQueue,
        env: Any,
        trace: TraceWriter,
        health: HealthMonitor,
        segment: int = 0,
    ) -> None:
        """`segment` is 0 for a first run and k for the k-th resume, whose footage goes to new files."""
        runtime = self.bundle.run.runtime
        period = 1.0 / runtime.control_hz
        video = None
        if self.bundle.run.trace.video:
            video = EpisodeVideo(
                episode_video_path(self.episode_dir, segment), fps=self.bundle.run.trace.video_fps
            )
        frames = (
            EpisodeFrames(self.episode_dir, fps=runtime.control_hz, segment=segment)
            if self.bundle.run.trace.frames
            else None
        )
        try:
            self._ticks(supervisor, queue, env, trace, health, period, video, frames)
        finally:
            if video is not None:
                video.close()
            if frames is not None:
                frames.close()

    def _ticks(
        self,
        supervisor: Supervisor,
        queue: ChunkQueue,
        env: Any,
        trace: TraceWriter,
        health: HealthMonitor,
        period: float,
        video: EpisodeVideo | None,
        frames: EpisodeFrames | None = None,
    ) -> None:
        next_tick = self.clock.now()
        underrun_since: float | None = None
        errors = 0
        ticks = 0
        while not supervisor.finished:
            if self.max_ticks is not None and ticks >= self.max_ticks:
                logger.warning("stopping after %d ticks (max_ticks)", ticks)
                return
            ticks += 1
            started = time.perf_counter()
            now = self.clock.now()
            update_context(t=round(trace.elapsed, 3))
            try:
                obs = env.observe()
                obs.t = now
                if video is not None:
                    video.add(obs.images)
                if obs.terminated:
                    supervisor.end_by_env(obs)
                    self._record_terminal(frames, supervisor, obs, trace)
                    return
                supervisor.tick(obs)
                if supervisor.finished:
                    self._record_terminal(frames, supervisor, obs, trace)
                    return
                if supervisor.executing:
                    underrun_since, sent, source, action_call = self._step(
                        supervisor, queue, env, obs, trace, now, underrun_since
                    )
                else:
                    underrun_since, source = None, "hold"
                    sent = env.hold()
                    action_call = supervisor.active
                if frames is not None:
                    frames.add(
                        tick=ticks,
                        t=trace.elapsed,  # the same time base as events.jsonl
                        images=obs.images,
                        proprio=obs.proprio,
                        action=sent,
                        source=source,
                        call_id=action_call.call_id if action_call is not None else None,
                        call=action_call.call.key() if action_call is not None else None,
                        prompt=(
                            action_call.prompt.text
                            if action_call is not None and action_call.prompt
                            else None
                        ),
                    )
                if supervisor.state is State.HOLD:
                    hold_code, hold_message, held = supervisor.hold_info()
                    # A complete ledger is only a progress signal; the native simulator still owns
                    # success.  Keep stepping the last qpos target until RoboTwin emits its terminal
                    # verdict instead of turning this wait into an interactive abort.
                    waiting_for_env = (
                        hold_code is ErrorCode.E_PLANNER_INVALID
                        and isinstance(hold_message, str)
                        and hold_message.startswith("the ledger shows all bottles released")
                    )
                    command = None if waiting_for_env else self.commands.poll(hold_code, hold_message, held)
                    if command:
                        supervisor.command(command)
                        if supervisor.finished:
                            self._record_terminal(frames, supervisor, env.observe(), trace)
                errors = 0
            except Exception:
                errors += 1
                logger.exception("control loop error", extra={"code": ErrorCode.E_INTERNAL})
                trace.event("error", code=ErrorCode.E_INTERNAL, error=traceback.format_exc(limit=8))
                if errors >= MAX_CONSECUTIVE_ERRORS:
                    raise
                supervisor.hold("harness error, see episode.log", ErrorCode.E_INTERNAL)
            health.record_tick(now, time.perf_counter() - started, len(queue))
            health.maybe_log(now)
            next_tick, skipped = next_deadline(next_tick, period, self.clock.now())
            if skipped:
                health.record_skipped(now, skipped)
                logger.debug(
                    "tick overran; skipped %d tick(s)", skipped, extra={"code": ErrorCode.E_CTRL_OVERRUN}
                )
            self.clock.sleep_until(next_tick)

    @staticmethod
    def _record_terminal(
        frames: EpisodeFrames | None, supervisor: Supervisor, obs: Observation, trace: TraceWriter
    ) -> None:
        result = supervisor.result
        if frames is None or result is None:
            return
        success = result.env_success
        if success is None and result.status != "ended":
            success = result.status == "done"
        frames.finish(
            t=trace.elapsed,
            images=obs.images,
            proprio=obs.proprio,
            status=result.status,
            success=success,
            ledger=result.ledger,
            memory=result.memory,
        )

    def _step(
        self,
        supervisor: Supervisor,
        queue: ChunkQueue,
        env: Any,
        obs: Observation,
        trace: TraceWriter,
        now: float,
        underrun_since: float | None,
    ) -> tuple[float | None, Any, str, ActiveCall | None]:
        """Send one action for the running call.

        Returns when the queue first ran dry (if it has), the action sent as the env reports it (else
        the one given), its source (`policy`, `primitive` or `hold`), and the call that produced it.
        A new chunk is inferred only once nothing is queued, the rest of a previous call's chunk
        included (finish_chunk), so it is computed from the observation of the tick it starts on.
        """
        active = supervisor.active
        assert active is not None
        if queue.pending(active.version) == 0 and not queue.finishing() and not active.executor.is_done():
            queue.push(active.executor.infer_chunk(obs))
        item = queue.pop()
        if item is not None:
            sent = env.act(item.action)
            trace.action(item.version, item.action)
            action_call = supervisor.call_for_version(item.version) or active
            return (
                None,
                item.action if sent is None else sent,
                "primitive" if action_call.skill.scripted else "policy",
                action_call,
            )
        sent = env.hold()
        if active.executor.is_done():
            # A finished primitive waits for its verdict; nothing is missing.
            return None, sent, "hold", active
        limit = self.bundle.run.runtime.underrun_limit_s
        if underrun_since is None:
            logger.warning("executor produced no actions", extra={"code": ErrorCode.E_EXEC_UNDERRUN})
            return now, sent, "hold", active
        if now - underrun_since > limit:
            supervisor.hold(f"no actions from the executor for {limit:g}s", ErrorCode.E_EXEC_UNDERRUN)
            return None, sent, "hold", active
        return underrun_since, sent, "hold", active
