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
from ..core.clock import Clock, RealClock, SimClock
from ..core.types import DecisionReason, Observation
from ..executor.chunk_queue import ChunkQueue
from ..logs.codes import ErrorCode
from ..logs.context import clear_context, update_context
from ..logs.health import HealthMonitor
from ..logs.setup import close_file_log, open_file_log
from ..memory.ledger import Ledger
from ..planner.runner import InlinePlannerRunner, PlannerRunner, ThreadPlannerRunner
from ..supervisor.supervisor import EpisodeResult, State, Supervisor
from ..trace.reader import RestoredState, load_restored
from ..trace.writer import TraceWriter
from .commands import AutoCommands, CommandSource
from .factory import bind_planner, build_env, build_planner, build_scripted, build_vla

logger = logging.getLogger("fmc3_harness.runtime")

MAX_CONSECUTIVE_ERRORS = 3


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
        vla: Any = None,
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
        self.vla = vla
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
            trace = TraceWriter(
                self.episode_dir,
                self.clock,
                t0=None if restored is None else self.clock.now() - restored.last_t,
                start_seq=0 if restored is None else restored.last_seq,
            )
            env = self.env if self.env is not None else build_env(bundle)
            planner = self.planner if self.planner is not None else build_planner(bundle)
            bind_planner(planner, bundle)
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
            raise RuntimeError(f"{self.episode_dir} already finished; there is nothing to resume")
        return restored

    def _run(
        self, trace: TraceWriter, env: Any, planner_runner: PlannerRunner, restored: RestoredState | None
    ) -> EpisodeRun:
        bundle = self.bundle
        runtime = bundle.run.runtime
        if restored is None:
            trace.write_yaml("config.snapshot.yaml", {"harness_version": __version__, **bundle.snapshot()})
            trace.event("episode_start", seed=self.seed, task=bundle.task.id, run=bundle.run.name)
            logger.info("episode %s started (task %s, seed %s)", self.episode_id, bundle.task.id, self.seed)
        else:
            trace.event("episode_resume", ledger_ops=len(restored.ledger_ops), memory=restored.memory)
            logger.info("episode %s resumed from %s", self.episode_id, self.episode_dir)

        vla = self.vla if self.vla is not None else build_vla(bundle)
        scripted = build_scripted(bundle)
        env.reset(self.seed)
        if restored is not None and restored.env_snapshot is not None and hasattr(env, "restore"):
            env.restore(restored.env_snapshot)
        vla.reset()
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
            task=bundle.task,
            planner_config=bundle.run.planner,
            runner=planner_runner,
            vla=vla,
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
        )
        supervisor.start(DecisionReason.START if restored is None else DecisionReason.RESUME)
        self._loop(supervisor, queue, env, trace, health)

        result = supervisor.result
        if result is not None:
            trace.write_json(
                "outcome.json",
                {**result.to_dict(), "attribution": None, "health": health.summary()},
            )
            logger.info("health totals: %s", health.summary())
        return EpisodeRun(self.episode_dir, result)

    def _loop(
        self, supervisor: Supervisor, queue: ChunkQueue, env: Any, trace: TraceWriter, health: HealthMonitor
    ) -> None:
        runtime = self.bundle.run.runtime
        period = 1.0 / runtime.control_hz
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
                supervisor.tick(obs)
                if supervisor.executing:
                    underrun_since = self._step(supervisor, queue, env, obs, trace, now, underrun_since)
                else:
                    underrun_since = None
                    env.hold()
                    if supervisor.state is State.HOLD:
                        command = self.commands.poll(*supervisor.hold_info())
                        if command:
                            supervisor.command(command)
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
            next_tick += period
            self.clock.sleep_until(next_tick)

    def _step(
        self,
        supervisor: Supervisor,
        queue: ChunkQueue,
        env: Any,
        obs: Observation,
        trace: TraceWriter,
        now: float,
        underrun_since: float | None,
    ) -> float | None:
        """Send one action for the running call. Returns when the queue first ran dry, if it has."""
        active = supervisor.active
        assert active is not None
        if queue.pending(active.version) == 0 and not active.executor.is_done():
            queue.push(active.executor.infer_chunk(obs))
        item = queue.pop()
        if item is not None:
            env.act(item.action)
            trace.action(item.version, item.action)
            return None
        env.hold()
        if active.executor.is_done():
            return None  # a finished primitive waits for its verdict; nothing is missing
        limit = self.bundle.run.runtime.underrun_limit_s
        if underrun_since is None:
            logger.warning("executor produced no actions", extra={"code": ErrorCode.E_EXEC_UNDERRUN})
            return now
        if now - underrun_since > limit:
            supervisor.hold(f"no actions from the executor for {limit:g}s", ErrorCode.E_EXEC_UNDERRUN)
            return None
        return underrun_since
