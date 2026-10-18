"""planner.call: periodic — the planner is asked again at chunk boundaries and may switch skills."""

import json
import threading
import time

import numpy as np
import pytest
import yaml

from conftest import make_bundle, of_type, read_events, run_episode, small_task
from embodied_harness.config.loader import load_bundle
from embodied_harness.core.types import Observation
from embodied_harness.planner.base import PlannerOutput
from embodied_harness.planner.oracle import MockSortingOracle
from embodied_harness.planner.runner import InlinePlannerRunner
from embodied_harness.policy.executor.mock import MockPolicyExecutor
from embodied_harness.runtime.factory import check_bundle

SUBGOALS = ["reach the cube", "lift the cube", "press the button"]


def write_config(
    tmp_path,
    *,
    every_steps=4,
    chunk=4,
    call="periodic",
    switch_when=None,
    subgoal=None,
    extra_skills=None,
    planner=None,
    runtime=None,
    done_when=None,
):
    """`subgoal`, `planner` and `runtime` are merged into those sections; `extra_skills` adds skills."""
    root = tmp_path / "configs"
    root.mkdir()
    ids = {text: f"sg{i}" for i, text in enumerate(SUBGOALS)}
    library = {"library": "sg_lib", "prompts": [{"id": pid, "text": text} for text, pid in ids.items()]}
    skills = {
        "version": 1,
        "library": "sg_lib",
        "skills": {
            "subgoal": {
                "args": {"text": {"type": "enum", "values": SUBGOALS}},
                "prompt": ids,
                "timeout_s": 1000,
                **({"verify": {"switch_when": switch_when}} if switch_when else {}),
                **(subgoal or {}),
            },
            **(extra_skills or {}),
        },
    }
    task_data = {"id": "t", "goal": "config goal", "goal_source": "episode", "skills": ["subgoal"]}
    if done_when is not None:
        task_data["done_when"] = done_when
    task = {"task": task_data}
    run = {
        "name": "periodic",
        "task": "task.yaml",
        "skills": "skills.yaml",
        "env": {"type": "mock_sorting", "params": {}},
        "executor": {
            "type": "mock",
            "library": "library.yaml",
            "handoff": "hard",
            "params": {"chunk_size": chunk},
        },
        "planner": {
            "type": "sequence",
            "call": call,
            "every_steps": every_steps,
            "params": {"steps": []},
            **(planner or {}),
        },
        "runtime": {"clock": "sim", "control_hz": 10, **(runtime or {})},
    }
    for name, data in (("library", library), ("skills", skills), ("task", task), ("run", run)):
        (root / f"{name}.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    bundle = load_bundle(root / "run.yaml")
    check_bundle(bundle)
    return bundle


class TickEnv:
    """Counts actions; ends the episode by itself after `end_after` actions."""

    def __init__(self, end_after=40, closes_after=None):
        self.end_after = end_after
        self.closes_after = closes_after
        self.actions = []

    def reset(self, seed=None):
        self.actions = []
        return self.observe()

    def instruction(self):
        return "press the button after lifting the cube"

    def observe(self):
        done = len(self.actions) >= self.end_after
        return Observation(
            images={"front": np.zeros((4, 4, 3), np.uint8)},
            proprio={
                "gripper_width": 0.04,
                "gripper_closed": self.closes_after is not None and len(self.actions) >= self.closes_after,
            },
            info={"success": done, "status": "success" if done else "ongoing"},
            terminated=done,
        )

    def act(self, action):
        self.actions.append(action)

    def hold(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass


class ScheduledPlanner:
    """Answers from a schedule, one entry per call; records what it was given."""

    name = "scheduled"

    def __init__(self, schedule):
        self.schedule = list(schedule)
        self.contexts = []

    def plan(self, context):
        self.contexts.append(context)
        text = self.schedule[min(len(self.contexts) - 1, len(self.schedule) - 1)]
        return PlannerOutput(raw=json.dumps({"skill": "subgoal", "args": {"text": text}, "memory": ""}))


class RawScheduledPlanner:
    """Returns raw planner decisions from a fixed schedule, including ``{"done": true}``."""

    name = "raw_scheduled"

    def __init__(self, schedule):
        self.schedule = list(schedule)
        self.contexts = []

    def plan(self, context):
        self.contexts.append(context)
        raw = self.schedule[min(len(self.contexts) - 1, len(self.schedule) - 1)]
        if isinstance(raw, str):
            raw = {"skill": "subgoal", "args": {"text": raw}, "memory": ""}
        return PlannerOutput(raw=json.dumps(raw))


def test_periodic_checks_confirm_and_switch_at_chunk_boundaries(tmp_path):
    bundle = write_config(tmp_path, every_steps=4, chunk=4)
    schedule = [SUBGOALS[0], SUBGOALS[0], SUBGOALS[1], SUBGOALS[1], SUBGOALS[2]]
    planner = ScheduledPlanner(schedule)
    env = TickEnv(end_after=30)
    policy = MockPolicyExecutor(bundle.library, chunk_size=4)
    run = run_episode(bundle, tmp_path, env=env, planner=planner, policy=policy)
    assert run.result.status == "ended" and run.result.env_success is True

    events = read_events(run)
    requests = of_type(events, "planner_request")
    # Asked at the start and then every 4 steps, each time the executor had used up its chunk.
    assert [round(e["t"], 3) for e in requests][:6] == [0.0, 0.4, 0.8, 1.2, 1.6, 2.0]
    assert [e["reason"] for e in requests[:2]] == ["start", "periodic"]
    assert len(of_type(events, "planner_confirm")) >= 2
    preempted = [e for e in of_type(events, "skill_end") if e["outcome"]["status"] == "preempted"]
    assert [e["call"]["args"]["text"] for e in preempted] == [SUBGOALS[0], SUBGOALS[1]]
    started = [e["call"]["args"]["text"] for e in of_type(events, "skill_start")]
    assert started == SUBGOALS
    # The planner sees the calls dispatched so far and the episode instruction.
    history = [[c["args"]["text"] for c in ctx.history] for ctx in planner.contexts]
    assert history[0] == [] and history[1] == [SUBGOALS[0]] and history[3] == SUBGOALS[:2]
    assert planner.contexts[0].task.goal == "press the button after lifting the cube"
    # Every action belongs to the call that was active when its chunk was inferred.
    prompts = [a["prompt"] for a in env.actions]
    assert prompts[:8] == ["sg0"] * 8 and prompts[8:16] == ["sg1"] * 8 and set(prompts[16:]) == {"sg2"}


def test_invalid_periodic_answers_keep_the_running_skill(tmp_path):
    bundle = write_config(tmp_path, every_steps=2, chunk=2)
    planner = ScheduledPlanner([SUBGOALS[0], "juggle", "juggle", SUBGOALS[2]])
    run = run_episode(
        bundle,
        tmp_path,
        env=TickEnv(end_after=12),
        planner=planner,
        policy=MockPolicyExecutor(bundle.library, chunk_size=2),
    )
    events = read_events(run)
    rejected = of_type(events, "decision_rejected")
    assert len(rejected) == 2 and all(e["periodic"] for e in rejected)
    assert [e["call"]["args"]["text"] for e in of_type(events, "skill_start")] == [SUBGOALS[0], SUBGOALS[2]]
    assert run.result.resamples == 2


def test_periodic_done_policy_continue_keeps_skill_running(tmp_path):
    bundle = write_config(
        tmp_path,
        every_steps=2,
        chunk=2,
        done_when="False",
        planner={"done_policy": "continue"},
    )
    reach = {"skill": "subgoal", "args": {"text": SUBGOALS[0]}, "memory": "reaching"}
    planner = RawScheduledPlanner([reach, {"done": True, "memory": "premature"}, SUBGOALS[0]])
    run = run_episode(
        bundle,
        tmp_path,
        env=TickEnv(end_after=10),
        planner=planner,
        policy=MockPolicyExecutor(bundle.library, chunk_size=2),
    )
    events = read_events(run)
    assert run.result.status == "ended" and run.result.env_success is True
    assert len(of_type(events, "planner_done_ignored")) == 1
    assert [event["passed"] for event in of_type(events, "check")] == [False]
    assert not [e for e in of_type(events, "skill_end") if e["outcome"].get("reason") == "planner_done"]
    assert run.result.error_code is None
    # The next check asks again with the refusal, as a resample of the refused answer (show_rejections
    # renders it); the one after is an ordinary check again.
    refusal = "done: done_when is not met: False"
    asked = [(c.reason.value, c.resample, c.rejected) for c in planner.contexts[1:4]]
    assert asked == [("periodic", 0, []), ("check_failed", 1, [refusal]), ("periodic", 0, [])]
    # Like any refused answer's, the refused done's memory is dropped.
    assert planner.contexts[2].memory == "reaching"
    assert "premature" not in [event["memory"] for event in of_type(events, "memory_update")]


def test_periodic_done_policy_continue_still_finishes_when_done_when_holds(tmp_path):
    # ``continue`` is only a guard for a false done_when. Once the ledger condition holds, a planner
    # done claim follows the normal completion path.
    bundle = write_config(
        tmp_path,
        every_steps=2,
        chunk=2,
        done_when="True",
        planner={"done_policy": "continue"},
    )
    planner = RawScheduledPlanner([SUBGOALS[0], {"done": True, "memory": "done"}])
    run = run_episode(
        bundle,
        tmp_path,
        env=TickEnv(end_after=10),
        planner=planner,
        policy=MockPolicyExecutor(bundle.library, chunk_size=2),
    )
    events = read_events(run)
    assert run.result.status == "done"
    assert not of_type(events, "planner_done_ignored")
    assert of_type(events, "check")[0]["passed"] is True


@pytest.mark.parametrize("recovery", ["planner", "retry"])
def test_an_ignored_done_is_not_held_against_the_next_call(tmp_path, recovery):
    # The call times out while the refusal of its second ignored done waits for the next check. Neither the
    # call the planner picks next nor a local retry of the same call is asked about that done.
    subgoal = {
        "timeout_s": 0.95,
        **({"on_failure": {"timeout": {"retry": 1}}} if recovery == "retry" else {}),
    }
    bundle = write_config(tmp_path, done_when="False", subgoal=subgoal, planner={"done_policy": "continue"})
    done = {"done": True, "memory": ""}
    planner = RawScheduledPlanner([SUBGOALS[0], done, done, SUBGOALS[1]])
    run = run_episode(
        bundle,
        tmp_path,
        env=TickEnv(end_after=30),
        planner=planner,
        policy=MockPolicyExecutor(bundle.library, chunk_size=4),
    )
    events = read_events(run)
    assert of_type(events, "skill_end")[0]["outcome"]["reason"] == "timeout"
    assert len(of_type(events, "planner_done_ignored")) == 2
    asked = [(c.reason.value, c.resample, c.rejected) for c in planner.contexts]
    refusal = ["done: done_when is not met: False"]
    assert asked[:3] == [("start", 0, []), ("periodic", 0, []), ("check_failed", 1, refusal)]
    after = [("skill_failed", 0, [])] if recovery == "planner" else []
    assert asked[3 : 4 + len(after)] == [*after, ("periodic", 0, [])]
    assert run.result.status == "ended"


@pytest.mark.parametrize("done_policy", ["continue", "resample"])
def test_allow_done_false_refuses_a_periodic_done_without_done_when(tmp_path, done_policy):
    # allow_done: false hands completion to the env's terminal verdict, whatever the planner or its output
    # mode. Without a done_when the planner's done used to end the episode.
    planner_config = {"done_policy": done_policy, "allow_done": False}
    bundle = write_config(tmp_path, every_steps=2, chunk=2, planner=planner_config)
    planner = RawScheduledPlanner([SUBGOALS[0], {"done": True, "memory": "premature"}, SUBGOALS[0]])
    run = run_episode(
        bundle,
        tmp_path,
        env=TickEnv(end_after=10),
        planner=planner,
        policy=MockPolicyExecutor(bundle.library, chunk_size=2),
    )
    events = read_events(run)
    assert run.result.status == "ended" and run.result.env_success is True
    assert [event["passed"] for event in of_type(events, "check")] == [False]
    refusal = "done: planner.allow_done is false; the env ends this episode"
    assert (planner.contexts[2].reason.value, planner.contexts[2].rejected) == ("check_failed", [refusal])
    assert "premature" not in [event["memory"] for event in of_type(events, "memory_update")]
    ignored = [event["reason"] for event in of_type(events, "planner_done_ignored")]
    preempted = [e for e in of_type(events, "skill_end") if e["outcome"].get("reason") == "planner_done"]
    if done_policy == "continue":
        assert ignored == ["done_not_allowed"] and not preempted
    else:  # the done ended the running call and the round asked again
        assert not ignored and len(preempted) == 1


def test_boundary_mode_never_asks_while_a_skill_runs(tmp_path):
    bundle = write_config(tmp_path, call="boundary")
    planner = ScheduledPlanner([SUBGOALS[0], SUBGOALS[1]])
    run = run_episode(
        bundle,
        tmp_path,
        env=TickEnv(end_after=20),
        planner=planner,
        policy=MockPolicyExecutor(bundle.library, chunk_size=4),
    )
    assert len(planner.contexts) == 1
    assert run.result.status == "ended"


def test_switch_when_gates_planner_switches(tmp_path):
    bundle = write_config(tmp_path, every_steps=4, chunk=4, switch_when=["gripper_closed"])
    planner = ScheduledPlanner([SUBGOALS[0], SUBGOALS[1]])
    env = TickEnv(end_after=20, closes_after=10)
    run = run_episode(
        bundle, tmp_path, env=env, planner=planner, policy=MockPolicyExecutor(bundle.library, chunk_size=4)
    )
    events = read_events(run)
    blocked = of_type(events, "switch_blocked")
    # Proposed at 4 and 8 actions, while the gripper was still open; allowed at 12.
    assert [round(e["t"], 1) for e in blocked] == [0.4, 0.8]
    assert all(e["proposed"] == f"subgoal(text={SUBGOALS[1]})" for e in blocked)
    starts = of_type(events, "skill_start")
    assert [e["call"]["args"]["text"] for e in starts] == SUBGOALS[:2]
    assert round(starts[1]["t"], 1) == 1.2
    # A blocked proposal is not dispatched, so it is not in the planner's history either.
    assert [c["args"]["text"] for c in planner.contexts[2].history] == [SUBGOALS[0]]
    assert {e["reason"] for e in blocked} == {"switch_when"}


# Mock sorting, place as in docs/configuration.md: success and the switch gate are both `released`.
DOCS_PLACE = {"verify": {"success": [{"when": "released", "hold_s": 0.3}], "switch_when": ["released"]}}
SAME_TICK = {"verify": {"success": [{"when": "released", "hold_s": 0}]}}
PLACE_IN_20 = {
    "steps": {"pick_next": 30, "flip": 20, "place_bin_a": 20, "place_bin_b": 20, "place_bin_c": 20}
}


@pytest.mark.parametrize(
    ("place", "world", "every_steps"),
    [
        # Checks fall due while `released` holds but not yet for its hold_s: they wait for the verdict.
        (DOCS_PLACE, None, 2),
        # A check is due on the very tick the place is released: the verdict on that tick goes first.
        (SAME_TICK, PLACE_IN_20, 16),
    ],
    ids=["hold_window", "same_tick"],
)
def test_a_periodic_switch_never_drops_a_verified_success(tmp_path, place, world, every_steps):
    # The planner sees the package released and proposes the next pick. That switch used to preempt the
    # place before its verifier said success, so its effect never counted and done_when never held.
    bundle = make_bundle(
        tmp_path,
        task=small_task(3),
        skills={"skills": {"place": place}},
        run={"planner": {"call": "periodic", "every_steps": every_steps}},
    )
    run = run_episode(bundle, tmp_path, world=world)
    events = read_events(run)
    assert run.result.status == "done"
    places = [e["outcome"]["status"] for e in of_type(events, "skill_end") if e["call"]["skill"] == "place"]
    assert places == ["success"] * 3
    assert len(of_type(events, "ledger_update")) == 3 and sum(run.result.ledger["sorted"].values()) == 3
    # No check is asked while a success is pending, so no answer has to be blocked either.
    assert not of_type(events, "switch_blocked")


class RetractFirst(MockSortingOracle):
    """Chooses the retract primitive first, then plays the oracle."""

    def __init__(self):
        super().__init__()
        self.first = True

    def plan(self, context):
        if self.first:
            self.first = False
            return PlannerOutput(raw=json.dumps({"skill": "retract", "memory": ""}))
        return super().plan(context)


def test_a_finished_primitive_waiting_for_min_duration_is_not_switched_away(tmp_path):
    # Without success conditions a call succeeds once its executor is done and min_duration_s has passed.
    # The retract's 5 steps end at 0.5 s and its verdict comes at 1 s: the planner, which would pick, is not
    # asked in between.
    bundle = make_bundle(
        tmp_path,
        task=small_task(1),
        skills={"skills": {"retract": {"min_duration_s": 1.0}}},
        run={"planner": {"call": "periodic", "every_steps": 2}},
    )
    run = run_episode(bundle, tmp_path, world={"num_packages": 1}, planner=RetractFirst())
    events = read_events(run)
    retract = of_type(events, "skill_end")[0]
    assert retract["call"]["skill"] == "retract" and retract["outcome"]["status"] == "success"
    assert retract["outcome"]["evidence"]["condition"] == "executor_done"
    checks = [e for e in of_type(events, "planner_request") if e["reason"] == "periodic"]
    assert checks and not [e for e in checks if e["t"] < retract["t"]]
    assert run.result.status == "done"


class LaggedRunner(InlinePlannerRunner):
    """Hands each answer out a few polls after its submit, as a planner on its own thread would, but
    deterministically."""

    lag = 3

    def __init__(self, planner):
        super().__init__(planner)
        self.wait = 0

    def submit(self, context):
        super().submit(context)
        self.wait = self.lag

    def poll(self):
        if self.wait > 0:
            self.wait -= 1
            return None
        return super().poll()


def test_a_check_answered_while_a_success_is_pending_does_not_switch(tmp_path, monkeypatch):
    # A threaded planner answers some ticks after it was asked. The switch it proposed while the gripper
    # was open arrives once `gripper_closed` holds but not yet for its hold_s: the success goes first.
    monkeypatch.setattr("embodied_harness.runtime.runner.InlinePlannerRunner", LaggedRunner)
    success = {"when": "gripper_closed", "hold_s": 0.5}
    bundle = write_config(tmp_path, subgoal={"verify": {"success": [success]}})
    run = run_episode(
        bundle,
        tmp_path,
        env=TickEnv(end_after=30, closes_after=6),
        planner=ScheduledPlanner(SUBGOALS),
        policy=MockPolicyExecutor(bundle.library, chunk_size=4),
    )
    events = read_events(run)
    first = of_type(events, "skill_end")[0]
    assert first["call"]["args"]["text"] == SUBGOALS[0] and first["outcome"]["status"] == "success"
    blocked = [(e["proposed"], e["reason"]) for e in of_type(events, "switch_blocked")]
    assert blocked == [(f"subgoal(text={SUBGOALS[1]})", "success_pending")]


def test_a_failure_condition_on_its_way_does_not_hold_back_a_switch(tmp_path):
    # Only a pending success holds checks back: a failure condition that has not held for its hold_s yet
    # leaves the planner free to move on.
    failure = {"when": "gripper_closed", "hold_s": 1.0, "reason": "slip"}
    bundle = write_config(tmp_path, subgoal={"verify": {"failure": [failure]}})
    run = run_episode(
        bundle,
        tmp_path,
        env=TickEnv(end_after=20, closes_after=6),
        planner=RawScheduledPlanner([SUBGOALS[0], SUBGOALS[0], SUBGOALS[1]]),
        policy=MockPolicyExecutor(bundle.library, chunk_size=4),
    )
    events = read_events(run)
    assert not of_type(events, "switch_blocked")
    first = of_type(events, "skill_end")[0]["outcome"]
    assert (first["status"], first["reason"]) == ("preempted", "planner")


def test_a_periodic_answer_may_confirm_a_call_whose_preconditions_it_made_false(tmp_path):
    # Preconditions are start conditions. The running call closes the gripper, so confirming it must not be
    # rejected for `not gripper_closed`; another call is still checked against them.
    bundle = write_config(tmp_path, subgoal={"preconditions": ["not gripper_closed"]})
    texts = [SUBGOALS[0], SUBGOALS[0], SUBGOALS[0], SUBGOALS[1], SUBGOALS[0]]
    answers = [{"skill": "subgoal", "args": {"text": t}, "memory": f"m{i}"} for i, t in enumerate(texts)]
    run = run_episode(
        bundle,
        tmp_path,
        env=TickEnv(end_after=20, closes_after=6),
        planner=RawScheduledPlanner(answers),
        policy=MockPolicyExecutor(bundle.library, chunk_size=4),
    )
    events = read_events(run)
    rejected = of_type(events, "decision_rejected")
    assert [e["errors"] for e in rejected] == [
        [f"subgoal(text={SUBGOALS[1]}): precondition not met: not gripper_closed"]
    ]
    assert len(of_type(events, "planner_confirm")) == 3 and run.result.resamples == 1
    # A confirmation is an accepted answer: its memory is kept.
    assert [e["memory"] for e in of_type(events, "memory_update")] == ["m0", "m1", "m2", "m4"]
    assert [e["call"]["args"]["text"] for e in of_type(events, "skill_start")] == [SUBGOALS[0]]


class SlipOnceEnv(TickEnv):
    """The gripper reads closed right after the 6th action only: the first attempt slips once. The
    retry's first policy action sets `retry_started`."""

    def __init__(self, retry_started):
        super().__init__(end_after=40)
        self.retry_started = retry_started

    def observe(self):
        obs = super().observe()
        obs.proprio["gripper_closed"] = len(self.actions) == 6
        return obs

    def act(self, action):
        super().act(action)
        if "prompt" in action and action["version"] > 1 and not self.retry_started.is_set():
            self.retry_started.set()
            time.sleep(0.2)  # time passes on a robot: the check asked earlier has answered by now


class SlowCheck:
    """Confirms the first subgoal, except the periodic check asked during the first attempt: that one
    answers `lift the cube` once the retry is running (a slow VLM)."""

    name = "slow_check"

    def __init__(self, retry_started):
        self.retry_started = retry_started
        self.calls = 0

    def plan(self, context):
        self.calls += 1
        text = SUBGOALS[0]
        if self.calls == 2:
            self.retry_started.wait(10)
            text = SUBGOALS[1]
        return PlannerOutput(raw=json.dumps({"skill": "subgoal", "args": {"text": text}, "memory": ""}))


@pytest.mark.parametrize("before_retry", [False, True])
def test_a_retry_is_not_preempted_by_a_check_asked_about_the_failed_attempt(tmp_path, before_retry):
    retry = {"retry": 1, **({"before_retry": ["open_gripper"]} if before_retry else {})}
    bundle = write_config(
        tmp_path,
        subgoal={
            "verify": {"failure": [{"when": "gripper_closed", "reason": "slip"}]},
            "on_failure": {"slip": retry},
        },
        extra_skills={"open_gripper": {"executor": "scripted", "timeout_s": 1000}},
        planner={"threaded": True, "timeout_s": 1000},
        runtime={"primitive_steps": {"open_gripper": 3}},
    )
    retry_started = threading.Event()
    run = run_episode(
        bundle,
        tmp_path,
        env=SlipOnceEnv(retry_started),
        planner=SlowCheck(retry_started),
        policy=MockPolicyExecutor(bundle.library, chunk_size=4),
    )
    events = read_events(run)
    assert retry_started.is_set()
    main = [e for e in of_type(events, "skill_start") if e["purpose"] == "main"]
    assert [e["call"]["args"]["text"] for e in main] == [SUBGOALS[0], SUBGOALS[0]]
    assert main[1]["retries"] == {"slip": 1}
    # The check asked during the failed attempt was dropped with it: its answer never switched the retry.
    assert "pc_0002" not in [e["call_id"] for e in of_type(events, "planner_response")]
    assert not [e for e in of_type(events, "skill_end") if e["outcome"]["status"] == "preempted"]
    assert run.result.status == "ended"
