"""What a HOLD is about: the call it interrupts or the call that failed, and what humans can do then."""

import json
import logging

from conftest import make_bundle, of_type, read_events, run_episode, small_task
from embodied_harness.env.mock_sorting import MockSortingWorld
from embodied_harness.planner.base import PlannerOutput
from embodied_harness.policy.executor.mock import MockPolicyExecutor
from embodied_harness.runtime.commands import ScriptedCommands
from fake_openai import chat_reply, user_text
from test_vlm_planner import vlm_bundle

HOME = {"name": "home", "when": "previous.skill == 'flip' and next.skill == 'pick'", "run": ["retract"]}


class Operator(ScriptedCommands):
    """Scripted commands that also record what each HOLD showed: (code, message, held call)."""

    def __init__(self, commands):
        super().__init__(commands)
        self.seen = []

    def poll(self, code, message, failed_call):
        self.seen.append((code.value if code else None, message, failed_call))
        return super().poll(code, message, failed_call)


class FlipAfterFailedPlace:
    """pick, place(A); once place(A) has failed back to the planner, flip; then place(A) or done."""

    name = "flip_after_failed_place"

    def __init__(self):
        self.answers = [{"skill": "pick"}, {"skill": "place", "args": {"bin": "A"}}, {"skill": "flip"}]

    def plan(self, context):
        if self.answers:
            decision = self.answers.pop(0)
        elif context.info.get("held"):
            decision = {"skill": "place", "args": {"bin": "A"}}
        else:
            decision = {"done": True}
        return PlannerOutput(raw=json.dumps({**decision, "memory": ""}))


class CrashOnFlip(MockPolicyExecutor):
    def infer_chunk(self, obs):
        if self._prompt is not None and self._prompt.id == "flip":
            raise ConnectionError("policy server went away")
        return super().infer_chunk(obs)


def crash_during_flip(tmp_path, commands):
    """place(A) times out twice and goes back to the planner, which picks flip; the policy server dies
    during the flip. Before the fix, the hold still pointed at the abandoned place(A)."""
    bundle = make_bundle(tmp_path, task=small_task(1))
    env = MockSortingWorld(
        {"num_packages": 1, "barcode_down_prob": 0.0, "scripted_outcomes": {"place_bin_a": ["timeout"] * 2}}
    )
    operator = Operator(commands)
    run = run_episode(
        bundle,
        tmp_path,
        env=env,
        planner=FlipAfterFailedPlace(),
        policy=CrashOnFlip(bundle.library),
        commands=operator,
    )
    return run, env, operator


def test_a_runner_error_closes_the_running_call_and_skip_acts_on_it(tmp_path):
    run, env, operator = crash_during_flip(tmp_path, ["skip"])
    events = read_events(run)
    assert operator.seen[0] == ("E_INTERNAL", "harness error, see episode.log", "flip")
    flip_end = next(e for e in of_type(events, "skill_end") if e["call"]["skill"] == "flip")
    assert flip_end["outcome"]["status"] == "preempted" and flip_end["outcome"]["reason"] == "hold"
    assert flip_end["outcome"]["evidence"]["code"] == "E_INTERNAL"
    assert flip_end["seq"] < of_type(events, "hold")[0]["seq"]
    command = of_type(events, "human_command")[0]
    assert command["failed_call"] == {"skill": "flip", "args": {}}
    # Skipping the flip changes no ledger entry; place(A) only counts once it is really done.
    assert [e["seq"] > command["seq"] for e in of_type(events, "ledger_update")] == [True]
    assert run.result.status == "done"
    assert run.result.ledger["sorted"]["A"] == env.observe().info["bins"]["A"] == 1
    # Every started call has an end, so a resume or the report never sees a dangling one.
    assert len(of_type(events, "skill_start")) == len(of_type(events, "skill_end"))


def test_retry_runs_the_interrupted_call_again(tmp_path):
    run, _, operator = crash_during_flip(tmp_path, ["retry"])
    events = read_events(run)
    command = of_type(events, "human_command")[0]
    assert command["command"] == "retry" and command["failed_call"]["skill"] == "flip"
    after = [e for e in events if e["seq"] > command["seq"]]
    assert of_type(after, "skill_start")[0]["call"]["skill"] == "flip"  # not the abandoned place(A)
    assert operator.seen[1][2] == "flip"
    assert run.result.status == "aborted" and run.result.ledger["sorted"]["A"] == 0


class CrashAfterGrasp(MockPolicyExecutor):
    """The policy server dies once, on the pick's 7th chunk: the package is already in the gripper."""

    def __init__(self, library):
        super().__init__(library, chunk_size=5)
        self.picks = 0

    def infer_chunk(self, obs):
        if self._prompt.id == "pick_next":
            self.picks += 1
            if self.picks == 7:
                raise ConnectionError("policy server went away")
        return super().infer_chunk(obs)


def test_retry_is_refused_while_the_held_calls_preconditions_do_not_hold(tmp_path, caplog):
    bundle = make_bundle(tmp_path, task=small_task(1))
    operator = Operator(["retry", "skip"])
    with caplog.at_level(logging.WARNING, logger="embodied_harness.supervisor"):
        run = run_episode(
            bundle,
            tmp_path,
            world={"num_packages": 1},
            policy=CrashAfterGrasp(bundle.library),
            commands=operator,
        )
    events = read_events(run)
    assert [held for _, _, held in operator.seen] == ["pick", "pick"]  # the refused retry kept the hold
    assert "cannot retry pick: precondition not met: not holding" in caplog.text
    assert run.result.human_commands == ["skip"]
    assert [e["command"] for e in of_type(events, "human_command")] == ["skip"]
    assert [e["call"]["skill"] for e in of_type(events, "skill_start")] == ["pick", "place"]
    assert run.result.status == "done"


def test_a_hold_that_is_not_about_a_call_refuses_skip_and_retry(bundle, tmp_path):
    class Broken:
        name = "broken"

        def plan(self, context):
            return PlannerOutput(raw="place C")

    operator = Operator(["retry", "skip"])
    run = run_episode(bundle, tmp_path, planner=Broken(), commands=operator)
    events = read_events(run)
    assert [held for _, _, held in operator.seen] == [None, None, None]
    assert run.result.human_commands == ["abort"]
    assert run.result.error_code == "E_PLANNER_INVALID"
    # A refused retry no longer turns into a replan.
    assert "human_resume" not in [e["reason"] for e in of_type(events, "planner_request")]


def test_a_failed_transition_step_holds_on_the_call_it_prepares(tmp_path):
    # flip drops the package and goes back to the planner, which picks again; the transition's retract
    # times out. The hold is about that pick, not the abandoned flip.
    skills = {"transitions": [HOME], "skills": {"retract": {"timeout_s": 0.1}}}
    bundle = make_bundle(tmp_path, task=small_task(1), skills=skills)
    world = {"num_packages": 1, "barcode_down_prob": 1.0, "scripted_outcomes": {"flip": ["dropped"]}}
    operator = Operator(["skip"])
    run = run_episode(bundle, tmp_path, world=world, commands=operator)
    events = read_events(run)
    hold = of_type(events, "hold")[0]
    assert hold["message"] == "transition step retract failed (timeout) before pick"
    assert operator.seen[0][2] == "pick"
    command = of_type(events, "human_command")[0]
    assert command["failed_call"] == {"skill": "pick", "args": {}}
    assert not [e for e in of_type(events, "ledger_update") if e["call"]["skill"] == "flip"]


def test_a_hold_during_a_transition_step_closes_it_and_retry_prepares_the_call_again(tmp_path):
    class BrokenRetract(MockSortingWorld):
        def act(self, action):
            if action and action.get("primitive") == "retract":
                raise RuntimeError("arm driver fault")
            return super().act(action)

    bundle = make_bundle(tmp_path, task=small_task(1), skills={"transitions": [HOME]})
    world = {"num_packages": 1, "barcode_down_prob": 1.0, "scripted_outcomes": {"flip": ["dropped"]}}
    env = BrokenRetract(world)
    operator = Operator(["retry"])
    run = run_episode(bundle, tmp_path, env=env, commands=operator)
    events = read_events(run)
    first_hold = of_type(events, "hold")[0]
    closed = [e for e in of_type(events, "skill_end") if e["seq"] < first_hold["seq"]][-1]
    assert closed["call"]["skill"] == "retract" and closed["purpose"] == "transition"
    assert closed["outcome"]["status"] == "preempted" and closed["outcome"]["reason"] == "hold"
    assert operator.seen[0][:2] == ("E_INTERNAL", "harness error, see episode.log")
    assert operator.seen[0][2] == "pick"
    command = of_type(events, "human_command")[0]
    after = [e["type"] for e in events if e["seq"] > command["seq"]]
    assert after[:4] == ["transition", "state_change", "boundary", "skill_start"]  # retract again, then pick
    assert len(of_type(events, "skill_start")) == len(of_type(events, "skill_end"))


# --- a "done" that done_when contradicts ---------------------------------------------------------------


class AlwaysDone:
    name = "always_done"

    def __init__(self):
        self.contexts = []

    def plan(self, context):
        self.contexts.append(context)
        return PlannerOutput(raw=json.dumps({"done": True, "memory": "all sorted"}))


def test_a_planner_that_keeps_claiming_done_holds_after_max_resamples(tmp_path):
    # No budget: before the fix nothing bounded the loop (one planner request per tick, forever).
    bundle = make_bundle(tmp_path, task=small_task(3, budget={"__replace__": True}))
    planner = AlwaysDone()
    run = run_episode(bundle, tmp_path, planner=planner, max_ticks=500)
    events = read_events(run)
    assert run.result is not None and run.result.status == "aborted"
    assert run.result.error_code == "E_PLANNER_INVALID"
    assert "planner declared done" in run.result.message
    requests = of_type(events, "planner_request")
    assert [(e["reason"], e["resample"]) for e in requests] == [
        ("start", 0),
        ("check_failed", 1),
        ("check_failed", 2),
    ]
    assert [c.resample for c in planner.contexts] == [0, 1, 2]
    assert planner.contexts[0].rejected == []
    refusal = "done: done_when is not met: sum(ledger.sorted) == task.expected_total"
    assert planner.contexts[1].rejected == planner.contexts[2].rejected == [refusal]
    assert [e["passed"] for e in of_type(events, "check")] == [False] * 3
    assert run.result.resamples == 3
    # A refused done is a refused answer: its memory ("all sorted") is never kept or shown again.
    assert not of_type(events, "memory_update")
    assert [c.memory for c in planner.contexts] == [""] * 3 and run.result.memory == ""


def test_allow_done_false_refuses_a_planner_done_without_done_when(tmp_path):
    # planner.allow_done: false leaves completion to the env, whatever the planner or its output mode: with
    # no done_when to check it, a planner's done used to end the episode.
    bundle = make_bundle(tmp_path, task=small_task(3, done_when=None), run={"planner": {"allow_done": False}})
    planner = AlwaysDone()
    run = run_episode(bundle, tmp_path, planner=planner)
    events = read_events(run)
    assert run.result.status == "aborted" and run.result.error_code == "E_PLANNER_INVALID"
    assert "planner.allow_done is false" in run.result.message
    assert [e["passed"] for e in of_type(events, "check")] == [False] * 3
    refusal = "done: planner.allow_done is false; the env ends this episode"
    assert [c.rejected for c in planner.contexts] == [[], [refusal], [refusal]]
    assert not of_type(events, "memory_update")


def test_allow_done_false_still_lets_done_when_decide(tmp_path):
    # A done_when decides a planner's done as it does after a commit: here it already holds (nothing to sort).
    bundle = make_bundle(tmp_path, task=small_task(0), run={"planner": {"allow_done": False}})
    run = run_episode(bundle, tmp_path, planner=AlwaysDone())
    assert run.result.status == "done" and run.result.memory == "all sorted"
    assert [e["passed"] for e in of_type(read_events(run), "check")] == [True]


def test_a_done_that_passes_keeps_its_memory(tmp_path):
    # Without done_when (and with allow_done left on) the planner's done ends the episode with its memory.
    bundle = make_bundle(tmp_path, task=small_task(3, done_when=None))
    run = run_episode(bundle, tmp_path, planner=AlwaysDone())
    assert run.result.status == "done" and run.result.memory == "all sorted"
    assert [e["memory"] for e in of_type(read_events(run), "memory_update")] == ["all sorted"]


def test_a_refused_done_is_resampled_hotter_and_shown_to_the_planner(servers, tmp_path):
    server = servers(lambda body, headers: chat_reply('{"done": true, "memory": ""}'))
    bundle = vlm_bundle(tmp_path, server.base_url, show_rejections=True)
    run = run_episode(bundle, tmp_path, world={"cameras": ["head", "wrist"], "image_size": [32, 24]})
    assert run.result.error_code == "E_PLANNER_INVALID"
    assert [body["temperature"] for body in server.requests] == [0.0, 0.7, 0.7]
    assert "done_when is not met" not in user_text(server.requests[0])
    assert "上一次的回答没有通过检查：done: done_when is not met" in user_text(server.requests[1])


def test_a_refused_done_still_lets_a_corrected_answer_through(bundle, tmp_path):
    class DoneOnce:
        name = "done_once"

        def __init__(self):
            self.first = True
            self.oracle = None

        def plan(self, context):
            if self.first:
                self.first = False
                return PlannerOutput(raw=json.dumps({"done": True, "memory": ""}))
            from embodied_harness.planner.oracle import MockSortingOracle

            self.oracle = self.oracle or MockSortingOracle()
            return self.oracle.plan(context)

    run = run_episode(bundle, tmp_path, planner=DoneOnce())
    requests = of_type(read_events(run), "planner_request")
    assert [(e["reason"], e["resample"]) for e in requests[:2]] == [("start", 0), ("check_failed", 1)]
    assert run.result.status == "done"
