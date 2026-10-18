import json
import threading
import time

import pytest

from conftest import RUN_CONFIG, make_bundle, write_configs
from embodied_harness.config.loader import load_bundle
from embodied_harness.core.types import DecisionReason, SkillCall
from embodied_harness.planner.base import PlannerContext, PlannerOutput
from embodied_harness.planner.decision import DecisionError, parse_decision
from embodied_harness.planner.oracle import MockSortingOracle
from embodied_harness.planner.runner import InlinePlannerRunner, ThreadPlannerRunner
from embodied_harness.planner.sequence import SequencePlanner
from embodied_harness.runtime.factory import bind_planner
from embodied_harness.runtime.logs.context import clear_context, current_context, update_context

TASK = load_bundle(RUN_CONFIG).task


def context(reason=DecisionReason.START, resample=0, info=None) -> PlannerContext:
    return PlannerContext(
        task=TASK,
        images={},
        proprio={},
        memory="",
        reason=reason,
        skills=list(TASK.skills),
        resample=resample,
        info=info or {},
    )


def wait_for(runner, seconds=5.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        result = runner.poll()
        if result is not None:
            return result
        time.sleep(0.01)
    return None


def test_parses_a_skill_call():
    decision = parse_decision('{"skill": "place", "args": {"bin": "C"}, "memory": "m"}')
    assert decision.call == SkillCall("place", {"bin": "C"})
    assert decision.memory == "m"
    assert not decision.done


def test_parses_done():
    assert parse_decision('{"done": true, "memory": ""}').done


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("place C", "not JSON"),
        ('["pick"]', "JSON object"),
        ('{"skill": "pick"}', "memory"),
        ('{"skill": "pick", "memory": 3}', "memory"),
        ('{"done": true, "skill": "pick", "memory": ""}', "both"),
        ('{"done": "yes", "memory": ""}', "true or false"),
        ('{"memory": ""}', "'skill'"),
        ('{"skill": "place", "args": ["C"], "memory": ""}', "'args'"),
        ('{"skill": "pick", "memory": "", "why": "x"}', "unexpected"),
    ],
)
def test_rejects_malformed_output(raw, message):
    with pytest.raises(DecisionError, match=message):
        parse_decision(raw)


def world(held=None, on_table=0):
    return {"packages": [], "held": held, "on_table": on_table, "bins": {"A": 0, "B": 0, "C": 0}}


@pytest.mark.parametrize(
    ("info", "expected"),
    [
        (world(on_table=2), {"skill": "pick"}),
        (world(held={"size": "large", "barcode_up": False}), {"skill": "flip"}),
        (world(held={"size": "large", "barcode_up": True}), {"skill": "place", "args": {"bin": "C"}}),
        (world(), {"done": True}),
    ],
)
def test_oracle(info, expected):
    decision = json.loads(MockSortingOracle().plan(context(info=info)).raw)
    memory = decision.pop("memory")
    assert decision == expected
    assert memory.startswith("已放")


def test_sequence_planner_repeats_a_failed_step_and_then_finishes():
    planner = SequencePlanner([{"skill": "pick"}, {"skill": "place", "args": {"bin": "A"}}])

    def next_decision(reason, resample=0):
        return json.loads(planner.plan(context(reason, resample)).raw)

    assert next_decision(DecisionReason.START)["skill"] == "pick"
    assert next_decision(DecisionReason.SKILL_FAILED)["skill"] == "pick"
    assert next_decision(DecisionReason.SKILL_DONE)["skill"] == "place"
    assert next_decision(DecisionReason.SKILL_DONE, resample=1)["skill"] == "place"
    assert next_decision(DecisionReason.SKILL_DONE)["done"] is True


def test_sequence_planner_restores_a_completed_step_from_trace():
    planner = SequencePlanner([{"skill": "pick"}, {"skill": "place"}])
    events = [
        {"type": "planner_request", "call_id": "pc_1", "reason": "start", "resample": 0},
        {
            "type": "planner_response",
            "call_id": "pc_1",
            "meta": {"sequence_index": 0},
            "decision": {"skill": "pick", "args": {}, "memory": ""},
            "errors": [],
        },
        {
            "type": "skill_start",
            "call_id": "sc_1",
            "planner_call_id": "pc_1",
            "purpose": "main",
            "call": {"skill": "pick", "args": {}},
        },
        {
            "type": "skill_end",
            "call_id": "sc_1",
            "purpose": "main",
            "outcome": {"status": "success"},
        },
    ]
    planner.restore_trace(events)
    assert json.loads(planner.plan(context(DecisionReason.RESUME)).raw)["skill"] == "place"


def test_sequence_planner_repeats_an_interrupted_duplicate_step():
    planner = SequencePlanner([{"skill": "pick"}, {"skill": "pick"}, {"skill": "place"}])
    events = [
        {"type": "planner_request", "call_id": "pc_1", "reason": "start", "resample": 0},
        {
            "type": "planner_response",
            "call_id": "pc_1",
            "meta": {"sequence_index": 0},
            "decision": {"skill": "pick", "args": {}, "memory": ""},
            "errors": [],
        },
        {
            "type": "skill_start",
            "call_id": "sc_1",
            "planner_call_id": "pc_1",
            "purpose": "main",
            "call": {"skill": "pick", "args": {}},
        },
    ]
    planner.restore_trace(events)
    output = json.loads(planner.plan(context(DecisionReason.RESUME)).raw)
    assert output["skill"] == "pick"
    assert output["memory"] == "第 1/3 步"


@pytest.mark.parametrize("ending", ["failure", "rejected", "skipped"])
def test_sequence_resume_handles_failures_rejections_and_human_skips(ending):
    calls = [{"skill": "pick"}, {"skill": "place"}, {"skill": "retract"}]
    events = []

    def proposal(number, index, reason, errors=None):
        events.extend(
            [
                {"type": "planner_request", "call_id": f"pc{number}", "reason": reason, "resample": 0},
                {
                    "type": "planner_response",
                    "call_id": f"pc{number}",
                    "decision": {**calls[index], "memory": ""},
                    "errors": errors or [],
                },
            ]
        )

    def execution(number, index, status):
        events.extend(
            [
                {"type": "skill_start", "call_id": f"sc{number}", "call": calls[index]},
                {"type": "skill_end", "call_id": f"sc{number}", "outcome": {"status": status}},
            ]
        )

    proposal(1, 0, "start")
    execution(1, 0, "success")
    proposal(2, 1, "skill_done", errors=["precondition"] if ending == "rejected" else None)
    if ending != "rejected":
        execution(2, 1, "failure")
    if ending == "skipped":
        events.append({"type": "human_command", "command": "skip", "failed_call": calls[1]})
    elif ending == "failure":
        # The failed step was proposed again before interruption. It must not consume a new index.
        proposal(3, 1, "skill_failed")

    planner = SequencePlanner(calls)
    planner.restore_trace(events)
    output = planner.plan(context(DecisionReason.RESUME))
    index = 2 if ending == "skipped" else 1
    assert output.meta["sequence_index"] == index
    assert json.loads(output.raw)["skill"] == calls[index]["skill"]
    if ending == "rejected":
        assert planner.plan(context(DecisionReason.RESUME, resample=1)).meta["sequence_index"] == index


def test_sequence_resume_rejects_a_trace_with_different_calls():
    planner = SequencePlanner([{"skill": "pick"}])
    events = [
        {"type": "planner_request", "call_id": "pc1", "reason": "start"},
        {"type": "planner_response", "call_id": "pc1", "decision": {"skill": "place"}},
    ]
    with pytest.raises(ValueError, match="does not match configured step"):
        planner.restore_trace(events)


class Echo:
    name = "echo"

    def plan(self, context):
        return PlannerOutput(raw='{"done": true, "memory": ""}')


class Raising:
    name = "raising"

    def plan(self, context):
        raise RuntimeError("model server down")


def test_inline_runner_returns_the_output_or_the_exception():
    runner = InlinePlannerRunner(Echo())
    runner.submit(context())
    output = runner.poll()
    assert isinstance(output, PlannerOutput) and output.latency_s >= 0
    assert runner.poll() is None

    runner = InlinePlannerRunner(Raising())
    runner.submit(context())
    assert isinstance(runner.poll(), RuntimeError)


def test_thread_runner_keeps_the_log_context():
    seen = {}

    class Recording:
        name = "recording"

        def plan(self, context):
            seen.update(current_context())
            return PlannerOutput(raw="{}")

    clear_context()
    update_context(episode_id="ep_x")
    runner = ThreadPlannerRunner(Recording())
    try:
        runner.submit(context())
        assert isinstance(wait_for(runner), PlannerOutput)
        assert seen["episode_id"] == "ep_x"
    finally:
        runner.close()
        clear_context()


def test_thread_runner_cancel_does_not_block_the_next_call():
    release = threading.Event()

    class Blocking:
        name = "blocking"
        calls = 0

        def plan(self, context):
            Blocking.calls += 1
            if Blocking.calls == 1:
                release.wait(5)
            return PlannerOutput(raw="{}")

    runner = ThreadPlannerRunner(Blocking())
    try:
        runner.submit(context())
        time.sleep(0.05)
        runner.cancel()  # the first call is stuck
        runner.submit(context())
        assert isinstance(wait_for(runner, 2.0), PlannerOutput)
    finally:
        release.set()
        runner.close()


# --- the sequence planner on a task plan: rollback, following human commands, resume ------------------


EMPTY = {"gripper_width": 0.0, "gripper_closed": True}  # closed on nothing: the package fell
HOLDING = {"gripper_width": 0.05, "gripper_closed": True}
PICK_OK = {"call": "pick", "status": "success", "reason": None}
FLIP_DROPPED = {"call": "flip", "status": "failure", "reason": "dropped"}
ONE_PACKAGE = ["pick", "flip", {"skill": "place", "args": {"bin": "A"}}]
ONE_PACKAGE_CALLS = [{"skill": "pick"}, {"skill": "flip"}, {"skill": "place", "args": {"bin": "A"}}]


def bound_sequence(tmp_path, calls=None, **task):
    bundle = make_bundle(
        tmp_path,
        task={"task": {"expected_total": 1, "plan": ONE_PACKAGE, **task}},
        run={"planner": {"type": "sequence", "params": {"__replace__": True}}},
    )
    planner = SequencePlanner(calls)
    bind_planner(planner, bundle)
    return planner, bundle


def sorted_ledger(a=0):
    return {"sorted": {"A": a, "B": 0, "C": 0}}


def planned(planner, reason, *, last=None, proprio=EMPTY, resample=0, ledger=None):
    context = PlannerContext(
        task=TASK,
        images={},
        proprio=dict(proprio),
        memory="",
        reason=reason,
        skills=list(TASK.skills),
        resample=resample,
        ledger=sorted_ledger() if ledger is None else ledger,
        last=last,
    )
    output = planner.plan(context)
    return json.loads(output.raw), output.meta


def test_sequence_planner_rolls_back_only_after_a_matching_failure(tmp_path):
    planner, _ = bound_sequence(tmp_path)
    planned(planner, DecisionReason.START)
    planned(planner, DecisionReason.SKILL_DONE, last=PICK_OK)
    decision, meta = planned(planner, DecisionReason.SKILL_FAILED, last=FLIP_DROPPED)
    assert decision["skill"] == "pick"
    assert meta == {"planner": "sequence", "sequence_index": 0, "rollback": {"from": 1, "reason": "dropped"}}
    # A resample of the rollback proposal re-issues it; `last` is not a failure of the pick.
    assert planned(planner, DecisionReason.SKILL_FAILED, last=FLIP_DROPPED, resample=1)[1] == {
        "planner": "sequence",
        "sequence_index": 0,
    }

    planner, _ = bound_sequence(tmp_path / "timeout")
    planned(planner, DecisionReason.START)
    planned(planner, DecisionReason.SKILL_DONE, last=PICK_OK)
    timeout = {"call": "flip", "status": "failure", "reason": "timeout"}
    assert planned(planner, DecisionReason.SKILL_FAILED, last=timeout)[1]["sequence_index"] == 1

    unbound = SequencePlanner(ONE_PACKAGE_CALLS)  # without bind() it never rolls back
    planned(unbound, DecisionReason.START)
    planned(unbound, DecisionReason.SKILL_DONE, last=PICK_OK)
    assert planned(unbound, DecisionReason.SKILL_FAILED, last=FLIP_DROPPED)[1]["sequence_index"] == 1


def test_a_rollback_waits_until_its_target_can_start(tmp_path):
    planner, _ = bound_sequence(tmp_path)
    planned(planner, DecisionReason.START)
    planned(planner, DecisionReason.SKILL_DONE, last=PICK_OK)
    decision, meta = planned(planner, DecisionReason.SKILL_FAILED, last=FLIP_DROPPED, proprio=HOLDING)
    assert decision["skill"] == "flip"
    assert meta["rollback_blocked"] == {"to": 0, "reason": "dropped", "unmet": ["not holding"]}
    # The supervisor refused the repeated flip (the gripper did turn out empty): the resample rolls back.
    decision, meta = planned(planner, DecisionReason.SKILL_FAILED, last=FLIP_DROPPED, resample=1)
    assert decision["skill"] == "pick" and meta["rollback"] == {"from": 1, "reason": "dropped"}


@pytest.mark.parametrize("status", ["success", "skipped"])
def test_the_cursor_follows_a_human_retry_or_skip_of_a_later_step(tmp_path, status):
    planner, _ = bound_sequence(tmp_path)
    planned(planner, DecisionReason.START)
    planned(planner, DecisionReason.SKILL_DONE, last=PICK_OK)
    assert planned(planner, DecisionReason.SKILL_FAILED, last=FLIP_DROPPED)[0]["skill"] == "pick"
    # HOLD before the pick ran; the human retried (or skipped) the flip instead.
    decision, meta = planned(planner, DecisionReason.SKILL_DONE, last={"call": "flip", "status": status})
    assert (decision["skill"], meta["sequence_index"]) == ("place", 2)


@pytest.mark.parametrize("resample", [0, 1])  # the supervisor may ask again as a resample of the round
def test_a_refused_done_raises_instead_of_looping(tmp_path, resample):
    planner, _ = bound_sequence(tmp_path)
    planned(planner, DecisionReason.START)
    for call in ("pick", "flip", "place(bin=A)"):
        decision, _ = planned(planner, DecisionReason.SKILL_DONE, last={"call": call, "status": "success"})
    assert decision["done"] is True
    with pytest.raises(RuntimeError, match="the plan has ended but done_when"):  # not place(bin=A) again
        planned(planner, DecisionReason.CHECK_FAILED, resample=resample)


def test_a_human_resume_goes_on_after_a_step_that_already_succeeded(tmp_path):
    planner, _ = bound_sequence(tmp_path)
    planned(planner, DecisionReason.START)
    # The pick was verified, then a HOLD (a budget, an e-stop) came before the planner was asked again.
    decision, meta = planned(planner, DecisionReason.HUMAN_RESUME, last=PICK_OK)
    assert (decision["skill"], meta["sequence_index"]) == ("flip", 1)
    # A HOLD while that flip runs: nothing finished since it was proposed, so it is issued again.
    decision, meta = planned(planner, DecisionReason.HUMAN_RESUME, last=PICK_OK)
    assert (decision["skill"], meta["sequence_index"]) == ("flip", 1)
    # The plan ended with done_when false: a resume answers done again, not the last step once more.
    planned(planner, DecisionReason.SKILL_DONE, last={"call": "flip", "status": "success"})
    placed = {"call": "place(bin=A)", "status": "success"}
    assert planned(planner, DecisionReason.SKILL_DONE, last=placed)[0]["done"] is True
    with pytest.raises(RuntimeError, match="the plan has ended"):
        planned(planner, DecisionReason.CHECK_FAILED, last=placed)
    assert planned(planner, DecisionReason.HUMAN_RESUME, last=placed)[0]["done"] is True


def test_a_human_resume_does_not_take_an_earlier_success_for_the_proposal(tmp_path):
    # The flip was proposed after the first pick and a HOLD came before it finished, so `last` still shows
    # that pick's success. The plan's second pick further on must not be taken for it.
    planner, _ = bound_sequence(tmp_path, plan=[*ONE_PACKAGE, *ONE_PACKAGE], expected_total=2)
    planned(planner, DecisionReason.START)
    planned(planner, DecisionReason.SKILL_DONE, last=PICK_OK)
    decision, meta = planned(planner, DecisionReason.HUMAN_RESUME, last=PICK_OK)
    assert (decision["skill"], meta["sequence_index"]) == ("flip", 1)


def test_a_human_resume_tells_two_equal_steps_apart_by_the_ledger(tmp_path):
    placed = {"call": "place(bin=A)", "status": "success"}
    planner, _ = bound_sequence(tmp_path, plan=[*ONE_PACKAGE, ONE_PACKAGE[-1]], expected_total=2)
    planned(planner, DecisionReason.START)
    planned(planner, DecisionReason.SKILL_DONE, last=PICK_OK)
    planned(planner, DecisionReason.SKILL_DONE, last={"call": "flip", "status": "success"})
    second = planned(planner, DecisionReason.SKILL_DONE, last=placed, ledger=sorted_ledger(1))
    assert second[1]["sequence_index"] == 3
    # A HOLD while the second place runs: `last` and the ledger are what the proposal was made with.
    resumed = planned(planner, DecisionReason.HUMAN_RESUME, last=placed, ledger=sorted_ledger(1))
    assert resumed[1]["sequence_index"] == 3
    # A HOLD after it was verified: `last` reads the same, but its effect moved the ledger on.
    resumed = planned(planner, DecisionReason.HUMAN_RESUME, last=placed, ledger=sorted_ledger(2))
    assert resumed[0]["done"] is True


def test_bind_takes_task_plan_and_refuses_a_different_one(tmp_path):
    planner, bundle = bound_sequence(tmp_path)
    assert planned(planner, DecisionReason.START)[0]["skill"] == "pick"
    with pytest.raises(ValueError, match="differs from task"):
        bind_planner(SequencePlanner([{"skill": "pick"}]), bundle)
    bind_planner(SequencePlanner(ONE_PACKAGE_CALLS), bundle)  # the same steps: accepted
    without = load_bundle(RUN_CONFIG)  # no task.plan, no params.calls
    with pytest.raises(ValueError, match="no steps"):
        bind_planner(SequencePlanner(), without)
    periodic = load_bundle(write_configs(tmp_path / "p", run={"planner": {"call": "periodic"}}))
    with pytest.raises(ValueError, match="call: boundary"):
        bind_planner(SequencePlanner([{"skill": "pick"}]), periodic)


def _rollback_trace():
    """pick done, flip dropped, rollback proposal pick (pc3), a HOLD, then the human's retried flip."""

    def request(number, reason):
        return {"type": "planner_request", "call_id": f"pc{number}", "reason": reason, "resample": 0}

    def response(number, index, call, **meta):
        decision = {"skill": call, "args": {}, "memory": ""}
        return {
            "type": "planner_response",
            "call_id": f"pc{number}",
            "meta": {"sequence_index": index, **meta},
            "decision": decision,
            "errors": [],
        }

    def start(number, planner_call, call):
        return {
            "type": "skill_start",
            "call_id": f"sc{number}",
            "planner_call_id": f"pc{planner_call}",
            "purpose": "main",
            "call": {"skill": call, "args": {}},
        }

    def end(number, status, reason=None):
        return {
            "type": "skill_end",
            "call_id": f"sc{number}",
            "purpose": "main",
            "outcome": {"status": status, "reason": reason},
        }

    return (
        [
            request(1, "start"),
            response(1, 0, "pick"),
            start(1, 1, "pick"),
            end(1, "success"),
            request(2, "skill_done"),
            response(2, 1, "flip"),
            start(2, 2, "flip"),
            end(2, "failure", "dropped"),
            request(3, "skill_failed"),
            response(3, 0, "pick", rollback={"from": 1, "reason": "dropped"}),
            {"type": "hold", "message": "transition step retract failed (timeout)"},
        ],
        start,
        end,
    )


@pytest.mark.parametrize(
    ("ending", "expected"),
    [("retry_running", "pick"), ("retry_succeeded", "place"), ("retry_failed", "pick"), ("skipped", "place")],
)
def test_sequence_resume_after_a_human_command_that_followed_a_rollback(ending, expected):
    events, start, end = _rollback_trace()
    flip = {"skill": "flip", "args": {}}
    if ending == "skipped":
        events.append({"type": "human_command", "command": "skip", "failed_call": flip})
    else:
        events += [{"type": "human_command", "command": "retry", "failed_call": flip}, start(3, 3, "flip")]
        if ending == "retry_succeeded":
            events.append(end(3, "success"))
        elif ending == "retry_failed":
            events.append(end(3, "failure", "dropped"))
    planner = SequencePlanner(ONE_PACKAGE_CALLS)
    planner.restore_trace(events)
    decision, _ = planned(planner, DecisionReason.RESUME)
    assert decision["skill"] == expected
