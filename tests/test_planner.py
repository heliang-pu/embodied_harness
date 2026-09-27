import json
import threading
import time

import pytest

from conftest import RUN_CONFIG
from fmc3_harness.config.loader import load_bundle
from fmc3_harness.core.types import DecisionReason, SkillCall
from fmc3_harness.logs.context import clear_context, current_context, update_context
from fmc3_harness.planner.base import PlannerContext, PlannerOutput
from fmc3_harness.planner.decision import DecisionError, parse_decision
from fmc3_harness.planner.oracle import MockSortingOracle
from fmc3_harness.planner.runner import InlinePlannerRunner, ThreadPlannerRunner
from fmc3_harness.planner.sequence import SequencePlanner

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
