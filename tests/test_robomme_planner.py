"""RoboMME's subgoal predictor backend: the exact prompt of their Qwen3VLModel, and text -> skill call."""

import base64
import io

import numpy as np
import pytest
from PIL import Image

from conftest import REPO
from embodied_harness.config.loader import load_bundle
from embodied_harness.core.types import DecisionReason
from embodied_harness.planner.base import PlannerContext
from embodied_harness.planner.decision import parse_decision
from embodied_harness.planner.robomme_subgoal import RoboMMESubgoalPlanner, wrap_history
from fake_openai import chat_reply

CONFIG = REPO / "configs" / "robomme" / "mme" / "run_mme_pickx.yaml"
GOAL = (
    "pick up the red cube and place it on the target, repeating this action two times, "
    "then press the button to stop"
)


def planner(url):
    bundle = load_bundle(CONFIG)
    backend = RoboMMESubgoalPlanner(base_url=url, model="simple_subgoal")
    backend.bind(registry=bundle.registry, task=bundle.task, planner_config=bundle.run.planner)
    return backend, bundle


def context(bundle, history=()):
    front = np.arange(256 * 256 * 3, dtype=np.uint32).reshape(256, 256, 3).astype(np.uint8)
    return PlannerContext(
        task=bundle.task.model_copy(update={"goal": GOAL}),
        images={"front": front, "wrist": np.zeros((256, 256, 3), np.uint8)},
        proprio={"gripper_width": 0.04},
        memory="",
        reason=DecisionReason.START,
        skills=["subgoal"],
        history=[{"skill": "subgoal", "args": {"text": text}} for text in history],
    )


def test_first_turn_prompt_matches_robomme(servers):
    server = servers(lambda body, headers: chat_reply("Pick up the red cube for the first time"))
    backend, bundle = planner(server.base_url)
    ctx = context(bundle)
    output = backend.plan(ctx)
    body = server.requests[0]
    assert body["model"] == "simple_subgoal" and body["temperature"] == 0.0 and body["max_tokens"] == 128
    system, user = body["messages"]
    assert system == {
        "role": "system",
        "content": (
            "You are a helpful assistant to help guide the robot to complete the task "
            "by predicting a sequence of language subgoals"
        ),
    }
    before, image, after = user["content"]
    assert before == {
        "type": "text",
        "text": f"The task goal is: {GOAL}\nThis is the initial turn for prediction\n",
    }
    assert after == {"type": "text", "text": "What's the next language subgoal based on current observation?"}
    url = image["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    sent = np.asarray(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))))
    np.testing.assert_array_equal(sent, ctx.images["front"])  # lossless, full resolution, front only
    decision = parse_decision(output.raw)
    assert decision.call.skill == "subgoal"
    assert decision.call.args == {"text": "pick up the red cube for the first time"}
    assert bundle.registry.check_call(decision.call) == []
    assert output.meta["response"] == "Pick up the red cube for the first time"
    assert output.attachments["front.png"] == base64.b64decode(url.split(",", 1)[1])


def test_history_is_numbered_like_robomme(servers):
    server = servers(lambda body, headers: chat_reply("press the button to stop"))
    backend, bundle = planner(server.base_url)
    history = ["pick up the red cube for the first time", "place the red cube onto the target"]
    backend.plan(context(bundle, history))
    before = server.requests[0]["messages"][1]["content"][0]["text"]
    assert before == (
        f"The task goal is: {GOAL}\nThe history of previous predicted language subgoals are: "
        "1. pick up the red cube for the first time; 2. place the red cube onto the target\n"
    )
    assert wrap_history(["a"]) == "1. a"


def test_unknown_subgoal_is_left_for_the_supervisor_to_reject(servers):
    server = servers(lambda body, headers: chat_reply("dance with the cube"))
    backend, bundle = planner(server.base_url)
    decision = parse_decision(backend.plan(context(bundle)).raw)
    assert decision.call.args == {"text": "dance with the cube"}
    assert bundle.registry.check_call(decision.call)  # not in the enum: rejected


def test_bind_checks_the_subgoal_skill():
    bundle = load_bundle(CONFIG)
    backend = RoboMMESubgoalPlanner(base_url="http://127.0.0.1:9/v1", model="m", skill="nope")
    with pytest.raises(ValueError, match="no skill"):
        backend.bind(registry=bundle.registry, task=bundle.task, planner_config=bundle.run.planner)
    with pytest.raises(RuntimeError, match="bind"):
        RoboMMESubgoalPlanner(base_url="http://x/v1", model="m").plan(context(bundle))


def test_subgoals_map_onto_split_skills(servers):
    replies = iter(["place the red cube onto the target", "pick up the red cube for the second time"])
    server = servers(lambda body, headers: chat_reply(next(replies)))
    bundle = load_bundle(CONFIG.with_name("run_mme_pickx_gated.yaml"))
    backend = RoboMMESubgoalPlanner(base_url=server.base_url, model="simple_subgoal")
    backend.bind(registry=bundle.registry, task=bundle.task, planner_config=bundle.run.planner)
    ctx = context(bundle)
    ctx.history = [{"skill": "pick", "args": {"text": "pick up the red cube for the first time"}}]
    first = parse_decision(backend.plan(ctx).raw).call
    assert (first.skill, first.args) == ("place", {"text": "place the red cube onto the target"})
    second = parse_decision(backend.plan(ctx).raw).call
    assert second.skill == "pick" and bundle.registry.check_call(second) == []
    before = server.requests[0]["messages"][1]["content"][0]["text"]
    assert "are: 1. pick up the red cube for the first time\n" in before
    assert bundle.registry.skill("place").switch_when[0].source == "released"
