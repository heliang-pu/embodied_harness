"""M1: the shared renderer, the decision schema, image encoding, the vLLM planner and replay."""

import dataclasses
import io
import json

import numpy as np
import pytest
from PIL import Image

from conftest import (
    REPLACE,
    RUN_CONFIG,
    make_bundle,
    of_type,
    read_events,
    run_episode,
    small_task,
    write_configs,
)
from fake_openai import chat_reply, image_urls, user_text
from fmc3_harness.cli import main
from fmc3_harness.config.errors import ConfigError
from fmc3_harness.config.loader import load_bundle
from fmc3_harness.core.types import DecisionReason
from fmc3_harness.planner.base import PlannerContext, PlannerOutput
from fmc3_harness.planner.decision import parse_decision
from fmc3_harness.planner.images import encode_image
from fmc3_harness.planner.openai_compat import OpenAICompatPlanner
from fmc3_harness.planner.render import (
    RENDER_VERSION,
    RenderConfig,
    render_prompt,
    render_target,
    skill_signatures,
    to_openai_messages,
)
from fmc3_harness.planner.schema import decision_schema
from fmc3_harness.runtime.factory import check_bundle
from fmc3_harness.trace.replay import replay, summarize

CAMERAS = ["head", "wrist"]
BUNDLE = load_bundle(RUN_CONFIG)


def sorting_reply(body, headers):
    """Picks when the hand is empty, otherwise places into A; decided from the proprio text only."""
    holding = "gripper_closed=true" in user_text(body) and "gripper_width=0.000" not in user_text(body)
    decision = {"skill": "place", "args": {"bin": "A"}} if holding else {"skill": "pick"}
    return chat_reply(json.dumps({**decision, "memory": f"calls so far: {len(body['messages'])}"}))


def vlm_bundle(tmp_path, base_url, **planner_params):
    params = {"base_url": base_url, "model": "fake-planner", "cameras": CAMERAS, "image_size": [32, 24]}
    params.update(planner_params)
    return make_bundle(
        tmp_path,
        task=small_task(3),
        run={
            "env": {"params": {"cameras": CAMERAS, "image_size": [32, 24]}},
            "planner": {REPLACE: True, "type": "openai_compat", "params": params},
        },
    )


def context(images=None, resample=0):
    return PlannerContext(
        task=BUNDLE.task,
        images=images if images is not None else {c: np.zeros((24, 32, 3), np.uint8) for c in CAMERAS},
        proprio={"gripper_width": 0.09, "gripper_closed": False},
        memory="old memory",
        reason=DecisionReason.START,
        skills=list(BUNDLE.task.skills),
        resample=resample,
    )


# --- renderer and schema -------------------------------------------------------------------------


def test_skill_signatures():
    assert skill_signatures(BUNDLE.registry, BUNDLE.task.skills) == [
        "pick",
        "flip",
        "place(bin: A|B|C)",
        "retract",
    ]


def test_render_prompt_layout():
    prompt = render_prompt(
        task=BUNDLE.task,
        signatures=skill_signatures(BUNDLE.registry, BUNDLE.task.skills),
        images={"wrist": "W", "head": "H"},
        proprio={"gripper_width": 0.05, "gripper_closed": True},
        memory="",
        config=RenderConfig(cameras=("head", "wrist"), proprio_keys=("gripper_width", "gripper_closed")),
    )
    text = prompt.text()
    assert prompt.version == RENDER_VERSION
    assert text.index("<image:head>") < text.index("<image:wrist>")  # configured order, not dict order
    assert "本体状态：gripper_width=0.050, gripper_closed=true" in text
    assert "记忆：（空）" in text
    assert BUNDLE.task.goal in prompt.system
    assert "- place(bin: A|B|C)" in prompt.system


def test_render_prompt_needs_every_camera():
    with pytest.raises(KeyError, match="missing camera"):
        render_prompt(
            task=BUNDLE.task,
            signatures=[],
            images={"head": "H"},
            proprio={},
            memory="",
            config=RenderConfig(cameras=("head", "wrist")),
        )


@pytest.mark.parametrize(
    "decision",
    [
        {"skill": "place", "args": {"bin": "C"}, "memory": "m"},
        {"skill": "pick", "memory": "x"},
        {"done": True, "memory": ""},
    ],
)
def test_render_target_round_trips_through_the_parser(decision):
    parsed = parse_decision(render_target(decision)).to_dict()
    assert parsed["memory"] == decision["memory"]
    assert parsed.get("skill") == decision.get("skill")
    assert parsed.get("args", {}) == decision.get("args", {})


def test_openai_messages_interleave_labels_and_images():
    prompt = render_prompt(
        task=BUNDLE.task,
        signatures=[],
        images={"head": "H", "wrist": "W"},
        proprio={},
        memory="m",
        config=RenderConfig(cameras=("head", "wrist")),
    )
    messages = to_openai_messages(prompt, lambda camera, image: f"url:{camera}")
    kinds = [part["type"] for part in messages[1]["content"]]
    assert kinds == ["text", "image_url", "text", "image_url", "text"]
    assert messages[1]["content"][1]["image_url"]["url"] == "url:head"


def test_decision_schema():
    variants = decision_schema(BUNDLE.registry, BUNDLE.task.skills, 50)["anyOf"]
    skills = [v["properties"]["skill"]["enum"][0] for v in variants if "skill" in v["properties"]]
    assert skills == ["pick", "flip", "place", "retract"]
    place = variants[2]
    assert place["properties"]["args"]["properties"]["bin"]["enum"] == ["A", "B", "C"]
    assert place["required"] == ["skill", "args", "memory"]
    assert variants[-1]["required"] == ["done", "memory"]
    assert all(v["properties"]["memory"]["maxLength"] == 50 for v in variants)


def test_encode_image():
    array = np.random.default_rng(0).integers(0, 255, (24, 32, 3), dtype=np.uint8)
    data = encode_image(array, size=(16, 12))
    assert data[:2] == b"\xff\xd8"
    assert Image.open(io.BytesIO(data)).size == (16, 12)
    floats = encode_image(np.full((8, 8, 3), 0.5, dtype=np.float32), fmt="png")
    assert Image.open(io.BytesIO(floats)).getpixel((0, 0)) == (128, 128, 128)
    assert encode_image(data) is not None and encode_image(data) == data  # bytes pass through untouched


# --- the vLLM-backed planner ---------------------------------------------------------------------


def bound_planner(base_url, **kwargs):
    planner = OpenAICompatPlanner(
        base_url=base_url, model="fake-planner", cameras=CAMERAS, image_size=[32, 24], **kwargs
    )
    planner.bind(registry=BUNDLE.registry, task=BUNDLE.task, planner_config=BUNDLE.run.planner)
    return planner


def test_request_carries_images_schema_and_key(servers, monkeypatch):
    server = servers(lambda body, headers: chat_reply('{"skill": "pick", "memory": "m1"}'))
    monkeypatch.setenv("FMC3_TEST_KEY", "secret")
    planner = bound_planner(server.base_url, api_key_env="FMC3_TEST_KEY")
    output = planner.plan(context())

    assert json.loads(output.raw) == {"skill": "pick", "memory": "m1"}
    body = server.requests[0]
    assert server.paths[0] == "/v1/chat/completions"
    assert body["model"] == "fake-planner"
    assert body["temperature"] == 0.0
    urls = image_urls(body)
    assert len(urls) == 2 and all(url.startswith("data:image/jpeg;base64,") for url in urls)
    assert "记忆：old memory" in user_text(body)
    assert body["response_format"]["json_schema"]["schema"]["anyOf"]
    assert server.headers[0]["authorization"] == "Bearer secret"
    assert set(output.attachments) == {"head.jpg", "wrist.jpg"}
    assert output.meta["render"] == RENDER_VERSION
    assert "<image:head>" in output.meta["prompt"]


def test_resampling_raises_the_temperature(servers):
    server = servers(lambda body, headers: chat_reply('{"skill": "pick", "memory": ""}'))
    bound_planner(server.base_url).plan(context(resample=1))
    assert server.requests[0]["temperature"] == 0.7


def test_http_errors_raise(servers):
    server = servers(lambda body, headers: (500, {"error": "model crashed"}))
    with pytest.raises(RuntimeError, match="HTTP 500"):
        bound_planner(server.base_url).plan(context())


def test_plan_before_bind_fails():
    planner = OpenAICompatPlanner(base_url="http://127.0.0.1:9/v1", model="m", cameras=CAMERAS)
    with pytest.raises(RuntimeError, match="bind"):
        planner.plan(context())


def test_planner_cameras_must_exist_in_the_env(tmp_path):
    with pytest.raises(ConfigError) as info:
        check_bundle(
            load_bundle(
                write_configs(
                    tmp_path,
                    run={
                        "env": {"params": {"cameras": ["head"]}},
                        "planner": {
                            REPLACE: True,
                            "type": "openai_compat",
                            "params": {"base_url": "http://x/v1", "model": "m", "cameras": ["head", "side"]},
                        },
                    },
                )
            )
        )
    assert any("side" in problem for problem in info.value.problems)


# --- closed loop and replay ----------------------------------------------------------------------


def test_vlm_planner_closes_the_loop(servers, tmp_path):
    server = servers(sorting_reply)
    bundle = vlm_bundle(tmp_path, server.base_url)
    run = run_episode(bundle, tmp_path, world={"cameras": CAMERAS, "image_size": [32, 24]})

    assert run.result.status == "done"
    assert run.result.ledger["sorted"]["A"] == 3
    assert len(server.requests) == 6
    assert all(len(image_urls(body)) == 2 for body in server.requests)
    record = json.loads((run.episode_dir / "planner" / "pc_0001.json").read_text())
    assert record["attachments"] == ["planner/pc_0001/head.jpg", "planner/pc_0001/wrist.jpg"]
    assert (run.episode_dir / "planner" / "pc_0001" / "head.jpg").read_bytes()[:2] == b"\xff\xd8"
    assert record["meta"]["render"] == RENDER_VERSION
    memories = [event["memory"] for event in of_type(read_events(run), "memory_update")]
    assert memories[0].startswith("calls so far")


def test_replay_compares_a_new_planner_with_the_recording(servers, tmp_path):
    server = servers(sorting_reply)
    bundle = vlm_bundle(tmp_path, server.base_url)
    run = run_episode(bundle, tmp_path, world={"cameras": CAMERAS, "image_size": [32, 24]})

    seen_images = []

    class AlwaysPick:
        name = "always_pick"

        def plan(self, ctx):
            seen_images.append({camera: type(image) for camera, image in ctx.images.items()})
            return PlannerOutput(raw='{"skill": "pick", "memory": ""}')

    rows = replay(AlwaysPick(), [run.episode_dir], bundle.task)
    summary = summarize(rows)
    assert summary["calls"] == 6 and summary["same_skill"] == 3 and summary["same_call"] == 3
    assert seen_images[0] == {"head": bytes, "wrist": bytes}  # the recorded images, byte for byte


def test_replay_cli_with_the_same_planner_agrees(servers, tmp_path, capsys):
    server = servers(sorting_reply)
    bundle = vlm_bundle(tmp_path, server.base_url)
    run = run_episode(bundle, tmp_path, world={"cameras": CAMERAS, "image_size": [32, 24]})
    before = len(server.requests)
    report = tmp_path / "replay.json"

    assert main(["replay", str(bundle.path), str(run.episode_dir), "--out", str(report)]) == 0
    assert "same call 6/6" in capsys.readouterr().out
    assert len(server.requests) == before + 6
    # The replayed requests carry exactly the recorded images.
    assert image_urls(server.requests[0]) == image_urls(server.requests[before])
    assert json.loads(report.read_text())["summary"]["same_call_rate"] == 1.0


def test_context_is_a_plain_dataclass():
    assert dataclasses.replace(context(), resample=2).resample == 2
