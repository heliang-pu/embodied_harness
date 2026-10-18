"""M1: the shared renderer, the decision schema, image encoding, the vLLM planner and replay."""

import dataclasses
import io
import json
import logging

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
from embodied_harness.cli import main
from embodied_harness.config.errors import ConfigError
from embodied_harness.config.loader import load_bundle
from embodied_harness.core.types import DecisionReason
from embodied_harness.planner.base import PlannerContext, PlannerOutput
from embodied_harness.planner.decision import parse_decision
from embodied_harness.planner.images import encode_image
from embodied_harness.planner.openai_compat import OpenAICompatPlanner
from embodied_harness.planner.render import (
    RENDER_VERSION,
    RenderConfig,
    render_prompt,
    render_target,
    skill_signatures,
    to_openai_messages,
)
from embodied_harness.planner.schema import decision_schema
from embodied_harness.runtime.factory import bind_planner, build_planner, check_bundle
from embodied_harness.runtime.trace.replay import replay, summarize
from fake_openai import chat_reply, image_urls, user_text

CAMERAS = ["head", "wrist"]
BUNDLE = load_bundle(RUN_CONFIG)


def sorting_reply(body, headers):
    """Picks when the hand is empty, otherwise places into A; decided from the proprio text only."""
    holding = "gripper_closed=true" in user_text(body) and "gripper_width=0.000" not in user_text(body)
    decision = {"skill": "place", "args": {"bin": "A"}} if holding else {"skill": "pick"}
    return chat_reply(json.dumps({**decision, "memory": f"calls so far: {len(body['messages'])}"}))


def vlm_bundle(tmp_path, base_url, planner_fields=None, task=None, **planner_params):
    params = {"base_url": base_url, "model": "fake-planner", "cameras": CAMERAS, "image_size": [32, 24]}
    params.update(planner_params)
    return make_bundle(
        tmp_path,
        task=task or small_task(3),
        run={
            "env": {"params": {"cameras": CAMERAS, "image_size": [32, 24]}},
            "planner": {REPLACE: True, "type": "openai_compat", "params": params, **(planner_fields or {})},
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


def test_render_prompt_can_show_the_steps_so_far():
    def render(show_history, **extra):
        config = RenderConfig(cameras=("head",), show_history=show_history)
        prompt = render_prompt(
            task=BUNDLE.task,
            signatures=[],
            images={"head": "H"},
            proprio={},
            memory="m",
            config=config,
            **extra,
        )
        return prompt.text()

    shown = render(True, history=["pick", "place(bin=A)"], reason="skill_failed")
    assert "已执行的步骤：1. pick；2. place(bin=A)\n这次调用的原因：上一步失败了\n记忆：m" in shown
    assert "已执行的步骤：（无）\n这次调用的原因：任务开始" in render(True, reason="start")
    assert "已执行的步骤" not in render(False, history=["pick"], reason="skill_done")


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
    monkeypatch.setenv("PLANNER_TEST_KEY", "secret")
    planner = bound_planner(server.base_url, api_key_env="PLANNER_TEST_KEY")
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


def trained_export(tmp_path, base_url, without=()):
    """A training export's manifest.json holding the default planner's settings, less `without`."""
    trained = vlm_bundle(tmp_path / "trained", base_url)
    planner = build_planner(trained)
    bind_planner(planner, trained)
    settings = {key: value for key, value in planner.settings().items() if key not in without}
    export = tmp_path / ("old_export" if without else "export")
    export.mkdir()
    (export / "manifest.json").write_text(json.dumps({"planner": settings}), encoding="utf-8")
    return export


def test_a_run_warns_about_every_setting_that_differs_from_the_training_export(servers, tmp_path, caplog):
    server = servers(sorting_reply)
    export = trained_export(tmp_path, server.base_url)

    def run(name, params=None, task=None, manifest=export, **planner_fields):
        fields = {"trained_manifest": str(manifest), **planner_fields}
        bundle = vlm_bundle(tmp_path / name, server.base_url, fields, task, **(params or {}))
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="embodied_harness.runtime"):
            result = run_episode(bundle, tmp_path / name, world={"cameras": CAMERAS, "image_size": [32, 24]})
        assert result.result.status == "done"  # a warning, not a refusal
        return result, [r.getMessage() for r in caplog.records if "trained_manifest" in r.getMessage()]

    assert run("same")[1] == []
    result, warnings = run("drifted", {"show_progress": True}, allow_done=False)
    # One per setting; the completion sentence of the system prompt changed too.
    assert [warning.split()[2] for warning in warnings] == ["no_done", "show_progress", "system"]
    assert "trained False, now True; the model is not seeing what it was trained on" in warnings[1]
    assert "(the system prompt: rules, lessons, skills or the done sentence)" in warnings[2]
    assert warnings[1] in (result.episode_dir / "episode.log").read_text(encoding="utf-8")
    # Every call sends the configured goal.
    warnings = run("goal", task=small_task(3, goal="把所有包裹分拣好"))[1]
    assert [warning.split()[2] for warning in warnings] == ["goal"]
    # The export may be on another machine: nothing is compared, which the warning says.
    elsewhere = (tmp_path / "elsewhere").resolve()
    assert run("missing", manifest=elsewhere)[1] == [
        f"planner.trained_manifest: {elsewhere}: file not found; the planner settings are not compared"
    ]
    # An export made before the goal and the system prompt were fingerprinted cannot show they changed.
    old = trained_export(tmp_path, server.base_url, without=("goal", "system"))
    warnings = run("old_rules", task=small_task(3, rules=["先抓大的"]), manifest=old)[1]
    assert warnings == [
        f"planner.trained_manifest ({old}): the training export has no goal or system fingerprint (it was "
        "made before they were recorded): a changed goal, rule, lesson, skill or done sentence goes "
        "unnoticed; export again to compare them"
    ]


def test_validate_compares_the_planner_with_its_training_export(tmp_path, capsys):
    base_url = "http://127.0.0.1:9/v1"  # building and binding the planner sends nothing
    export = trained_export(tmp_path, base_url)

    def validate(name, **params):
        bundle = vlm_bundle(tmp_path / name, base_url, {"trained_manifest": str(export)}, **params)
        code = main(["validate", str(bundle.path)])
        printed = capsys.readouterr()
        return code, printed.out + printed.err

    code, printed = validate("same")
    assert code == 0 and f"planner settings match planner.trained_manifest ({export})" in printed
    code, printed = validate("drifted", show_progress=True)
    assert code == 0 and "warning: planner setting show_progress differs" in printed
    # What the run would stop at, now reported ahead of it.
    code, printed = validate("broken", show_state=True, state_predicates=["grasped"])
    assert code == 2 and "the planner cannot be built" in printed and "grasped not defined" in printed


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


def test_replayed_calls_keep_the_step_history(tmp_path):
    from embodied_harness.runtime.trace.replay import context_from_record

    history = [{"skill": "pick", "args": {}}, {"skill": "place", "args": {"bin": "A"}}]
    record = {
        "input": {
            "proprio": {"gripper_width": 0.05},
            "memory": "m",
            "reason": "skill_done",
            "skills": list(BUNDLE.task.skills),
            "history": history,
        },
        "attachments": [],
    }
    context = context_from_record(record, tmp_path, BUNDLE.task)
    assert context.history == history  # a show_history planner renders the same prompt as the original call
