"""Structured planner input: predicates and the last result (show_state), task progress (show_progress),
lessons from earlier runs, and thinking models."""

import json

import numpy as np
import pytest
import yaml

from conftest import REPLACE, RUN_CONFIG, make_bundle, run_episode, small_task, write_configs
from embodied_harness.cli import main
from embodied_harness.config.errors import ConfigError
from embodied_harness.config.loader import load_bundle
from embodied_harness.core.types import DecisionReason
from embodied_harness.planner.base import PlannerContext
from embodied_harness.planner.http import split_reasoning
from embodied_harness.planner.oracle import MockSortingOracle
from embodied_harness.planner.render import RenderConfig, format_last, format_progress, render_prompt
from embodied_harness.runtime.factory import bind_planner, build_planner
from embodied_harness.runtime.trace.reader import restore_state
from fake_openai import chat_reply, user_text

BUNDLE = load_bundle(RUN_CONFIG)
CAMERAS = ["head"]


def render(**options):
    extra = {key: options.pop(key) for key in ("predicates", "last", "ledger") if key in options}
    prompt = render_prompt(
        task=BUNDLE.task,
        signatures=["pick"],
        images={"head": "H"},
        proprio={"gripper_width": 0.05},
        memory="m",
        config=RenderConfig(cameras=("head",), **options),
        **extra,
    )
    return prompt


def test_state_and_progress_lines():
    text = render(
        show_state=True,
        show_progress=True,
        predicates={"holding": True, "released": False},
        last={"call": "pick", "status": "failure", "reason": "empty_grasp"},
        ledger={"sorted": {"A": 1, "B": 0, "C": 2}},
    ).text()
    assert (
        "本体状态：gripper_width=0.050\n"
        "当前状态：holding=true, released=false\n"
        "上一步：pick 失败（empty_grasp）\n"
        "进度：sorted: A=1, B=0, C=2（共 3）；目标总数 10\n"
        "记忆：m"
    ) in text


def test_options_off_leave_the_default_layout_alone():
    plain = render().text()
    assert "当前状态" not in plain and "进度" not in plain and "上一步" not in plain
    with pytest.raises(ValueError, match="ledger"):
        render(show_progress=True)
    with pytest.raises(ValueError, match="predicate"):
        render(show_state=True)


def test_formatting_helpers():
    assert format_last(None) == "（无）"
    assert format_last({"call": "place(bin=A)", "status": "success"}) == "place(bin=A) 成功"
    assert format_last({"call": "flip", "status": "preempted", "reason": "planner"}) == "flip 被打断"
    assert format_last({"call": "pick", "status": "skipped"}) == "pick 人工完成"
    ledger = {"sorted": {"A": 1}, "done": True, "count": 3, "phase": "pick"}
    assert format_progress(ledger, None, None) == "sorted: A=1（共 1）；done=true；count=3；phase=pick"
    assert format_progress(ledger, ["count"], 5) == "count=3；目标总数 5"
    assert format_progress({}, None, None) == "（无）"
    with pytest.raises(KeyError, match="no field 'missing'"):
        format_progress(ledger, ["missing"], None)


def test_lessons_are_listed_after_the_rules():
    task = BUNDLE.task.model_copy(update={"lessons": ["放完 A 区先 retract 再 pick"]})
    prompt = render_prompt(
        task=task,
        signatures=["pick"],
        images={"head": "H"},
        proprio={},
        memory="",
        config=RenderConfig(cameras=("head",)),
    )
    lessons = "经验（以往运行的总结）：\n- 放完 A 区先 retract 再 pick\n"
    assert f"规则：\n- 条码朝下的包裹先翻面再放\n{lessons}可用技能" in prompt.system
    assert "经验" not in render().system


def test_predicate_values_read_only_what_they_need():
    registry = BUNDLE.registry
    assert registry.predicate_values({"gripper_width": 0.05, "gripper_closed": True}) == {
        "holding": True,
        "empty_grasp": False,
        "released": False,
    }
    # `released` reads only gripper_width, so gripper_closed may be missing.
    assert registry.predicate_values({"gripper_width": 0.09}, ["released"]) == {"released": True}


# --- the vLLM planner ------------------------------------------------------------------------------


def vlm_run(base_url, **params):
    planner_params = {
        "base_url": base_url,
        "model": "fake-planner",
        "cameras": CAMERAS,
        "image_size": [32, 24],
    }
    planner_params.update(params)
    return {
        "env": {"params": {"cameras": CAMERAS, "image_size": [32, 24]}},
        "planner": {REPLACE: True, "type": "openai_compat", "params": planner_params},
    }


def bound(tmp_path, base_url="http://127.0.0.1:9/v1", task=None, **params):
    bundle = make_bundle(tmp_path, task=task or small_task(3), run=vlm_run(base_url, **params))
    planner = build_planner(bundle)
    bind_planner(planner, bundle)
    return bundle, planner


def context(bundle, **fields):
    return PlannerContext(
        task=bundle.task,
        images={"head": np.zeros((24, 32, 3), np.uint8)},
        proprio={"gripper_width": 0.05, "gripper_closed": True},
        memory="",
        reason=DecisionReason.SKILL_DONE,
        skills=list(bundle.task.skills),
        **fields,
    )


def test_settings_record_every_option(tmp_path):
    _, planner = bound(
        tmp_path,
        show_state=True,
        state_predicates=["holding"],
        show_progress=True,
        progress_fields=["sorted"],
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )
    settings = planner.settings()
    assert settings["show_state"] is True and settings["state_predicates"] == ["holding"]
    assert settings["show_progress"] is True and settings["progress_fields"] == ["sorted"]
    assert settings["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}
    assert settings["lessons"] is None


def test_settings_record_the_done_mode_the_call_mode_and_the_system_prompt(tmp_path):
    def settings(name, task=None, **planner_fields):
        run = vlm_run("http://127.0.0.1:9/v1")
        run["planner"].update(planner_fields)
        bundle = make_bundle(tmp_path / name, task=task or small_task(3), run=run)
        planner = build_planner(bundle)
        bind_planner(planner, bundle)
        return planner.settings()

    default = settings("default")
    # Off by default, so an export made before these were recorded compares equal.
    assert default["no_done"] is False and "call" not in default
    assert len(default["system"]) == 12
    assert settings("again")["system"] == default["system"]
    # allow_done false rewrites the completion sentence of the system prompt.
    no_done = settings("no_done", allow_done=False)
    assert no_done["no_done"] is True and no_done["system"] != default["system"]
    assert settings("periodic", call="periodic", every_steps=4)["call"] == "periodic"
    rules = settings("rules", task={"task": {"expected_total": 3, "rules": ["先抓大的"]}})
    assert rules["system"] != default["system"] and rules["goal"] == default["goal"]
    # The goal has a fingerprint of its own: every exported sample records its goal, so `data eval`
    # compares the rest of the system prompt alone.
    goal = settings("goal", task={"task": {"expected_total": 3, "goal": "b"}})
    assert len(default["goal"]) == 12 and goal["goal"] != default["goal"]
    assert goal["system"] == default["system"]
    # An episode's own goal replaces the configured one, so there is no configured goal to fingerprint.
    episode = settings("episode", task={"task": {"goal_source": "episode", "expected_total": 3, "goal": "b"}})
    assert episode["goal"] is None and episode["system"] == default["system"]


def test_bind_checks_predicate_and_ledger_names(tmp_path):
    with pytest.raises(ValueError, match="state_predicates: grasped not defined"):
        bound(tmp_path, show_state=True, state_predicates=["grasped"])
    with pytest.raises(ValueError, match="progress_fields: the ledger has no placed"):
        bound(tmp_path, show_progress=True, progress_fields=["placed"])
    with pytest.raises(ConfigError, match="extra_body cannot set messages"):
        bound(tmp_path, extra_body={"messages": []})


def test_the_request_carries_state_progress_and_extra_fields(tmp_path, servers):
    server = servers(lambda body, headers: chat_reply(json.dumps({"skill": "pick", "memory": ""})))
    bundle, planner = bound(
        tmp_path,
        server.base_url,
        show_state=True,
        state_predicates=["holding", "released"],
        show_progress=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}, "top_p": 0.9},
    )
    last = {"call": "place(bin=B)", "status": "success", "reason": None}
    planner.plan(context(bundle, ledger={"sorted": {"A": 0, "B": 1, "C": 0}}, last=last))
    body = server.requests[-1]
    text = user_text(body)
    assert "当前状态：holding=true, released=false\n上一步：place(bin=B) 成功\n" in text
    assert "进度：sorted: A=0, B=1, C=0（共 1）；目标总数 3\n" in text
    assert body["chat_template_kwargs"] == {"enable_thinking": False} and body["top_p"] == 0.9
    assert body["response_format"]["type"] == "json_schema"  # extra fields never replace the schema


def reasoning_reply(message):
    def reply(body, headers):
        return 200, {"choices": [{"index": 0, "message": {"role": "assistant", **message}}]}

    return reply


def test_thinking_output_is_split_from_the_decision(tmp_path, servers):
    decision = json.dumps({"skill": "pick", "memory": ""})
    split = servers(reasoning_reply({"content": decision, "reasoning_content": "hand is empty"}))
    _, planner = bound(tmp_path / "a", split.base_url)
    output = planner.plan(context(load_bundle(RUN_CONFIG)))
    assert output.raw == decision and output.meta["reasoning"] == "hand is empty"

    inline = servers(reasoning_reply({"content": f"<think>\nnothing held\n</think>\n\n{decision}"}))
    _, planner = bound(tmp_path / "b", inline.base_url)
    output = planner.plan(context(load_bundle(RUN_CONFIG)))
    assert output.raw == decision and output.meta["reasoning"] == "nothing held"

    plain = servers(reasoning_reply({"content": decision}))
    _, planner = bound(tmp_path / "c", plain.base_url)
    assert "reasoning" not in planner.plan(context(load_bundle(RUN_CONFIG))).meta


def test_split_reasoning():
    assert split_reasoning('{"done": true}') == ('{"done": true}', None)
    assert split_reasoning("<think>a</think>{}") == ("{}", "a")
    assert split_reasoning("opened by the template</think> {}") == ("{}", "opened by the template")


# --- what the supervisor passes ----------------------------------------------------------------------


class RecordingOracle(MockSortingOracle):
    def __init__(self):
        super().__init__()
        self.contexts = []

    def plan(self, context):
        self.contexts.append(context)
        return super().plan(context)


def test_the_supervisor_passes_the_ledger_and_the_last_result(bundle, tmp_path):
    planner = RecordingOracle()
    # The first pick closes on nothing; the recovery table retries it without asking the planner.
    world = {"scripted_outcomes": {"pick_next": ["empty_grasp"]}}
    run = run_episode(bundle, tmp_path, planner=planner, world=world)
    assert run.result.status == "done"
    first, *rest = planner.contexts
    assert first.last is None and first.ledger == {"sorted": {"A": 0, "B": 0, "C": 0}}
    assert [c.last["call"].split("(")[0] for c in rest] == ["pick", "place", "pick", "place", "pick"]
    assert all(c.last["status"] == "success" for c in rest)
    # The third place fills the ledger, so done_when ends the episode without asking again.
    assert [sum(c.ledger["sorted"].values()) for c in planner.contexts] == [0, 0, 1, 1, 2, 2]
    recorded = json.loads((run.episode_dir / "planner" / "pc_0002.json").read_text(encoding="utf-8"))
    assert recorded["input"]["last"] == rest[0].last and recorded["input"]["ledger"] == rest[0].ledger


def test_a_failure_reaches_the_planner_with_its_reason(tmp_path):
    bundle = make_bundle(
        tmp_path,
        task=small_task(3),
        skills={"skills": {"pick": {"on_failure": {"empty_grasp": {"retry": 0}}}}},
    )
    planner = RecordingOracle()
    run = run_episode(
        bundle, tmp_path, planner=planner, world={"scripted_outcomes": {"pick_next": ["empty_grasp"]}}
    )
    assert run.result.status == "done"
    failed = [c for c in planner.contexts if c.reason is DecisionReason.SKILL_FAILED]
    assert failed and failed[0].last == {"call": "pick", "status": "failure", "reason": "empty_grasp"}


def test_resume_restores_the_history_and_the_last_result():
    events = [
        {"seq": 1, "t": 0.0, "type": "skill_start", "call": {"skill": "pick", "args": {}}, "purpose": "main"},
        {
            "seq": 2,
            "t": 3.0,
            "type": "skill_end",
            "call": {"skill": "pick", "args": {}},
            "purpose": "main",
            "outcome": {"status": "success", "reason": None},
        },
        {
            "seq": 3,
            "t": 3.1,
            "type": "skill_start",
            "call": {"skill": "retract", "args": {}},
            "purpose": "transition",
        },
        {
            "seq": 4,
            "t": 3.6,
            "type": "skill_end",
            "call": {"skill": "retract", "args": {}},
            "purpose": "transition",
            "outcome": {"status": "success", "reason": None},
        },
        {
            "seq": 5,
            "t": 3.7,
            "type": "skill_start",
            "call": {"skill": "place", "args": {"bin": "A"}},
            "purpose": "main",
        },
        {
            "seq": 6,
            "t": 5.0,
            "type": "skill_end",
            "call": {"skill": "place", "args": {"bin": "A"}},
            "purpose": "main",
            "outcome": {"status": "failure", "reason": "timeout"},
        },
        {
            "seq": 7,
            "t": 5.1,
            "type": "skill_start",
            "call": {"skill": "place", "args": {"bin": "A"}},
            "purpose": "main",
        },
    ]
    restored = restore_state(events)
    assert [call.key() for call in restored.history] == ["pick", "place(bin=A)"]
    assert restored.last == {"call": "place(bin=A)", "status": "failure", "reason": "timeout"}


# --- lessons files -----------------------------------------------------------------------------------


def write_lessons(tmp_path, entries):
    path = tmp_path / "configs" / "lessons.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"lessons": entries}, allow_unicode=True), encoding="utf-8")
    return path


def test_reviewed_lessons_join_the_task(tmp_path):
    write_lessons(
        tmp_path,
        [
            {"text": "先翻面再放", "reviewed": True, "source": "report"},
            {"text": "还没看过的", "reviewed": False},
        ],
    )
    config = write_configs(
        tmp_path,
        task={"task": {"lessons": ["写在任务里的"]}},
        run={"planner": {"lessons": "../lessons.yaml"}},
    )
    bundle = load_bundle(config)
    assert bundle.task.lessons == ["写在任务里的", "先翻面再放"]
    assert bundle.unreviewed_lessons == 1


def test_lessons_change_the_planner_settings(tmp_path):
    write_lessons(tmp_path, [{"text": "先翻面再放", "reviewed": True}])
    _, planner = bound(tmp_path, task={"task": {"expected_total": 3}})
    assert planner.settings()["lessons"] is None
    run = vlm_run("http://127.0.0.1:9/v1")
    run["planner"]["lessons"] = "../lessons.yaml"
    bundle = make_bundle(tmp_path, task=small_task(3), run=run)
    planner = build_planner(bundle)
    bind_planner(planner, bundle)
    assert planner.settings()["lessons"] and len(planner.settings()["lessons"]) == 12


def test_a_missing_lessons_file_is_a_config_error(tmp_path, capsys):
    config = write_configs(tmp_path, run={"planner": {"lessons": "../nowhere.yaml"}})
    with pytest.raises(ConfigError, match=r"nowhere\.yaml: file not found"):
        load_bundle(config)
    write_lessons(tmp_path, [{"text": "x", "reviewed": False}])
    config = write_configs(tmp_path, run={"planner": {"lessons": "../lessons.yaml"}})
    assert main(["validate", str(config)]) == 0
    assert "lessons: 0 shown to the planner, 1 not reviewed yet (left out)" in capsys.readouterr().out


def test_replay_keeps_the_ledger_and_the_last_result(tmp_path):
    from embodied_harness.runtime.trace.replay import context_from_record

    last = {"call": "pick", "status": "success", "reason": None}
    original = PlannerContext(
        task=BUNDLE.task,
        images={},
        proprio={"gripper_width": 0.05},
        memory="",
        reason=DecisionReason.SKILL_DONE,
        skills=list(BUNDLE.task.skills),
        ledger={"sorted": {"A": 1, "B": 0, "C": 0}},
        last=last,
    )
    restored = context_from_record({"input": original.summary()}, tmp_path, BUNDLE.task)
    assert restored.ledger == original.ledger and restored.last == last


def test_a_refused_answer_is_explained_on_the_resample(tmp_path, servers):
    answers = iter([{"skill": "place", "args": {"bin": "A"}}, {"skill": "pick"}])

    def reply(body, headers):
        return chat_reply(json.dumps({**next(answers, {"skill": "pick"}), "memory": ""}))

    server = servers(reply)
    bundle = make_bundle(tmp_path, task=small_task(1), run=vlm_run(server.base_url, show_rejections=True))
    planner = build_planner(bundle)
    run = run_episode(bundle, tmp_path, planner=planner, world={"num_packages": 1, "cameras": CAMERAS})
    first, second = (user_text(body) for body in server.requests[:2])
    assert "上一次的回答" not in first
    assert (
        "上一次的回答没有通过检查：place(bin=A): precondition not met: holding。请换一个回答。\n记忆"
        in second
    )
    recorded = json.loads((run.episode_dir / "planner" / "pc_0002.json").read_text(encoding="utf-8"))
    assert recorded["input"]["rejected"] == ["place(bin=A): precondition not met: holding"]
    # The next round starts clean.
    third = json.loads((run.episode_dir / "planner" / "pc_0003.json").read_text(encoding="utf-8"))
    assert third["input"]["rejected"] == []
    assert planner.settings()["show_rejections"] is True
