"""Transition rules: primitives the harness runs between two skill calls, out of the planner's sight."""

import json

import pytest

from conftest import make_bundle, of_type, read_events, run_episode, small_task
from embodied_harness.config.errors import ConfigError

RETREAT = {
    "name": "retreat",
    "when": "previous.skill == 'place' and next.skill == 'pick'",
    "run": ["retract"],
}


def bundle_with(tmp_path, rule=RETREAT, **skills):
    return make_bundle(tmp_path, task=small_task(3), skills={"transitions": [rule], **skills})


def test_the_rule_runs_its_primitive_between_place_and_pick(tmp_path):
    run = run_episode(bundle_with(tmp_path), tmp_path)
    assert run.result.status == "done"
    events = read_events(run)
    starts = [f"{e['call']['skill']}/{e['purpose']}" for e in of_type(events, "skill_start")]
    sequence = " ".join(starts)
    assert "place/main pick/main" not in sequence  # a pick never follows a place directly
    assert sequence.count("place/main retract/transition pick/main") == 2  # before the 2nd and 3rd pick
    assert starts[0] == "pick/main"  # no previous call: the rule does not fire before the first pick
    rules = of_type(events, "transition")
    assert [(e["rule"], e["previous"]["skill"], e["next"]["skill"]) for e in rules] == [
        ("retreat", "place", "pick"),
        ("retreat", "place", "pick"),
    ]
    # The planner never saw the primitive: its call history holds only what it chose.
    requests = of_type(events, "planner_response")
    assert all("retract" not in str(e.get("decision")) for e in requests)


def test_a_failing_transition_primitive_holds(tmp_path):
    bundle = bundle_with(tmp_path, skills={"retract": {"timeout_s": 0.1}})
    run = run_episode(bundle, tmp_path)
    assert run.result.status == "aborted"
    hold = of_type(read_events(run), "hold")[0]
    assert "transition step retract failed (timeout)" in hold["message"]


@pytest.mark.parametrize(
    ("rule", "message"),
    [
        ({"when": "true", "run": ["pick"]}, "'pick' is not a scripted primitive"),
        ({"when": "true", "run": ["jump"]}, "'jump' is not defined"),
        ({"when": "previous.skill ==", "run": ["retract"]}, "transitions[0].when"),
    ],
)
def test_rules_are_checked_when_the_config_loads(tmp_path, rule, message):
    with pytest.raises(ConfigError) as info:
        bundle_with(tmp_path, rule=rule)
    assert any(message in problem for problem in info.value.problems)


def test_rules_may_only_read_known_names(tmp_path):
    with pytest.raises(ConfigError) as info:
        bundle_with(tmp_path, rule={"when": "previous.skill == 'place' and banana", "run": ["retract"]})
    assert any("unknown name(s) banana" in problem for problem in info.value.problems)


APPROACH = {
    "executor": "scripted",
    "description": "move above the bin the last package went to",
    "args": {"bin": {"type": "enum", "values": ["A", "B", "C"]}},
    "timeout_s": 5,
}
APPROACH_RULE = {
    "name": "back_from_the_bin",
    "when": "previous.skill == 'place' and next.skill == 'pick'",
    "run": ["retract", {"skill": "approach", "args": {"bin": "previous.args.bin"}}],
}


def test_steps_take_arguments_from_the_calls_around_them(tmp_path):
    bundle = bundle_with(tmp_path, rule=APPROACH_RULE, skills={"approach": APPROACH})
    run = run_episode(bundle, tmp_path)
    assert run.result.status == "done"
    events = read_events(run)
    rules = of_type(events, "transition")
    places = [
        e["call"]["args"]["bin"] for e in of_type(events, "skill_start") if e["call"]["skill"] == "place"
    ]
    assert [e["run"] for e in rules] == [["retract", f"approach(bin={bin_})"] for bin_ in places[:2]]
    approach = [e for e in of_type(events, "skill_start") if e["call"]["skill"] == "approach"]
    assert [e["purpose"] for e in approach] == ["transition", "transition"]
    actions = [json.loads(line) for line in (run.episode_dir / "actions.jsonl").read_text().splitlines()]
    sent = [a["action"] for a in actions if a["action"].get("primitive") == "approach"]
    assert sent and all(a["args"] == {"bin": places[0]} for a in sent[:1])
    retract = [a["action"] for a in actions if a["action"].get("primitive") == "retract"]
    assert retract and all("args" not in a for a in retract)  # no arguments, no args key: as before


@pytest.mark.parametrize(
    ("step", "message"),
    [
        ({"skill": "approach"}, "approach needs argument(s) bin"),
        ({"skill": "approach", "args": {"bin": "'A'", "speed": "1"}}, "approach has no argument(s) speed"),
        ({"skill": "approach", "args": {"bin": "previous.args["}}, "run.approach.args.bin"),
        ({"skill": "pick", "args": {}}, "'pick' is not a scripted primitive"),
    ],
)
def test_step_arguments_are_checked_when_the_config_loads(tmp_path, step, message):
    rule = {"when": "true", "run": [step]}
    with pytest.raises(ConfigError) as info:
        bundle_with(tmp_path, rule=rule, skills={"approach": APPROACH})
    assert any(message in problem for problem in info.value.problems)


def test_step_arguments_may_only_read_known_names(tmp_path):
    rule = {"when": "true", "run": [{"skill": "approach", "args": {"bin": "target_bin"}}]}
    with pytest.raises(ConfigError) as info:
        bundle_with(tmp_path, rule=rule, skills={"approach": APPROACH})
    assert any("unknown name(s) target_bin" in problem for problem in info.value.problems)


def test_an_argument_outside_its_values_holds(tmp_path):
    rule = {"when": "next.skill == 'pick'", "run": [{"skill": "approach", "args": {"bin": "'D'"}}]}
    run = run_episode(bundle_with(tmp_path, rule=rule, skills={"approach": APPROACH}), tmp_path)
    assert run.result.status == "aborted"
    hold = of_type(read_events(run), "hold")[0]
    assert "transition rule transitions[0]: approach: bin='D' is not one of" in hold["message"]
