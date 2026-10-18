import json

import pytest
import yaml

from conftest import (
    CONFIGS,
    RUN_CONFIG,
    make_bundle,
    of_type,
    read_events,
    run_episode,
    small_task,
    write_configs,
)
from embodied_harness.config.errors import ConfigError
from embodied_harness.config.loader import load_bundle
from embodied_harness.config.schema import SuiteEntry
from embodied_harness.core.types import SkillCall
from embodied_harness.runtime.factory import build_planner, check_bundle, trained_settings
from embodied_harness.runtime.logs.codes import ErrorCode
from embodied_harness.runtime.suite import prepare_entry


def test_mock_config_is_valid():
    bundle = load_bundle(RUN_CONFIG)
    check_bundle(bundle)
    assert bundle.registry.names == ["pick", "flip", "place", "retract"]


def test_resolves_prompts_from_the_library():
    registry = load_bundle(RUN_CONFIG).registry
    assert registry.resolve_prompt(SkillCall("place", {"bin": "C"})).text == "place the package into bin C"
    assert registry.resolve_prompt(SkillCall("pick")).id == "pick_next"
    assert registry.resolve_prompt(SkillCall("retract")) is None


def test_timeouts():
    registry = load_bundle(RUN_CONFIG).registry
    assert registry.timeout_for("retract") == 5  # explicit
    assert registry.timeout_for("pick") == pytest.approx(7.0)  # p95 3.5 s x factor 2
    assert registry.timeout_for("place") == pytest.approx(6.0)  # the largest p95 among the mapped prompts


def test_check_call():
    registry = load_bundle(RUN_CONFIG).registry
    assert registry.check_call(SkillCall("place", {"bin": "A"})) == []
    assert "not one of" in registry.check_call(SkillCall("place", {"bin": "D"}))[0]
    assert "missing" in registry.check_call(SkillCall("place"))[0]
    assert "unexpected" in registry.check_call(SkillCall("pick", {"bin": "A"}))[0]


def _problems(tmp_path, **patches) -> list[str]:
    with pytest.raises(ConfigError) as info:
        check_bundle(load_bundle(write_configs(tmp_path, **patches)))
    return info.value.problems


def test_unknown_prompt_id(tmp_path):
    problems = _problems(tmp_path, skills={"skills": {"pick": {"prompt": "pick_anything"}}})
    assert any("'pick_anything' is not in prompt library" in p for p in problems)


def test_prompt_map_must_cover_every_value(tmp_path):
    problems = _problems(
        tmp_path, skills={"skills": {"place": {"args": {"bin": {"values": ["A", "B", "C", "D"]}}}}}
    )
    assert any("no prompt for bin=D" in p for p in problems)


def test_executor_vla_is_an_older_name_for_policy(tmp_path):
    bundle = load_bundle(write_configs(tmp_path, skills={"skills": {"pick": {"executor": "vla"}}}))
    assert bundle.skills.skills["pick"].executor == "policy"
    assert not bundle.registry.skill("pick").scripted


def test_policy_skills_need_a_prompt(tmp_path):
    problems = _problems(tmp_path, skills={"skills": {"flip": {"prompt": None}}})
    assert any("skills run by the policy need a prompt" in p for p in problems)


def test_skills_written_for_another_library(tmp_path):
    problems = _problems(tmp_path, skills={"library": "pi05_sort_v3"})
    assert any("pi05_sort_v3" in p for p in problems)


def test_planner_trained_on_another_library(tmp_path):
    with pytest.raises(ConfigError) as info:
        load_bundle(write_configs(tmp_path, run={"planner": {"trained_library": "pi05_sort_v2"}}))
    assert info.value.code is ErrorCode.E_LIBRARY_MISMATCH


def _vlm_run(params=None, **planner_fields):
    """The mock run with the openai_compat planner."""
    params = {"base_url": "http://127.0.0.1:9/v1", "model": "m", "cameras": ["head"], **(params or {})}
    return {
        "env": {"params": {"cameras": ["head"]}},
        "planner": {"type": "openai_compat", "params": {"__replace__": True, **params}, **planner_fields},
    }


def test_allow_done_among_the_planner_params_is_refused(tmp_path):
    # It used to be accepted there and ignored: the schema and the prompt still offered done.
    run = _vlm_run({"allow_done": False})
    problems = _problems(tmp_path, run=run)
    assert any(p.startswith("planner.params.allow_done:") and "set planner.allow_done" in p for p in problems)
    with pytest.raises(ConfigError, match="allow_done"):
        build_planner(load_bundle(write_configs(tmp_path / "built", run=run)))
    check_bundle(load_bundle(write_configs(tmp_path / "top", run=_vlm_run(allow_done=False))))


def test_the_trained_manifest_is_read_only_to_compare_settings(tmp_path):
    export = tmp_path / "configs" / "export"
    run = {"planner": {"trained_manifest": "../export"}}  # relative to configs/runs/
    # A config may name an export this machine lacks, or one the data commands are about to make.
    bundle = make_bundle(tmp_path, run=run)
    assert bundle.trained_manifest == export.resolve()
    assert load_bundle(RUN_CONFIG).trained_manifest is None
    with pytest.raises(ValueError, match="file not found"):
        trained_settings(bundle.trained_manifest)
    export.mkdir(parents=True)
    (export / "manifest.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot read it as JSON"):
        trained_settings(export)
    (export / "manifest.json").write_text(json.dumps({"splits": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="no planner settings"):
        trained_settings(export)
    settings = {"render": "render_v1", "show_history": True}
    (export / "manifest.json").write_text(json.dumps({"planner": settings}), encoding="utf-8")
    assert trained_settings(export) == trained_settings(export / "manifest.json") == settings


def test_a_suite_entry_keeps_the_trained_manifest(tmp_path):
    export = tmp_path / "configs" / "export"
    config = write_configs(tmp_path, run={"planner": {"trained_manifest": "../export"}})
    suite = tmp_path / "suite.yaml"
    entry = SuiteEntry(name="a", config=str(config.relative_to(tmp_path)), category="A")
    # The entry's run config is written into the entry directory, so its relative paths are made absolute.
    bundle = prepare_entry(suite, entry, tmp_path / "entry")
    assert bundle.trained_manifest == export.resolve()


def test_unknown_names_are_reported_once_the_env_is_known(tmp_path):
    problems = _problems(tmp_path, skills={"skills": {"pick": {"preconditions": ["not holdng"]}}})
    assert any("holdng" in p for p in problems)


def test_effects_must_target_ledger_fields(tmp_path):
    problems = _problems(
        tmp_path, skills={"skills": {"place": {"effects": {"on_success": ["ledger.placed += 1"]}}}}
    )
    assert any("no field 'placed'" in p for p in problems)


def test_before_retry_skill_must_exist(tmp_path):
    problems = _problems(
        tmp_path,
        skills={"skills": {"pick": {"on_failure": {"empty_grasp": {"retry": 1, "before_retry": ["home"]}}}}},
    )
    assert any("'home' is not defined" in p for p in problems)


def test_all_problems_are_reported_together(tmp_path):
    problems = _problems(
        tmp_path,
        skills={"skills": {"pick": {"prompt": "nope", "preconditions": ["(("]}}},
        task={"task": {"skills": ["pick", "fly"]}},
    )
    assert len(problems) >= 3


def test_schema_errors_name_the_field(tmp_path):
    problems = _problems(tmp_path, run={"planner": {"max_resamples": -1}})
    assert any("planner.max_resamples" in p for p in problems)


# --- task.plan and rollback rules ---------------------------------------------------------------------

SEQUENCE_RUN = {"planner": {"type": "sequence", "params": {"__replace__": True}}}
PLACE_A = {"skill": "place", "args": {"bin": "A"}}
PLACE_B = {"skill": "place", "args": {"bin": "B"}}
TWO_UNITS = ["pick", "flip", PLACE_A, "pick", "flip", PLACE_B]


def _plan_problems(tmp_path, plan, *, total=2, skills=None, run=SEQUENCE_RUN, **task):
    patches = {"task": {"task": {"expected_total": total, "plan": plan, **task}}, "run": run}
    if skills:
        patches["skills"] = skills
    return _problems(tmp_path, **patches)


def test_a_plan_config_is_valid(tmp_path):
    bundle = load_bundle(write_configs(tmp_path, task={"task": {"expected_total": 2, "plan": TWO_UNITS}}))
    check_bundle(bundle)
    assert [call.key() for call in bundle.registry.plan] == [
        "pick",
        "flip",
        "place(bin=A)",
        "pick",
        "flip",
        "place(bin=B)",
    ]
    assert bundle.registry.plan_rollbacks(bundle.registry.plan) == [
        {},
        {"dropped": 0},
        {},
        {},
        {"dropped": 3},
        {},
    ]


@pytest.mark.parametrize(
    ("plan", "extra", "message"),
    [
        (["pick", "dance"], {}, "task.plan[1]: 'dance' is not one of the task's skills"),
        (
            ["pick", {"skill": "place", "args": {"bin": "D"}}],
            {},
            "task.plan[1]: place: bin='D' is not one of",
        ),
        (["pick", "flip", PLACE_A], {"total": 2}, "is still false after all 3 steps"),
        (["pick", "flip", PLACE_A, "retract"], {"total": 1}, "already holds before [3] retract"),
        (
            TWO_UNITS,
            {"budget": {"max_skill_calls": 4}},
            "max_skill_calls (4) is smaller than task.plan (6 steps)",
        ),
        # a unit without its pick: rolling back to the first pick would place package A again
        (["pick", "flip", PLACE_A, "flip", PLACE_B], {}, "would redo [2] place(bin=A)"),
    ],
)
def test_plan_problems(tmp_path, plan, extra, message):
    problems = _plan_problems(tmp_path, plan, **extra)
    assert any(message in problem for problem in problems), problems


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ({"rollback": "jump"}, "rollback skill 'jump' is not defined"),
        ({"rollback": "flip"}, "rollback names 'flip' itself"),
        ({"escalate": "hold"}, "rollback needs escalate: planner"),
    ],
)
def test_rollback_rules_are_checked(tmp_path, entry, message):
    problems = _problems(tmp_path, skills={"skills": {"flip": {"on_failure": {"dropped": entry}}}})
    assert any(f"skills.flip.on_failure.dropped: {message}" in problem for problem in problems)


def test_rollback_is_not_a_default(tmp_path):
    problems = _problems(tmp_path, skills={"defaults": {"on_failure": {"rollback": "pick"}}})
    assert any(problem.startswith("defaults.on_failure: rollback") for problem in problems)


def test_a_rollback_may_not_redo_an_effect_that_reads_the_ledger(tmp_path):
    counting = {
        "skills": {
            "place": {"effects": {"on_success": ["ledger.sorted[args.bin] = ledger.sorted[args.bin] + 1"]}}
        }
    }
    problems = _plan_problems(tmp_path, ["pick", "flip", PLACE_A, "flip", PLACE_B], skills=counting)
    assert any("would redo [2] place(bin=A)" in problem for problem in problems)


def test_the_done_when_check_is_skipped_when_an_effect_reads_robot_state(tmp_path):
    reads_proprio = {
        "skills": {
            "place": {"effects": {"on_success": ["ledger.sorted[args.bin] += 1 if gripper_width > 0 else 0"]}}
        }
    }
    bundle = load_bundle(
        write_configs(
            tmp_path,
            task={"task": {"expected_total": 2, "plan": ["pick", "flip", PLACE_A]}},
            skills=reads_proprio,
        )
    )
    assert len(bundle.registry.plan) == 3


def test_plan_problems_are_reported_together(tmp_path):
    problems = _plan_problems(
        tmp_path,
        ["pick", "dance", {"skill": "place", "args": {"bin": "D"}}],
        skills={"skills": {"flip": {"on_failure": {"dropped": {"rollback": "jump"}}}}},
    )
    assert len(problems) >= 3


@pytest.mark.parametrize(
    ("run", "message"),
    [
        (
            {"planner": {"type": "sequence", "call": "periodic", "params": {"__replace__": True}}},
            "planner.call",
        ),
        (
            {"planner": {"type": "sequence", "params": {"__replace__": True, "calls": [{"skill": "pick"}]}}},
            "planner.params.calls differs from task.plan",
        ),
    ],
)
def test_the_sequence_planner_checks_its_plan_at_validate(tmp_path, run, message):
    problems = _plan_problems(tmp_path, TWO_UNITS, run=run)
    assert any(message in problem for problem in problems)


def test_params_calls_get_the_plan_checks(tmp_path):
    run = {"planner": {"type": "sequence", "params": {"__replace__": True, "calls": [{"skill": "dance"}]}}}
    problems = _problems(tmp_path, run=run)
    assert any("planner.params.calls[0]: 'dance' is not one of the task's skills" in p for p in problems)


# --- recovery tables: retries count towards max_consecutive_failures ----------------------------------


@pytest.mark.parametrize(
    ("skills", "where"),
    [
        (
            {"skills": {"pick": {"on_failure": {"empty_grasp": {"retry": 2}}}}},
            "skills.pick.on_failure.empty_grasp",
        ),
        ({"defaults": {"on_failure": {"retry": 2}}}, "defaults.on_failure"),
        ({"defaults": {"max_consecutive_failures": 2}}, "skills.place.on_failure.timeout"),  # its retry: 1
    ],
)
def test_retries_that_use_up_the_failure_cap_are_refused(tmp_path, skills, where):
    problems = _problems(tmp_path, skills=skills)
    assert any(p.startswith(f"{where}: escalate: planner is never reached") for p in problems), problems


def test_the_failure_cap_check_names_both_ways_out(tmp_path):
    problems = _problems(tmp_path, skills={"skills": {"pick": {"on_failure": {"empty_grasp": {"retry": 2}}}}})
    assert (
        "skills.pick.on_failure.empty_grasp: escalate: planner is never reached: after retry: 2 the failure "
        "would go to the planner at failure 3 in a row, but defaults.max_consecutive_failures (3) holds at "
        "failure 3; lower retry to 1 or raise max_consecutive_failures to 4"
    ) in problems


def test_retries_under_the_cap_or_ending_in_a_hold_are_accepted(tmp_path):
    two_retries = {"skills": {"pick": {"on_failure": {"empty_grasp": {"retry": 2}}}}}
    raised = {**two_retries, "defaults": {"max_consecutive_failures": 4}}
    check_bundle(load_bundle(write_configs(tmp_path / "raised", skills=raised)))
    held = {"skills": {"pick": {"on_failure": {"empty_grasp": {"retry": 2, "escalate": "hold"}}}}}
    check_bundle(load_bundle(write_configs(tmp_path / "held", skills=held)))


def test_the_second_empty_grasp_in_a_row_reaches_the_planner(tmp_path):
    # The shipped table retracts and retries once; with retry: 2 the third miss held before the planner heard.
    bundle = make_bundle(tmp_path, task=small_task(3))
    run = run_episode(bundle, tmp_path, world={"scripted_outcomes": {"pick_next": ["empty_grasp"] * 3}})
    events = read_events(run)
    assert [e["action"] for e in of_type(events, "recovery")] == ["retry", "planner"]
    assert "skill_failed" in [e["reason"] for e in of_type(events, "planner_request")]
    assert run.result.status == "aborted" and "failed 3 times in a row" in run.result.message


def test_the_insertion_verifiers_fit_the_grasp_readings_and_the_8_s_episode():
    registry = load_bundle(CONFIGS / "aloha_sim_insertion" / "run_insertion_seq.yaml").registry

    def holding_both(left, right):
        return registry.predicate_values({"left_gripper": left, "right_gripper": right}, ["holding_both"])

    # Good socket grasps read 0.6-0.84 in closed-loop runs; open reads about 0.97.
    assert holding_both(0.65, 0.3) == holding_both(0.82, 0.3) == {"holding_both": True}
    assert holding_both(0.97, 0.3) == {"holding_both": False}
    assert registry.skill("insert").spec.stall_s == 1.0
    assert registry.timeout_for("insert") == 3  # it starts at about 4.8 s of the 8 s
    assert registry.recovery_for("pick_up", "timeout").before_retry == []  # go_home would open both grippers


def _run_configs():
    for path in sorted(CONFIGS.rglob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and {"executor", "planner"} <= data.keys():
            yield path


@pytest.mark.parametrize("path", list(_run_configs()), ids=lambda path: f"{path.parent.name}/{path.name}")
def test_every_shipped_run_config_validates(path):
    check_bundle(load_bundle(path))
