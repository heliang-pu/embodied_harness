import pytest

from conftest import RUN_CONFIG, write_configs
from fmc3_harness.config.errors import ConfigError
from fmc3_harness.config.loader import load_bundle
from fmc3_harness.core.types import SkillCall
from fmc3_harness.logs.codes import ErrorCode
from fmc3_harness.runtime.factory import check_bundle


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


def test_skills_written_for_another_library(tmp_path):
    problems = _problems(tmp_path, skills={"library": "pi05_sort_v3"})
    assert any("pi05_sort_v3" in p for p in problems)


def test_planner_trained_on_another_library(tmp_path):
    with pytest.raises(ConfigError) as info:
        load_bundle(write_configs(tmp_path, run={"planner": {"trained_library": "pi05_sort_v2"}}))
    assert info.value.code is ErrorCode.E_LIBRARY_MISMATCH


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
