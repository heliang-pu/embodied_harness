"""Long tasks in a fixed order: the sequence planner plays task.plan and rolls back after failures that undo
an earlier step (skills' on_failure.<reason>.rollback)."""

import pytest

from conftest import make_bundle, of_type, read_events, run_episode
from embodied_harness.runtime.commands import ScriptedCommands
from embodied_harness.runtime.trace.progress import episode_progress

PLACE_A = {"skill": "place", "args": {"bin": "A"}}
PLACE_B = {"skill": "place", "args": {"bin": "B"}}
UNIT = ["pick", "flip"]
PLAN = [*UNIT, PLACE_A, *UNIT, PLACE_B]  # two packages, barcode down: 6 steps
KEYS = ["pick", "flip", "place(bin=A)", "pick", "flip", "place(bin=B)"]
NO_ROLLBACK = {"skills": {"flip": {"on_failure": {"dropped": {"rollback": None}}}}}
HOME = {"name": "home", "when": "previous.skill == 'flip' and next.skill == 'pick'", "run": ["retract"]}


def plan_bundle(tmp_path, plan=PLAN, skills=None, **task):
    total = sum(1 for step in plan if isinstance(step, dict) and step["skill"] == "place")
    return make_bundle(
        tmp_path,
        task={"task": {"expected_total": total, "plan": plan, **task}},
        skills=skills,
        run={"planner": {"type": "sequence", "params": {"__replace__": True}}},
    )


def world(packages=2, **outcomes):
    return {"num_packages": packages, "barcode_down_prob": 1.0, "scripted_outcomes": outcomes}


def main_calls(events):
    return [
        e["call"]["skill"] + (f"(bin={e['call']['args']['bin']})" if e["call"]["args"] else "")
        for e in of_type(events, "skill_start")
        if e["purpose"] == "main"
    ]


def responses_after(events, event):
    return [e for e in of_type(events, "planner_response") if e["seq"] > event["seq"]]


def test_the_sequence_planner_plays_task_plan(tmp_path):
    run = run_episode(plan_bundle(tmp_path), tmp_path, world=world())
    events = read_events(run)
    assert run.result.status == "done"
    assert main_calls(events) == KEYS
    assert [e["meta"]["sequence_index"] for e in of_type(events, "planner_response")] == list(range(6))
    # The last commit satisfies done_when, so the planner is not asked again.
    assert of_type(events, "planner_request")[-1]["seq"] < of_type(events, "ledger_update")[-1]["seq"]
    assert episode_progress(run.episode_dir).completed == 6


def test_a_flip_drop_rolls_back_to_the_pick(tmp_path):
    run = run_episode(plan_bundle(tmp_path), tmp_path, world=world(flip=["dropped"]))
    events = read_events(run)
    assert run.result.status == "done"
    assert main_calls(events) == [
        "pick",
        "flip",
        "pick",
        "flip",
        "place(bin=A)",
        "pick",
        "flip",
        "place(bin=B)",
    ]
    failed = next(e for e in of_type(events, "skill_end") if e["outcome"]["reason"] == "dropped")
    response = responses_after(events, failed)[0]
    assert response["decision"]["skill"] == "pick"
    assert response["meta"]["sequence_index"] == 0
    assert response["meta"]["rollback"] == {"from": 1, "reason": "dropped"}
    assert not of_type(events, "decision_rejected")
    assert sum(run.result.ledger["sorted"].values()) == 2
    progress = episode_progress(run.episode_dir)
    assert (progress.completed, progress.furthest, progress.first_failure, progress.rollbacks) == (6, 6, 1, 1)


def test_without_a_rollback_rule_a_drop_still_ends_in_hold(tmp_path):
    run = run_episode(plan_bundle(tmp_path, skills=NO_ROLLBACK), tmp_path, world=world(flip=["dropped"]))
    events = read_events(run)
    assert run.result.status == "aborted" and run.result.error_code == "E_PLANNER_INVALID"
    rejected = of_type(events, "decision_rejected")
    assert len(rejected) == 3 and all(
        "flip: precondition not met: holding" in e["errors"][0] for e in rejected
    )
    assert episode_progress(run.episode_dir).completed == 1


def test_a_drop_in_a_later_unit_rolls_back_to_its_own_pick(tmp_path):
    run = run_episode(plan_bundle(tmp_path), tmp_path, world=world(flip=["success", "dropped"]))
    events = read_events(run)
    assert run.result.status == "done"
    assert main_calls(events) == [
        "pick",
        "flip",
        "place(bin=A)",
        "pick",
        "flip",
        "pick",
        "flip",
        "place(bin=B)",
    ]
    rolled = [e for e in of_type(events, "planner_response") if "rollback" in e["meta"]]
    assert [(e["meta"]["sequence_index"], e["meta"]["rollback"]["from"]) for e in rolled] == [(3, 4)]


def test_a_transition_rule_runs_before_the_redone_pick(tmp_path):
    bundle = plan_bundle(tmp_path, skills={"transitions": [HOME]})
    run = run_episode(bundle, tmp_path, world=world(flip=["dropped"]))
    events = read_events(run)
    assert run.result.status == "done"
    starts = [f"{e['call']['skill']}/{e['purpose']}" for e in of_type(events, "skill_start")]
    assert starts[:4] == ["pick/main", "flip/main", "retract/transition", "pick/main"]


def test_drops_that_keep_coming_hold_at_the_streak_cap(tmp_path):
    run = run_episode(plan_bundle(tmp_path), tmp_path, world=world(flip=["dropped"] * 3))
    events = read_events(run)
    assert run.result.status == "aborted"
    assert run.result.message == "flip failed 3 times in a row (last: dropped)"
    assert main_calls(events) == ["pick", "flip"] * 3
    assert len([e for e in of_type(events, "planner_response") if "rollback" in e["meta"]]) == 2
    progress = episode_progress(run.episode_dir)
    assert (progress.completed, progress.furthest, progress.rollbacks, progress.stopped_at) == (1, 1, 2, 1)


def test_human_resume_after_the_streak_hold_rolls_back(tmp_path):
    run = run_episode(
        plan_bundle(tmp_path),
        tmp_path,
        world=world(flip=["dropped"] * 3),
        commands=ScriptedCommands(["resume"]),
    )
    events = read_events(run)
    assert run.result.status == "done"
    command = of_type(events, "human_command")[0]
    response = responses_after(events, command)[0]
    assert response["decision"]["skill"] == "pick"
    assert response["meta"]["rollback"] == {"from": 1, "reason": "dropped"}


def test_a_rollback_is_blocked_when_its_target_cannot_start(tmp_path):
    # A false drop: flip reports `dropped` while the gripper still holds the package, so pick cannot start.
    skills = {
        "skills": {"flip": {"verify": {"failure": [{"reason": "dropped", "when": "holding", "hold_s": 1}]}}}
    }
    bundle = plan_bundle(tmp_path, plan=[*UNIT, PLACE_A], skills=skills)
    run = run_episode(bundle, tmp_path, world=world(packages=1))
    events = read_events(run)
    blocked = [e for e in of_type(events, "planner_response") if "rollback_blocked" in e["meta"]]
    assert [e["decision"]["skill"] for e in blocked] == ["flip", "flip"]
    assert blocked[0]["meta"]["rollback_blocked"] == {"to": 0, "reason": "dropped", "unmet": ["not holding"]}
    assert not of_type(events, "decision_rejected")  # the repeated flip could start
    assert run.result.message == "flip failed 3 times in a row (last: dropped)"


@pytest.mark.parametrize(
    ("max_ticks", "expected"), [(60, ("pick", {"from": 1, "reason": "dropped"})), (70, ("pick", None))]
)
def test_resume_after_a_drop_continues_the_rollback(tmp_path, max_ticks, expected):
    bundle = plan_bundle(tmp_path)
    reference = run_episode(bundle, tmp_path, world=world(flip=["dropped"]), episode_id="reference")
    first = run_episode(bundle, tmp_path, world=world(flip=["dropped"]), max_ticks=max_ticks)
    before = read_events(first)
    assert first.result is None
    if max_ticks == 60:  # stopped after the drop, before the planner was asked
        assert of_type(before, "skill_end")[-1]["outcome"]["reason"] == "dropped"
        assert of_type(before, "planner_request")[-1]["reason"] == "skill_done"
    else:  # stopped while the redone pick runs
        assert "rollback" in of_type(before, "planner_response")[-1]["meta"]
        assert of_type(before, "skill_start")[-1]["call"]["skill"] == "pick"
    resumed = run_episode(bundle, tmp_path, world=world(flip=["dropped"]), resume=True)
    assert resumed.result.status == "done"
    assert resumed.result.ledger == reference.result.ledger
    response = of_type(read_events(resumed)[len(before) :], "planner_response")[0]
    assert (response["decision"]["skill"], response["meta"].get("rollback")) == expected


def test_resume_after_a_human_retry_that_followed_a_rollback(tmp_path):
    # The rollback's transition primitive times out (HOLD, about the pick it prepares); the human retries,
    # which runs the transition again, and the process stops right after it started. Resuming proposes the
    # pending rollback target again instead of refusing the trace.
    skills = {"transitions": [HOME], "skills": {"retract": {"timeout_s": 0.1}}}
    bundle = plan_bundle(tmp_path, plan=[*UNIT, PLACE_A], skills=skills)
    reference = run_episode(
        bundle,
        tmp_path,
        world=world(1, flip=["dropped"]),
        commands=ScriptedCommands(["retry"]),
        episode_id="reference",
    )
    retried = of_type(read_events(reference), "human_command")[0]
    assert retried["failed_call"]["skill"] == "pick"
    ticks = round(retried["t"] * bundle.run.runtime.control_hz) + 1  # up to the tick of the retry
    commands = ScriptedCommands(["retry"])
    first = run_episode(
        bundle, tmp_path, world=world(1, flip=["dropped"]), commands=commands, max_ticks=ticks
    )
    before = read_events(first)
    assert [e["type"] for e in before[-4:]] == ["transition", "state_change", "boundary", "skill_start"]
    assert before[-1]["call"]["skill"] == "retract"
    resumed = run_episode(bundle, tmp_path, world=world(1), resume=True)
    response = of_type(read_events(resumed)[len(before) :], "planner_response")[0]
    assert response["meta"]["sequence_index"] == 0  # the pending rollback target, proposed again


def test_a_skip_after_a_rollback_continues_after_the_skipped_step(tmp_path):
    # A time-budget HOLD while the redone pick is being verified (the package is back in the gripper); the
    # human says that pick is done: the plan goes on with the flip.
    bundle = plan_bundle(tmp_path, plan=[*UNIT, PLACE_A], budget={"max_time_s": 9.2})
    run = run_episode(
        bundle, tmp_path, world=world(packages=1, flip=["dropped"]), commands=ScriptedCommands(["skip"])
    )
    events = read_events(run)
    command = of_type(events, "human_command")[0]
    assert command["failed_call"]["skill"] == "pick"
    response = responses_after(events, command)[0]
    assert (response["decision"]["skill"], response["meta"]["sequence_index"]) == ("flip", 1)
    assert run.result.status == "done"


def test_a_refused_retry_after_a_rollback_leaves_the_skip(tmp_path):
    # The same HOLD: retrying the pick is refused (the package is already in the gripper), and after the
    # human's skip the plan goes on with the flip and the place.
    bundle = plan_bundle(tmp_path, plan=[*UNIT, PLACE_A], budget={"max_time_s": 9.2})
    commands = ScriptedCommands(["retry", "skip"])
    run = run_episode(bundle, tmp_path, world=world(packages=1, flip=["dropped"]), commands=commands)
    events = read_events(run)
    assert [e["command"] for e in of_type(events, "human_command")] == ["skip"]
    assert run.result.status == "done"
    assert main_calls(events) == ["pick", "flip", "pick", "flip", "place(bin=A)"]
    assert episode_progress(run.episode_dir).completed == 3


def test_a_plan_that_ends_with_done_when_false_holds_instead_of_looping(tmp_path):
    # An effect that reads robot state skips the load-time done_when check; the plan ends one package short.
    effect = "ledger.sorted[args.bin] += 1 if gripper_width > 0 else 0"
    skills = {"skills": {"place": {"effects": {"on_success": [effect]}}}}
    bundle = plan_bundle(tmp_path, plan=[*UNIT, PLACE_A], skills=skills, expected_total=2)
    run = run_episode(bundle, tmp_path, world=world(packages=2))
    events = read_events(run)
    assert run.result.status == "aborted" and run.result.error_code == "E_PLANNER_TIMEOUT"
    assert "the plan has ended but done_when" in run.result.message
    assert len(of_type(events, "planner_error")) == 4  # the first call and 3 retries
    assert [e["reason"] for e in of_type(events, "planner_request")][-4:] == ["check_failed"] * 4


def test_a_human_resume_after_a_verified_step_goes_on_with_the_next_one(tmp_path):
    # The time budget runs out on the tick after place(bin=A) is verified, before the planner is asked for the
    # next step. Resuming goes on with the second pick; proposing place(bin=A) again was refused by its
    # precondition (the gripper is empty) until the run held for good.
    reference = run_episode(plan_bundle(tmp_path / "reference"), tmp_path / "reference", world=world())
    placed = of_type(read_events(reference), "ledger_update")[0]
    bundle = plan_bundle(tmp_path, budget={"max_time_s": placed["t"] + 0.05})
    run = run_episode(bundle, tmp_path, world=world(), commands=ScriptedCommands(["resume"]))
    events = read_events(run)
    hold = of_type(events, "hold")[0]
    assert hold["code"] == "E_BUDGET" and hold["seq"] > of_type(events, "ledger_update")[0]["seq"]
    response = responses_after(events, of_type(events, "human_command")[0])[0]
    assert (response["decision"]["skill"], response["meta"]["sequence_index"]) == ("pick", 3)
    assert not of_type(events, "decision_rejected")
    assert run.result.status == "done" and run.result.ledger == reference.result.ledger
    assert main_calls(events) == KEYS
