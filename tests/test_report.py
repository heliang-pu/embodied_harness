"""`embodied-harness report`: per-skill and per-transition tables, and one cause per failed episode."""

import json

import pytest
import yaml

from conftest import make_bundle, run_episode, small_task
from embodied_harness.cli import main
from embodied_harness.config.errors import ConfigError
from embodied_harness.runtime.trace.report import (
    build_report,
    episode_succeeded,
    format_report,
    read_episode,
    report_json,
)


def start(t, skill, call_id, **args):
    return {
        "type": "skill_start",
        "t": t,
        "call_id": call_id,
        "call": {"skill": skill, "args": args},
        "purpose": "main",
    }


def end(t, call_id, status, reason=None):
    return {"type": "skill_end", "t": t, "call_id": call_id, "outcome": {"status": status, "reason": reason}}


def write_episode(root, name, events, **outcome):
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    (directory / "outcome.json").write_text(json.dumps(outcome))
    return directory


def episodes(tmp_path):
    runs = tmp_path / "run"
    write_episode(
        runs,
        "ep_0001",
        [start(0, "pick", "a"), end(3, "a", "success"), start(3.1, "place", "b"), end(6, "b", "success")],
        status="done",
    )
    write_episode(  # the env fails the episode 1 s after switching from pick to place
        runs,
        "ep_0002",
        [
            start(0, "pick", "a"),
            end(3, "a", "success"),
            start(3.0, "place", "b"),
            {"type": "env_end", "t": 4.0, "success": False},
        ],
        status="ended",
        env_success=False,
    )
    write_episode(  # pick fails, is retried, fails again: the skill, not the transition
        runs,
        "ep_0003",
        [
            start(0, "pick", "a"),
            end(5, "a", "failure", "empty_grasp"),
            start(5.1, "pick", "b"),
            end(10, "b", "failure", "empty_grasp"),
        ],
        status="aborted",
        message="pick failed 2 times in a row (last: empty_grasp)",
    )
    write_episode(
        runs,
        "ep_0004",
        [{"type": "decision_rejected", "t": 0.1}] * 3,
        status="aborted",
        error_code="E_PLANNER_INVALID",
        message="planner output still invalid",
    )
    return runs


def test_each_failed_episode_gets_one_cause(tmp_path):
    runs = episodes(tmp_path)
    reports = [read_episode(path) for path in sorted(runs.iterdir())]
    build_report(reports)
    assert [(r.cause, r.detail) for r in reports] == [
        ("success", ""),
        ("transition", "place after pick: ended_failure 1.0s in"),
        ("skill", "pick after pick: empty_grasp 4.9s in"),
        ("planner", "E_PLANNER_INVALID: planner output still invalid"),
    ]
    assert reports[3].rejected == 3


def test_transitions_are_counted_for_the_next_call(tmp_path):
    report = build_report([read_episode(path) for path in sorted(episodes(tmp_path).iterdir())])
    assert report["causes"] == {
        "success": 1,
        "planner": 1,
        "transition": 1,
        "skill": 1,
        "harness": 0,
        "other": 0,
    }
    after_pick = report["transitions"][("pick", "place")]
    assert (after_pick.starts, after_pick.success, after_pick.failure, after_pick.rate) == (2, 1, 1, 0.5)
    first = report["transitions"][("START", "pick")]
    assert (first.starts, first.success, first.failure) == (3, 2, 1)
    assert report["skills"]["pick"].reasons == {"empty_grasp": 2}


def test_the_command_reads_real_runs(tmp_path, capsys):
    rule = {"when": "previous.skill == 'place' and next.skill == 'pick'", "run": ["retract"]}
    bundle = make_bundle(tmp_path, task=small_task(3), skills={"transitions": [rule]})
    run = run_episode(bundle, tmp_path)
    out = tmp_path / "report.json"
    assert main(["report", str(run.episode_dir.parent), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "1 episode(s): success 1" in printed
    assert "| retract -> pick |" in printed and "| place -> retract |" in printed
    data = json.loads(out.read_text())
    assert data["causes"]["success"] == 1
    assert {row["next"] for row in data["transitions"]} >= {"pick", "place", "retract"}


def test_a_failure_sooner_than_the_skill_normally_takes_is_a_transition_failure(tmp_path):
    runs = tmp_path / "run"
    for index in (1, 2):  # place normally takes 6 s to succeed
        write_episode(
            runs,
            f"ep_000{index}",
            [start(0, "pick", "a"), end(3, "a", "success"), start(3, "place", "b"), end(9, "b", "success")],
            status="done",
        )
    write_episode(  # fails 5 s after the switch: longer than the 3 s floor, sooner than a normal place
        runs,
        "ep_0003",
        [
            start(0, "pick", "a"),
            end(3, "a", "success"),
            start(3, "place", "b"),
            {"type": "env_end", "t": 8.0, "success": False},
        ],
        status="ended",
        env_success=False,
    )
    report = build_report([read_episode(path) for path in sorted(runs.iterdir())])
    assert report["typical_s"]["place"] == 6.0
    assert report["failed"][0].cause == "transition"


def test_failed_calls_get_a_kind(tmp_path):
    runs = tmp_path / "run"
    for index in (1, 2):  # pick normally takes 6 s
        write_episode(runs, f"ep_000{index}", [start(0, "pick", "a"), end(6, "a", "success")], status="done")
    tagged = end(4, "t", "failure", "wrong_bin")
    tagged["outcome"]["evidence"] = {"kind": "grounding"}
    cases = {
        "ep_0003": [start(0, "pick", "a"), end(1, "a", "failure", "empty_grasp")],  # within 1/3 of 6 s
        "ep_0004": [start(0, "pick", "a"), end(5, "a", "failure", "empty_grasp")],  # a condition, later
        "ep_0005": [start(0, "pick", "a"), end(4, "a", "failure", "stall")],  # stopped moving
        "ep_0006": [start(0, "pick", "a"), end(12, "a", "failure", "timeout")],  # slow or lost: unknown
        "ep_0007": [start(0, "pick", "t"), tagged],  # the verifier condition names its kind
    }
    for name, events in cases.items():
        write_episode(runs, name, events, status="aborted", message="failed")
    report = build_report([read_episode(path) for path in sorted(runs.iterdir())])
    kinds = {episode.directory.rsplit("/", 1)[-1]: episode.kind for episode in report["failed"]}
    assert kinds == {
        "ep_0003": "readiness",
        "ep_0004": "control",
        "ep_0005": "readiness",
        "ep_0006": "unknown",
        "ep_0007": "grounding",
    }
    assert report["skills"]["pick"].kinds == {"readiness": 2, "control": 1, "unknown": 1, "grounding": 1}
    assert report["kinds"]["skill"] == {"readiness": 2, "control": 1, "unknown": 1, "grounding": 1}
    text = format_report(report)
    assert "skill 5 (readiness 2, grounding 1, control 1, unknown 1)" in text
    assert "| START -> pick | 7 | 2 | 5 | 29% | empty_grasp x2, stall x1, timeout x1 | readiness 2" in text


def test_a_failure_condition_can_name_its_kind(tmp_path):
    skills = {
        "skills": {
            "flip": {
                "verify": {
                    "failure": [
                        {"reason": "dropped", "when": "not holding", "hold_s": 0.3, "kind": "control"}
                    ]
                }
            }
        }
    }
    bundle = make_bundle(tmp_path, task=small_task(3), skills=skills)
    world = {"barcode_down_prob": 1.0, "scripted_outcomes": {"flip": ["dropped"]}}
    run = run_episode(bundle, tmp_path, world=world)
    events = [json.loads(line) for line in (run.episode_dir / "events.jsonl").read_text().splitlines()]
    dropped = [e for e in events if e["type"] == "skill_end" and e["outcome"]["reason"] == "dropped"]
    assert dropped and dropped[0]["outcome"]["evidence"]["kind"] == "control"


def test_kind_is_only_for_failure_conditions(tmp_path):
    skills = {"skills": {"pick": {"verify": {"success": [{"when": "holding", "kind": "control"}]}}}}
    with pytest.raises(ConfigError, match="`kind` describes a failure"):
        make_bundle(tmp_path, task=small_task(3), skills=skills)


def test_a_retry_through_a_recovery_primitive_is_still_a_retry(tmp_path):
    retract = {**start(3.5, "retract", "r"), "purpose": "before_retry"}
    events = [
        start(0, "pick", "a"),
        end(3.4, "a", "failure", "empty_grasp"),
        retract,
        end(4.0, "r", "success"),
        start(4.0, "pick", "b"),
        end(7.4, "b", "failure", "empty_grasp"),
    ]
    write_episode(
        tmp_path / "run", "ep_0001", events, status="aborted", message="pick failed 2 times in a row"
    )
    report = build_report([read_episode(path) for path in sorted((tmp_path / "run").iterdir())])
    assert report["failed"][0].cause == "skill"  # pick -> retract -> pick is a retry, not a transition
    assert ("pick", "pick") in report["transitions"] and ("pick", "retract") in report["transitions"]


SNAPSHOT = {
    "task": {"plan": ["pick", {"skill": "place", "args": {"bin": "A"}}]},
    "run": {"planner": {"type": "mock_oracle", "params": {}}},
}


def test_the_report_measures_episodes_against_the_plan(tmp_path):
    runs = tmp_path / "run"
    cases = {
        "ep_0001": (
            [
                start(0, "pick", "a"),
                end(3, "a", "success"),
                start(3, "place", "b", bin="A"),
                end(6, "b", "success"),
            ],
            {"status": "done"},
        ),
        "ep_0002": (
            [
                start(0, "pick", "a"),
                end(3, "a", "success"),
                start(3, "place", "b", bin="A"),
                end(4, "b", "failure", "dropped"),
            ],
            {"status": "aborted", "message": "x"},
        ),
    }
    for name, (events, outcome) in cases.items():
        directory = write_episode(runs, name, events, **outcome)
        (directory / "config.snapshot.yaml").write_text(yaml.safe_dump(SNAPSHOT), encoding="utf-8")
    episodes_ = [read_episode(path) for path in sorted(runs.iterdir())]
    report = build_report(episodes_)
    (plan,) = report["plan"]
    assert (plan["episodes"], plan["done"], plan["avg_completed"], plan["success_at"]) == (
        2,
        1,
        1.5,
        [1.0, 0.5],
    )
    text = format_report(report)
    assert "plan: 2 steps, 2 episode(s), 1 done (positions matched by call)" in text
    assert "| 2 | place(bin=A) | 2 | 1 | 1/2 | 2 | 0 | 1 | dropped x1 |" in text
    data = report_json(report, episodes_)
    assert data["plan"][0]["per_step"][1]["failures"] == {"dropped": 1}
    assert data["episodes"][1]["progress"]["completed"] == 1


def test_the_plan_summary_scores_a_done_the_env_rejected_as_unfinished(tmp_path):
    planner_done = {"type": "check", "t": 3.1, "claimed_by_planner": True, "passed": True}
    events = [start(0, "pick", "a"), end(3, "a", "success"), planner_done]
    directory = write_episode(tmp_path / "run", "ep_0001", events, status="done", env_success=False)
    (directory / "config.snapshot.yaml").write_text(yaml.safe_dump(SNAPSHOT), encoding="utf-8")
    report = build_report([read_episode(directory)])
    (plan,) = report["plan"]
    # The same episode the cause column calls a planner failure: one step done, stopped at the place.
    assert report["causes"]["planner"] == 1
    assert (plan["done"], plan["avg_completed"], plan["success_at"]) == (0, 1.0, [1.0, 0.0])
    assert [step["stopped"] for step in plan["per_step"]] == [0, 1]


def test_runs_without_a_plan_keep_their_report(tmp_path):
    reports = [read_episode(path) for path in sorted(episodes(tmp_path).iterdir())]
    report = build_report(reports)
    assert report["plan"] == [] and "plan:" not in format_report(report)
    data = report_json(report, reports)
    assert "plan" not in data and all("progress" not in episode for episode in data["episodes"])


def test_the_envs_own_verdict_decides_a_done_episode(tmp_path):
    runs = tmp_path / "run"
    planner_done = {"type": "check", "t": 3.1, "claimed_by_planner": True, "passed": True}
    ledger_done = {"type": "check", "t": 3.1, "claimed_by_planner": False, "passed": True}
    failed_pick = [start(0, "pick", "a"), end(3, "a", "failure", "timeout")]
    picked = [start(0, "pick", "a"), end(3, "a", "success")]
    write_episode(runs, "ep_0001", [*failed_pick, planner_done], status="done", env_success=False)
    write_episode(runs, "ep_0002", [*picked, ledger_done], status="done", env_success=False)
    write_episode(runs, "ep_0003", [*picked, ledger_done], status="done")  # a real robot: no verdict
    write_episode(runs, "ep_0004", [*picked, planner_done], status="done", env_success=True)
    reports = [read_episode(path) for path in sorted(runs.iterdir())]
    report = build_report(reports)
    assert [(r.cause, r.detail) for r in reports] == [
        ("planner", "the planner declared done; env_success=False"),
        ("other", "done_when passed; env_success=False"),
        ("success", ""),
        ("success", ""),
    ]
    assert report["causes"]["success"] == 2
    assert episode_succeeded("done", None) and episode_succeeded("ended", True)
    assert not episode_succeeded("done", False) and not episode_succeeded("ended", None)


def test_a_call_a_hold_stopped_counts_as_aborted_only_when_the_episode_ended_there(tmp_path):
    runs = tmp_path / "run"
    held = end(2, "a", "preempted", "hold")
    write_episode(
        runs,
        "ep_0001",
        [start(0, "pick", "a"), held, {"type": "hold", "t": 2}],
        status="aborted",
        error_code="E_EXEC_UNDERRUN",
        message="no actions from the executor for 1s",
    )
    write_episode(  # a human retried the pick after the hold, and the episode went on
        runs,
        "ep_0002",
        [start(0, "pick", "a"), held, start(3, "pick", "b"), end(6, "b", "success")],
        status="done",
    )
    first, second = [read_episode(path) for path in sorted(runs.iterdir())]
    build_report([first, second])
    assert (first.calls[0].status, first.calls[0].elapsed) == ("aborted", 2.0)
    assert (first.cause, first.detail) == ("skill", "pick after START: hold 2.0s in")
    assert [row.status for row in second.calls] == ["preempted", "success"]
