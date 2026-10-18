"""`embodied-harness lessons`: drafts from failure patterns that keep coming back, merged into a file."""

import json

import yaml

from conftest import write_configs
from embodied_harness.cli import main
from embodied_harness.config.loader import load_bundle


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


def write_runs(tmp_path):
    runs = tmp_path / "run"
    for index in range(5):
        directory = runs / f"ep_000{index + 1}"
        directory.mkdir(parents=True)
        pick_ok = index == 4  # four of five picks after the place close on nothing within a second
        events = [
            start(0, "pick", "a"),
            end(3, "a", "success"),
            start(3, "place", "b", bin="A"),
            end(6, "b", "success"),
            start(6, "pick", "c"),
            end(
                9 if pick_ok else 6.5,
                "c",
                "success" if pick_ok else "failure",
                None if pick_ok else "empty_grasp",
            ),
            {"type": "check", "t": 9.5, "claimed_by_planner": True, "passed": False},
        ]
        (directory / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
        (directory / "outcome.json").write_text(json.dumps({"status": "aborted" if not pick_ok else "done"}))
    return runs


def test_patterns_become_unreviewed_drafts(tmp_path, capsys):
    runs = write_runs(tmp_path)
    out = tmp_path / "configs" / "lessons.yaml"
    assert main(["lessons", str(runs), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "在 place(bin=A) 之后直接执行 pick 常常失败：4/5 次（empty_grasp），多半是起点不对" in printed
    assert "你声称完成了 5 次" in printed
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert [entry["reviewed"] for entry in data["lessons"]] == [False, False]
    assert data["lessons"][0]["source"] == "report: place(bin=A) -> pick, empty_grasp, 4/5 failed, readiness"

    # A reviewed, rewritten lesson survives the next round, and the pattern is not drafted twice.
    data["lessons"][0].update(text="放完之后先 retract 再 pick", reviewed=True)
    out.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    assert main(["lessons", str(runs), "--out", str(out)]) == 0
    assert "0 draft(s) added, 2 already there" in capsys.readouterr().out
    lessons = yaml.safe_load(out.read_text(encoding="utf-8"))["lessons"]
    assert lessons[0]["text"] == "放完之后先 retract 再 pick" and len(lessons) == 2

    # The run config picks the reviewed one up.
    config = write_configs(tmp_path, run={"planner": {"lessons": "../lessons.yaml"}})
    bundle = load_bundle(config)
    assert bundle.task.lessons == ["放完之后先 retract 再 pick"] and bundle.unreviewed_lessons == 1


def test_rare_failures_are_not_drafted(tmp_path, capsys):
    runs = write_runs(tmp_path)
    out = tmp_path / "lessons.yaml"
    assert main(["lessons", str(runs), "--out", str(out), "--min-count", "6"]) == 0
    assert "no failure pattern came back 6 times" in capsys.readouterr().out
    assert yaml.safe_load(out.read_text(encoding="utf-8"))["lessons"] == []


def test_patterns_name_the_planners_own_previous_call(tmp_path):
    from embodied_harness.runtime.trace.lessons import draft_lessons
    from embodied_harness.runtime.trace.report import read_episode

    runs = tmp_path / "run"
    for index in range(3):  # place -> [retract, approach: a transition rule] -> pick fails
        directory = runs / f"ep_000{index + 1}"
        directory.mkdir(parents=True)
        events = [
            start(0, "place", "a", bin="A"),
            end(3, "a", "success"),
            {**start(3, "retract", "r"), "purpose": "transition"},
            end(3.5, "r", "success"),
            {**start(3.5, "approach", "p", bin="A"), "purpose": "transition"},
            end(4, "p", "success"),
            start(4, "pick", "b"),
            end(4.5, "b", "failure", "stall"),
        ]
        (directory / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
        (directory / "outcome.json").write_text(json.dumps({"status": "aborted"}))
    (draft,) = draft_lessons([read_episode(path) for path in sorted(runs.iterdir())])
    assert draft["source"] == "report: place(bin=A) -> pick, stall, 3/3 failed, readiness"
