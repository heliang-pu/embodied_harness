"""RoboTwin success requires the simulator's terminal verdict."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _scorer():
    path = Path(__file__).resolve().parents[1] / "scripts" / "score_robotwin_eval.py"
    spec = importlib.util.spec_from_file_location("score_robotwin_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.score_run


def _episode(root, name, outcome, events):
    directory = root / name
    directory.mkdir()
    if outcome is not None:
        (directory / "outcome.json").write_text(json.dumps(outcome))
    (directory / "events.jsonl").write_text("\n".join(json.dumps(event) for event in events))


def test_only_confirmed_terminal_success_counts_and_missing_episodes_remain_in_denominator(tmp_path):
    start = {"type": "episode_start", "seed": 1000}
    ended = {"type": "env_end", "success": True}
    _episode(tmp_path, "ep_0001", {"status": "ended", "env_success": True}, [start, ended])
    # A controller's completion claim is insufficient, including an old trace's generic success flag.
    _episode(tmp_path, "ep_0002", {"status": "done", "env_success": None}, [start])
    _episode(tmp_path, "ep_0003", {"status": "ended", "env_success": True}, [start])
    _episode(tmp_path, "ep_0004", None, [start])
    result = _scorer()(tmp_path, expected_episodes=5)
    assert result["successful_episodes"] == 1
    assert result["attempted_episodes"] == 5
    assert result["failed_episodes"] == 4
    assert result["unrecorded_episodes"] == 1
    assert result["success_rate"] == 0.2
    assert [row["success"] for row in result["episodes"]] == [True, False, False, False]


def test_premature_done_refusals_are_recorded_as_failure_evidence(tmp_path):
    _episode(
        tmp_path,
        "ep_0001",
        {"status": "aborted", "env_success": None, "error_code": "E_PLANNER_INVALID"},
        [
            {"type": "episode_start", "seed": 1001},
            {"type": "check", "claimed_by_planner": True, "passed": False},
        ],
    )
    result = _scorer()(tmp_path)
    assert result["successful_episodes"] == 0
    assert result["episodes"][0]["false_done_claims"] == 1
    assert result["episodes"][0]["seed"] == 1001


def test_planner_cameras_are_read_from_each_episodes_config_snapshot(tmp_path):
    start = {"type": "episode_start", "seed": 1000}
    _episode(tmp_path, "ep_0001", {"status": "ended", "env_success": True}, [start])
    (tmp_path / "ep_0001" / "config.snapshot.yaml").write_text(
        "run:\n  planner:\n    params:\n      cameras: [front, head, left_wrist, right_wrist]\n"
    )
    _episode(tmp_path, "ep_0002", None, [start])  # killed before anything but its first event
    result = _scorer()(tmp_path)
    cameras = [row["planner_cameras"] for row in result["episodes"]]
    assert cameras == [["front", "head", "left_wrist", "right_wrist"], None]
    assert "planner_camera_note" not in result  # a fixed claim about one early smoke run
