"""Runs as training data: per-tick capture (`trace.frames`) and export to a LeRobot dataset."""

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import make_bundle, run_episode, small_task
from embodied_harness.data.from_runs import RecordedEpisode, check_episode, language_rows
from embodied_harness.env.mock_sorting import MockSortingWorld
from embodied_harness.runtime.trace.frames import FRAMES_FILE, numeric_action
from embodied_harness.runtime.trace.report import episode_succeeded

CAMERAS = ["head", "wrist"]


class NumericWorld(MockSortingWorld):
    """The mock world plus what a robot adapter reports: a state vector and the actions it sends."""

    def observe(self):
        obs = super().observe()
        width = float(obs.proprio["gripper_width"])
        obs.proprio = {**obs.proprio, "state": [width, 1.0 if obs.proprio["gripper_closed"] else 0.0]}
        return obs

    def act(self, action):
        super().act(action)
        key = (action or {}).get("prompt") or (action or {}).get("primitive") or ""
        return np.array([float(len(key)), 1.0], dtype=np.float32)

    def hold(self):
        super().hold()
        return np.zeros(2, dtype=np.float32)


def recorded_run(tmp_path, packages=2):
    bundle = make_bundle(tmp_path, task=small_task(packages), run={"trace": {"frames": True}})
    params = {"num_packages": packages, "barcode_down_prob": 0.0, "cameras": CAMERAS, "image_size": [64, 48]}
    return bundle, run_episode(bundle, tmp_path, env=NumericWorld(params))


def read_frames(episode_dir):
    return [json.loads(line) for line in (episode_dir / FRAMES_FILE).read_text().splitlines()]


def test_every_tick_is_recorded_with_the_action_sent(tmp_path):
    _, run = recorded_run(tmp_path)
    frames = read_frames(run.episode_dir)
    assert [frame["tick"] for frame in frames] == list(range(1, len(frames) + 1))
    assert {frame["source"] for frame in frames} >= {"policy", "hold"}
    assert all(isinstance(frame["action"], list) and len(frame["action"]) == 2 for frame in frames)
    assert all(len(frame["proprio"]["state"]) == 2 for frame in frames)
    policy = next(frame for frame in frames if frame["source"] == "policy")
    assert policy["prompt"] == "pick up the next package" and policy["call"] == "pick"
    hold = next(frame for frame in frames if frame["source"] == "hold")
    assert hold["action"] == [0.0, 0.0]
    events = [json.loads(line) for line in (run.episode_dir / "events.jsonl").read_text().splitlines()]
    start = next(e for e in events if e["type"] == "skill_start")
    assert any(frame["t"] == start["t"] for frame in frames)  # the same time base as the events


def test_numeric_actions_only():
    assert numeric_action(np.array([1, 2], dtype=np.float32)) == [1.0, 2.0]
    assert numeric_action({"primitive": "retract"}) is None
    assert numeric_action("pick_next") is None


def test_finishing_old_chunks_keeps_their_original_call_labels(tmp_path):
    _, run = recorded_run(tmp_path)
    frames = [row for row in read_frames(run.episode_dir) if row["source"] != "hold"]
    actions = [json.loads(line) for line in (run.episode_dir / "actions.jsonl").read_text().splitlines()]
    events = [json.loads(line) for line in (run.episode_dir / "events.jsonl").read_text().splitlines()]
    calls = {event["version"]: event for event in events if event["type"] == "skill_start"}
    assert len(frames) == len(actions)
    for frame, action in zip(frames, actions, strict=True):
        origin = calls[action["version"]]
        assert frame["call_id"] == origin["call_id"]
        assert frame["prompt"] == origin["prompt"]["text"]


def episode(frames, events, skills=None):
    snapshot = {
        "run": {"runtime": {"control_hz": 10}},
        "skills": {"skills": skills or {}},
        "task": {"goal": "sort"},
    }
    return RecordedEpisode(Path("ep"), frames, events, {}, snapshot)


def test_language_rows_land_on_the_frames():
    frames = [{"t": round(0.1 * i, 1)} for i in range(6)]
    events = [
        {
            "type": "skill_start",
            "t": 0.0,
            "call": {"skill": "pick"},
            "prompt": {"text": "pick up the next package"},
        },
        {"type": "memory_update", "t": 0.0, "memory": "first package"},
        {"type": "skill_start", "t": 0.3, "call": {"skill": "retract"}, "prompt": None},
        {"type": "memory_update", "t": 0.3, "memory": "stale"},
        {"type": "memory_update", "t": 0.3, "memory": "retracting"},
        {"type": "memory_update", "t": 0.4, "memory": ""},
    ]
    rows = language_rows(episode(frames, events, {"retract": {"executor": "scripted"}}), fps=10.0)
    assert [(r["style"], r["timestamp"], r["content"]) for r in rows] == [
        ("memory", 0.0, "first package"),
        ("subtask", 0.0, "pick up the next package"),
        ("memory", pytest.approx(0.3), "retracting"),  # the last memory written at a frame wins
        ("subtask", pytest.approx(0.3), "retract"),  # a scripted primitive: its description, else its name
        ("memory", pytest.approx(0.4), ""),  # clearing memory is part of the runtime history
    ]


@pytest.mark.parametrize(
    ("outcome", "kept"),
    [
        ({"status": "done", "env_success": True}, True),
        ({"status": "done", "env_success": False}, False),  # done, but the simulator judged a failure
        ({"status": "done", "env_success": None}, True),  # a real robot gives no verdict
        ({"status": "done"}, True),  # a trace from before done episodes recorded the verdict
        ({"status": "ended", "env_success": True}, True),
        ({"status": "ended", "env_success": None}, False),
        ({"status": "aborted", "env_success": None}, False),
        ({}, False),
    ],
)
def test_success_only_keeps_what_run_and_report_count_as_a_success(outcome, kept):
    assert RecordedEpisode(Path("ep"), [], [], outcome, {}).succeeded is kept
    assert episode_succeeded(outcome.get("status"), outcome.get("env_success")) is kept


class JudgedWorld(NumericWorld):
    """A simulator that reports its verdict on every tick (as aloha_sim, libero and robomme do): failed."""

    def observe(self):
        obs = super().observe()
        obs.info = {**obs.info, "success": False}
        return obs


def test_success_only_leaves_out_a_done_the_env_judged_a_failure(tmp_path):
    pytest.importorskip("av")
    pytest.importorskip("lerobot.datasets.lerobot_dataset")
    from embodied_harness.cli import main

    bundle = make_bundle(tmp_path, task=small_task(2), run={"trace": {"frames": True}})
    params = {"num_packages": 2, "barcode_down_prob": 0.0, "cameras": CAMERAS, "image_size": [64, 48]}
    judged = run_episode(bundle, tmp_path, env=JudgedWorld(params), episode_id="ep_0001")
    assert (judged.result.status, judged.result.env_success) == ("done", False)  # `run` counts a failure
    kept = run_episode(bundle, tmp_path, env=NumericWorld(params), episode_id="ep_0002")
    runs, out = str(judged.episode_dir.parent), tmp_path / "dataset"
    assert main(["data", "from-runs", runs, "--out", str(out), "--success-only"]) == 0
    sidecar = json.loads((out / ".embodied_harness" / "from_runs.json").read_text(encoding="utf-8"))
    assert [episode["source"] for episode in sidecar["episodes"]] == [str(kept.episode_dir)]
    assert sidecar["skipped_unsuccessful"] == [str(judged.episode_dir)]


def test_episodes_without_numeric_actions_are_refused():
    frames = [{"t": 0.0, "action": None, "proprio": {"state": [0.0]}}]
    with pytest.raises(ValueError, match="no numeric action"):
        check_episode(episode(frames, []), "state")


def test_a_recorded_run_becomes_a_lerobot_dataset(tmp_path):
    pytest.importorskip("av")
    pytest.importorskip("lerobot.datasets.lerobot_dataset")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from embodied_harness.cli import main
    from embodied_harness.data.annotations import read_annotated_episodes

    bundle, run = recorded_run(tmp_path)
    assert run.result.status == "done"
    out = tmp_path / "dataset"
    assert main(["data", "from-runs", str(run.episode_dir.parent), "--out", str(out)]) == 0

    frames = read_frames(run.episode_dir)
    dataset = LeRobotDataset("local/dataset", root=out)
    assert dataset.meta.total_episodes == 1 and dataset.meta.total_frames == len(frames)
    item = dataset[0]
    assert item["observation.images.head"].shape[-2:] == (48, 64)
    assert item["task"] == bundle.task.goal
    assert item["action"].tolist() == frames[0]["action"]

    (annotated,) = read_annotated_episodes(out, planner=True)
    labels = [decision.label for decision in annotated.decisions]
    assert labels[0] == "pick up the next package" and labels[-1] is None  # the final "done"
    assert annotated.decisions[1].memory  # the planner's memory rows came along

    # Use the actual LeRobot decoder through the CLI, including the sidecar episode goal.
    import yaml

    from embodied_harness.data.evaluate import evaluate_samples
    from embodied_harness.planner.base import PlannerOutput
    from embodied_harness.planner.render import to_sft_record
    from test_data import bound_planner, read_jsonl, vlm_run

    run_config = vlm_run()
    run_config["planner"]["params"].update(show_history=True, show_progress=True, show_state=True)
    training = make_bundle(tmp_path / "training", task=small_task(2, goal_source="episode"), run=run_config)
    labels, mapping = tmp_path / "labels.yaml", tmp_path / "mapping.yaml"
    mapping.write_text(
        yaml.safe_dump(
            {
                "instruction_key": "instruction",
                "cameras": {camera: f"observation.images.{camera}" for camera in CAMERAS},
                "proprio": {"gripper_width": "state[0]", "gripper_closed": "state[1] > 0.5"},
            }
        )
    )
    assert main(["data", "labels", str(training.path), str(out), "--out", str(labels), "--accept-exact"]) == 0
    exported = tmp_path / "planner-export"
    assert (
        main(
            [
                "data",
                "planner",
                str(training.path),
                str(out),
                "--labels",
                str(labels),
                "--mapping",
                str(mapping),
                "--out",
                str(exported),
                "--val-fraction",
                "0",
            ]
        )
        == 0
    )
    records = read_jsonl(exported / "train.jsonl")
    samples = read_jsonl(exported / "samples.jsonl")
    assert all(sample["goal"] == bundle.task.goal for sample in samples)
    assert samples[-1]["terminal"] and samples[-1]["target"]["done"]
    assert sum(samples[-1]["ledger"]["sorted"].values()) == 2

    rendered = []
    backend = bound_planner(training)

    class RecordingPlanner:
        def settings(self):
            return backend.settings()

        def plan(self, context):
            expected = records[len(rendered)]
            request = backend.render_request(context)
            actual = to_sft_record(request.prompt, expected["messages"][2]["content"], expected["images"])
            rendered.append(actual["messages"][:2])
            assert rendered[-1] == expected["messages"][:2]
            return PlannerOutput(raw=expected["messages"][2]["content"])

    rows = evaluate_samples(RecordingPlanner(), training, exported, split="all")
    assert len(rows) == len(records) and all(row.same_call for row in rows)


def test_from_runs_rejects_fractional_control_rate():
    episode_data = episode(
        [{"t": 0.0, "action": [0.0], "proprio": {"state": [0.0]}}],
        [],
    )
    episode_data.snapshot["run"]["runtime"]["control_hz"] = 7.5
    with pytest.raises(ValueError, match=r"control_hz=7\.5 is not an integer"):
        check_episode(episode_data, "state")
