import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from conftest import make_bundle, of_type, read_events, run_episode, small_task
from embodied_harness.core.clock import SimClock
from embodied_harness.core.types import ActionChunk
from embodied_harness.env.mock_sorting import MockSortingWorld
from embodied_harness.env.robomme import RoboMMEEnv
from embodied_harness.planner.oracle import MockSortingOracle
from embodied_harness.planner.render import format_proprio
from embodied_harness.policy.executor.mock import MockPolicyExecutor
from embodied_harness.runtime import runner as runner_module
from embodied_harness.runtime.runner import ResumeError
from embodied_harness.runtime.trace import frames as frames_module
from embodied_harness.runtime.trace import writer as writer_module
from embodied_harness.runtime.trace.frames import row_video, segment_dir
from embodied_harness.runtime.trace.reader import load_restored
from embodied_harness.runtime.trace.replay import context_from_record
from embodied_harness.runtime.trace.report import build_report, read_episode
from embodied_harness.runtime.trace.writer import TraceWriter

WORLD = {"num_packages": 3, "barcode_down_prob": 0.0}


def test_resume_finishes_an_interrupted_episode(bundle, tmp_path):
    first = run_episode(bundle, tmp_path, world=WORLD, max_ticks=100)  # stops about 10 s in
    assert first.result is None
    assert not (first.episode_dir / "outcome.json").exists()
    before = read_events(first)
    sorted_before = sum(op["value"] for event in of_type(before, "ledger_update") for op in event["ops"])
    assert 0 < sorted_before < 3

    second = run_episode(bundle, tmp_path, world=WORLD, resume=True)
    assert second.result.status == "done"
    assert sum(second.result.ledger["sorted"].values()) == 3

    events = read_events(second)
    assert [event["seq"] for event in events] == list(range(1, len(events) + 1))
    times = [event["t"] for event in events]
    assert times == sorted(times)  # the resumed part continues the original timeline
    assert len(of_type(events, "episode_resume")) == 1
    requests = of_type(events, "planner_request")
    assert len({event["call_id"] for event in requests}) == len(requests)
    assert [event["reason"] for event in requests].count("resume") == 1


def test_resume_refuses_a_finished_episode(bundle, tmp_path):
    run_episode(bundle, tmp_path, world=WORLD)
    with pytest.raises(RuntimeError, match="already finished"):
        run_episode(bundle, tmp_path, world=WORLD, resume=True)


def test_a_new_run_refuses_an_existing_trace(bundle, tmp_path):
    run_episode(bundle, tmp_path, world=WORLD, max_ticks=10)
    with pytest.raises(FileExistsError):
        run_episode(bundle, tmp_path, world=WORLD)


def test_a_torn_last_line_is_ignored(tmp_path):
    directory = tmp_path / "ep"
    directory.mkdir()
    (directory / "events.jsonl").write_text(
        '{"seq": 1, "t": 0.0, "type": "memory_update", "memory": "x"}\n{"seq": 2, "t": 0.1, "ty',
        encoding="utf-8",
    )
    restored = load_restored(directory)
    assert restored.memory == "x"
    assert restored.last_seq == 1


def test_resume_repairs_a_torn_tail_before_appending(bundle, tmp_path):
    first = run_episode(bundle, tmp_path, world=WORLD, max_ticks=100)
    assert first.result is None
    before = read_events(first)
    with (first.episode_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"seq": 9999, "type": "episode_end"')

    resumed = run_episode(bundle, tmp_path, world=WORLD, resume=True)
    assert resumed.result is not None and resumed.result.status == "done"
    events = read_events(resumed)
    assert events[: len(before)] == before
    assert [event["seq"] for event in events] == list(range(1, len(events) + 1))
    assert len(of_type(events, "episode_resume")) == 1
    assert events[-1]["type"] == "episode_end"
    assert load_restored(resumed.episode_dir).finished


@pytest.mark.parametrize("name", ["events.jsonl", "actions.jsonl"])
def test_resume_keeps_a_complete_json_tail_without_newline(tmp_path, name):
    directory = tmp_path / "ep"
    directory.mkdir()
    first = (
        {"seq": 1, "t": 0.0, "type": "memory_update", "memory": "x"}
        if name == "events.jsonl"
        else {"t": 0.0, "version": 1, "action": [0.5]}
    )
    (directory / name).write_text(json.dumps(first), encoding="utf-8")

    writer = TraceWriter(directory, SimClock(), start_seq=1)
    try:
        if name == "events.jsonl":
            writer.event("episode_resume", memory="x")
        else:
            writer.action(1, [0.25])
    finally:
        writer.close()

    entries = [json.loads(line) for line in (directory / name).read_text(encoding="utf-8").splitlines()]
    assert len(entries) == 2
    assert entries[0] == first
    if name == "events.jsonl":
        assert entries[1]["type"] == "episode_resume" and entries[1]["seq"] == 2
    else:
        assert entries[1]["action"] == [0.25]


def test_trace_files_share_one_tail_repair_and_one_normalizer():
    # events/actions (the writer) and frames.jsonl used to have their own copies, whose numpy handling
    # had drifted apart: the writer stored a numpy value's printed text, frames its numbers.
    assert frames_module.repair_tail is writer_module.repair_tail
    assert frames_module.last_record is writer_module.last_record
    state = np.array([0.1, 1 / 3], np.float32)
    value = {"state": state, "count": np.int64(3), "on": np.bool_(True), "xy": (1, 2)}
    exact = {"state": [float(state[0]), float(state[1])], "count": 3, "on": True, "xy": [1, 2]}
    assert writer_module.to_jsonable(value) == frames_module.plain(value) == exact
    # frames.jsonl is dumped without a fallback, so plain() still turns any other value into its text.
    assert frames_module.plain({"path": Path("a/b"), "z": complex(1, 2)}) == {"path": "a/b", "z": "(1+2j)"}


JOINTS = np.array([0.1, -0.2, 1 / 3, 0.0012345679], dtype=np.float32)  # numpy prints the last one inexactly


class NumpyWorld(MockSortingWorld):
    """The mock world reporting a float32 state vector, the way the robot adapters do."""

    def observe(self):
        obs = super().observe()
        obs.proprio = {**obs.proprio, "state": np.array([obs.proprio["gripper_width"], 1 / 3], np.float32)}
        return obs


class NumpyPolicy(MockPolicyExecutor):
    """The mock policy, with a float32 joint target in every action, the way real policies send them."""

    def infer_chunk(self, obs):
        chunk = super().infer_chunk(obs)
        return ActionChunk(chunk.version, [{**action, "joints": JOINTS} for action in chunk.actions])


class ProprioRecordingOracle(MockSortingOracle):
    def __init__(self):
        super().__init__()
        self.proprio = []

    def plan(self, context):
        self.proprio.append(dict(context.proprio))
        return super().plan(context)


def numpy_run(bundle, tmp_path, **kwargs):
    planner = ProprioRecordingOracle()
    env, policy = NumpyWorld(WORLD), NumpyPolicy(bundle.library)
    return run_episode(bundle, tmp_path, env=env, policy=policy, planner=planner, **kwargs), planner


def traced_joints(actions_jsonl):
    rows = [json.loads(line) for line in actions_jsonl.splitlines()]
    return [row["action"]["joints"] for row in rows if "joints" in (row["action"] or {})]


def test_numpy_actions_and_proprio_are_traced_as_their_exact_numbers(bundle, tmp_path):
    writer = TraceWriter(tmp_path / "rows", SimClock())
    writer.action(1, JOINTS)  # a bare ndarray row, as the websocket policy queues them
    writer.close()
    (row,) = [json.loads(line) for line in (tmp_path / "rows" / "actions.jsonl").read_text().splitlines()]
    assert isinstance(row["action"], list) and np.array_equal(np.array(row["action"], np.float32), JOINTS)

    run, planner = numpy_run(bundle, tmp_path)
    assert run.result.status == "done"
    joints = traced_joints((run.episode_dir / "actions.jsonl").read_text(encoding="utf-8"))
    assert joints and all(isinstance(value, list) for value in joints)  # not numpy's printed text
    assert all(np.array_equal(np.array(value, np.float32), JOINTS) for value in joints)  # bit for bit
    boundaries = of_type(read_events(run), "boundary")
    assert boundaries and all(e["proprio"]["state"][1] == float(np.float32(1 / 3)) for e in boundaries)

    # A planner record keeps the proprio text the prompt showed, so replay renders the live prompt's.
    records = sorted((run.episode_dir / "planner").glob("pc_*.json"))
    assert len(records) == len(planner.proprio)
    for path, live in zip(records, planner.proprio, strict=True):
        record = json.loads(path.read_text(encoding="utf-8"))
        replayed = context_from_record(record, run.episode_dir, bundle.task)
        assert format_proprio(replayed.proprio, None, 3) == format_proprio(live, None, 3)


def test_numpy_values_without_an_exact_json_number_are_traced_as_their_text(tmp_path):
    # A longdouble's tolist() is a longdouble again, and a numpy class has no instance to list: the trace
    # keeps their text, next to exact numbers, instead of recursing or raising inside the control loop.
    values = {
        "longdouble": np.longdouble(1.5),
        "longdoubles": np.array([1.5, 2.25], np.longdouble),
        "clongdouble": np.clongdouble(1 + 2j),
        "scalar_class": np.float32,
        "array_class": np.ndarray,
        "exact": np.float32(0.1),
    }
    expected = {
        "longdouble": "1.5",
        "longdoubles": ["1.5", "2.25"],
        "clongdouble": "(1+2j)",
        "scalar_class": "<class 'numpy.float32'>",
        "array_class": "<class 'numpy.ndarray'>",
        "exact": float(np.float32(0.1)),
    }
    writer = TraceWriter(tmp_path, SimClock())
    writer.event("boundary", proprio=values)
    writer.action(1, values)
    writer.write_json("outcome.json", values)
    writer.close()
    (event,) = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    (row,) = [json.loads(line) for line in (tmp_path / "actions.jsonl").read_text().splitlines()]
    assert event["proprio"] == row["action"] == expected
    assert json.loads((tmp_path / "outcome.json").read_text()) == expected
    assert json.loads(json.dumps(frames_module.plain(values))) == expected  # frames.jsonl has no fallback


def test_a_trace_with_numpy_text_still_resumes_and_reports(bundle, tmp_path):
    first, _ = numpy_run(bundle, tmp_path, max_ticks=100)
    assert first.result is None
    directory = first.episode_dir

    def old_writer(record):  # what the writer recorded before it kept numbers: numpy's printed text
        for holder, key in ((record.get("action"), "joints"), (record.get("proprio"), "state")):
            if isinstance(holder, dict) and isinstance(holder.get(key), list):
                holder[key] = str(np.array(holder[key], np.float32))
        return json.dumps(record, ensure_ascii=False)

    recorded = {}
    for name in ("actions.jsonl", "events.jsonl"):
        lines = (directory / name).read_text(encoding="utf-8").splitlines()
        recorded[name] = "".join(old_writer(json.loads(line)) + "\n" for line in lines)
        (directory / name).write_text(recorded[name], encoding="utf-8")
    assert isinstance(traced_joints(recorded["actions.jsonl"])[0], str)

    resumed, _ = numpy_run(bundle, tmp_path, resume=True)
    assert resumed.result.status == "done"
    for name, text in recorded.items():
        assert (directory / name).read_text(encoding="utf-8").startswith(text)  # kept as recorded
    appended = (directory / "actions.jsonl").read_text(encoding="utf-8")[len(recorded["actions.jsonl"]) :]
    joints = traced_joints(appended)
    assert joints and all(isinstance(value, list) for value in joints)
    assert build_report([read_episode(directory)])["causes"]["success"] == 1


class _GoalWorld(MockSortingWorld):
    """A snapshot-capable world whose instruction identifies the original reset seed."""

    def reset(self, seed=None):
        self._goal = f"goal-{seed}"
        return super().reset(seed)

    def instruction(self):
        return self._goal

    def snapshot(self):
        return {**super().snapshot(), "goal": self._goal}

    def restore(self, state):
        super().restore(state)
        self._goal = state["goal"]


class _RecordingOracle(MockSortingOracle):
    def __init__(self):
        super().__init__()
        self.goals = []

    def plan(self, context):
        self.goals.append(context.task.goal)
        return super().plan(context)


@pytest.mark.parametrize("max_ticks", [10, 40])
def test_resume_restores_the_recorded_episode_instruction(tmp_path, max_ticks):
    bundle = make_bundle(tmp_path, task=small_task(1, goal_source="episode"))
    first_planner = _RecordingOracle()
    first = run_episode(
        bundle,
        tmp_path,
        seed=7,
        env=_GoalWorld({"num_packages": 1, "barcode_down_prob": 0.0}),
        planner=first_planner,
        max_ticks=max_ticks,
    )
    assert first.result is None
    assert (first.episode_dir / "env_snapshot.json").is_file() is (max_ticks == 40)
    assert first_planner.goals and set(first_planner.goals) == {"goal-7"}

    resumed_planner = _RecordingOracle()
    resumed = run_episode(
        bundle,
        tmp_path,
        seed=None,
        env=_GoalWorld({"num_packages": 1, "barcode_down_prob": 0.0}),
        planner=resumed_planner,
        resume=True,
    )
    assert resumed.result is not None and resumed.result.status == "done"
    assert resumed_planner.goals and set(resumed_planner.goals) == {"goal-7"}


def test_resume_reads_a_legacy_instruction_after_restoring_the_world(tmp_path):
    bundle = make_bundle(tmp_path, task=small_task(1, goal_source="episode"))
    first = run_episode(
        bundle,
        tmp_path,
        seed=7,
        env=_GoalWorld({"num_packages": 1, "barcode_down_prob": 0.0}),
        max_ticks=40,
    )
    assert first.result is None and (first.episode_dir / "env_snapshot.json").is_file()
    # Old traces may have a world snapshot but no recorded instruction event.
    event_path = first.episode_dir / "events.jsonl"
    event_path.write_text(
        "".join(json.dumps(event) + "\n" for event in read_events(first) if event["type"] != "instruction"),
        encoding="utf-8",
    )

    resumed_planner = _RecordingOracle()
    resumed = run_episode(
        bundle,
        tmp_path,
        seed=None,
        env=_GoalWorld({"num_packages": 1, "barcode_down_prob": 0.0}),
        planner=resumed_planner,
        resume=True,
    )
    assert resumed.result is not None and resumed.result.status == "done"
    assert resumed_planner.goals and set(resumed_planner.goals) == {"goal-7"}


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize(("max_ticks", "expected_index"), [(6, 1), (8, 1), (12, 2), (18, 3)])
def test_sequence_resume_preserves_duplicate_step_positions(tmp_path, legacy, max_ticks, expected_index):
    # Each retract consumes five action ticks, then a verifier tick. Adjacent duplicate calls
    # deliberately collapse to a single history entry; only the full trace distinguishes them.
    bundle = make_bundle(
        tmp_path,
        task={"task": {"skills": ["retract"], "done_when": None}},
        run={
            "executor": {"handoff": "hard"},
            "planner": {
                "type": "sequence",
                "params": {"__replace__": True, "calls": [{"skill": "retract"}] * 3},
            },
        },
    )
    first = run_episode(bundle, tmp_path, max_ticks=max_ticks)
    assert first.result is None
    before = read_events(first)
    assert len(load_restored(first.episode_dir).history) == 1
    if legacy:
        for event in before:
            event.pop("meta", None)
        (first.episode_dir / "events.jsonl").write_text(
            "".join(json.dumps(event) + "\n" for event in before), encoding="utf-8"
        )

    # Both runs construct a fresh planner through the factory, as the CLI does in a new process.
    resumed = run_episode(bundle, tmp_path, resume=True)
    assert resumed.result is not None and resumed.result.status == "done"
    events = read_events(resumed)[len(before) :]
    response = of_type(events, "planner_response")[0]
    assert response["meta"]["sequence_index"] == expected_index
    assert of_type(events, "planner_request")[0]["reason"] == "resume"
    if expected_index < 3:
        assert response["decision"]["skill"] == "retract"
    else:
        assert response["decision"]["done"] is True
    assert len(of_type(read_events(resumed), "skill_end")) == 3


def test_sequence_resume_reissues_a_proposal_interrupted_in_transition(tmp_path):
    bundle = make_bundle(
        tmp_path,
        task={"task": {"skills": ["retract", "next_step"], "done_when": None}},
        skills={
            "skills": {"next_step": {"executor": "scripted", "timeout_s": 5}},
            "transitions": [{"when": "next.skill == 'next_step'", "run": ["retract"]}],
        },
        run={
            "executor": {"handoff": "hard"},
            "planner": {
                "type": "sequence",
                "params": {
                    "__replace__": True,
                    "calls": [{"skill": "retract"}, {"skill": "next_step"}],
                },
            },
        },
    )
    first = run_episode(bundle, tmp_path, max_ticks=8)
    before = read_events(first)
    assert first.result is None
    assert of_type(before, "skill_start")[-1]["purpose"] == "transition"
    assert [call.key() for call in load_restored(first.episode_dir).history] == ["retract"]

    resumed = run_episode(bundle, tmp_path, resume=True)
    assert resumed.result is not None and resumed.result.status == "done"
    response = of_type(read_events(resumed)[len(before) :], "planner_response")[0]
    assert response["decision"]["skill"] == "next_step"
    assert response["meta"]["sequence_index"] == 1


class SeedRecordingWorld(MockSortingWorld):
    def __init__(self, params):
        super().__init__(params)
        self.seeds = []

    def reset(self, seed=None):
        self.seeds.append(seed)
        return super().reset(seed)


@pytest.mark.parametrize("seed", [None, 3])
def test_resume_resets_the_env_with_the_recorded_seed(bundle, tmp_path, seed):
    # `run --resume` without --seed (the documented command) passes None; a different seed is ignored.
    first = run_episode(bundle, tmp_path, env=SeedRecordingWorld(WORLD), seed=7, max_ticks=100)
    assert first.result is None
    world = SeedRecordingWorld(WORLD)
    resumed = run_episode(bundle, tmp_path, env=world, seed=seed, resume=True)
    assert world.seeds == [7]
    assert of_type(read_events(resumed), "episode_resume")[0]["seed"] == 7
    assert resumed.result.status == "done"


class SimulatorWithoutRestore(MockSortingWorld):
    """Like RoboMME: a simulator whose world cannot be saved and brought back."""

    simulated = True
    snapshot = None
    restore = None


def test_resume_refuses_a_simulator_that_cannot_restore_its_world(bundle, tmp_path):
    first = run_episode(bundle, tmp_path, env=SimulatorWithoutRestore(WORLD), max_ticks=100)
    assert first.result is None and not (first.episode_dir / "env_snapshot.json").exists()
    trace = (first.episode_dir / "events.jsonl").read_bytes()
    with pytest.raises(ResumeError, match="simulator without restore"):
        run_episode(bundle, tmp_path, env=SimulatorWithoutRestore(WORLD), resume=True)
    assert (first.episode_dir / "events.jsonl").read_bytes() == trace  # nothing was appended
    assert RoboMMEEnv.simulated and not hasattr(RoboMMEEnv, "restore")


class StampedWorld(MockSortingWorld):
    """Each tick's camera image and proprio carry the same running number."""

    ticks = 0

    def observe(self):
        obs = super().observe()
        StampedWorld.ticks += 1
        obs.images = {"front": np.full((4, 4, 3), StampedWorld.ticks % 256, np.uint8)}
        obs.proprio = {**obs.proprio, "stamp": StampedWorld.ticks}
        return obs


class FakeVideo:
    """Stands in for the camera video writer; refuses to reopen (overwrite) a recorded file."""

    frames: ClassVar[dict[Path, list[int]]] = {}

    def __init__(self, path, fps):
        self.path = path

    def add(self, frame):
        if self.path not in FakeVideo.frames:
            assert not self.path.exists(), f"{self.path} would be overwritten"
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.touch()
            FakeVideo.frames[self.path] = []
        FakeVideo.frames[self.path].append(int(frame[0, 0, 0]))

    def close(self):
        pass


def test_resume_records_frames_as_a_new_segment(tmp_path, monkeypatch):
    monkeypatch.setattr(frames_module, "_Video", FakeVideo)
    monkeypatch.setattr(FakeVideo, "frames", {})
    videos = []

    class EpisodeVideo:  # trace.video's side-by-side video: only where it is written matters here
        def __init__(self, path, fps):
            videos.append(path)

        def add(self, images):
            pass

        def close(self):
            pass

    monkeypatch.setattr(runner_module, "EpisodeVideo", EpisodeVideo)
    bundle = make_bundle(tmp_path, task=small_task(3), run={"trace": {"frames": True, "video": True}})
    first = run_episode(bundle, tmp_path, env=StampedWorld(WORLD), max_ticks=40)
    directory = first.episode_dir
    with (directory / "frames.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"tick": 41, "t": 4.0, "proprio": {"gripper')  # a crash tore the last line

    resumed = run_episode(bundle, tmp_path, env=StampedWorld(WORLD), resume=True)
    assert resumed.result.status == "done"
    assert videos == [directory / "episode.mp4", directory / "episode_resume_1.mp4"]
    rows = [json.loads(line) for line in (directory / "frames.jsonl").read_text().splitlines()]
    assert [row["tick"] for row in rows] == list(range(1, len(rows) + 1))  # numbering continues
    times = [row["t"] for row in rows]
    assert times == sorted(times)  # the resumed timeline starts after the last recorded tick
    assert [row["segment"] for row in rows[:40]] == [0] * 40 and {row["segment"] for row in rows[40:]} == {1}
    for index, row in enumerate(rows):
        segment, frame = row_video(row, index)
        video = FakeVideo.frames[segment_dir(directory, segment) / "front.mp4"]
        assert video[frame] == row["proprio"]["stamp"] % 256  # every row finds its own image
    assert len(FakeVideo.frames[directory / "frames" / "front.mp4"]) == 40  # the first run's video is kept
    events = read_events(resumed)
    last_frame_t = rows[39]["t"]
    assert of_type(events, "episode_resume")[0]["t"] >= last_frame_t
