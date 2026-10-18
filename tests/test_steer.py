"""`embodied-harness policy steer-check`: does a policy answer different prompts differently?"""

import asyncio
import json
import threading
import zlib

import numpy as np
import pytest

from conftest import REPLACE, make_bundle, small_task
from embodied_harness.cli import main
from embodied_harness.core.types import SkillCall
from embodied_harness.policy.serving.websocket_server import serve_async
from embodied_harness.runtime import steer
from embodied_harness.runtime.steer import analyze, observations_from_frames, policy_calls


class PromptPolicy:
    """Actions: a direction per prompt (when it listens) plus sampling noise (when it samples)."""

    def __init__(self, listens=True, noise=0.01):
        self.listens, self.noise = listens, noise
        self.rng = np.random.default_rng(0)
        self.prompts = []

    def infer(self, obs):
        self.prompts.append(obs["prompt"])
        base = np.zeros((10, 4))
        if self.listens:
            base += (zlib.crc32(obs["prompt"].encode()) % 97) / 97.0
        base += float(obs["observation.images.head"].mean()) / 255.0  # the scene matters too
        return {"actions": base + self.noise * self.rng.standard_normal(base.shape)}


@pytest.fixture
def serve():
    started = []

    def start(policy):
        ready, port, stop = threading.Event(), [], asyncio.Event()
        loop = asyncio.new_event_loop()

        def run():
            asyncio.set_event_loop(loop)
            loop.run_until_complete(
                serve_async(
                    policy,
                    host="127.0.0.1",
                    port=0,
                    metadata={"chunk_size": 10},
                    ready=lambda bound: (port.append(bound), ready.set()),
                    stop=stop,
                )
            )

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        assert ready.wait(10)
        started.append((loop, stop, thread))
        return f"ws://127.0.0.1:{port[0]}"

    yield start
    for loop, stop, thread in started:
        loop.call_soon_threadsafe(stop.set)
        thread.join(10)


def config(tmp_path, url, texts=None):
    bundle = make_bundle(
        tmp_path,
        task=small_task(3),
        run={
            "env": {"params": {"cameras": ["head"], "image_size": [32, 24]}},
            "executor": {
                REPLACE: True,
                "type": "websocket",
                "library": "../libraries/mock_sort_v1.yaml",
                "params": {
                    "url": url,
                    "images": {"head": "observation.images.head"},
                    **({"texts": texts} if texts is not None else {}),
                },
            },
        },
    )
    return bundle


def test_policy_calls_cover_every_prompt(bundle):
    assert [call.key() for call in policy_calls(bundle)] == [
        "pick",
        "flip",
        "place(bin=A)",
        "place(bin=B)",
        "place(bin=C)",
    ]


def test_a_listening_policy_passes(tmp_path, serve, capsys):
    policy = PromptPolicy()
    bundle = config(tmp_path, serve(policy))
    out = tmp_path / "steer.json"
    code = main(
        ["policy", "steer-check", str(bundle.path), "--samples", "3", "--repeats", "2", "--out", str(out)]
    )
    printed = capsys.readouterr().out
    assert code == 0, printed
    assert (
        "3 observation(s) x 5 prompt(s) x 2 repeat(s)" in printed
        and "10/10 prompt pair(s) distinguished" in printed
    )
    assert set(policy.prompts) == {
        "pick up the next package",
        "flip the package over",
        "place the package into bin A",
        "place the package into bin B",
        "place the package into bin C",
    }
    data = json.loads(out.read_text())
    assert data["passed"] is True and len(data["pairs"]) == 10


def test_steer_check_passes_the_task_goal_to_the_policy(tmp_path, serve, capsys):
    policy = PromptPolicy()
    bundle = config(tmp_path, serve(policy), {"prompt": "{goal}", "subgoal": "{prompt}"})
    # This fake policy reads only the unchanged goal, so it cannot distinguish subgoals.
    assert main(["policy", "steer-check", str(bundle.path), "--samples", "1"]) == 1
    assert policy.prompts[0] == bundle.task.goal


def test_steer_check_uses_each_observations_episode_goal(tmp_path, monkeypatch):
    from embodied_harness.core.types import ActionChunk
    from embodied_harness.env.mock_sorting import MockSortingWorld

    bundle = make_bundle(tmp_path, task=small_task(1, goal_source="episode"))

    class InstructedWorld(MockSortingWorld):
        def reset(self, seed=None):
            self.goal = f"episode {seed}"
            return super().reset(seed)

        def instruction(self):
            return self.goal

    observations = steer.observations_from_env(bundle, 2, env=InstructedWorld({"num_packages": 1}))

    class Recorder:
        def __init__(self):
            self.goals = []

        def begin_episode(self, task):
            self.goals.append(task.goal)

        def set_instruction(self, call, prompt, version):
            self.version = version

        def infer_chunk(self, obs):
            return ActionChunk(self.version, [[0.0]])

    recorder = Recorder()
    steer.query(recorder, bundle, observations, [SkillCall("pick")], 1)
    assert recorder.goals == ["episode 0", "episode 1"]

    episode_dir = tmp_path / "recorded"
    (episode_dir / "frames").mkdir(parents=True)
    (episode_dir / "frames/head.mp4").touch()
    (episode_dir / "frames.jsonl").write_text(json.dumps({"t": 0, "proprio": {}}) + "\n")
    events = [{"type": "instruction", "text": "recorded goal"}]
    (episode_dir / "events.jsonl").write_text(json.dumps(events[0]) + "\n")
    monkeypatch.setattr(steer, "read_video_frames", lambda path, indices: {0: np.zeros((2, 2, 3))})
    recorded = steer.observations_from_frames([episode_dir], 1)
    steer.query(recorder, bundle, recorded, [SkillCall("pick")], 1)
    assert recorder.goals[-1] == "recorded goal"


def test_a_policy_that_ignores_its_prompt_fails(tmp_path, serve, capsys):
    bundle = config(tmp_path, serve(PromptPolicy(listens=False)))
    calls = ["--call", "place(bin=A)", "--call", "place(bin=B)"]
    code = main(["policy", "steer-check", str(bundle.path), "--samples", "2", *calls])
    printed = capsys.readouterr().out
    assert code == 1
    assert "| place(bin=A) vs place(bin=B) |" in printed and "| NO |" in printed


def test_deterministic_policies_are_judged_against_the_action_scale():
    rng = np.random.default_rng(1)
    scene = rng.standard_normal((4, 1, 1, 10, 3))  # four observations, one answer per prompt
    listening = np.concatenate([scene, scene + 0.5], axis=1)
    deaf = np.concatenate([scene, scene], axis=1)
    passed = analyze(listening, ["a", "b"])
    assert passed.noise == 0 and passed.pairs[0].ratio is None and passed.passed
    failed = analyze(deaf, ["a", "b"])
    assert failed.pairs[0].relative == 0 and not failed.passed


def test_bad_calls_are_refused(tmp_path, serve, capsys):
    bundle = config(tmp_path, serve(PromptPolicy()))
    assert main(["policy", "steer-check", str(bundle.path), "--call", "retract", "--call", "pick"]) == 2
    assert "scripted primitive" in capsys.readouterr().err
    assert main(["policy", "steer-check", str(bundle.path), "--call", "pick"]) == 2
    assert "at least two" in capsys.readouterr().err


def test_observations_come_from_recorded_frames(tmp_path, monkeypatch):
    episode = tmp_path / "run" / "ep_0001"
    (episode / "frames").mkdir(parents=True)
    (episode / "frames" / "head.mp4").write_bytes(b"")
    calls = ["pick"] * 4 + ["place(bin=A)"] * 4
    records = [{"t": 0.1 * i, "proprio": {"state": [float(i)]}, "call": call} for i, call in enumerate(calls)]
    (episode / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    requested = []

    def fake_reader(path, indices):
        requested.append(sorted(indices))
        return {index: np.full((24, 32, 3), index, np.uint8) for index in indices}

    monkeypatch.setattr(steer, "read_video_frames", fake_reader)
    during_place = observations_from_frames([tmp_path / "run"], 2, during="place")
    assert requested[-1] == [4, 7] and [o.proprio["state"] for o in during_place] == [[4.0], [7.0]]
    everything = observations_from_frames([episode], 3)
    assert [int(o.images["head"][0, 0, 0]) for o in everything] == [0, 4, 7]  # spread over the whole run
    with pytest.raises(ValueError, match="no recorded tick while 'flip' ran"):
        observations_from_frames([episode], 2, during="flip")
    assert SkillCall("pick").key() == "pick"


def test_a_resumed_episodes_rows_take_their_images_from_their_own_segment(tmp_path, monkeypatch):
    episode = tmp_path / "ep_0001"
    (episode / "frames" / "resume_1").mkdir(parents=True)
    for video in ("frames/head.mp4", "frames/resume_1/head.mp4"):
        (episode / video).write_bytes(b"")
    rows = [
        {
            "tick": i + 1,
            "t": 0.1 * i,
            "proprio": {"state": [float(i)]},
            "call": "pick",
            "segment": i // 3,  # three ticks before the resume, three after it
            "frame": i % 3,
        }
        for i in range(6)
    ]
    text = "".join(json.dumps(row) + "\n" for row in rows) + '{"tick": 7, "prompt": "拿'
    (episode / "frames.jsonl").write_bytes(text.encode()[:-1])  # a crash cut the last line mid-character
    (episode / "events.jsonl").write_text(json.dumps({"type": "episode_resume", "seed": 0}) + "\n")

    def fake_reader(path, indices):
        segment = 1 if path.parent.name == "resume_1" else 0
        return {index: np.full((2, 2, 3), 10 * segment + index, np.uint8) for index in indices}

    monkeypatch.setattr(steer, "read_video_frames", fake_reader)
    observations = observations_from_frames([episode], 6)
    assert [int(o.images["head"][0, 0, 0]) for o in observations] == [0, 1, 2, 10, 11, 12]
    assert [o.proprio["state"] for o in observations] == [[float(i)] for i in range(6)]


def test_frames_recorded_before_segments_are_refused_once_the_episode_was_resumed(tmp_path):
    # Back then a resume rewrote the camera videos, so the rows no longer match them.
    episode = tmp_path / "ep_0001"
    (episode / "frames").mkdir(parents=True)
    (episode / "frames" / "head.mp4").write_bytes(b"")
    row = {"tick": 1, "t": 0.0, "proprio": {}, "call": "pick"}  # no segment: recorded before they existed
    (episode / "frames.jsonl").write_text(json.dumps(row) + "\n")
    (episode / "events.jsonl").write_text(json.dumps({"type": "episode_resume"}) + "\n")
    with pytest.raises(ValueError, match="no longer match the camera videos"):
        observations_from_frames([episode], 1)
