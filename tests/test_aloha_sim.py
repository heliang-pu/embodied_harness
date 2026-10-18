"""The gym-aloha env adapter against a fake gymnasium env: no MuJoCo needed."""

from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest

from conftest import CONFIGS, of_type, read_events, run_episode
from embodied_harness.config.loader import load_bundle
from embodied_harness.core.types import ActionChunk
from embodied_harness.env.aloha_sim import STATE_KEYS, AlohaSimEnv

ARM = [index for index, key in enumerate(STATE_KEYS) if not key.endswith("gripper")]
GRIPPERS = [STATE_KEYS.index("left_gripper"), STATE_KEYS.index("right_gripper")]
SHOULDERS = [STATE_KEYS.index("left_shoulder"), STATE_KEYS.index("right_shoulder")]
INSERTION = CONFIGS / "aloha_sim_insertion" / "run_insertion_seq.yaml"


class FakePhysics:
    """dm_control's physics as the adapter uses it: `data.ptr` is what the fake mujoco functions read."""

    def __init__(self, sim):
        self.model = types.SimpleNamespace(ptr=None)
        self.data = types.SimpleNamespace(ptr=sim)
        self.forwards = 0

    def forward(self):
        self.forwards += 1


class FakeMuJoCo(types.ModuleType):
    """mujoco's state functions over a FakeAloha: the state is its 14 joints and the peg's position."""

    class mjtState:  # mujoco's own name
        mjSTATE_INTEGRATION = 0x3FFF

    @staticmethod
    def mj_stateSize(model, sig):
        return 15

    @staticmethod
    def mj_getState(model, sim, out, sig):
        out[:] = [*sim.pos, sim.peg]

    @staticmethod
    def mj_setState(model, sim, values, sig):
        sim.pos, sim.peg = np.asarray(values[:14], dtype=np.float32).copy(), float(values[14])


class FakeAloha:
    """Moves the joints to the commanded targets, arm joints `sag` below them (gravity against the position
    controllers); a gripper commanded shut reads its `grasp` value (the object it closed on) when set;
    succeeds once the right gripper reaches 0.35. `unwrapped._env` stands in for gym-aloha's dm_control env,
    whose state also has a peg that only the physics state and the image show."""

    grasp: tuple[float, float] | None = None

    def __init__(self, env_id, **kwargs):
        self.env_id, self.kwargs = env_id, kwargs
        self.actions: list[np.ndarray] = []
        self.pos = np.zeros(14, dtype=np.float32)
        self.peg = 0.0
        self.sag = 0.0
        self.unwrapped = self
        task = types.SimpleNamespace(get_observation=lambda physics: self._raw())
        self._env = types.SimpleNamespace(physics=FakePhysics(self), task=task)

    def _raw(self):
        image = np.full((4, 6, 3), round(self.peg * 100) % 256, dtype=np.uint8)
        return {"images": {"top": image}, "qpos": self.pos.copy()}

    def _format_raw_obs(self, raw):
        return {"pixels": {"top": raw["images"]["top"]}, "agent_pos": raw["qpos"]}

    def _obs(self):
        return self._format_raw_obs(self._raw())

    def reset(self, seed=None):
        self.seed = seed
        self.pos = np.full(14, 0.1, dtype=np.float32)
        self.peg = 0.1 * (seed or 0)  # the seed places the objects
        return self._obs(), {"is_success": False}

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        self.pos = np.asarray(action, dtype=np.float32).copy()
        self.pos[ARM] -= self.sag
        for index, reading in zip(GRIPPERS, self.grasp or (), strict=False):
            if action[index] < 0.5:
                self.pos[index] = reading
        self.peg += 0.01
        success = abs(float(self.pos[13]) - 0.35) < 1e-6
        return self._obs(), 4 if success else 0, success, False, {"is_success": success}

    def close(self):
        self.closed = True


@pytest.fixture
def made(monkeypatch):
    envs: list[FakeAloha] = []

    def make(env_id, **kwargs):
        envs.append(FakeAloha(env_id, **kwargs))
        return envs[-1]

    monkeypatch.setitem(sys.modules, "gymnasium", types.SimpleNamespace(make=make))
    monkeypatch.setitem(sys.modules, "gym_aloha", types.ModuleType("gym_aloha"))
    monkeypatch.setitem(sys.modules, "mujoco", FakeMuJoCo("mujoco"))
    return envs


def test_observation_names_the_state_for_verifiers_and_keeps_it_whole_for_the_policy(made):
    env = AlohaSimEnv({"task": "insertion", "max_steps": 50})
    obs = env.reset(seed=3)
    assert made[0].env_id == "gym_aloha/AlohaInsertion-v0" and made[0].seed == 3
    assert made[0].kwargs["max_episode_steps"] == 50  # not gymnasium's registered limit
    assert set(obs.proprio) == {*STATE_KEYS, "state"} and obs.proprio["right_gripper"] == pytest.approx(0.1)
    assert obs.proprio["state"].shape == (14,) and obs.images["top"].shape == (4, 6, 3)
    assert set(AlohaSimEnv.proprio_keys({})) == set(obs.proprio)
    assert AlohaSimEnv.camera_keys({}) == ["top"]


def test_hold_keeps_the_last_gripper_command_so_a_grasp_stays_closed(made):
    env = AlohaSimEnv({})
    env.reset(seed=0)
    target = np.linspace(0.0, 1.3, 14, dtype=np.float32)
    target[6], target[13] = 0.0, 0.2
    env.act(target)
    made[0].pos[6] = 0.66  # the fingers stop on the socket, short of the command
    sent = env.hold()
    assert sent[6] == pytest.approx(0.0) and sent[13] == pytest.approx(0.2)
    assert sent[0] == pytest.approx(target[0])  # the joints stay where they are


def test_hold_sends_the_last_targets_so_the_arms_do_not_sag(made):
    # Under gravity every arm joint settles a little below its target. Targets rebuilt from those readings
    # sank by that much on every tick (0.7 rad at the wrist after 300 ticks in gym-aloha).
    env = AlohaSimEnv({"max_steps": 1000})
    start = env.reset(seed=0).proprio["state"]
    made[0].sag = 0.01
    for _ in range(300):
        sent = env.hold()
    np.testing.assert_array_equal(sent, start)  # right after a reset: the start pose
    np.testing.assert_allclose(env.observe().proprio["state"][ARM], start[ARM] - 0.01, atol=1e-6)

    target = np.linspace(0.0, 1.3, 14, dtype=np.float32)
    target[6], target[13] = 0.0, 0.2
    env.act(target)
    for _ in range(300):
        sent = env.hold()
    np.testing.assert_array_equal(sent, target)
    np.testing.assert_allclose(env.observe().proprio["state"][ARM], target[ARM] - 0.01, atol=1e-6)


def test_success_or_the_step_limit_ends_the_episode(made):
    env = AlohaSimEnv({"max_steps": 3})
    env.reset(seed=0)
    grasp = np.zeros(14, dtype=np.float32)
    grasp[13] = 0.35
    env.act(grasp)
    obs = env.observe()
    assert obs.terminated and obs.info["success"]
    assert env.act(grasp) is None and env.hold() is None
    env.reset(seed=1)
    for _ in range(3):
        env.act(np.zeros(14))
    assert env.observe().terminated and not env.observe().info["success"]


def test_bad_parameters_and_actions_are_rejected(made):
    with pytest.raises(ValueError, match="no task"):
        AlohaSimEnv({"task": "stacking"})
    env = AlohaSimEnv({})
    env.reset()
    with pytest.raises(ValueError, match="14 joint targets"):
        env.act(np.zeros(7))


def test_go_home_opens_the_grippers_and_walks_the_joints_back_to_the_start_pose(made):
    env = AlohaSimEnv({})
    env.reset(seed=0)  # the fake starts every joint at 0.1
    far = np.full(14, 0.1, dtype=np.float32)
    far[1], far[8] = 0.5, -0.02  # shoulders away from home
    env.act(far)
    sent = env.act({"primitive": "go_home", "version": 3})
    assert sent[6] == pytest.approx(1.0) and sent[13] == pytest.approx(1.0)
    assert sent[1] == pytest.approx(0.45) and sent[8] == pytest.approx(0.03)  # 0.05 per step at most
    for _ in range(10):
        sent = env.act({"primitive": "go_home"})
    assert sent[1] == pytest.approx(0.1) and sent[8] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="no primitive"):
        env.act({"primitive": "wave"})


def test_a_snapshot_restores_the_sim_and_the_adapter_in_a_fresh_env(made):
    assert AlohaSimEnv.simulated
    env = AlohaSimEnv({"max_steps": 5})
    assert env.snapshot() is None  # no sim before the first reset
    env.reset(seed=2)
    target = np.full(14, 0.3, dtype=np.float32)
    target[6], target[13] = 0.0, 0.2  # a grasp
    env.act(np.zeros(14, dtype=np.float32))
    env.act(target)
    saved = json.loads(json.dumps(env.snapshot()))  # what env_snapshot.json holds
    assert saved["steps"] == 2 and saved["mujoco"]["sig"] == 0x3FFF and len(saved["mujoco"]["state"]) == 15
    before = env.observe()
    env.act(np.ones(14, dtype=np.float32))  # the world moves on after the snapshot

    resumed = AlohaSimEnv({"max_steps": 5})
    resumed.reset(seed=9)  # `run --resume` builds the sim with reset(), which places the objects elsewhere
    resumed.restore(saved)
    sim = made[-1]
    assert sim.peg == pytest.approx(0.22) and sim._env.physics.forwards == 1
    obs = resumed.observe()
    np.testing.assert_array_equal(obs.proprio["state"], before.proprio["state"])
    np.testing.assert_array_equal(obs.images["top"], before.images["top"])  # rendered from the restored state
    assert obs.info == {"success": False, "steps": 2} and not obs.terminated
    np.testing.assert_array_equal(resumed.hold(), target)  # the grasp command survives the resume
    for _ in range(2):
        resumed.act(target)
    assert resumed.observe().terminated  # the step limit counts on from the snapshot: 2 + 3 = 5
    sent = resumed.act(np.ones(14, dtype=np.float32))
    assert sent is None


def test_a_snapshot_of_another_sim_is_refused(made):
    env = AlohaSimEnv({"task": "insertion"})
    with pytest.raises(RuntimeError, match="reset"):
        env.restore({})
    env.reset(seed=0)
    saved = env.snapshot()
    cube = AlohaSimEnv({"task": "transfer_cube"})
    cube.reset(seed=0)
    with pytest.raises(ValueError, match="AlohaInsertion"):
        cube.restore(saved)
    with pytest.raises(ValueError, match="expects 15"):
        env.restore({**saved, "mujoco": {"sig": saved["mujoco"]["sig"], "state": [0.0] * 3}})


class ReachAndClose:
    """A policy for the insertion skills: pick_up reaches down with both grippers open for 2 s, then closes
    them (keeps them open with close=False); insert holds still with the grippers closed."""

    def __init__(self, library, close=True):
        self.library, self.close = library, close
        self.prompt, self.version, self.steps = None, 0, 0

    def prompt_library(self):
        return self.library

    def capabilities(self):
        return types.SimpleNamespace(chunk_size=50)

    def reset(self):
        self.prompt, self.version, self.steps = None, 0, 0

    def set_instruction(self, call, prompt, version):
        self.prompt, self.version, self.steps = prompt, version, 0

    def infer_chunk(self, obs):
        actions = []
        for _ in range(50):
            action = np.asarray(obs.proprio["state"], dtype=np.float32).copy()
            action[SHOULDERS] = -0.5
            opening = self.prompt.id == "pick_up" and (self.steps < 100 or not self.close)
            action[GRIPPERS] = 1.0 if opening else 0.0
            self.steps += 1
            actions.append(action)
        return ActionChunk(self.version, actions)

    def is_done(self):
        return False


def run_insertion(tmp_path, monkeypatch, grasp, *, close=True, env_class=AlohaSimEnv, **kwargs):
    """run_insertion_seq.yaml (pick_up, then insert; 400 steps) on the fake sim."""
    monkeypatch.setattr(FakeAloha, "grasp", grasp)
    bundle = load_bundle(INSERTION)
    env = env_class(dict(bundle.run.env.params))
    run = run_episode(bundle, tmp_path, env=env, policy=ReachAndClose(bundle.library, close=close), **kwargs)
    return run, env


def test_a_good_grasp_verifies_and_a_still_insert_reaches_the_planner(made, monkeypatch, tmp_path):
    # Good socket grasps read 0.80-0.84 in gym-aloha. Above the old 0.8 bound pick_up timed out at 7 s, and
    # its go_home opened both grippers.
    run, _ = run_insertion(tmp_path, monkeypatch, (0.82, 0.3))
    events = read_events(run)
    ends = [
        (e["call"]["skill"], e["outcome"]["status"], e["outcome"]["reason"])
        for e in of_type(events, "skill_end")
    ]
    assert ends == [("pick_up", "success", None), *[("insert", "failure", "stall")] * 3]
    assert [e["action"] for e in of_type(events, "recovery")] == ["retry", "planner"]
    assert run.result.status == "aborted" and "insert failed 3 times in a row" in run.result.message


def test_a_pick_up_timeout_keeps_what_is_in_hand_and_goes_home_empty_handed(made, monkeypatch, tmp_path):
    # Both grippers closed on their objects, the socket reading just above the holding range: the retry
    # keeps them; go_home would open both, 7 s into an 8 s episode.
    run, _ = run_insertion(tmp_path / "holding", monkeypatch, (0.9, 0.3))
    events = read_events(run)
    recovery = of_type(events, "recovery")
    assert [(e["reason"], e["action"], e["before_retry"]) for e in recovery] == [("timeout", "retry", [])]
    starts = [(e["call"]["skill"], e["purpose"]) for e in of_type(events, "skill_start")]
    assert starts == [("pick_up", "main"), ("pick_up", "main")]
    # Both hands empty: the retried pick_up starts from the start pose.
    run, _ = run_insertion(tmp_path / "empty", monkeypatch, None, close=False)
    events = read_events(run)
    assert [e["rule"] for e in of_type(events, "transition")] == ["home_before_picking_from_nothing"]
    starts = [(e["call"]["skill"], e["purpose"]) for e in of_type(events, "skill_start")]
    assert starts == [("pick_up", "main"), ("go_home", "transition")]


def test_a_resume_continues_in_the_sim_where_the_last_skill_ended(made, monkeypatch, tmp_path):
    class Recording(AlohaSimEnv):
        def restore(self, state):
            super().restore(state)
            self.restored = self.observe()

    first, _ = run_insertion(tmp_path, monkeypatch, (0.82, 0.3), max_ticks=150)  # stopped during insert
    saved = json.loads((first.episode_dir / "env_snapshot.json").read_text())  # written as pick_up ended
    _, env = run_insertion(tmp_path, monkeypatch, (0.82, 0.3), env_class=Recording, resume=True)
    assert env.restored.info["steps"] == saved["steps"] > 100
    assert env.restored.proprio["left_gripper"] == pytest.approx(0.82)  # a fresh reset reads 0.1
