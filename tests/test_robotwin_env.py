"""RoboTwin adapter tests without importing SAPIEN or the simulator itself."""

from __future__ import annotations

import sys
import types

import numpy as np

from embodied_harness.env.robotwin import STATE_NAMES, RobotwinEnv


class FakeLeRobotRoboTwin:
    """Small stand-in for LeRobot's Gym wrapper and its RGB/state observations."""

    task_description = "put bottles in dustbin"

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.step_count = 0
        self.closed = False

    def reset(self, seed=None):
        self.seed = seed
        self.step_count = 0
        return self._raw(0), {"is_success": False, "task": "privileged metadata"}

    def step(self, action):
        self.step_count += 1
        assert np.asarray(action).shape == (14,)
        success = self.step_count >= 2
        return (
            self._raw(self.step_count),
            1.0 if success else 0.0,
            success,
            False,
            {
                "task": "put_bottles_dustbin",
                "is_success": success,
                "step": self.step_count,
            },
        )

    def _raw(self, value):
        pixels = {
            "head_camera": np.full((3, 4, 3), value, dtype=np.uint8),
            "left_camera": np.full((3, 4, 3), value + 1, dtype=np.uint8),
            "right_camera": np.full((3, 4, 3), value + 2, dtype=np.uint8),
        }
        return {"pixels": pixels, "agent_pos": np.arange(14, dtype=np.float32) + value}

    def close(self):
        self.closed = True


def install_fake_lerobot(monkeypatch):
    lerobot = types.ModuleType("lerobot")
    envs = types.ModuleType("lerobot.envs")
    robotwin = types.ModuleType("lerobot.envs.robotwin")
    robotwin.RoboTwinEnv = FakeLeRobotRoboTwin
    monkeypatch.setitem(sys.modules, "lerobot", lerobot)
    monkeypatch.setitem(sys.modules, "lerobot.envs", envs)
    monkeypatch.setitem(sys.modules, "lerobot.envs.robotwin", robotwin)


def test_reset_exposes_only_rgb_and_joint_state(monkeypatch):
    install_fake_lerobot(monkeypatch)
    env = RobotwinEnv({"task": "put_bottles_dustbin", "episode_index": 7})

    obs = env.reset(seed=3)

    assert env._env.kwargs["task_name"] == "put_bottles_dustbin"
    assert env._env.kwargs["episode_index"] == 7
    assert env._env.seed == 3
    assert set(obs.images) == {"head", "left_wrist", "right_wrist"}
    assert set(obs.proprio) == {*STATE_NAMES, "state"}
    np.testing.assert_array_equal(obs.proprio["state"], np.arange(14, dtype=np.float32))
    # Reset metadata must not leak simulator/task internals to the policy or verifier.
    assert obs.info == {}
    assert not obs.terminated


def test_terminal_verdict_maps_is_success_and_is_hidden_before_episode_end(monkeypatch):
    install_fake_lerobot(monkeypatch)
    env = RobotwinEnv({})
    env.reset(seed=0)

    env.act(np.zeros(14, dtype=np.float32))
    assert env.observe().info == {}
    assert not env.observe().terminated

    env.act(np.zeros(14, dtype=np.float32))
    obs = env.observe()
    assert obs.terminated
    assert obs.info["success"] is True
    assert obs.info["status"] == "success"
    assert obs.info["steps"] == 2


def test_action_is_returned_and_close_releases_wrapped_env(monkeypatch):
    install_fake_lerobot(monkeypatch)
    env = RobotwinEnv({})
    env.reset()
    action = np.linspace(-1, 1, 14, dtype=np.float32)

    sent = env.act(action)

    np.testing.assert_array_equal(sent, action)
    env.close()
    assert env._env is None


def test_xpolicy_joint_action_map_is_reordered_at_robotwin_boundary():
    action = {
        "left_arm_joint_state": np.arange(6),
        "left_ee_joint_state": np.array([6]),
        "right_arm_joint_state": np.arange(7, 13),
        "right_ee_joint_state": np.array([13]),
    }
    np.testing.assert_array_equal(RobotwinEnv._action(action), np.arange(14, dtype=np.float32))
