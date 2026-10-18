"""Native RoboTwin adapter tests using a fake task module."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from embodied_harness.config.loader import load_bundle
from embodied_harness.env.robotwin import RobotwinEnv
from embodied_harness.env.robotwin_native import NativeRobotwinEnv

ROBOTWIN_CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "robotwin"
OBSERVER = 100  # pixel offset of the fake observer camera, a view no camera_names entry asks for


class FakeTask:
    eval_success = False
    step_lim = 3
    take_action_cnt = 0

    def __init__(self):
        self.closed = False
        self.setup_kwargs = None

    def setup_demo(self, **kwargs):
        self.setup_kwargs = kwargs
        self.eval_success = False
        self.take_action_cnt = 0

    def get_obs(self):
        value = self.take_action_cnt
        raw = {
            "observation": {
                "head_camera": {"rgb": np.full((2, 3, 3), value, dtype=np.uint8)},
                "left_camera": {"rgb": np.full((2, 3, 3), value + 1, dtype=np.uint8)},
                "right_camera": {"rgb": np.full((2, 3, 3), value + 2, dtype=np.uint8)},
                "front_camera": {"rgb": np.full((2, 3, 3), value + 3, dtype=np.uint8)},
            },
            "joint_action": {"vector": np.arange(14, dtype=np.float32) + value},
        }
        # Like RoboTwin's get_obs: the observer camera only with data_type.third_view.
        if (self.setup_kwargs.get("data_type") or {}).get("third_view"):
            raw["third_view_rgb"] = np.full((2, 3, 3), value + OBSERVER, dtype=np.uint8)
        return raw

    def take_action(self, action, action_type="qpos"):
        assert action_type == "qpos"
        np.testing.assert_equal(np.asarray(action).shape, (14,))
        self.take_action_cnt += 1
        self.eval_success = self.take_action_cnt >= 2

    def check_success(self):
        return self.eval_success

    def close_env(self):
        self.closed = True


def install_fake_robotwin(monkeypatch, tmp_path):
    """A fake RoboTwin source tree under tmp_path; returns the task objects it creates, in order."""
    config_dir = tmp_path / "task_config"
    robot_dir = tmp_path / "assets" / "embodiments" / "aloha-agilex"
    config_dir.mkdir(parents=True)
    robot_dir.mkdir(parents=True)
    (config_dir / "demo_clean.yml").write_text(
        "embodiment: [aloha-agilex]\ncamera:\n  head_camera_type: D435\n", encoding="utf-8"
    )
    (config_dir / "_embodiment_config.yml").write_text(
        "aloha-agilex:\n  file_path: assets/embodiments/aloha-agilex\n", encoding="utf-8"
    )
    (config_dir / "_camera_config.yml").write_text("D435:\n  h: 2\n  w: 3\n", encoding="utf-8")
    (robot_dir / "config.yml").write_text("arm_joints_name: [[], []]\n", encoding="utf-8")

    created = []

    class RecordedTask(FakeTask):
        def __init__(self):
            super().__init__()
            created.append(self)

    package = types.ModuleType("envs")
    package.__path__ = [str(tmp_path / "envs")]
    package.CONFIGS_PATH = str(config_dir)
    task_module = types.ModuleType("envs.fake_task")
    task_module.fake_task = RecordedTask
    monkeypatch.setitem(sys.modules, "envs", package)
    monkeypatch.setitem(sys.modules, "envs.fake_task", task_module)
    return created


def test_native_rgb_qpos_and_terminal_verdict(monkeypatch, tmp_path):
    install_fake_robotwin(monkeypatch, tmp_path)
    env = NativeRobotwinEnv(task_name="fake_task", root=tmp_path, episode_length=10)

    raw, info = env.reset(seed=9)

    assert info == {}
    assert set(raw["pixels"]) == {"head_camera", "left_camera", "right_camera"}
    np.testing.assert_array_equal(raw["agent_pos"], np.arange(14, dtype=np.float32))
    raw, reward, terminated, truncated, info = env.step(np.zeros(14, dtype=np.float32))
    assert reward == 0.0 and not terminated and not truncated and info == {}
    _, reward, terminated, truncated, info = env.step(np.zeros(14, dtype=np.float32))
    assert reward == 1.0 and terminated and not truncated and info == {"success": True, "step": 2}
    env.close()


def test_robotwin_native_backend_can_expose_optional_front_camera(monkeypatch, tmp_path):
    install_fake_robotwin(monkeypatch, tmp_path)
    env = RobotwinEnv(
        {
            "backend": "native",
            "task": "fake_task",
            "root": str(tmp_path),
            "camera_names": ("head_camera", "left_camera", "right_camera", "front_camera"),
        }
    )
    # Native source returns four cameras, while the harness names the optional fourth ``front``.
    env._ensure_env()
    raw, _ = env._env.reset(seed=0)
    obs = env._observation(raw)
    assert set(obs.images) == {"head", "left_wrist", "right_wrist", "front"}


def test_front_is_robotwins_front_camera_not_the_observer_view(monkeypatch, tmp_path):
    # The planner's training data has cam_front = RoboTwin's front_camera (robotwin_conversion_manifest.json),
    # low over the table facing the bottles. The observer camera (third_view_rgb) looks back at the robot from
    # across the table: a view the planner never saw.
    tasks = install_fake_robotwin(monkeypatch, tmp_path)
    env = RobotwinEnv(
        {
            "backend": "native",
            "task": "fake_task",
            "root": str(tmp_path),
            "camera_names": ("front_camera", "head_camera", "left_camera", "right_camera"),
        }
    )
    obs = env.reset(seed=1000)
    np.testing.assert_array_equal(obs.images["front"], np.full((2, 3, 3), 3, dtype=np.uint8))
    assert not (tasks[0].setup_kwargs.get("data_type") or {}).get("third_view")  # nothing extra rendered
    env.act(np.zeros(14, dtype=np.float32))
    np.testing.assert_array_equal(env.observe().images["front"], np.full((2, 3, 3), 4, dtype=np.uint8))


def test_a_camera_robotwin_does_not_render_is_named(monkeypatch, tmp_path):
    install_fake_robotwin(monkeypatch, tmp_path)
    env = NativeRobotwinEnv(task_name="fake_task", root=tmp_path, camera_names=("head_camera", "top_camera"))
    with pytest.raises(KeyError, match="no RGB image for 'top_camera'"):
        env.reset(seed=0)


def test_the_eval_config_runs_scenes_the_training_demonstrations_never_used():
    # The 50 demonstrations were recorded with RoboTwin seeds 0-79 (episode_seeds in the conversion
    # manifest). `run` without --seed starts at runtime.seed and adds the episode index.
    assert load_bundle(ROBOTWIN_CONFIGS / "run_bottles_eval.yaml").run.runtime.seed >= 1000
