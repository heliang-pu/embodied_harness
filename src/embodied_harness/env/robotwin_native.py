from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import yaml


class NativeRobotwinEnv:
    """Small native RoboTwin wrapper with RGB/qpos observations and terminal scoring.

    No expert trajectory or seed filtering is used. The task's verdict remains in terminal
    metadata, while the returned observation contains only cameras and robot joint state.
    """

    def __init__(
        self,
        *,
        task_name: str,
        episode_index: int = 0,
        camera_names: Sequence[str] = ("head_camera", "left_camera", "right_camera"),
        episode_length: int = 1200,
        root: str | Path | None = None,
        task_config: str = "demo_clean",
    ) -> None:
        if episode_length <= 0:
            raise ValueError("RoboTwin episode_length must be positive")
        self.task_name = task_name
        self.task_description = task_name.replace("_", " ")
        self.episode_index = episode_index
        self.camera_names = tuple(camera_names)
        self.episode_length = episode_length
        self.task_config = task_config
        self._root = Path(root).expanduser().resolve() if root is not None else None
        self._env: Any | None = None
        self._config_path: Path | None = None
        self._steps = 0
        self._ended = False
        self._ready = False

    @contextmanager
    def _runtime_directory(self) -> Iterator[None]:
        # Native RoboTwin assets and embodiment configs contain repository-relative paths.
        previous = Path.cwd()
        try:
            if self._root is not None:
                os.chdir(self._root)
            yield
        finally:
            os.chdir(previous)

    @staticmethod
    def _enable_grad() -> Any:
        try:
            import torch
        except ModuleNotFoundError:
            return nullcontext()
        return torch.enable_grad()

    def _ensure_env(self) -> None:
        if self._env is not None:
            return
        if self._root is not None and str(self._root) not in sys.path:
            sys.path.insert(0, str(self._root))
        with self._runtime_directory():
            configs = importlib.import_module("envs")
            self._config_path = Path(configs.CONFIGS_PATH).resolve()
            if self._root is None:
                self._root = self._config_path.parent
            module = importlib.import_module(f"envs.{self.task_name}")
            self._env = getattr(module, self.task_name)()

    @staticmethod
    def _read_yaml(path: Path) -> dict[str, Any]:
        with path.open(encoding="utf-8") as source:
            return yaml.safe_load(source)

    def _setup_kwargs(self) -> dict[str, Any]:
        assert self._config_path is not None and self._root is not None
        args = self._read_yaml(self._config_path / f"{self.task_config}.yml")
        embodiments = self._read_yaml(self._config_path / "_embodiment_config.yml")
        selected = args.get("embodiment", ["aloha-agilex"])
        if len(selected) == 1:
            left, right = selected[0], selected[0]
            args["dual_arm_embodied"] = True
        elif len(selected) == 3:
            left, right = selected[:2]
            args["embodiment_dis"] = selected[2]
            args["dual_arm_embodied"] = False
        else:
            raise ValueError("RoboTwin embodiment must contain one or three items")
        for side, name in (("left", left), ("right", right)):
            robot_file = Path(embodiments[name]["file_path"])
            if not robot_file.is_absolute():
                robot_file = self._root / robot_file
            args[f"{side}_robot_file"] = str(robot_file)
            args[f"{side}_embodiment_config"] = self._read_yaml(robot_file / "config.yml")
        cameras = self._read_yaml(self._config_path / "_camera_config.yml")
        head_camera = cameras[args["camera"]["head_camera_type"]]
        args.update(
            head_camera_h=head_camera["h"],
            head_camera_w=head_camera["w"],
            task_name=self.task_name,
            task_config=self.task_config,
            render_freq=0,
            eval_mode=True,
            collect_data=False,
            save_data=False,
            eval_video_save_dir=None,
        )
        return args

    def _get_obs(self) -> dict[str, Any]:
        raw = self._env.get_obs()
        # Every camera is in raw["observation"], front_camera too: the static camera low on the robot's side
        # of the table, facing the objects, that the demonstrations' cam_front was recorded from (XPolicyLab's
        # HDF5 names it cam_third_view). RoboTwin's own data_type.third_view adds a different camera as
        # raw["third_view_rgb"]: the observer, a wide view from across the table back at the robot.
        cameras = raw["observation"]
        pixels = {}
        for name in self.camera_names:
            if "rgb" not in cameras.get(name, {}):
                rendered = ", ".join(key for key, value in cameras.items() if "rgb" in value) or "none"
                raise KeyError(f"RoboTwin returned no RGB image for {name!r}; it rendered {rendered}")
            image = np.asarray(cameras[name]["rgb"], dtype=np.uint8)
            if image.ndim != 3 or image.shape[-1] < 3:
                raise ValueError(f"RoboTwin camera {name!r} must contain an HWC RGB image")
            pixels[name] = image[..., :3].copy()
        state = np.asarray(raw["joint_action"]["vector"], dtype=np.float32).reshape(-1)
        if state.size != 14 or not np.isfinite(state).all():
            raise ValueError("RoboTwin joint observation must contain 14 finite values")
        return {"pixels": pixels, "agent_pos": state.copy()}

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        self._ensure_env()
        actual_seed = self.episode_index if seed is None else seed
        args = self._setup_kwargs()
        args.update(seed=actual_seed, now_ep_num=self.episode_index, is_test=True)
        with self._runtime_directory(), self._enable_grad():
            if self._ready:
                self._env.close_env()
                # Base_Task.close() does not rebuild SAPIEN's engine. Recreate the task object
                # before the next episode to avoid reusing a closed scene or stale actors.
                self._env = type(self._env)()
            self._env.setup_demo(**args)
            raw = self._get_obs()
        self.episode_index += 1
        self._steps = 0
        self._ended = False
        self._ready = True
        return raw, {}

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if not self._ready or self._ended:
            raise RuntimeError("RoboTwin step requires an active episode; call reset()")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (14,) or not np.isfinite(action).all():
            raise ValueError("RoboTwin action must contain 14 finite joint targets")
        with self._runtime_directory(), self._enable_grad():
            self._env.take_action(action, action_type="qpos")
            self._steps += 1
            # eval_success is the simulator's own terminal signal, never an input to policy.
            success = bool(getattr(self._env, "eval_success", False))
            step_limit = getattr(self._env, "step_lim", None)
            simulator_ended = step_limit is not None and (
                getattr(self._env, "take_action_cnt", 0) >= step_limit
            )
            truncated = self._steps >= self.episode_length or simulator_ended
            if truncated and not success:
                success = bool(self._env.check_success())
            raw = self._get_obs()
        self._ended = success or truncated
        info = {"success": success, "step": self._steps} if self._ended else {}
        return raw, float(success), success, bool(truncated and not success), info

    def close(self) -> None:
        if self._env is not None:
            with self._runtime_directory():
                self._env.close_env()
            self._env = None
        self._ready = False
        self._ended = False
