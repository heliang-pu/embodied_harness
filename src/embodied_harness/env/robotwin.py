"""RoboTwin 2.0 task adapter using LeRobot's simulator wrapper.

The wrapper exposes only RGB images and the commanded 14-D joint state to the harness. RoboTwin's
own terminal verdict is retained only in ``Observation.info`` after the simulator ends an episode;
it is never placed in the supervisor predicate namespace.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ..core.types import Observation

STATE_NAMES = (
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
)
CAMERAS = ("head", "left_wrist", "right_wrist")


class RobotwinEnv:
    """Adapt ``lerobot.envs.robotwin.RoboTwinEnv`` to the harness ``Env`` protocol."""

    simulated = True

    @classmethod
    def proprio_keys(cls, params: Mapping[str, Any] | None = None) -> tuple[str, ...]:
        return (*STATE_NAMES, "state")

    @classmethod
    def camera_keys(cls, params: Mapping[str, Any] | None = None) -> tuple[str, ...]:
        params = params or {}
        camera_names = tuple(params.get("camera_names", ("head_camera", "left_camera", "right_camera")))
        return (*CAMERAS, "front") if "front_camera" in camera_names else CAMERAS

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        params = dict(params or {})
        self._task = str(params.get("task", "put_bottles_dustbin"))
        # RoboTwin's scene seed for a reset without one; run, suite and probe always pass a seed.
        self._episode_index = int(params.get("episode_index", 0))
        self._episode_length = int(params.get("episode_length", 1200))
        self._backend = str(params.get("backend", "lerobot"))
        if self._backend not in {"lerobot", "native"}:
            raise ValueError("RoboTwin backend must be 'lerobot' or 'native'")
        self._root = params.get("root")
        self._task_config = str(params.get("task_config", "demo_clean"))
        self._camera_names = tuple(params.get("camera_names", ("head_camera", "left_camera", "right_camera")))
        self._env: Any | None = None
        self._last_raw: dict[str, Any] | None = None
        self._last_observation: Observation | None = None
        self._last_action: np.ndarray | None = None
        self._instruction = self._task.replace("_", " ")
        self._steps = 0

    def _ensure_env(self) -> Any:
        if self._env is None:
            if self._backend == "native":
                from .robotwin_native import NativeRobotwinEnv

                self._env = NativeRobotwinEnv(
                    task_name=self._task,
                    episode_index=self._episode_index,
                    camera_names=self._camera_names,
                    episode_length=self._episode_length,
                    root=self._root,
                    task_config=self._task_config,
                )
                return self._env
            from lerobot.envs.robotwin import RoboTwinEnv as LeRobotRobotwinEnv

            self._env = LeRobotRobotwinEnv(
                task_name=self._task,
                episode_index=self._episode_index,
                camera_names=self._camera_names,
                episode_length=self._episode_length,
                action_mode="joint",
            )
        return self._env

    @staticmethod
    def _state(raw: Mapping[str, Any]) -> np.ndarray:
        return np.asarray(raw.get("agent_pos", np.zeros(14)), dtype=np.float32).reshape(-1)[:14]

    @staticmethod
    def _action(action: Any) -> np.ndarray:
        """Convert an XPolicyLab joint-action map to RoboTwin's 14-D order.

        Numeric vectors are already in the environment contract.  XPolicyLab's
        standard server returns action dictionaries, so this adapter performs the
        robot-specific ordering at the simulator boundary.
        """

        if not isinstance(action, Mapping):
            vector = np.asarray(action, dtype=np.float32).reshape(-1)
        else:
            fields = (
                "left_arm_joint_state",
                "left_ee_joint_state",
                "right_arm_joint_state",
                "right_ee_joint_state",
            )
            missing = [name for name in fields if name not in action]
            if missing:
                raise ValueError(f"RoboTwin XPolicy action is missing field(s): {', '.join(missing)}")
            vector = np.concatenate(
                [np.asarray(action[name], dtype=np.float32).reshape(-1) for name in fields]
            )
        if vector.size != 14:
            raise ValueError(f"RoboTwin joint action must have 14 values, got {vector.size}")
        return np.asarray(vector, dtype=np.float32)

    def _observation(
        self, raw: Mapping[str, Any], *, terminated: bool = False, info: Mapping[str, Any] | None = None
    ) -> Observation:
        pixels = raw.get("pixels", {})
        images = {
            "head": np.asarray(pixels.get("head_camera"), dtype=np.uint8),
            "left_wrist": np.asarray(pixels.get("left_camera"), dtype=np.uint8),
            "right_wrist": np.asarray(pixels.get("right_camera"), dtype=np.uint8),
        }
        if "front_camera" in self._camera_names:
            front = pixels.get("front_camera")
            if front is None:
                raise KeyError("RoboTwin requested front_camera but the simulator did not return it")
            images["front"] = np.asarray(front, dtype=np.uint8)
        state = self._state(raw)
        proprio = {name: float(state[index]) for index, name in enumerate(STATE_NAMES)}
        proprio["state"] = state.copy()
        # Keep simulator metadata out of normal observations.  The harness only allows the
        # environment's own terminal verdict to be consumed for scoring (see ``Observation``),
        # never as a planner/verifier input.  LeRobot names this field ``is_success`` while the
        # harness contract uses ``success``; normalize it at this boundary and expose it only
        # on the terminal observation.
        terminal_info: dict[str, Any] = {}
        if terminated:
            source = dict(info or {})
            success = source.get("success", source.get("is_success"))
            if success is not None:
                terminal_info["success"] = bool(success)
            status = source.get("status")
            if status is None:
                status = "success" if terminal_info.get("success") else "timeout"
            terminal_info["status"] = str(status)
            terminal_info["steps"] = int(source.get("step", self._steps))
        return Observation(
            images=images,
            proprio=proprio,
            info=terminal_info,
            terminated=terminated,
        )

    def reset(self, seed: int | None = None) -> Observation:
        env = self._ensure_env()
        raw, info = env.reset(seed=seed)
        self._last_raw = raw
        self._last_action = self._state(raw).copy()
        self._steps = 0
        self._instruction = str(getattr(env, "task_description", self._task.replace("_", " ")))
        self._last_observation = self._observation(raw, info=info)
        return self._last_observation

    def instruction(self) -> str | None:
        return self._instruction

    def observe(self) -> Observation:
        if self._last_observation is not None:
            return self._last_observation
        if self._last_raw is None:
            return self.reset()
        self._last_observation = self._observation(self._last_raw)
        return self._last_observation

    def act(self, action: Any) -> Any:
        env = self._ensure_env()
        sent_action = self._action(action)
        raw, _reward, terminated, truncated, info = env.step(sent_action)
        self._last_action = sent_action.copy()
        self._last_raw = raw
        self._steps += 1
        ended = bool(terminated or truncated)
        # The simulator verdict is available only at the terminal boundary for scoring. It is not
        # copied into proprio or used by the verifier to choose a skill.
        self._last_observation = self._observation(raw, terminated=ended, info=info)
        return sent_action

    def hold(self) -> Any:
        # RoboTwin advances its terminal checks only when a qpos action is stepped.  When the
        # supervisor is deliberately waiting for the environment verdict (for example after all
        # verified releases), repeat the last safe target instead of freezing the simulator.
        if self._env is None or self._last_action is None:
            return None
        raw, _reward, terminated, truncated, info = self._env.step(self._last_action)
        self._last_raw = raw
        self._steps += 1
        ended = bool(terminated or truncated)
        self._last_observation = self._observation(raw, terminated=ended, info=info)
        return self._last_action.copy()

    def stop(self) -> None:
        self.close()

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
        self._last_raw = None
        self._last_observation = None
        self._last_action = None
